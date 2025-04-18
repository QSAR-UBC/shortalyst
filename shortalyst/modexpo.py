"""Modular exponentiation circuits and necessary subroutines for
Shor's algorithm, designed to be qjit compatible"""

import jax
from jax import numpy as jnp
from .utils import modular_inverse, repeated_squaring

import pennylane as qml


def QFT(wires):
    for i in range(len(wires)):
        qml.Hadamard(wires[i])

        for j in range(len(wires) - 1 - i):
            shift = 2 * jnp.pi / (2 ** (j + 2))
            qml.ControlledPhaseShift(shift, wires=[wires[(i + 1) + j], wires[i]])


def fourier_adder_phase_shift(a, wires):
    """Adds phases on a Fourier-transformed basis state for addition.

    Used as a subroutine by other parts of the modular exponentiation circuits.
    This subroutine assumes that the input is QFT|b>, where |b> is a register
    with n = floor(log2(b)) + 1 bits (the first of which is overflow).

    After this subroutine, applying QFT^{-1} will yield the state |a + b>
    on n + 1 wires.

    Args:
        a (int): An n-bit integer to be added to a register.
        wires (Wires): The set of wires in the register we are adding to. The
            register should have n+1 wires to prevent overflow.
    """
    # Compute the phases
    n = len(wires)
    a_bits = jnp.unpackbits(jnp.array([a]).view("uint8"), bitorder="little")[:n][::-1]
    powers_of_two = 1 / (2 ** jnp.arange(1, n + 1))
    for i in range(len(wires)):
        slice_size = n - i
        phase = jnp.dot(
            jax.lax.dynamic_slice(a_bits.astype(float), [i], [slice_size]),
            jax.lax.dynamic_slice(powers_of_two, [0], [slice_size]),
        )
        if phase != 0:
            qml.PhaseShift(2 * jnp.pi * phase, wires=wires[i])


def doubly_controlled_adder(N, a, control_wires, wires, aux_wire):
    """Doubly controlled Fourier adder, Figure 5 of
    https://arxiv.org/abs/quant-ph/0205095.

    Args:
        N (int): The modulus (number we are trying to factor).
        a (int): An n-bit integer to be added to a register.
        control_wires (Wires): Two wires that this operation is being controlled on.
        wires (Wires): The set of wires in the register we are adding to. The
            register should have n+1 wires to prevent overflow, prepared in some
            basis state QFT|b>.
        aux_wire (Wires): A single wire, used as an auxiliary bit.
    """
    qml.ctrl(fourier_adder_phase_shift, control=control_wires)(a, wires)
    qml.adjoint(fourier_adder_phase_shift)(N, wires)

    qml.adjoint(QFT)(wires)
    qml.CNOT(wires=[wires[0], aux_wire])
    QFT(wires)

    qml.ctrl(fourier_adder_phase_shift, control=aux_wire)(N, wires)

    qml.adjoint(qml.ctrl(fourier_adder_phase_shift, control=control_wires))(a, wires)

    qml.adjoint(QFT)(wires)
    qml.PauliX(wires=wires[0])
    qml.CNOT(wires=[wires[0], aux_wire])
    qml.PauliX(wires=wires[0])
    QFT(wires)

    qml.ctrl(fourier_adder_phase_shift, control=control_wires)(a, wires)


def controlled_ua(
    N,
    a,
    control_wire,
    target_wires,
    aux_wires,
    mult_a_mask,
    mult_a_inv_mask,
    max_pow_a,
    do_skip_oflow_optim=True,
    do_bitwise_optim=True,
):
    """Figure 7 of https://arxiv.org/abs/quant-ph/0205095.

    This operation sends |c>|x>|0> to |c>|ax mod N>|0> if c = 1; it is the
    key controlled U_a being applied during phase estimation.

    Note that a must have an inverse modulo N for this function to work.

    Args:
        N (int): The modulus (number we are trying to factor).
        a (int): An n-bit integer to be added to a register.
        control_wire (Wires): The wire this operation is being controlled on.
        target_wires (Wires): The register |x> which should contain the results
            after the subroutine.
        aux_wires (Wires): A set of n + 2 auxiliary wires prepared in |0>
        mult_a_mask (array): an array indicating which bits in the superposition
            of powers of a can be 1; doubly-controlled adders in the M_a are applied only for
            these bit indices.
        mult_a_inv_mask (array): same as mult_a_mask, but for the adjoint of addition of the
            modular inverse of a.
        max_pow_a (int): largest power of a; used to determine whether overflow/underflow
            will occur in addition modulo N
        do_skip_oflow_optim (bool): indicates if removal of overflow correction circuits
            should take place at runtime
        do_bitwise_optim (bool): indicates whether a-specific runtime optimizations
            that remove controlled operations should be done at runtime.
    """
    n = len(target_wires)

    current_sum = 0
    skip_overflow_correction_circuit = do_skip_oflow_optim
    do_bitwise_optim = jnp.asarray(do_bitwise_optim)
    for i in range(n):
        if (mult_a_mask[n - i - 1] > 0) or (not do_bitwise_optim):
            pow_a = (a * repeated_squaring(2, i, N)) % N

            # Register always start in |0>, so we only have overflow
            # if running sum exceeds N
            if skip_overflow_correction_circuit:
                current_sum += pow_a
                if not current_sum < N:
                    skip_overflow_correction_circuit = False

            args = [
                N,
                pow_a,
                [control_wire, target_wires[n - i - 1]],
                aux_wires[:-1],
                aux_wires[-1],
            ]

            if skip_overflow_correction_circuit:
                qml.ctrl(
                    fourier_adder_phase_shift,
                    control=[control_wire, target_wires[n - i - 1]],
                )(pow_a, aux_wires[:-1])
            else:
                doubly_controlled_adder(*args)

    qml.adjoint(QFT)(wires=aux_wires[:-1])

    # Controlled SWAP the target and aux wires; note that the top-most aux wire
    # is only to catch overflow, so we ignore it here.
    for i in range(n):
        qml.CNOT(wires=[aux_wires[i + 1], target_wires[i]])
        qml.Toffoli(wires=[control_wire, target_wires[i], aux_wires[i + 1]])
        qml.CNOT(wires=[aux_wires[i + 1], target_wires[i]])

    # Adjoint of controlled multiplication with the modular inverse of a
    a_mod_inv = modular_inverse(a, N)
    QFT(wires=aux_wires[:-1])

    current_sum = jnp.int64(max_pow_a)
    skip_overflow_correction_circuit = do_skip_oflow_optim
    for i in range(n):
        if (mult_a_inv_mask[i] > 0) or (not do_bitwise_optim):
            pow_a_inv = (-a_mod_inv * repeated_squaring(2, (n - i - 1), N)) % N

            # In this case max_pow_a is the largest term possible in the register
            # so we account for that in current_sum when deciding if overflow possible
            if skip_overflow_correction_circuit:
                current_sum += pow_a_inv
                if not current_sum < N:
                    skip_overflow_correction_circuit = False

            args = [
                N,
                pow_a_inv,
                [control_wire, target_wires[i]],
                aux_wires[:-1],
                aux_wires[-1],
            ]

            if skip_overflow_correction_circuit:
                qml.ctrl(
                    fourier_adder_phase_shift,
                    control=[control_wire, target_wires[i]],
                )(pow_a_inv, aux_wires[:-1])
            else:
                doubly_controlled_adder(*args)


def controlled_ua_no_optims(
    N,
    a,
    control_wire,
    target_wires,
    aux_wires,
):
    """Figure 7 of https://arxiv.org/abs/quant-ph/0205095.

    This operation sends |c>|x>|0> to |c>|ax mod N>|0> if c = 1; it is the
    key controlled U_a being applied during phase estimation.

    Note that a must have an inverse modulo N for this function to work.

    Args:
        N (int): The modulus (number we are trying to factor).
        a (int): An n-bit integer to be added to a register.
        control_wire (Wires): The wire this operation is being controlled on.
        target_wires (Wires): The register |x> which should contain the results
            after the subroutine.
        wires (Wires): A set of n + 2 auxiliary wires prepared in |0>
    """
    n = len(target_wires)

    for i in range(n):
        pow_a = (a * repeated_squaring(2, i, N)) % N

        args = [
            N,
            pow_a,
            [control_wire, target_wires[n - i - 1]],
            aux_wires[:-1],
            aux_wires[-1],
        ]

        doubly_controlled_adder(*args)

    qml.adjoint(QFT)(wires=aux_wires[:-1])

    # Controlled SWAP the target and aux wires; note that the top-most aux wire
    # is only to catch overflow, so we ignore it here.
    for i in range(n):
        qml.CNOT(wires=[aux_wires[i + 1], target_wires[i]])
        qml.Toffoli(wires=[control_wire, target_wires[i], aux_wires[i + 1]])
        qml.CNOT(wires=[aux_wires[i + 1], target_wires[i]])

    # Adjoint of controlled multiplication with the modular inverse of a
    a_mod_inv = modular_inverse(a, N)
    QFT(wires=aux_wires[:-1])

    for i in range(n):
        pow_a_inv = (-a_mod_inv * repeated_squaring(2, (n - i - 1), N)) % N

        args = [
            N,
            pow_a_inv,
            [control_wire, target_wires[i]],
            aux_wires[:-1],
            aux_wires[-1],
        ]

        doubly_controlled_adder(*args)
