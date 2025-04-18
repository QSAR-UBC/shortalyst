"""Quantum just-in-time compilable Shor's algorithm with native Python control
flow and AutoGraph integration.
"""

import jax.numpy as jnp
from jax import random

import pennylane as qml

import catalyst
from catalyst import measure

catalyst.autograph_strict_conversion = True

from .utils import fractional_binary_to_float, phase_to_order, repeated_squaring
from .modexpo import QFT, fourier_adder_phase_shift, controlled_ua


@qml.qjit(autograph=True, static_argnums=(2))
def shors_algorithm(
    N, a, n_bits, n_trials, key=random.PRNGKey(0), do_skip_oflow_optim=True, do_bitwise_optim=True
):
    """Execute Shor's algorithm and return a solution.

    This function is the core of the implementation. The whole function can
    be quantum just-in-time compiled, with the current caveat that the n_bits
    parameter must be passed as a static argument due to limitations with how
    JAX handles dynamically-sized arrays during JIT compilation.

    Args:
        N (int): The number we are trying to factor. Guaranteed to be the product
            of two unique prime numbers.
        a (int): Random integer for finding a non-trivial square root.
        n_bits (int): The number of bits in N
        n_trials (int): The number of shots to take for each candidate value of a
        key (jax.random.PRNGKey): If random numbers will be generated (a=0), a key
            must also be passed in.
        do_skip_oflow_optim (bool): indicates if removal of overflow correction circuits
            should take place at runtime
        do_bitwise_optim (bool): indicates whether a-specific runtime optimizations
            that remove controlled operations should be done at runtime.

    Returns:
        int, int, float, int, jax.random.PRNGKey: If a solution is found,
        returns p, q such that N = pq. Otherwise returns 0, 0. Remaining arguments are the
        success probability of the algorithm for n_trials shots, value of a, and the updated
        key for random number generation.
    """
    # If no explicit a is passed (denoted by a = 0), randomly choose a
    # non-trivial value of a that does not have a common factor with N.
    if a == 0:
        key, subkey = random.split(key)
        a = random.randint(subkey, (1,), 2, N - 1)[0]

        while jnp.gcd(a, N) != 1:
            key, subkey = random.split(key)
            a = random.randint(subkey, (1,), 2, N - 1)[0]

    # We need 3 registers with 2n + 3 qubits total.
    # - one wire at the top which we measure and reset for QPE (0)
    # - the target wires, upon which ctrl'ed mod. expo. is applied (1, ..., n+1)
    # - a set of aux wires for mod. expo. (n+1, ..., 2n + 2)
    est_wire = 0
    target_wires = jnp.arange(n_bits) + 1
    aux_wires = jnp.arange(n_bits + 2) + n_bits + 1

    dev = qml.device("lightning.qubit", wires=2 * n_bits + 3, shots=1)

    @qml.qnode(dev)
    def run_qpe():
        # Perform QPE using a single estimation qubit. After controlling the modular
        # exponentation, the qubit is rotated based on previous measurement results,
        # measured, and reset. The measurement outcomes are used to estimate the phase.
        meas_results = jnp.zeros((n_bits,), dtype=jnp.int32)
        cumulative_phase = jnp.array(0.0)
        phase_divisors = 2.0 ** jnp.arange(n_bits + 1, 1, -1)

        # We use a couple "masks" to perform some higher-than-gate-level
        # optimizations of the controlled-Ua, depending on the power of a. One
        # is for the controlled mult of a, and the second is for the adjoint of
        # the controlled mult of the inverse of a.
        a_mask = jnp.zeros(n_bits, dtype=jnp.int64)
        a_mask = a_mask.at[0].set(1) + jnp.array(
            jnp.unpackbits(jnp.array([a]).view("uint8"), bitorder="little")[:n_bits]
        )
        a_inv_mask = a_mask

        qml.PauliX(wires=target_wires[-1])

        QFT(wires=aux_wires[:-1])

        # In the first iteration, (|0> + |1>)|1> -> |0>|1> + |1>|a>. Since a is
        # between 2 and N - 2 (inclusive), we never have overflow, so simply add
        # a - 1 using the Fourier adder.
        qml.Hadamard(wires=est_wire)

        QFT(wires=target_wires)
        qml.ctrl(fourier_adder_phase_shift, control=est_wire)(a - 1, target_wires)
        qml.adjoint(QFT)(wires=target_wires)

        qml.Hadamard(wires=est_wire)
        meas_results[0] = measure(est_wire, reset=True)
        cumulative_phase = -2 * jnp.pi * jnp.sum(meas_results / jnp.roll(phase_divisors, 1))

        # For subsequent iterations, determine powers of a, and controlled U_a when needed
        # (i.e., when the power is not equal to 1)
        # powers_cua = jnp.array([repeated_squaring(a, 2**p, N) for p in range(n_bits)])
        max_pow_a_target_reg = jnp.array([a], dtype=jnp.int64)
        next_max_pow_a_targ_reg = jnp.array([a], dtype=jnp.int64)

        pow_a_idx = 1
        pow_cua = repeated_squaring(a, 2**pow_a_idx, N)

        while pow_a_idx < n_bits and pow_cua != 1:
            # Update masks if not all the operations are being applied
            # yet. The mask for controlled M this iteration is the same as
            # the adjoint's from the last iteration, so we only need to
            # update one of them.

            # For controlled U_{a^{2^k}}, controlled mult. by a^{2^k} is on
            # a control register with a superposition of |a^j>, j = 0, 1, ..., 2^{k} - 1
            # (inclusive), and the adjoint has the powers j = 2^{k}, ..., 2^{k+1} - 1
            if not jnp.all(a_inv_mask):
                for power in range(2**pow_a_idx, 2 ** (pow_a_idx + 1)):
                    next_pow_a = jnp.array([repeated_squaring(a, power, N)])
                    # Get largest element (mod N) in superposition
                    next_max_pow_a_targ_reg = jnp.maximum(next_pow_a, max_pow_a_target_reg)
                    a_inv_mask = a_inv_mask + jnp.array(
                        jnp.unpackbits(next_pow_a.view("uint8"), bitorder="little")[:n_bits]
                    )

            qml.Hadamard(wires=est_wire)

            controlled_ua(
                N,
                pow_cua,
                est_wire,
                target_wires,
                aux_wires,
                a_mask,
                a_inv_mask,
                max_pow_a_target_reg[0],
                do_skip_oflow_optim,
                do_bitwise_optim,
            )

            # Next iteration, all powers in regular mult, but only new ones in adjoint
            a_mask = a_mask + a_inv_mask
            a_inv_mask = jnp.zeros_like(a_inv_mask)

            # Update max power in control register
            max_pow_a_target_reg = next_max_pow_a_targ_reg

            # Measure then compute corrective phase for next round
            qml.PhaseShift(cumulative_phase, wires=est_wire)
            qml.Hadamard(wires=est_wire)
            meas_results[pow_a_idx] = measure(est_wire, reset=True)
            cumulative_phase = (
                -2 * jnp.pi * jnp.sum(meas_results / jnp.roll(phase_divisors, pow_a_idx + 1))
            )

            pow_a_idx += 1
            pow_cua = repeated_squaring(a, 2**pow_a_idx, N)

        qml.adjoint(QFT)(wires=aux_wires[:-1])

        return meas_results

    p, q = jnp.array(0, dtype=jnp.int32), jnp.array(0, dtype=jnp.int32)
    successful_trials = jnp.array(0, dtype=jnp.int32)

    for _ in range(n_trials):
        sample = run_qpe()
        phase = fractional_binary_to_float(sample)
        guess_r = phase_to_order(phase, N)

        # If the guess order is even, we may have a non-trivial square root.
        # If so, try to compute p and q.
        if guess_r % 2 == 0:
            guess_square_root = repeated_squaring(a, guess_r // 2, N)

            if guess_square_root != 1 and guess_square_root != N - 1:
                candidate_p = jnp.gcd(guess_square_root - 1, N).astype(jnp.int32)

                if candidate_p != 1:
                    candidate_q = N // candidate_p
                else:
                    candidate_q = jnp.gcd(guess_square_root + 1, N).astype(jnp.int32)

                    if candidate_q != 1:
                        candidate_p = N // candidate_q

                if candidate_p * candidate_q == N:
                    p, q = candidate_p, candidate_q
                    successful_trials += 1

    return p, q, successful_trials / n_trials, a, key
