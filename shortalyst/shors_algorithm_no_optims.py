"""QJIT-table Shor's algorithm with native Python control
flow and AutoGraph integration. This version does not contain 
any a- or N-specific runtime optimizations."""

import jax.numpy as jnp
from jax import random

import pennylane as qml

import catalyst
from catalyst import measure

catalyst.autograph_strict_conversion = True

from .utils import fractional_binary_to_float, phase_to_order, repeated_squaring
from .modexpo import QFT, fourier_adder_phase_shift, controlled_ua_no_optims


@qml.qjit(autograph=True, static_argnums=(2))
def shors_algorithm(N, a, n_bits, n_trials, key=random.PRNGKey(0)):
    """Execute Shor's algorithm and return a solution.

    This function is the core of the implementation. The whole function can
    be quantum just-in-time compiled, with the current caveat that the n_bits
    parameter must be passed as a static argument due to limitations with how
    JAX handles dynamically-sized arrays during JIT compilation.

    This version of the function does not contain any of the code for a- or
    N-based runtime optimizations. It is in all other regards identical to
    the version in shors_algorithm.py.

    Args:
        N (int): The number we are trying to factor. Guaranteed to be the product
            of two unique prime numbers.
        a (int): Random integer guess for finding a non-trivial square root.
        n_bits (int): The number of bits in N
        shots (int): The number of shots to take for each candidate value of a
        key (jax.random.PRNGKey): If random numbers will be generated (a=0), a key
            must also be passed in.

    Returns:
        int, int, float, int, jax.random.PRNGKey: If a solution is found,
        returns p, q such that N = pq. Otherwise returns 0, 0. Remaining arguments are the
        success probability of the algorithm for n_trials shots, value of a, and the updated
        key for random number generation.
    """
    # We need 3 registers with 2n + 3 qubits total.
    # - one wire at the top which we measure and reset for QPE (0)
    # - the target wires, upon which ctrl'ed mod. expo. is applied (1, ..., n+1)
    # - a set of aux wires for mod. expo. (n+1, ..., 2n + 2)
    if a == 0:
        key, subkey = random.split(key)
        a = random.randint(subkey, (1,), 2, N - 1)[0]

        while jnp.gcd(a, N) != 1:
            key, subkey = random.split(key)
            a = random.randint(subkey, (1,), 2, N - 1)[0]

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

        pow_a_idx = 1
        pow_cua = repeated_squaring(a, 2**pow_a_idx, N)

        while pow_a_idx < n_bits and pow_cua != 1:
            qml.Hadamard(wires=est_wire)

            controlled_ua_no_optims(
                N,
                pow_cua,
                est_wire,
                target_wires,
                aux_wires,
            )

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
