"""Utility functions for Shor implementation"""

import jax
import jax.numpy as jnp

# These functions are written based on the exercises in
# Xanadu Quantum Codebook nodes S.3 and S.4
# https://codebook.xanadu.ai


def repeated_squaring(a, exponent, N):
    """QJIT-compatible function to determine (a ** power) % N using repeated
    squaring, to prevent overflow."""
    exp_bits = jnp.array(jnp.unpackbits(jnp.array([exponent]).view("uint8"), bitorder="little"))
    total_bits_one = jnp.sum(exp_bits)

    result = jnp.array(1, dtype=jnp.int64)
    x = jnp.array(a, dtype=jnp.int64)

    idx, num_bits_added = 0, 0

    while num_bits_added < total_bits_one:
        if exp_bits[idx] == 1:
            result = (result * x) % N
            num_bits_added += 1
        x = (x**2) % N
        idx += 1

    return result


def modular_inverse(a, N):
    """QJIT compatible modular multiplicative inverse routine.

    Source: https://en.wikipedia.org/wiki/Extended_Euclidean_algorithm#Modular_integers
    """
    t = jnp.array(0, dtype=jnp.int32)
    newt = jnp.array(1, dtype=jnp.int32)
    r = jnp.array(N, dtype=jnp.int32)
    newr = jnp.array(a, dtype=jnp.int32)

    while newr != 0:
        quotient = r // newr
        t, newt = newt, t - quotient * newt
        r, newr = newr, r - quotient * newr

    if t < 0:
        t = t + N

    return t


def fractional_binary_to_float(sample):
    """Convert an n-bit sample [k1, k2, ..., kn] to a floating point
    value using fractional binary representation,

        k = (k1 / 2) + (k2 / 2 ** 2) + ... + (kn / 2 ** n)

    Args:
        sample (list[int] or array[int]): A list or array of bits, e.g.,
            the sample output of quantum circuit.

    Returns:
        float: The floating point value corresponding computed from the
        fractional binary representation.
    """
    powers_of_two = 2 ** (jnp.arange(len(sample)) + 1)
    return jnp.sum(sample / powers_of_two)


def as_integer_ratio(f):
    """QJIT compatible version of the float.as_integer_ratio() function in Python.

    Converts a floating point number to two 64-bit integers such that their quotient
    equals the input to available precision.
    """
    mantissa, exponent = jnp.frexp(f)

    i = 0
    while jnp.logical_and(i < 300, mantissa != jnp.floor(mantissa)):
        mantissa = mantissa * 2.0
        exponent = exponent - 1
        i += 1

    numerator = jnp.asarray(mantissa, dtype=jnp.int64)
    denominator = jnp.asarray(1, dtype=jnp.int64)
    abs_exponent = jnp.abs(exponent)

    if exponent > 0:
        num_to_return, denom_to_return = numerator << abs_exponent, denominator
    else:
        num_to_return, denom_to_return = numerator, denominator << abs_exponent

    return num_to_return, denom_to_return


def phase_to_order(phase, max_denominator):
    """Estimating which integer values divide to produce a float.

    Given some floating-point phase, estimate integers s, r such
    that s / r = phase, where r is no greater than some specified value.

    Uses a rewritten implementation of the Fraction.limit_denominator method from Python
    suitable for JIT compilation.

    Args:
        phase (float): Some fractional value (here, will be the output
            of running QPE).
        max_denominator (int): The largest r to be considered when looking
            for s, r such that s / r = phase.

    Returns:
        int: The estimated value of r.
    """

    numerator, denominator = as_integer_ratio(phase)

    order = 0

    if denominator <= max_denominator:
        order = denominator

    else:
        p0, q0, p1, q1 = 0, 1, 1, 0

        a = numerator // denominator
        q2 = q0 + a * q1

        while q2 < max_denominator:
            p0, q0, p1, q1 = p1, q1, p0 + a * p1, q2
            numerator, denominator = denominator, numerator - a * denominator

            a = numerator // denominator
            q2 = q0 + a * q1

        k = (max_denominator - q0) // q1
        bound1 = p0 + k * p1 / q0 + k * q1
        bound2 = p1 / q1

        loop_res = 0

        if jnp.abs(bound2 - phase) <= jnp.abs(bound1 - phase):
            loop_res = q1
        else:
            loop_res = q0 + k * q1

        order = loop_res

    return order
