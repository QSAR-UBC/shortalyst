"""Integration test for qjitted Shor's algo."""

import pytest

import jax.numpy as jnp
from jax import random


class TestShor:
    @pytest.mark.parametrize("N", [15, 21, 33, 65, 143])
    def test_shors_algorithm(self, N):
        """Test that the qjitted Shor routine correctly computes prime factors."""
        from shortalyst.shors_algorithm import shors_algorithm

        trials = 100
        n_bits = int(jnp.floor(jnp.log2(N)) + 1)

        a_choices = jnp.array(list(range(2, N - 1)))
        non_trivial_a = []

        key = random.PRNGKey(123456789)
        num_a = 3

        while len(non_trivial_a) < num_a:
            key, subkey = random.split(key)
            a = random.choice(subkey, a_choices)

            if jnp.gcd(a, N) == 1 and a not in non_trivial_a:
                non_trivial_a.append(a)

        for a in non_trivial_a:
            p, q, _, _, _ = shors_algorithm(N, a, n_bits, trials)
            if p * q == N:
                break

        assert int(p * q) == N
