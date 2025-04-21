import pytest

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from shortalyst.utils import (
    repeated_squaring,
    modular_inverse,
    fractional_binary_to_float,
    as_integer_ratio,
    phase_to_order,
)


class TestUtils:
    @pytest.mark.parametrize(
        "a, power, N",
        [
            (2, 8, 15),
            (5, 4, 21),
            (5, 8, 21),
            (5, 16, 21),
            (17, 7, 21),
            (18, 19, 23),
            (15, 24, 323),
            (53, 28, 2937),
        ],
    )
    def test_repeated_squaring(self, a, power, N):
        """Test that jittable modular exponentiation with JAX is performed correctly."""
        power_of_a = repeated_squaring(a, power, N)
        assert power_of_a == pow(a, power, N)

    @pytest.mark.parametrize(
        "a,N",
        [(2, 5), (8, 31), (4, 17), (53, 37), (635, 137)],
    )
    def test_modular_inverse(self, a, N):
        """Test that modular inverses are correctly computed.."""
        a_inv = modular_inverse(a, N)
        assert (a * a_inv) % N == 1

    @pytest.mark.parametrize(
        "sample,expected",
        [
            ([0], 0),
            ([1], 0.5),
            ([0, 1], 0.25),
            ([1, 0, 0], 0.5),
            ([0, 0, 0, 0], 0),
            ([1, 0, 1, 0, 0], 0.625),
        ],
    )
    def test_fractional_binary_to_float(self, sample, expected):
        """Test fractional binary representation is correctly convert to phases."""
        obtained = fractional_binary_to_float(jnp.array(sample, dtype=jnp.int32))
        assert jnp.isclose(obtained, expected)

    @pytest.mark.parametrize(
        "f",
        [0.25, 0.5, 0.3, 0.65, 0.12384],
    )
    def test_as_integer_ratio(self, f):
        """Test fractional binary representation is correctly convert to phases."""
        obtained_num, obtained_denom = as_integer_ratio(jnp.array(f))
        assert jnp.isclose(jnp.array(obtained_num / obtained_denom), jnp.array(f))
