import pytest

import pennylane as qml
from pennylane import numpy as np
from jax import numpy as jnp
from collections import Counter

from shortalyst.utils import modular_inverse
from shortalyst.modexpo import (
    QFT,
    fourier_adder_phase_shift,
    doubly_controlled_adder,
    controlled_ua,
)


class TestModExpo:

    @pytest.mark.parametrize("use_qjit", [True, False])
    @pytest.mark.parametrize(
        "N,a,b",
        [
            (17, 0, 0),
            (5, 3, 0),
            (6, 5, 2),
            (6, 0, 2),
            (8, 5, 6),
            (3, 2, 2),
            (13, 11, 6),
            (28, 11, 6),
            (13, 8, 4),
            (7, 1, 6),
        ],
    )
    def test_fourier_adder(self, N, a, b, use_qjit):
        """Test that applying Fourier addition works as expected."""
        n_bits = int(jnp.floor(jnp.log2(N)) + 1)
        b_bits = [int(x) for x in np.binary_repr(b, n_bits + 1)]

        if use_qjit:
            wires = jnp.arange(n_bits + 1)
        else:
            wires = list(range(n_bits + 1))

        dev = qml.device("lightning.qubit", wires=len(wires), shots=1)

        @qml.qnode(dev)
        def add_integers():
            qml.BasisState(b_bits, wires=wires)
            QFT(wires=wires)
            fourier_adder_phase_shift(a, wires)
            qml.adjoint(QFT)(wires=wires)
            return qml.sample()

        if use_qjit:
            add_integers = qml.qjit(autograph=True)(add_integers)
            sample = add_integers()[0]
        else:
            sample = add_integers()

        res = int("".join([str(x) for x in sample]), 2)

        if a + b < 2**n_bits:
            assert int(sample[0]) == 0
        else:
            assert int(sample[0]) == 1
        assert res == a + b  # Regular adder doesn't work modulo N

    @pytest.mark.parametrize("use_qjit", [True, False])
    @pytest.mark.parametrize(
        "N,a,b",
        [
            (17, 0, 0),
            (5, 3, 0),
            (6, 5, 2),
            (6, 0, 2),
            (8, 6, 2),
            (3, 2, 2),
            (13, 10, 6),
            (7, 1, 6),
        ],
    )
    def test_reverse_fourier_adder(self, N, a, b, use_qjit):
        """Test that applying adjoint of Fourier addition works as expected."""
        n_bits = int(jnp.floor(jnp.log2(N)) + 1)
        b_bits = [int(x) for x in np.binary_repr(b, n_bits + 1)]

        if use_qjit:
            wires = jnp.arange(n_bits + 1)
        else:
            wires = list(range(n_bits + 1))

        dev = qml.device("lightning.qubit", wires=len(wires), shots=1)

        @qml.qnode(dev)
        def inverse_add_integers():
            qml.BasisState(b_bits, wires=wires)
            QFT(wires=wires)
            qml.adjoint(fourier_adder_phase_shift)(a, wires)
            qml.adjoint(QFT)(wires=wires)
            return qml.sample()

        if use_qjit:
            inverse_add_integers = qml.qjit(autograph=True)(inverse_add_integers)
            sample = inverse_add_integers()[0]
        else:
            sample = inverse_add_integers()

        res = int("".join([str(x) for x in sample]), 2)

        # If b >= a, should get |b - a> and first bit is always 0
        if b >= a:
            assert int(sample[0]) == 0
            assert res == b - a
        # If b < a, should get |2^{n+1} - (a - b)>, and most significant bit is 1
        else:
            assert int(sample[0]) == 1
            assert res == (2 ** (n_bits + 1) - (a - b))

    @pytest.mark.parametrize("use_qjit", [True, False])
    @pytest.mark.parametrize("do_control", [True, False])
    @pytest.mark.parametrize(
        "N,a,b",
        [
            (17, 0, 0),
            (5, 3, 0),
            (6, 5, 2),
            (6, 0, 2),
            (8, 5, 6),
            (3, 2, 2),
            (13, 10, 6),
            (7, 1, 6),
        ],
    )
    def test_doubly_controlled_adder(self, N, a, b, do_control, use_qjit):
        """Test that the doubly-controlled adder of Fig. 5 does nothing
        to our register when the control bits are 0 and properly uncomputes."""
        n_bits = int(jnp.floor(jnp.log2(N)) + 1)
        b_bits = [int(x) for x in np.binary_repr(b, n_bits + 1)]

        if use_qjit:
            control_wires = jnp.array([0, 1])
            wires = jnp.arange(2, 2 + n_bits + 1)
            aux_wire = 2 + n_bits + 1
        else:
            control_wires = [0, 1]
            wires = list(range(2, 2 + n_bits + 1))
            aux_wire = 2 + n_bits + 1

        # Need to install PL v0.38.1 which contains the fix for issue
        # https://github.com/PennyLaneAI/pennylane/issues/6036
        # Otherwise we need to use more than one shot for non-qjitted MCM version
        dev = qml.device("lightning.qubit", wires=len(wires) + 3, shots=1)

        @qml.qnode(dev)
        def add_integers_mod_N():
            qml.BasisState(b_bits, wires=wires)

            if do_control:
                qml.PauliX(wires=control_wires[0])
                qml.PauliX(wires=control_wires[1])

            QFT(wires=wires)
            doubly_controlled_adder(N, a, control_wires, wires, aux_wire)
            qml.adjoint(QFT)(wires=wires)

            return qml.sample(wires=dev.wires)

        if use_qjit:
            add_integers_mod_N = qml.qjit(autograph=True)(add_integers_mod_N)
            sample = add_integers_mod_N()[0]
        else:
            sample = add_integers_mod_N()

        res = int("".join([str(x) for x in sample[2 : 2 + n_bits + 1]]), 2)

        # Check status of all wires
        if do_control:
            assert np.allclose([1, 1], sample[:2])
            assert res % N == (a + b) % N
        else:
            assert np.allclose([0, 0], sample[:2])
            assert res == b

        # Doubly-controlled adder never overflows; subtracts N if the sum is greater,
        # so the overflow bit will always be 0 at the end.
        assert int(sample[2]) == 0
        assert int(sample[-1]) == 0

    @pytest.mark.parametrize("use_qjit", [True, False])
    @pytest.mark.parametrize("do_control", [True, False])
    @pytest.mark.parametrize(
        "N,a,b",
        [
            (17, 0, 0),
            (5, 3, 0),
            (6, 5, 2),
            (6, 0, 2),
            (8, 5, 6),
            (3, 2, 2),
            (13, 10, 6),
            (7, 1, 6),
        ],
    )
    def test_adjoint_negative_doubly_controlled_adder(self, N, a, b, do_control, use_qjit):
        """Test that the adjoint of the doubly controlled adder gives same results
        as applying the doubly controlled adder with -a."""
        n_bits = int(jnp.floor(jnp.log2(N)) + 1)
        b_bits = [int(x) for x in np.binary_repr(b, n_bits + 1)]

        if use_qjit:
            control_wires = jnp.array([0, 1])
            wires = jnp.arange(2, 2 + n_bits + 1)
            aux_wire = 2 + n_bits + 1
        else:
            control_wires = [0, 1]
            wires = list(range(2, 2 + n_bits + 1))
            aux_wire = 2 + n_bits + 1

        dev = qml.device("lightning.qubit", wires=len(wires) + 3, shots=1)

        @qml.qnode(dev)
        def add_integers_mod_N(use_adjoint=True):
            qml.BasisState(b_bits, wires=wires)

            if do_control:
                qml.PauliX(wires=control_wires[0])
                qml.PauliX(wires=control_wires[1])

            QFT(wires=wires)
            if use_adjoint:
                qml.adjoint(doubly_controlled_adder)(N, a, control_wires, wires, aux_wire)
            else:
                doubly_controlled_adder(N, (-a % N), control_wires, wires, aux_wire)
            qml.adjoint(QFT)(wires=wires)

            return qml.sample(wires=dev.wires)

        if use_qjit:
            add_integers_mod_N = qml.qjit(autograph=True)(add_integers_mod_N)
            sample_adjoint = add_integers_mod_N()[0]
            sample_negative = add_integers_mod_N(False)[0]
        else:
            sample_adjoint = add_integers_mod_N()
            sample_negative = add_integers_mod_N(False)

        res_adj = int("".join([str(x) for x in sample_adjoint[2 : 2 + n_bits + 1]]), 2)

        # Check sample registers consistent
        assert np.allclose(sample_adjoint, sample_negative)

        # Check status of all wires
        if do_control:
            assert np.allclose([1, 1], sample_adjoint[:2])
            assert res_adj % N == (-a + b) % N
        else:
            assert np.allclose([0, 0], sample_adjoint[:2])
            assert res_adj == b

        assert int(sample_adjoint[2]) == 0
        assert int(sample_adjoint[-1]) == 0

    @pytest.mark.parametrize("use_qjit", [True, False])
    @pytest.mark.parametrize("do_control", [True, False])
    @pytest.mark.parametrize("do_optim", [True, False])
    @pytest.mark.parametrize(
        "N,a,x", [(5, 3, 2), (5, 0, 0), (75, 31, 31), (3, 2, 2), (15, 4, 6), (19, 1, 8)]
    )
    def test_controlled_ua(self, N, a, x, do_control, use_qjit, do_optim):
        """Test that the controlled-Ua performs |x> -> |ax mod N> correctly."""

        n_bits = int(jnp.floor(jnp.log2(N)) + 1)
        x_bits = [int(bit) for bit in np.binary_repr(x, n_bits)]

        if use_qjit:
            control_wire = 0
            x_wires = jnp.arange(1, 1 + len(x_bits))
            aux_wires = jnp.arange(x_wires[-1] + 1, x_wires[-1] + 1 + n_bits + 2)
        else:
            control_wire = 0
            x_wires = list(range(1, 1 + len(x_bits)))
            aux_wires = list(range(x_wires[-1] + 1, x_wires[-1] + 1 + n_bits + 2))

        dev = qml.device("lightning.qubit", wires=1 + len(x_wires) + len(aux_wires), shots=1)

        @qml.qnode(dev)
        def test_ua():
            # Prepare initial registers
            qml.BasisState(x_bits, wires=x_wires)

            if do_control:
                qml.PauliX(wires=control_wire)

            QFT(aux_wires[:-1])
            controlled_ua(
                N,
                a,
                control_wire,
                x_wires,
                aux_wires,
                jnp.ones(n_bits),
                jnp.ones(n_bits),
                x,
                do_optim,
            )
            qml.adjoint(QFT)(aux_wires[:-1])

            return qml.sample()

        if use_qjit:
            test_ua = qml.qjit(autograph=True)(test_ua)
            sample = test_ua()[0]
        else:
            sample = test_ua()

        res = int("".join([str(sample[i]) for i in x_wires]), 2)

        # Check status of all wires
        if do_control:
            assert np.allclose([1], sample[0])
            assert res == (a * x) % N
        else:
            assert np.allclose([0], sample[0])
            assert res == x

        assert np.allclose(sample[jnp.array(aux_wires)], jnp.zeros(len(aux_wires)))

    @pytest.mark.parametrize("use_qjit", [True, False])
    @pytest.mark.parametrize("do_control", [True, False])
    @pytest.mark.parametrize("do_optim", [True, False])
    @pytest.mark.parametrize(
        "N,a,x", [(5, 3, 2), (5, 0, 0), (75, 31, 31), (3, 2, 2), (15, 4, 6), (19, 1, 8)]
    )
    def test_controlled_ua_fewer_controls(self, N, a, x, do_control, use_qjit, do_optim):
        """Test that the controlled-Ua performs |x> -> |ax mod N> correctly
        when we apply controlled operations only when the bits of x are non-zero."""

        n_bits = int(jnp.floor(jnp.log2(N)) + 1)
        x_bits = jnp.array([int(bit) for bit in np.binary_repr(x, n_bits)])
        ax_bits = jnp.array([int(bit) for bit in np.binary_repr((a * x) % N, n_bits)])

        if use_qjit:
            control_wire = 0
            x_wires = jnp.arange(1, 1 + len(x_bits))
            aux_wires = jnp.arange(x_wires[-1] + 1, x_wires[-1] + 1 + n_bits + 2)
        else:
            control_wire = 0
            x_wires = list(range(1, 1 + len(x_bits)))
            aux_wires = list(range(x_wires[-1] + 1, x_wires[-1] + 1 + n_bits + 2))

        dev = qml.device("lightning.qubit", wires=1 + len(x_wires) + len(aux_wires), shots=1)

        @qml.qnode(dev)
        def test_ua():
            # Prepare initial registers
            qml.BasisState(x_bits, wires=x_wires)

            if do_control:
                qml.PauliX(wires=control_wire)

            QFT(aux_wires[:-1])
            controlled_ua(
                N,
                a,
                control_wire,
                x_wires,
                aux_wires,
                x_bits,
                ax_bits,
                x,
                do_optim,
            )
            qml.adjoint(QFT)(aux_wires[:-1])

            return qml.sample()

        if use_qjit:
            test_ua = qml.qjit(autograph=True)(test_ua)
            sample = test_ua()[0]
        else:
            sample = test_ua()

        res = int("".join([str(sample[i]) for i in x_wires]), 2)

        # Check status of all wires
        if do_control:
            assert np.allclose([1], sample[0])
            assert res == (a * x) % N
        else:
            assert np.allclose([0], sample[0])
            assert res == x

        assert np.allclose(sample[jnp.array(aux_wires)], jnp.zeros(len(aux_wires)))

    @pytest.mark.parametrize("do_control", [True, False])
    @pytest.mark.parametrize("do_optim", [True, False])
    @pytest.mark.parametrize(
        "N,a,xs",
        [(5, 3, [1, 2]), (75, 31, [1, 17, 31, 42]), (15, 2, [1, 2, 4, 5, 6, 7, 11])],
    )
    def test_superposition_in_x_skip_underflow_check(self, N, a, xs, do_control, do_optim):
        """Test that the controlled-Ua performs |x> -> |ax mod N> correctly
        when we apply controlled operations only when the bits of x are non-zero."""
        n_bits = int(jnp.floor(jnp.log2(N)) + 1)
        x_bits = jnp.array([[int(bit) for bit in np.binary_repr(x, n_bits)] for x in xs])
        ax_bits = jnp.array([[int(bit) for bit in np.binary_repr((a * x) % N, n_bits)] for x in xs])

        # create bitwise masks
        fwd_mask = jnp.sum(x_bits, axis=0)
        inv_mask = jnp.sum(ax_bits, axis=0)

        coefs = jnp.array([1 / jnp.sqrt(len(xs))] * len(xs))

        control_wire = 0
        # required bits for x = number of bits for max value
        x_wires = list(range(1, 1 + len(x_bits[-1])))
        aux_wires = list(range(x_wires[-1] + 1, x_wires[-1] + 1 + n_bits + 2))
        work_wire = x_wires[-1] + 1 + n_bits + 2

        # Add additional wire for superposition
        num_shots = 2000
        dev = qml.device(
            "lightning.qubit",
            wires=1 + len(x_wires) + len(aux_wires) + 1,
            shots=num_shots,
        )

        @qml.qnode(dev)
        def test_ua():
            # Prepare initial registers
            qml.Superposition(
                coefs,
                x_bits,
                x_wires,
                work_wire,
            )

            if do_control:
                qml.PauliX(wires=control_wire)

            QFT(aux_wires[:-1])
            controlled_ua(
                N,
                a,
                control_wire,
                x_wires,
                aux_wires,
                fwd_mask,
                inv_mask,
                xs[-1],
                do_optim,
            )
            qml.adjoint(QFT)(aux_wires[:-1])

            return qml.sample()

        samples = test_ua()
        results = []
        for sample in samples:
            res = int("".join([str(sample[i]) for i in x_wires]), 2)
            results.append(res)
        result_counts = Counter(results)

        if do_control:
            assert np.allclose([1], sample[0])
            poss_vals = [(a * x) % N for x in xs]
            for result, count in result_counts.items():
                assert result in poss_vals
                assert count == pytest.approx(int(num_shots / len(poss_vals)), rel=0.2)
        else:
            assert np.allclose([0], sample[0])
            for result, count in result_counts.items():
                assert result in xs
                assert count == pytest.approx(int(num_shots / len(xs)), rel=0.2)

        assert np.allclose(sample[jnp.array(aux_wires)], jnp.zeros(len(aux_wires)))
