import random
import jax.numpy as jnp
import sympy


from shortalyst.shors_algorithm import shors_algorithm
from shortalyst.shors_algorithm_no_optims import shors_algorithm as shors_algorithm_no_optim

random.seed(42)


scale_values = jnp.arange(2, 13)
p_values = [sympy.nextprime(2**scale) for scale in scale_values]
q_values = [sympy.nextprime(p) for p in p_values]
N_values = [p * q for p, q in zip(p_values, q_values)]


def get_valid_a_values(N, count=10):
    a_values = [2]
    tried = set()
    while len(a_values) < count:
        a = random.randint(2, N - 2)
        if a not in tried and jnp.gcd(a, N) == 1:
            a_values.append(a)
            tried.add(a)
    return a_values


results = []

for N in N_values:
    n_bits = int(jnp.floor(jnp.log2(N))) + 1
    a_vals = get_valid_a_values(N)
    for a in a_vals:
        print(f"RUNNING SHOR: N={N}, a={a}, ver={1}", flush=True)
        shors_algorithm(N, a, n_bits, 1)

        print(f"RUNNING SHOR: N={N}, a={a}, ver={0}", flush=True)
        shors_algorithm_no_optim(N, a, n_bits, 1)
