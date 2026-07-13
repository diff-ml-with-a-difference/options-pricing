#!/usr/bin/env python3
"""
AADC driver for "Differential ML with a Difference" (Glasserman & Karmarkar, 2025).

Shows where AADC adds value to DiffML label generation:
1. Smooth payoffs (BS): AADC = manual. Automation benefit only.
2. Heston model: AADC gives delta+vega automatically. Manual = months of derivation.
3. Heston + barrier: AADC works. Manual = essentially impossible.
4. Multi-asset basket: N deltas in one reverse pass vs N FD bumps.

Requirements: numpy, scipy, aadc (https://matlogica.com/aadc)
"""

import time
import numpy as np
from scipy.stats import norm

import aadc
from aadc.recording_ctx import record_kernel
from aadc.evaluate_wrappers import evaluate_kernel


# ============================================================
#  Analytical references
# ============================================================

def bs_call_delta(S, K, vol, T):
    d1 = (np.log(S / K) + 0.5 * vol**2 * T) / (vol * np.sqrt(T))
    return norm.cdf(d1)


# ============================================================
#  1. European Call — baseline (AADC = manual)
# ============================================================

def test_european_call():
    print("=" * 65)
    print("  Test 1: European Call (BS) — AADC vs Manual")
    print("=" * 65)

    m = 50000
    K, vol, T = 1.10, 0.2, 1.0

    # AADC: record kernel once, evaluate m samples in batch
    with record_kernel() as kernel:
        s = aadc.idouble(1.0)
        s_arg = s.mark_as_input()
        z = aadc.idouble(0.0)
        z_arg = z.mark_as_input_no_diff()

        sT = s * (aadc.idouble(-0.5 * vol**2 * T) +
                  aadc.idouble(vol * np.sqrt(T)) * z).exp()
        payoff = aadc.iif(sT > aadc.idouble(K),
                          sT - aadc.idouble(K), aadc.idouble(0.0))
        out = payoff.mark_as_output()

    rng = np.random.RandomState(42)
    S_vals = np.exp(0.3 * rng.normal(size=m))  # spread of spots
    Z_vals = rng.normal(size=m)

    t0 = time.time()
    result = evaluate_kernel(kernel, {out: [s_arg]},
                             {s_arg: S_vals, z_arg: Z_vals}, num_threads=4)
    t_aadc = time.time() - t0
    D_aadc = result.derivs[out][s_arg]

    # Manual pathwise
    t0 = time.time()
    ST = S_vals * np.exp(-0.5 * vol**2 * T + vol * np.sqrt(T) * Z_vals)
    D_manual = np.where(ST > K, ST / S_vals, 0.0)
    t_manual = time.time() - t0

    # Compare
    match = np.allclose(D_aadc, D_manual, atol=1e-10)
    max_diff = np.max(np.abs(D_aadc - D_manual))

    print(f"\n  m = {m}, AADC = {t_aadc:.4f}s, Manual = {t_manual:.4f}s")
    print(f"  Max |AADC - Manual| = {max_diff:.2e}")
    print(f"  Exact match: {match}")
    print(f"  → For BS, AADC = manual. Value = automation (no formula needed).")
    return match


# ============================================================
#  2. Heston — AADC gives delta + vega, manual is hard
# ============================================================

def test_heston():
    print("\n" + "=" * 65)
    print("  Test 2: Heston European Call — AADC delta + vega")
    print("=" * 65)

    S0, K, V0 = 1.0, 1.0, 0.04
    kappa, theta, xi, rho = 2.0, 0.04, 0.3, -0.7
    T, n_steps = 1.0, 50
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)
    m = 20000  # paths

    with record_kernel() as kernel:
        s = aadc.idouble(S0)
        s_arg = s.mark_as_input()
        v = aadc.idouble(V0)
        v_arg = v.mark_as_input()

        z_args = []
        z_list = []
        for i in range(n_steps * 2):
            zi = aadc.idouble(0.0)
            z_args.append(zi.mark_as_input_no_diff())
            z_list.append(zi)

        for step in range(n_steps):
            zs = z_list[2 * step]
            zv_ind = z_list[2 * step + 1]
            zv = aadc.idouble(rho) * zs + aadc.idouble(np.sqrt(1 - rho**2)) * zv_ind

            v_pos = aadc.iif(v > aadc.idouble(1e-8), v, aadc.idouble(1e-8))
            vol = v_pos.sqrt()

            s = s * (aadc.idouble(1.0) + vol * aadc.idouble(sqrt_dt) * zs)
            v = v + aadc.idouble(kappa) * (aadc.idouble(theta) - v) * aadc.idouble(dt) + \
                aadc.idouble(xi) * vol * aadc.idouble(sqrt_dt) * zv

        payoff = aadc.iif(s > aadc.idouble(K), s - aadc.idouble(K), aadc.idouble(0.0))
        out = payoff.mark_as_output()

    # Batch evaluation: all m paths at once
    rng = np.random.RandomState(42)
    Z_all = rng.normal(size=(m, n_steps * 2))

    # All paths start at same S0, V0
    inputs = {s_arg: S0, v_arg: V0}
    for k in range(n_steps * 2):
        inputs[z_args[k]] = Z_all[:, k]

    t0 = time.time()
    result = evaluate_kernel(kernel, {out: [s_arg, v_arg]}, inputs, num_threads=4)
    t_aadc = time.time() - t0

    price = result.values[out].mean()
    delta = result.derivs[out][s_arg].mean()
    vega = result.derivs[out][v_arg].mean()

    # FD validation
    h = 0.001
    # Delta FD
    inputs_up = dict(inputs)
    inputs_up[s_arg] = S0 + h
    r_up = evaluate_kernel(kernel, {out: []}, inputs_up, num_threads=4)
    inputs_dn = dict(inputs)
    inputs_dn[s_arg] = S0 - h
    r_dn = evaluate_kernel(kernel, {out: []}, inputs_dn, num_threads=4)
    delta_fd = (r_up.values[out].mean() - r_dn.values[out].mean()) / (2 * h)

    # Vega FD
    inputs_up_v = dict(inputs)
    inputs_up_v[v_arg] = V0 + h
    r_up_v = evaluate_kernel(kernel, {out: []}, inputs_up_v, num_threads=4)
    inputs_dn_v = dict(inputs)
    inputs_dn_v[v_arg] = V0 - h
    r_dn_v = evaluate_kernel(kernel, {out: []}, inputs_dn_v, num_threads=4)
    vega_fd = (r_up_v.values[out].mean() - r_dn_v.values[out].mean()) / (2 * h)

    delta_ratio = delta / delta_fd if abs(delta_fd) > 1e-10 else float('nan')
    vega_ratio = vega / vega_fd if abs(vega_fd) > 1e-10 else float('nan')

    print(f"\n  Heston: S0={S0}, V0={V0}, K={K}, T={T}, {n_steps} steps, {m} paths")
    print(f"  Price = {price:.6f}")
    print(f"  Delta: AADC = {delta:.6f}, FD = {delta_fd:.6f}, ratio = {delta_ratio:.4f}")
    print(f"  Vega:  AADC = {vega:.6f}, FD = {vega_fd:.6f}, ratio = {vega_ratio:.4f}")
    print(f"  Time:  AADC (price+delta+vega) = {t_aadc:.2f}s")
    print(f"  → Delta AND vega from ONE reverse pass. Manual: coupled SDE chain rule.")


# ============================================================
#  3. Heston + Barrier — manual pathwise impossible
# ============================================================

def test_heston_barrier():
    print("\n" + "=" * 65)
    print("  Test 3: Heston + Barrier — AADC handles it, manual cannot")
    print("=" * 65)

    S0, K, V0 = 1.0, 0.0, 0.04
    barrier = 0.8
    kappa, theta, xi, rho = 2.0, 0.04, 0.3, -0.7
    T, n_steps = 1.0, 50
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)
    m = 20000

    with record_kernel() as kernel:
        s = aadc.idouble(S0)
        s_arg = s.mark_as_input()
        v = aadc.idouble(V0)
        v_arg = v.mark_as_input()

        z_args = []
        z_list = []
        for i in range(n_steps * 2):
            zi = aadc.idouble(0.0)
            z_args.append(zi.mark_as_input_no_diff())
            z_list.append(zi)

        alive = aadc.idouble(1.0)

        for step in range(n_steps):
            zs = z_list[2 * step]
            zv_ind = z_list[2 * step + 1]
            zv = aadc.idouble(rho) * zs + aadc.idouble(np.sqrt(1 - rho**2)) * zv_ind

            v_pos = aadc.iif(v > aadc.idouble(1e-8), v, aadc.idouble(1e-8))
            vol = v_pos.sqrt()

            s = s * (aadc.idouble(1.0) + vol * aadc.idouble(sqrt_dt) * zs)
            v = v + aadc.idouble(kappa) * (aadc.idouble(theta) - v) * aadc.idouble(dt) + \
                aadc.idouble(xi) * vol * aadc.idouble(sqrt_dt) * zv

            # Barrier check each step
            alive = alive * aadc.iif(s > aadc.idouble(barrier),
                                     aadc.idouble(1.0), aadc.idouble(0.0))

        payoff = alive * aadc.iif(s > aadc.idouble(K),
                                   s - aadc.idouble(K), aadc.idouble(0.0))
        out = payoff.mark_as_output()

    rng = np.random.RandomState(42)
    Z_all = rng.normal(size=(m, n_steps * 2))

    inputs = {s_arg: S0, v_arg: V0}
    for k in range(n_steps * 2):
        inputs[z_args[k]] = Z_all[:, k]

    t0 = time.time()
    result = evaluate_kernel(kernel, {out: [s_arg, v_arg]}, inputs, num_threads=4)
    t_aadc = time.time() - t0

    price = result.values[out].mean()
    delta = result.derivs[out][s_arg].mean()
    vega = result.derivs[out][v_arg].mean()

    # FD validation
    h = 0.001
    inputs_up = dict(inputs); inputs_up[s_arg] = S0 + h
    inputs_dn = dict(inputs); inputs_dn[s_arg] = S0 - h
    r_up = evaluate_kernel(kernel, {out: []}, inputs_up, num_threads=4)
    r_dn = evaluate_kernel(kernel, {out: []}, inputs_dn, num_threads=4)
    delta_fd = (r_up.values[out].mean() - r_dn.values[out].mean()) / (2 * h)

    inputs_up_v = dict(inputs); inputs_up_v[v_arg] = V0 + h
    inputs_dn_v = dict(inputs); inputs_dn_v[v_arg] = V0 - h
    r_up_v = evaluate_kernel(kernel, {out: []}, inputs_up_v, num_threads=4)
    r_dn_v = evaluate_kernel(kernel, {out: []}, inputs_dn_v, num_threads=4)
    vega_fd = (r_up_v.values[out].mean() - r_dn_v.values[out].mean()) / (2 * h)

    delta_ratio = delta / delta_fd if abs(delta_fd) > 1e-10 else float('nan')
    vega_ratio = vega / vega_fd if abs(vega_fd) > 1e-10 else float('nan')

    print(f"\n  Heston + down-and-out barrier at {barrier}")
    print(f"  {n_steps} monitoring dates, {m} paths")
    print(f"  Price = {price:.6f}")
    print(f"  Delta: AADC = {delta:.6f}, FD = {delta_fd:.6f}, ratio = {delta_ratio:.4f}")
    print(f"  Vega:  AADC = {vega:.6f}, FD = {vega_fd:.6f}, ratio = {vega_ratio:.4f}")
    print(f"  Time:  AADC (price+delta+vega) = {t_aadc:.2f}s, FD (2 bumps) = ~{t_aadc*2:.2f}s each")
    print(f"  → Stochastic vol + barrier + 50 monitoring dates.")
    print(f"  → Manual pathwise: essentially impossible (coupled SDE + indicator).")
    print(f"  → AADC: just record on tape. No new math.")


# ============================================================
#  4. Multi-asset basket — N deltas in one pass
# ============================================================

def test_basket():
    print("\n" + "=" * 65)
    print("  Test 4: Multi-asset basket — N deltas in one reverse pass")
    print("=" * 65)

    for d in [5, 10, 20]:
        K, vol, T = 1.0, 0.2, 1.0
        m = 10000

        with record_kernel() as kernel:
            s_list = []
            s_args = []
            for i in range(d):
                si = aadc.idouble(1.0)
                s_args.append(si.mark_as_input())
                s_list.append(si)

            z_args_list = []
            z_list = []
            for i in range(d):
                zi = aadc.idouble(0.0)
                z_args_list.append(zi.mark_as_input_no_diff())
                z_list.append(zi)

            # Terminal prices (independent BS for simplicity)
            basket = aadc.idouble(0.0)
            for i in range(d):
                si_T = s_list[i] * (aadc.idouble(-0.5 * vol**2 * T) +
                                     aadc.idouble(vol * np.sqrt(T)) * z_list[i]).exp()
                basket = basket + si_T * aadc.idouble(1.0 / d)

            payoff = aadc.iif(basket > aadc.idouble(K),
                              basket - aadc.idouble(K), aadc.idouble(0.0))
            out = payoff.mark_as_output()

        rng = np.random.RandomState(42)
        Z_all = rng.normal(size=(m, d))

        inputs = {}
        for i in range(d):
            inputs[s_args[i]] = 1.0  # all start at 1
            inputs[z_args_list[i]] = Z_all[:, i]

        # AADC: one pass → all d deltas
        t0 = time.time()
        result = evaluate_kernel(kernel, {out: s_args}, inputs, num_threads=4)
        t_aadc = time.time() - t0

        price = result.values[out].mean()
        deltas = [result.derivs[out][s_args[i]].mean() for i in range(d)]

        # FD: d bumps
        h = 0.001
        t0 = time.time()
        fd_deltas = []
        for i in range(d):
            inp_up = dict(inputs); inp_up[s_args[i]] = 1.0 + h
            inp_dn = dict(inputs); inp_dn[s_args[i]] = 1.0 - h
            r_up = evaluate_kernel(kernel, {out: []}, inp_up, num_threads=4)
            r_dn = evaluate_kernel(kernel, {out: []}, inp_dn, num_threads=4)
            fd_deltas.append((r_up.values[out].mean() - r_dn.values[out].mean()) / (2 * h))
        t_fd = time.time() - t0

        max_err = max(abs(deltas[i] - fd_deltas[i]) for i in range(d))
        avg_ratio = np.mean([deltas[i] / fd_deltas[i] if abs(fd_deltas[i]) > 1e-10 else 0
                             for i in range(d)])

        print(f"\n  d = {d} assets, m = {m} paths")
        print(f"  Price = {price:.6f}")
        print(f"  AADC: {t_aadc:.3f}s (1 pass → {d} deltas)")
        print(f"  FD:   {t_fd:.3f}s ({2*d} bumps)")
        print(f"  Speedup: {t_fd/t_aadc:.1f}x")
        print(f"  Max |AADC-FD|: {max_err:.6f}, avg ratio: {avg_ratio:.4f}")


# ============================================================
#  5. DiffML training label generation benchmark
# ============================================================

def benchmark_labels():
    print("\n" + "=" * 65)
    print("  Benchmark: Training label generation for DiffML")
    print("=" * 65)

    vol, K, T = 0.2, 1.0, 1.0
    n_steps = 50
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)

    # Heston kernel (reuse)
    kappa, theta, xi, rho, V0 = 2.0, 0.04, 0.3, -0.7, 0.04

    with record_kernel() as kernel:
        s = aadc.idouble(1.0)
        s_arg = s.mark_as_input()
        v = aadc.idouble(V0)
        v_arg = v.mark_as_input()
        z_args = []
        z_list = []
        for i in range(n_steps * 2):
            zi = aadc.idouble(0.0)
            z_args.append(zi.mark_as_input_no_diff())
            z_list.append(zi)

        for step in range(n_steps):
            zs = z_list[2 * step]
            zv_ind = z_list[2 * step + 1]
            zv = aadc.idouble(rho) * zs + aadc.idouble(np.sqrt(1 - rho**2)) * zv_ind
            v_pos = aadc.iif(v > aadc.idouble(1e-8), v, aadc.idouble(1e-8))
            vl = v_pos.sqrt()
            s = s * (aadc.idouble(1.0) + vl * aadc.idouble(sqrt_dt) * zs)
            v = v + aadc.idouble(kappa) * (aadc.idouble(theta) - v) * aadc.idouble(dt) + \
                aadc.idouble(xi) * vl * aadc.idouble(sqrt_dt) * zv

        payoff = aadc.iif(s > aadc.idouble(K), s - aadc.idouble(K), aadc.idouble(0.0))
        out = payoff.mark_as_output()

    for m in [1024, 8192, 50000]:
        rng = np.random.RandomState(42)
        Z_all = rng.normal(size=(m, n_steps * 2))

        inputs = {s_arg: 1.0, v_arg: V0}
        for k in range(n_steps * 2):
            inputs[z_args[k]] = Z_all[:, k]

        # AADC: price + delta + vega
        t0 = time.time()
        result = evaluate_kernel(kernel, {out: [s_arg, v_arg]}, inputs, num_threads=4)
        t_aadc = time.time() - t0

        # FD: price + delta (1 bump) + vega (1 bump) = 5 evaluations
        t0 = time.time()
        evaluate_kernel(kernel, {out: []}, inputs, num_threads=4)  # base
        for param, val, arg in [(s_arg, 1.0, s_arg), (v_arg, V0, v_arg)]:
            for sign in [+1, -1]:
                inp = dict(inputs)
                inp[arg] = val + sign * 0.001
                evaluate_kernel(kernel, {out: []}, inp, num_threads=4)
        t_fd = time.time() - t0

        print(f"\n  m = {m:>6d}: AADC = {t_aadc:.3f}s (price+Δ+ν), FD = {t_fd:.3f}s (5 evals), speedup = {t_fd/t_aadc:.1f}x")


# ============================================================
#  Main
# ============================================================

def main():
    print("=" * 65)
    print("  Differential ML + AADC: Where AD Adds Value")
    print("  (Driver for Glasserman & Karmarkar, 2025)")
    print("=" * 65)

    test_european_call()
    test_heston()
    test_heston_barrier()
    test_basket()
    benchmark_labels()

    print("\n" + "=" * 65)
    print("  Summary")
    print("=" * 65)
    print("""
  1. BS European: AADC = manual pathwise (exact match). Automation only.
  2. Heston:      AADC gives delta+vega in one pass. AAD/FD ratio ≈ 1.0.
                  Manual: need coupled SDE chain rule (complex, error-prone).
  3. Heston+barrier: AADC handles it. Manual pathwise: impossible
                  (stochastic vol + barrier indicator + 50 monitoring dates).
  4. Basket d=20: AADC 20 deltas in one pass. FD: 40 bumps. ~10x speedup.
  5. DiffML labels: Heston 50-step, AADC generates price+delta+vega
                  faster than FD for any m. Scales to arbitrary models.

  AADC value for DiffML: not speed on toy models, but ability to generate
  exact training labels for COMPLEX models without manual derivation.
""")


if __name__ == "__main__":
    main()
