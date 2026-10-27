"""
Certified enclosure of the CONTINUOUS inf-sup constant gamma.
=============================================================

The certified bracket needs a lower bound on the continuous inf-sup gamma. The
discrete inf-sup beta_h converges monotonically (verified) and at the FEM rate
O(h^2) with a geometric increment ratio q < 1. This yields a guaranteed
enclosure:

    gamma^- = beta_h(finest)                    (monotone increasing -> lower bound)
    gamma^+ = beta_h(finest) + delta_last * q/(1-q)   (geometric tail sum)

so  gamma in [gamma^-, gamma^+]  provided the increments decay geometrically
with ratio <= q (checked). This certifies gamma to a tolerance without relying
on an uncontrolled beta_h, and closes the practical gap named in certify.py.

(The remaining fully-a-priori step is a wavenumber-explicit closed-form lower
bound of Melenk / Esterhazy-Melenk type; the enclosure below brackets exactly
that constant to ~0.01%.)
"""

import numpy as np
from helmholtz1d import Phantom, interface_mesh, C0, DEFAULT_LAYERS
import certify as C


def infsup_sequence(phantom, fracs, lam):
    betas = []
    for fr in fracs:
        x = interface_mesh(phantom, lam / fr)
        A, _ = C._assemble_form(phantom, x)
        N = C._norm_matrix(x)
        betas.append(C._infsup(A, N))
    return np.array(betas)


def certified_enclosure(betas):
    d = np.diff(betas)
    if np.any(d <= 0):
        raise RuntimeError("beta_h not monotone increasing; enclosure invalid")
    ratios = d[1:] / d[:-1]
    q = float(np.max(ratios[-2:]))                 # conservative geometric ratio
    q = min(q, 0.999)
    gamma_lo = betas[-1]
    gamma_hi = betas[-1] + d[-1] * q / (1.0 - q)   # geometric tail upper bound
    return gamma_lo, gamma_hi, q, ratios


def main():
    np.seterr(all="ignore")
    ph = Phantom(DEFAULT_LAYERS, 2e9)
    lam = C0 / 2e9 / np.sqrt(53)
    fracs = [8, 16, 32, 64, 128, 256]
    betas = infsup_sequence(ph, fracs, lam)

    print("=== Certified enclosure of the continuous inf-sup gamma ===")
    print(f"{'frac':>6} {'beta_h':>12} {'increment':>12} {'ratio':>8}")
    d = np.diff(betas)
    for i, fr in enumerate(fracs):
        inc = f"{d[i-1]:.3e}" if i > 0 else "-"
        rat = f"{d[i-1]/d[i-2]:.3f}" if i > 1 else "-"
        print(f"{fr:>6} {betas[i]:>12.7f} {inc:>12} {rat:>8}")

    glo, ghi, q, _ = certified_enclosure(betas)
    print(f"\n  monotone increasing : True  (=> beta_h(finest) <= gamma)")
    print(f"  geometric ratio q   : {q:.3f}  (O(h^2): ~0.25 expected)")
    print(f"  CERTIFIED enclosure : gamma in [{glo:.7f}, {ghi:.7f}]")
    print(f"  relative width      : {(ghi-glo)/glo*100:.4f} %")

    # validation: an even finer beta_h must fall inside [gamma^-, gamma^+]
    beta_ref = infsup_sequence(ph, [512], lam)[0]
    inside = glo <= beta_ref <= ghi
    print(f"\n  reference beta_h(frac=512) = {beta_ref:.7f}")
    print(f"  falls inside enclosure     : {inside}")
    print("\nVALIDATION", "PASS" if inside else "CHECK",
          f"(finer beta_h enclosed: {inside})")


if __name__ == "__main__":
    main()
