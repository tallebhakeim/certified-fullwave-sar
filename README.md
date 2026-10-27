# Computable two-sided error bounds on the SAR in full-wave dosimetry

Source code for the manuscript *Computable two-sided error bounds on the specific absorption rate in full-wave dosimetry: a goal-oriented framework validated in one, two and three dimensions* (H. Talleb, Physics in Medicine & Biology,
manuscript PMB-122013, revised version R1). Every number, table and figure of the revised
manuscript is produced by the scripts in this repository.

## What it computes

Full-wave (time-harmonic Maxwell) finite-element dosimetry with lowest-order Whitney edge
elements, a uniaxial PML and a scattered-field plane-wave source, and a goal-oriented
two-sided bracket on the absorbed power and on the peak 10 g spatial-average SAR:

    J(E_h) - 2 eta_p eta_d / gamma_-  <=  J(E)  <=  J(E_h) + 2 eta_p eta_d / gamma_- + (sigma_max/2) l^2

with residual bounds eta_p, eta_d, a lower bound gamma_- on the inf-sup constant (the
distance-to-resonance indicator) and a bound l on the L2 error.

* 1D (`sar_certified_poc/certify_rigorous.py`): equilibrated fluxes for the primal and the
  adjoint residuals, bounded remainder, exact integration across interfaces and window ends.
  On the 2 GHz skin/fat/muscle stack the bracket contains the transfer-matrix reference at
  every mesh and narrows from 231.8 % to 0.157 % (effectivity
  28-43). The psSAR10g bracket narrows to 0.65 % at lambda/128.
* 2D (`fullwave_sar_poc/certify2d.py`) and 3D (`fullwave_sar_poc/certify3d.py`,
  `certify3d_layered.py`): hierarchical residual estimates (saturation assumption). The 3D
  bracket on a heterogeneous skin/fat/muscle stack with an exact solution contains the exact
  value at every mesh (3 meshes) but is wide at these resolutions.

## Layout

    sar_certified_poc/   1D solver, exact transfer-matrix reference, brackets
    fullwave_sar_poc/    2D/3D edge-element solvers, PML, brackets, phantoms, Mie series
    revision/            scripts that produce the figures and tables of the manuscript
    results/             json output read by the manuscript
    figures/             generated figures (300 dpi)

## Reproducing

    pip install -r requirements.txt
    cd revision
    export OMP_NUM_THREADS=4          # avoid thread oversubscription
    python3 rev_bracket1d.py          # Fig. 4, inf-sup enclosure, resonance sweep (~2 min)
    python3 rev_peak10g.py            # Fig. 5, Table 4 (~30 s)
    python3 rev_pml1d.py              # Fig. 6 (~1 min)
    python3 rev_budget.py             # Table 6
    python3 rev_cartography.py        # Figs 1-2 (~1 min)
    python3 rev_certify2d.py          # Fig. 7, Table 5 (2D, ~20 min)
    (cd ../fullwave_sar_poc && python3 certify3d_layered.py)   # Table 5 (3D, long)
    python3 rev_postprocess.py        # applies the inf-sup rule to the 2D/3D brackets

The XCAT anatomical phantom (Segars et al 2010, Med. Phys. 37 4902) is distributed under
licence by its developers and is not included; `xcat_thorax.py` and `xcat3d.py` read it
from the folder given by the environment variable `XCAT_DIR`.

## Citation

DOI to be added after the first Zenodo release.

## Licence

MIT, see `LICENSE`.
