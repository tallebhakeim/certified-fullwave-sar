"""Common paths for the PMB-122013 revision scripts (figures + results)."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
R1 = os.path.dirname(HERE)
FIG = os.path.join(R1, "figures")
RES = os.path.join(R1, "results")
CODE = R1                       # repository root
sys.path.insert(0, os.path.join(CODE, "sar_certified_poc"))
sys.path.insert(0, os.path.join(CODE, "fullwave_sar_poc"))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.titlesize": 9,
                     "axes.labelsize": 9, "legend.fontsize": 8, "figure.dpi": 150,
                     "savefig.dpi": 300, "mathtext.fontset": "dejavuserif"})
ETA0 = 376.730313668
