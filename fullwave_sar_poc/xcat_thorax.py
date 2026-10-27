"""Load the XCAT thoracic slice, crop to the body, downsample to a square
label map, and map tissue labels to dielectric properties at 900 MHz.
Shared by the real-phantom SAR cartography."""
import json
import os
import numpy as np

SRC = os.path.join(os.environ.get("XCAT_DIR", "data"), "xcat_demo_thorax.json")

# tissue dielectric/mass properties @ 900 MHz (Gabriel / IT'IS database)
PROPS_900 = {                        # label -> (name, eps_r, sigma S/m, rho)
    0: ("Air",    1.0,  0.00, 1.2),
    1: ("Muscle", 55.0, 0.94, 1090),
    4: ("Air",    1.0,  0.00, 1.2),
    6: ("Lung",   22.0, 0.46, 394),
    7: ("Bone",   12.5, 0.14, 1908),
}


def load_square(n_target=150):
    d = json.load(open(SRC))
    nx, ny = d["nx"], d["ny"]
    lab = np.array(d["data"], dtype=np.int32).reshape(ny, nx)  # [row=y, col=x]
    # body bounding box (non-air)
    body = lab > 0
    ys, xs = np.where(body)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    # square crop with a small air margin, centred on the body
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    half = int(0.5 * max(y1 - y0, x1 - x0) * 1.45)
    y0, y1 = max(cy - half, 0), min(cy + half, ny)
    x0, x1 = max(cx - half, 0), min(cx + half, nx)
    crop = lab[y0:y1, x0:x1]
    side_mm = crop.shape[0] * d["dy"]                 # ~ physical side in mm
    # nearest-neighbour downsample to n_target x n_target (preserve labels)
    iy = np.linspace(0, crop.shape[0] - 1, n_target).round().astype(int)
    ix = np.linspace(0, crop.shape[1] - 1, n_target).round().astype(int)
    small = crop[np.ix_(iy, ix)]
    L = side_mm * 1e-3                                 # domain side [m]
    return small, L


def props(freq=900e6):
    return PROPS_900


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lab, L = load_square(150)
    print("label map", lab.shape, "domain", L * 100, "cm",
          "labels", np.unique(lab))
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.imshow(lab, origin="lower", cmap="tab10", extent=[0, L * 100, 0, L * 100])
    ax.set_title("XCAT thorax label map"); ax.set_xlabel("x [cm]")
    fig.savefig("xcat_labelmap_preview.png", dpi=120)
    print("saved xcat_labelmap_preview.png")
