"""
Generate desk_foreground.png from scene_kiracam.jfif.

Cuts the desk surface (bottom ~40%) into a transparent PNG for OBS layering
on top of the snow canvas.  Snow falls behind the desk; the desk occludes it.

Edge detection method: maximum |d(lum)/dy| per column within a 60px search
band centred on the visually-observed desk back edge (y≈0.577 in the 572px
image).  The result is smoothed with a horizontal median filter, then a 4px
generous buffer is taken above the edge and 8px of alpha feather added.
"""
from PIL import Image
import numpy as np

IMG_PATH      = r"web_dashboard\screens\scene_kiracam.jfif"
OUT_PATH      = r"web_dashboard\screens\desk_foreground.png"
BASELINE_FRAC = 0.577   # row ≈ 330 in 572px image (visually measured)
SEARCH_BAND   = 60       # ± pixels around baseline
BUFFER_PX     = 4
FEATHER_PX    = 8

img = Image.open(IMG_PATH).convert("RGBA")
W, H = img.size
print(f"Image: {W}×{H}")

arr = np.array(img, dtype=np.int32)
lum = (arr[:,:,0]*299 + arr[:,:,1]*587 + arr[:,:,2]*114) // 1000

baseline = int(BASELINE_FRAC * H)
band_top  = max(0, baseline - SEARCH_BAND)
band_bot  = min(H - 1, baseline + SEARCH_BAND)
print(f"Search band: y={band_top}–{band_bot}  baseline={baseline}")

edge_y = np.full(W, baseline, dtype=np.int32)
for x in range(W):
    col  = lum[band_top:band_bot, x].astype(float)
    grad = np.abs(np.diff(col))
    if grad.max() >= 8:
        edge_y[x] = band_top + int(np.argmax(grad))

try:
    from scipy.ndimage import median_filter
    edge_smooth = median_filter(edge_y.astype(float), size=51).astype(np.int32)
except ImportError:
    edge_smooth = edge_y.copy()
    hw = 25
    for x in range(W):
        lo, hi = max(0, x - hw), min(W, x + hw + 1)
        edge_smooth[x] = int(np.median(edge_y[lo:hi]))

print(f"Desk edge range: y={edge_smooth.min()}–{edge_smooth.max()}  "
      f"({edge_smooth.min()/H:.3f}–{edge_smooth.max()/H:.3f})")

alpha = np.zeros((H, W), dtype=np.uint8)
for x in range(W):
    e = edge_smooth[x] - BUFFER_PX
    alpha[e:, x] = 255
    for dy in range(FEATHER_PX):
        y = e - FEATHER_PX + dy
        if 0 <= y < H:
            alpha[y, x] = int(255 * dy / FEATHER_PX)

out = np.array(img, dtype=np.uint8)
out[:, :, 3] = alpha
Image.fromarray(out, mode="RGBA").save(OUT_PATH, format="PNG")

pct = 100 * int((alpha == 255).sum()) / (W * H)
print(f"Saved {OUT_PATH}  ({pct:.1f}% opaque)")
print("Done.")
