#!/usr/bin/env python3
"""strip_magenta.py — remove #FF00FF chroma-key backgrounds from PNG/JPEG images.

Usage:
    python scripts/strip_magenta.py <file_or_folder> [--tolerance 30] [--erode 0]

Options:
    --tolerance N   Colour distance from #FF00FF that counts as background.
                    Default 30 for PNG input; auto-raised to 40 for JPEG/JFIF
                    (JPEG compression scatters the key colour near edges).
    --erode N       Shave N pixels off the alpha mask edges after keying.
                    Useful for killing residual colour fringe/halo from AI gen.
                    Default 0 (no erosion).

Output:
    <original_name>_alpha.png placed next to each input file.
    Never overwrites the original; skips existing _alpha.png unless --force.
"""

import argparse
import os
import sys

try:
    from PIL import Image, ImageFilter
except ImportError:
    sys.exit("Pillow not installed.  Run: pip install pillow")


# ---------------------------------------------------------------------------
# Key colour (pure magenta)
# ---------------------------------------------------------------------------
KEY_R, KEY_G, KEY_B = 255, 0, 255


def _chroma_distance(r: int, g: int, b: int) -> float:
    """Euclidean distance in RGB space from the key colour."""
    dr = r - KEY_R
    dg = g - KEY_G
    db = b - KEY_B
    return (dr * dr + dg * dg + db * db) ** 0.5


_MAX_DIST = _chroma_distance(0, 255, 0)  # ~360 (worst case, pure green)


def strip_magenta(
    src_path: str,
    tolerance: int | None,
    erode: int,
    force: bool = False,
    feather: int | None = None,
) -> dict:
    """Process one image file.  Returns a stats dict."""

    name, ext = os.path.splitext(os.path.basename(src_path))
    ext_lower = ext.lower()

    # JPEG family: bump default tolerance
    is_jpeg = ext_lower in (".jpg", ".jpeg", ".jfif", ".jpe")
    if tolerance is None:
        effective_tol = 40 if is_jpeg else 30
    else:
        effective_tol = tolerance

    # Derived tolerance threshold as distance (0–255 component range → 0–441 dist)
    # We scale so that tolerance=30 on a 0-255 scale maps to a sensible distance.
    # Use component-wise max-deviation: pixel is BG if every component deviates
    # by no more than `effective_tol` from the key colour.
    # This is more intuitive than Euclidean for chroma-key work.
    tol = effective_tol

    dst_name = name + "_alpha.png"
    dst_path = os.path.join(os.path.dirname(src_path), dst_name)

    if os.path.exists(dst_path) and not force:
        return {
            "file": os.path.basename(src_path),
            "skipped": True,
            "reason": f"{dst_name} already exists (use --force to overwrite)",
        }

    img = Image.open(src_path).convert("RGBA")
    W, H = img.size
    data = list(img.getdata())

    # -------------------------------------------------------------------
    # Pass 1: key out full-bg pixels and soft-blend near-bg edge pixels
    # -------------------------------------------------------------------
    # For edge anti-aliasing we treat the alpha as proportional to how far
    # the pixel is from the key colour.  Full BG → alpha 0.  Art → alpha 255.
    # The transition (feather) zone spans [tol .. tol+feather] distance from the
    # key centre.  `feather` defaults to `tol` (the legacy 2*tol ramp) but can
    # be narrowed when background and art deviations are close together
    # (e.g. JPEG-scattered magenta vs a pink lid) so neither bleeds into the
    # other.
    ramp = feather if feather is not None else tol
    ramp = max(1, ramp)
    transition_half = tol          # full transparent below this
    transition_full = tol + ramp   # full opaque above this

    keyed_pixels = 0
    new_data = []
    for r, g, b, a_orig in data:
        # Max component deviation from key (intuitive; handles JPEG scatter)
        dev = max(abs(r - KEY_R), abs(g - KEY_G), abs(b - KEY_B))
        if dev <= transition_half:
            new_data.append((r, g, b, 0))
            keyed_pixels += 1
        elif dev <= transition_full:
            # Smooth ramp: 0 at transition_half, 255 at transition_full
            alpha = int(255 * (dev - transition_half) / ramp)
            new_data.append((r, g, b, alpha))
            keyed_pixels += 1  # counts partial as "touched"
        else:
            new_data.append((r, g, b, a_orig))

    img.putdata(new_data)

    # -------------------------------------------------------------------
    # Pass 2: erosion (optional) — shave N px off the alpha mask
    # -------------------------------------------------------------------
    if erode > 0:
        # Use the alpha channel as a mask and erode it
        alpha_ch = img.split()[3]
        for _ in range(erode):
            alpha_ch = alpha_ch.filter(ImageFilter.MinFilter(3))
        r_ch, g_ch, b_ch, _ = img.split()
        img = Image.merge("RGBA", (r_ch, g_ch, b_ch, alpha_ch))

    # -------------------------------------------------------------------
    # Compute bounding box of remaining art (non-transparent region)
    # -------------------------------------------------------------------
    alpha_ch = img.split()[3]
    bbox = alpha_ch.getbbox()  # (left, top, right, bottom) or None

    img.save(dst_path, format="PNG", optimize=False)

    return {
        "file": os.path.basename(src_path),
        "output": dst_name,
        "size": f"{W}x{H}",
        "tolerance": effective_tol,
        "erode": erode,
        "keyed_pixels": keyed_pixels,
        "art_bbox": bbox,
        "skipped": False,
    }


def _collect_images(path: str) -> list[str]:
    exts = {".png", ".jpg", ".jpeg", ".jfif", ".jpe"}
    if os.path.isfile(path):
        if os.path.splitext(path)[1].lower() in exts:
            return [path]
        return []
    # folder
    out = []
    for fn in sorted(os.listdir(path)):
        if os.path.splitext(fn)[1].lower() in exts:
            # Skip already-processed alpha outputs
            if fn.endswith("_alpha.png"):
                continue
            out.append(os.path.join(path, fn))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Strip #FF00FF magenta backgrounds, output <name>_alpha.png"
    )
    ap.add_argument("path", help="Image file or folder to process")
    ap.add_argument(
        "--tolerance", type=int, default=None,
        help="Max component deviation from #FF00FF to count as background "
             "(default 30 for PNG, 40 for JPEG/JFIF)",
    )
    ap.add_argument(
        "--erode", type=int, default=0,
        help="Pixels to erode from alpha edge after keying (default 0)",
    )
    ap.add_argument(
        "--feather", type=int, default=None,
        help="Width (in colour-deviation units) of the transparent→opaque "
             "anti-alias ramp above the tolerance threshold. Defaults to "
             "`tolerance` (legacy 2x ramp). Narrow it (e.g. 15) when the "
             "background colour sits close to the art colour.",
    )
    ap.add_argument(
        "--force", action="store_true",
        help="Overwrite existing _alpha.png outputs",
    )
    args = ap.parse_args()

    targets = _collect_images(args.path)
    if not targets:
        sys.exit(f"No supported image files found at: {args.path}")

    print(f"strip_magenta — processing {len(targets)} file(s)")
    print(f"  tolerance : {args.tolerance if args.tolerance is not None else 'auto'}")
    print(f"  erode     : {args.erode}")
    print()

    for src in targets:
        result = strip_magenta(src, args.tolerance, args.erode,
                               force=args.force, feather=args.feather)
        if result.get("skipped"):
            print(f"  SKIP  {result['file']}  — {result['reason']}")
        else:
            bbox_str = str(result["art_bbox"]) if result["art_bbox"] else "none (all transparent!)"
            print(
                f"  OK    {result['file']}  →  {result['output']}\n"
                f"        {result['size']}  tol={result['tolerance']}  "
                f"erode={result['erode']}  keyed={result['keyed_pixels']}px  "
                f"art_bbox={bbox_str}"
            )

    print("\nDone.")


if __name__ == "__main__":
    main()
