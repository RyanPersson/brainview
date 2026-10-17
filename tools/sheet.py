"""Tile screenshots into one contact sheet:  python tools/sheet.py out/sheet.png out/a.png out/b.png ...
Crops away the side panel (x < 330) unless SHEET_FULL=1."""
import os, sys
from PIL import Image
out, files = sys.argv[1], sys.argv[2:]
ims = [Image.open(f).convert("RGB") for f in files]
if not os.environ.get("SHEET_FULL"):
    ims = [im.crop((330, 0, im.width, im.height)) for im in ims]
scale = float(os.environ.get("SHEET_SCALE", "0.5"))
ims = [im.resize((int(im.width * scale), int(im.height * scale))) for im in ims]
cols = 2 if len(ims) > 1 else 1
rows = (len(ims) + cols - 1) // cols
w, h = ims[0].size
sheet = Image.new("RGB", (w * cols, h * rows), (40, 40, 40))
for i, im in enumerate(ims):
    sheet.paste(im, ((i % cols) * w, (i // cols) * h))
sheet.save(out)
