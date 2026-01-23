from pathlib import Path
from PIL import Image
import icnsutil

icns_path = Path("assets/CopyCat.icns")
out_ico = Path("assets/copycat.ico")

icns = icnsutil.IcnsFile(icns_path)
largest = max(icns.icons, key=lambda i: i.size[0])

img = Image.open(largest.open()).convert("RGBA")
sizes = [(256,256),(128,128),(64,64),(48,48),(32,32),(16,16)]
img.save(out_ico, format="ICO", sizes=sizes)

print(f"Wrote {out_ico.resolve()}")
