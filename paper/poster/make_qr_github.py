#!/usr/bin/env python3
"""Generate the GitHub-themed QR code for the poster.

Encodes the repo URL with error correction H so the GitHub mark can sit in the
center without breaking decodability. Output: figures/qr_github.png (+ a plain
fallback figures/qr_github_plain.png with no logo).
"""
import sys
from pathlib import Path

import qrcode
from qrcode.constants import ERROR_CORRECT_H
from qrcode.image.styledpil import StyledPilImage
from qrcode.image.styles.moduledrawers.pil import RoundedModuleDrawer
from PIL import Image, ImageDraw

URL = "https://github.com/eduardoloz/atli-inspection"
GITHUB_DARK = (13, 17, 23)  # GitHub dark-mode background (#0d1117)
HERE = Path(__file__).parent
OUT = HERE / "figures" / "qr_github.png"
OUT_PLAIN = HERE / "figures" / "qr_github_plain.png"


def build(logo_path: Path | None) -> None:
    qr = qrcode.QRCode(error_correction=ERROR_CORRECT_H, box_size=24, border=3)
    qr.add_data(URL)
    qr.make(fit=True)

    img = qr.make_image(
        image_factory=StyledPilImage,
        module_drawer=RoundedModuleDrawer(),
        fill_color=GITHUB_DARK,
        back_color="white",
    ).convert("RGB")
    img.save(OUT_PLAIN)

    if logo_path and logo_path.exists():
        logo = Image.open(logo_path).convert("RGBA")
        # White rounded plate behind the logo so it stays legible on the QR
        plate_size = img.size[0] // 4
        logo_size = int(plate_size * 0.82)
        logo = logo.resize((logo_size, logo_size), Image.LANCZOS)
        plate = Image.new("RGBA", (plate_size, plate_size), (0, 0, 0, 0))
        d = ImageDraw.Draw(plate)
        d.rounded_rectangle([0, 0, plate_size - 1, plate_size - 1],
                            radius=plate_size // 5, fill=(255, 255, 255, 255))
        off = (plate_size - logo_size) // 2
        plate.paste(logo, (off, off), logo)
        cx = (img.size[0] - plate_size) // 2
        img.paste(plate, (cx, cx), plate)

    img.save(OUT)
    print(f"wrote {OUT} ({img.size[0]}x{img.size[1]}) encoding {URL}")


if __name__ == "__main__":
    logo = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    build(logo)
