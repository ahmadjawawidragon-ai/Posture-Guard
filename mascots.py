"""mascots.py : aset maskot Posture Guard (Sapa pink, Apin biru, Awi abu) dan logo."""
import base64
import io
from pathlib import Path

import streamlit as st
from PIL import Image

ASSETS = Path(__file__).parent / "assets"
INFO = {
    "sapa": ("Sapa", "Selalu semangat", "Energik dan ceria. Tukang ngingetin buat gerak."),
    "apin": ("Apin", "Pintar, agak pemalu", "Suka baca dan analisis. Penjelas semua materi."),
    "awi": ("Awi", "Santai aja", "Tenang dan kalem. Pengingat supaya nggak terlalu tegang."),
}


def path(name):
    return str(ASSETS / f"{name}.png")


@st.cache_data(show_spinner=False)
def _b64(name, px):
    im = Image.open(ASSETS / f"{name}.png")
    if im.width > px:
        im = im.resize((px, round(im.height * px / im.width)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "WEBP", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


def img(name, width=160, alt=None):
    """<img> tajam (dikodekan 2x lebar tampil agar HD di layar retina). Lebar dalam piksel CSS."""
    alt = alt or INFO.get(name, (name,))[0]
    return (f'<img src="data:image/webp;base64,{_b64(name, width * 2)}" alt="{alt}" '
            f'style="width:{width}px;max-width:100%;height:auto;display:block">')


def small_html(name, width=96):
    return img(name, width)


def pagehead(eyebrow, title, lead, who="sapa", tone="pk"):
    """Kepala halaman berwarna dengan maskot melayang. who: sapa | apin | awi | trio. tone: pk | bl | gr."""
    if who == "trio":
        art = "".join(f'<div class="pg-float d{i + 1}">{img(k, 112)}</div>' for i, k in enumerate(("sapa", "apin", "awi")))
        art = f'<div class="pg-trio-row">{art}</div>'
    else:
        art = f'<div class="pg-float">{img(who, 150)}</div>'
    return (f'<div class="pg-pagehead {tone}"><div class="pg-ph-text"><span class="pg-eyebrow">{eyebrow}</span>'
            f'<h1 class="pg-h1">{title}</h1><p class="pg-lead">{lead}</p></div><div class="pg-ph-art">{art}</div></div>')
