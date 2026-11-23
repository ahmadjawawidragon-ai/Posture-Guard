"""ilustrasi.py : gambar vektor (SVG) lima postur PostureGuard: TUP, TLF, TLB (tampak samping), TLL, TLR (tampak depan)."""
import base64

POSTUR = {   # kode: (label, warna badan, tampak, rotasi derajat, wajah)
    "TUP": ("Tegak", "#3FB39A", "samping", 0, "senang"),
    "TLF": ("Condong depan", "#F48BB5", "samping", 27, "cemas"),
    "TLB": ("Condong belakang", "#8E86E6", "samping", -24, "cemas"),
    "TLL": ("Condong kiri", "#72A9E8", "depan", 18, "cemas"),
    "TLR": ("Condong kanan", "#F2B35E", "depan", -18, "cemas"),
}
SKIN, SKIN_LINE, HAIR = "#FFD9B8", "#C89A74", "#5A4A42"
CHAIR, CHAIR_D, LEG, SHOE = "#B4BAC8", "#9AA1B3", "#3F5E8E", "#2F4A73"


def _face(cx, cy, front, mood):
    ex = (cx - 9, cx + 9) if front else (cx + 7, cx + 19)
    eyes = "".join(f'<circle cx="{x}" cy="{cy - 2}" r="2.8" fill="#3b2f2a"/>' for x in ex)
    mx = cx if front else cx + 13
    mouth = (f'<path d="M{mx - 5} {cy + 9} Q{mx} {cy + 14} {mx + 5} {cy + 9}" stroke="#3b2f2a" stroke-width="2" fill="none" stroke-linecap="round"/>'
             if mood == "senang" else
             f'<path d="M{mx - 4} {cy + 12} Q{mx} {cy + 8} {mx + 4} {cy + 12}" stroke="#3b2f2a" stroke-width="2" fill="none" stroke-linecap="round"/>')
    bl = (cx - 15, cx + 15) if front else (cx + 3, cx + 23)
    blush = "".join(f'<circle cx="{x}" cy="{cy + 6}" r="4.5" fill="#F48BB5" opacity=".45"/>' for x in bl)
    return eyes + blush + mouth


def _head(cx, cy, front, mood):
    hair = f'<path d="M{cx - 26} {cy - 2} Q{cx - 24} {cy - 28} {cx} {cy - 28} Q{cx + 24} {cy - 28} {cx + 26} {cy - 4} Q{cx + 12} {cy - 16} {cx} {cy - 15} Q{cx - 14} {cy - 15} {cx - 26} {cy - 2}Z" fill="{HAIR}"/>'
    return (f'<circle cx="{cx}" cy="{cy}" r="26" fill="{SKIN}" stroke="{SKIN_LINE}" stroke-width="2.5"/>' + hair + _face(cx, cy, front, mood))


def svg(kode):
    label, col, view, rot, mood = POSTUR[kode]
    s = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 240 270">',
         '<ellipse cx="120" cy="258" rx="92" ry="7" fill="#2b3242" opacity=".08"/>']
    if view == "samping":
        hx, hy = 100, 166
        s += [f'<rect x="40" y="66" width="11" height="116" rx="5.5" fill="{CHAIR}"/>', f'<rect x="40" y="172" width="112" height="12" rx="6" fill="{CHAIR}"/>',
              f'<rect x="88" y="184" width="11" height="64" rx="5.5" fill="{CHAIR_D}"/>', f'<rect x="64" y="246" width="58" height="8" rx="4" fill="{CHAIR_D}"/>',
              f'<line x1="{hx}" y1="20" x2="{hx}" y2="172" stroke="#9298A5" stroke-width="2" stroke-dasharray="4 6" opacity=".75"/>',
              f'<rect x="{hx}" y="156" width="70" height="22" rx="11" fill="{LEG}"/>', f'<rect x="154" y="156" width="22" height="88" rx="11" fill="{LEG}"/>',
              f'<rect x="154" y="236" width="40" height="13" rx="6.5" fill="{SHOE}"/>',
              f'<g transform="rotate({rot} {hx} {hy})"><rect x="{hx - 18}" y="86" width="36" height="86" rx="15" fill="{col}"/>{_head(hx, 58, False, mood)}</g>']
    else:
        hx, hy = 120, 166
        s += [f'<rect x="52" y="172" width="136" height="13" rx="6.5" fill="{CHAIR}"/>',
              f'<rect x="82" y="178" width="28" height="68" rx="12" fill="{LEG}"/>', f'<rect x="130" y="178" width="28" height="68" rx="12" fill="{LEG}"/>',
              f'<rect x="76" y="238" width="38" height="13" rx="6.5" fill="{SHOE}"/>', f'<rect x="126" y="238" width="38" height="13" rx="6.5" fill="{SHOE}"/>',
              f'<line x1="{hx}" y1="20" x2="{hx}" y2="172" stroke="#9298A5" stroke-width="2" stroke-dasharray="4 6" opacity=".75"/>',
              f'<g transform="rotate({rot} {hx} {hy})"><rect x="{hx - 26}" y="86" width="52" height="88" rx="18" fill="{col}"/>{_head(hx, 58, True, mood)}</g>']
    s.append("</svg>")
    return "".join(s)


def img(kode, width="100%"):
    b = base64.b64encode(svg(kode).encode()).decode()
    return f'<img src="data:image/svg+xml;base64,{b}" alt="Ilustrasi postur {kode}: {POSTUR[kode][0]}" style="width:{width};height:auto;display:block;margin:0 auto">'
