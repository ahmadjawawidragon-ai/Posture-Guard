"""app.py : router utama PostureGuard.  Jalankan:  streamlit run app.py

Layanan utama : Deteksi Kamera AI (deteksi.py) dan Evaluasi Model
Pendamping    : halaman edukasi (edukasi.py)
Butuh Streamlit >= 1.40."""
import streamlit as st

import edukasi as ed

st.set_page_config(page_title="PostureGuard", page_icon=":material/accessibility_new:", layout="wide")
st.markdown(ed.CSS, unsafe_allow_html=True)

P = {
    "beranda": st.Page(ed.beranda, title="Beranda", icon=":material/home:", url_path="beranda", default=True),
    "deteksi": st.Page("deteksi.py", title="Deteksi Kamera AI", icon=":material/videocam:", url_path="deteksi"),
    "evaluasi": st.Page(ed.evaluasi, title="Evaluasi Model", icon=":material/analytics:", url_path="evaluasi"),
    "tentang": st.Page(ed.tentang_postur, title="Tentang Postur", icon=":material/accessibility_new:", url_path="tentang-postur"),
    "panduan": st.Page(ed.panduan, title="Panduan", icon=":material/menu_book:", url_path="panduan"),
    "latihan": st.Page(ed.latihan, title="Latihan", icon=":material/self_improvement:", url_path="latihan"),
    "kuis": st.Page(ed.kuis, title="Kuis", icon=":material/quiz:", url_path="kuis"),
    "kami": st.Page(ed.tentang_kami, title="Tentang Kami", icon=":material/info:", url_path="tentang-kami"),
}
ed.PAGES.update(P)

pg = st.navigation({
    "Layanan Utama": [P["beranda"], P["deteksi"], P["evaluasi"]],
    "Belajar": [P["tentang"], P["panduan"], P["latihan"], P["kuis"], P["kami"]],
})
pg.run()
