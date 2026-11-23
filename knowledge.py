"""Basis pengetahuan edukatif berbasis aturan (Tabel 2 proposal). BUKAN keluaran model dan BUKAN diagnosis medis."""

DISCLAIMER = ("Informasi ini bersifat edukatif dan bukan diagnosis medis. Kamera hanya mengamati postur, "
              "sehingga penyebab yang ditampilkan hanyalah kemungkinan umum. Konsultasikan keluhan "
              "kepada tenaga kesehatan.")

_LR = {
    "penyebab": "Bertumpu pada satu sisi atau satu siku, menyilangkan kaki, posisi meja atau layar tidak simetris.",
    "keluhan": "Ketegangan otot tidak seimbang, nyeri punggung sebelah.",
    "saran": "Posisikan layar di tengah, letakkan kaki rata di lantai, hindari bertumpu pada satu sisi.",
}

KNOWLEDGE = {
    "TLF": {
        "penyebab": "Layar terlalu rendah atau jauh, kursi dan meja tidak ergonomis, otot punggung lelah, terlalu fokus.",
        "keluhan": "Nyeri leher dan punggung atas/bawah, postur kepala maju, bungkuk postural.",
        "saran": "Naikkan layar sejajar mata, dekatkan kursi ke meja, istirahat dan peregangan tiap 30–60 menit.",
    },
    "TLB": {
        "penyebab": "Merosot di kursi, sandaran terlalu miring, tidak ada penyangga pinggang.",
        "keluhan": "Nyeri punggung bawah, tekanan berlebih pada pinggang.",
        "saran": "Duduk dengan bokong rapat ke sandaran, gunakan bantal penyangga pinggang.",
    },
    "TLL": _LR,
    "TLR": _LR,
    "TUP": {"penyebab": None, "keluhan": None,
            "saran": "Pertahankan postur dan tetap bergerak secara berkala."},
}
