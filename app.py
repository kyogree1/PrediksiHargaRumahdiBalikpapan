import warnings
from sklearn.exceptions import InconsistentVersionWarning

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)


import pandas as pd
import numpy as np
import joblib
from flask import Flask, render_template, request
import os

# =========================
# KONSTANTA DARI TRAINING
# =========================
MAE_FINAL = 120_000_000  # Prediksi ± MAE (IDR)

# Default fitur (fallback jika file kolom tidak ada)
DEFAULT_FEATURE_COLUMNS = [
    "Luas Tanah (m²)",
    "Luas Bangunan (m²)",
    "Kamar Tidur",
    "Kamar Mandi",
    "Daerah_Balikpapan Barat",
    "Daerah_Balikpapan Kota",
    "Daerah_Balikpapan Selatan",
    "Daerah_Balikpapan Tengah",
    "Daerah_Balikpapan Timur",
    "Daerah_Balikpapan Utara",
    "harga_per_m2_tanah",
    "rasio_lb_lt",
]

# =========================
# PATH MODEL & KOLOM FITUR
# (mendukung: di folder model/ atau di root)
# =========================
CANDIDATE_MODEL_PATHS = [
    "model/model_random_forest_harga_rumah.pkl",
    "model_random_forest_harga_rumah.pkl",
]

CANDIDATE_COLUMNS_PATHS = [
    "model/kolom_fitur_model1.pkl",
    "kolom_fitur_model1.pkl",
    "model/kolom_fitur_model.pkl",
    "kolom_fitur_model.pkl",
]


def pick_existing_path(candidates: list[str]) -> str | None:
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


# =========================
# LOAD MODEL
# =========================
MODEL_PATH = pick_existing_path(CANDIDATE_MODEL_PATHS)
if MODEL_PATH is None:
    print("❌ ERROR: File model tidak ditemukan. Coba taruh salah satu ini:")
    for p in CANDIDATE_MODEL_PATHS:
        print(" -", p)
    raise SystemExit(1)

try:
    model = joblib.load(MODEL_PATH)
    print(f"✅ Model berhasil dimuat dari: {MODEL_PATH}")
except Exception as e:
    print(f"❌ ERROR saat memuat model: {e}")
    raise SystemExit(1)

# =========================
# LOAD FEATURE COLUMNS
# =========================
COLUMNS_PATH = pick_existing_path(CANDIDATE_COLUMNS_PATHS)
FEATURE_COLUMNS = DEFAULT_FEATURE_COLUMNS

if COLUMNS_PATH is None:
    print("⚠️ File kolom fitur tidak ditemukan. Menggunakan kolom hardcoded (DEFAULT_FEATURE_COLUMNS).")
else:
    try:
        loaded_feature_names = joblib.load(COLUMNS_PATH)
        if isinstance(loaded_feature_names, (list, tuple)) and len(loaded_feature_names) >= 4:
            FEATURE_COLUMNS = list(loaded_feature_names)
            print(f"✅ Kolom fitur berhasil dimuat dari: {COLUMNS_PATH} ({len(FEATURE_COLUMNS)} kolom)")
        else:
            print(f"⚠️ Isi {COLUMNS_PATH} tidak valid. Menggunakan kolom hardcoded.")
    except Exception as e:
        print(f"⚠️ Gagal load kolom fitur dari {COLUMNS_PATH}: {e}")
        print("   Menggunakan kolom hardcoded (DEFAULT_FEATURE_COLUMNS).")


# =========================
# KONFIG DEFAULT FE (INFERENCE)
# =========================
# Ideal: isi dengan median harga_per_m2_tanah dari data training.
# Kalau belum ada, pakai angka realistis (mis. 12 juta/m²).
DEFAULT_HARGA_PER_M2_TANAH = 12_000_000


# =========================
# FLASK INIT
# =========================
app = Flask(__name__)

kecamatan_list = [
    "Balikpapan Kota",
    "Balikpapan Selatan",
    "Balikpapan Tengah",
    "Balikpapan Timur",
    "Balikpapan Utara",
    "Balikpapan Barat",
]


# =========================
# HELPER: buat DataFrame input sesuai fitur model
# =========================
def create_input_dataframe(kamar_tidur, kamar_mandi, luas_tanah, luas_bangunan, kecamatan):
    # Guard pembagian nol
    if luas_tanah <= 0:
        raise ValueError("Luas Tanah harus > 0.")

    rasio_lb_lt = luas_bangunan / luas_tanah

    input_data = {
        "Luas Tanah (m²)": float(luas_tanah),
        "Luas Bangunan (m²)": float(luas_bangunan),
        "Kamar Tidur": int(kamar_tidur),
        "Kamar Mandi": int(kamar_mandi),
        "rasio_lb_lt": float(rasio_lb_lt),
        # Tidak bisa dihitung dari input user (butuh Harga), jadi pakai default yang realistis
        "harga_per_m2_tanah": float(DEFAULT_HARGA_PER_M2_TANAH),
    }

    # Inisialisasi dummy daerah
    for col in FEATURE_COLUMNS:
        if col.startswith("Daerah_"):
            input_data[col] = 0

    daerah_col = f"Daerah_{kecamatan}"
    if daerah_col in FEATURE_COLUMNS:
        input_data[daerah_col] = 1

    df_input = pd.DataFrame([input_data])

    # Pastikan kolom & urutan sama persis dengan model
    df_input = df_input.reindex(columns=FEATURE_COLUMNS, fill_value=0)

    return df_input


@app.route("/")
def index():
    return render_template("index.html", kecamatan_list=kecamatan_list)


@app.route("/predict", methods=["POST"])
def predict():
    # Set default agar tidak UnboundLocalError saat error
    kamar_tidur = kamar_mandi = None
    luas_tanah = luas_bangunan = None
    kecamatan = None

    try:
        kamar_tidur = int(request.form.get("kamar_tidur", "").strip())
        kamar_mandi = int(request.form.get("kamar_mandi", "").strip())
        luas_tanah = float(request.form.get("luas_tanah", "").strip())
        luas_bangunan = float(request.form.get("luas_bangunan", "").strip())
        kecamatan = request.form.get("kecamatan", "").strip()

        if kecamatan not in kecamatan_list:
            raise ValueError("Kecamatan tidak valid.")

        # Validasi batas wajar (sesuaikan dengan data training kamu)
        if not (1 <= kamar_tidur <= 10):
            raise ValueError("Kamar Tidur harus 1–10.")
        if not (1 <= kamar_mandi <= 10):
            raise ValueError("Kamar Mandi harus 1–10.")
        if not (30 <= luas_tanah <= 2000):
            raise ValueError("Luas Tanah harus 30–2000 m².")
        if not (20 <= luas_bangunan <= 1500):
            raise ValueError("Luas Bangunan harus 20–1500 m².")
        if luas_bangunan > luas_tanah * 5:
            # optional sanity check
            raise ValueError("Luas Bangunan terlalu besar dibanding Luas Tanah (cek input).")

        fitur = create_input_dataframe(
            kamar_tidur, kamar_mandi, luas_tanah, luas_bangunan, kecamatan
        )

        # Model memprediksi log(Harga)
        log_pred = float(model.predict(fitur)[0])
        prediksi = float(np.exp(log_pred))

        bawah = max(0.0, prediksi - MAE_FINAL)
        atas = prediksi + MAE_FINAL

        return render_template(
            "index.html",
            kamar_tidur=kamar_tidur,
            kamar_mandi=kamar_mandi,
            luas_tanah=luas_tanah,
            luas_bangunan=luas_bangunan,
            kecamatan=kecamatan,
            prediksi_tengah=f"Rp{prediksi:,.0f}",
            range_harga=f"Rp{bawah:,.0f} – Rp{atas:,.0f}",
            kecamatan_list=kecamatan_list,
        )

    except Exception as e:
        return render_template(
            "index.html",
            error=str(e),
            kamar_tidur=kamar_tidur,
            kamar_mandi=kamar_mandi,
            luas_tanah=luas_tanah,
            luas_bangunan=luas_bangunan,
            kecamatan=kecamatan,
            kecamatan_list=kecamatan_list,
        )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=True)
