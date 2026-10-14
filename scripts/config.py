"""Konfigurasi bersama (path, daftar kolom, hyper-parameter) untuk seluruh proyek."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]   # folder root repository
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
SUBMISSION_DIR = ROOT / "submissions"
FIGURE_DIR = ROOT / "reports" / "figures"

SEED_SPLIT = 42      # seed pembagian train/validasi
SEED_OOF = 7         # seed pembagi fold out-of-fold
N_FOLDS = 10         # jumlah fold out-of-fold
NUM_CLASS = 5        # damage_grade 1..5

# kolom kategorik bangunan (teks) dan kolom ID wilayah (diperlakukan sebagai kategori)
CATS = ["land_surface_condition", "foundation_type", "roof_type", "ground_floor_type",
        "other_floor_type", "position", "plan_configuration", "superstructure"]
GEO = ["district_id", "vdcmun_id", "ward_id"]

# parameter LightGBM (multiclass)
LGB_PARAMS = dict(
    objective="multiclass", num_class=NUM_CLASS,
    learning_rate=0.06,                    # langkah belajar kecil -> lebih halus
    num_leaves=63,                         # kompleksitas tiap pohon
    min_data_in_leaf=60,                   # minimal sampel per daun -> cegah overfit
    feature_fraction=0.5,                  # tiap pohon hanya pakai 50% fitur
    bagging_fraction=0.8, bagging_freq=1,  # tiap iterasi pakai 80% baris acak
    lambda_l2=5,                           # regularisasi L2
    cat_smooth=20, max_cat_to_onehot=8,    # penanganan fitur kategorik
    verbose=-1, num_threads=1, max_bin=63,
)
MAX_ROUNDS = 1500        # iterasi maksimum (dibatasi early stopping)
EARLY_STOPPING = 40      # berhenti jika 40 iterasi tak membaik
FINAL_ITER_FACTOR = 1.1  # iterasi model akhir = 1,1 x iterasi terbaik
