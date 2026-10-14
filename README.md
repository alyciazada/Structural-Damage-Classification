# Structural Damage Classification — Gempa Gorkha, Nepal 2015

Memprediksi **tingkat kerusakan bangunan (`damage_grade` 1–5)** pasca gempa Mw 7,8 Gorkha (25 April 2015) dari karakteristik struktural bangunan dan lokasinya, menggunakan data survei *Central Bureau of Statistics (CBS) Nepal*. Kompetisi Kaggle: **Structural Damage Classification** — metrik **Macro F1**.

| | |
|---|---|
| Model | LightGBM multikelas di dalam scikit-learn `Pipeline` + `ColumnTransformer` |
| Macro F1 validasi (20% train) | |
| Public / Private score Kaggle |  |
| Nama Kaggle / Nama |  |

---

## 1. Struktur repository

```
structural-damage-classification/
│
├── README.md
├── .gitignore
├── requirements.txt
│
├── data/
│   ├── .gitkeep
│   ├── train.csv              # (tidak di-commit, unduh dari Kaggle)
│   ├── test.csv               # (tidak di-commit)
│   └── sample_submission.csv  # (tidak di-commit)
│
├── notebooks/
│   ├── 01_eda.ipynb           # eksplorasi data
│   ├── 02_modeling.ipynb      # fitur, pipeline, validasi, tuning bobot kelas, training akhir -> simpan model
│   └── 03_submission.ipynb    # muat model, prediksi test, tulis submission.csv
│
├── scripts/
│   ├── __init__.py
│   ├── config.py              # path, daftar kolom, hyper-parameter
│   ├── preprocess.py          # load data, BaseFeatureBuilder, CategoryEncoder, ColumnTransformer tipe kategori
│   ├── feature_engineering.py # target encoding, fitur tetangga, out-of-fold, LocationFeatureBuilder
│   ├── train.py               # LGBMBoosterClassifier, build_pipeline, validasi, tuning bobot kelas, training akhir
│   └── predict.py             # prediksi + penulisan submission.csv (CLI)
│
├── models/
│   └── pipeline_model_lgbm.pkl  # pipeline terlatih (dibuat oleh notebook/train.py)
│
├── submissions/
│   └── submission.csv
│
└── reports/
    └── figures/               # gambar EDA, feature importance, confusion matrix
```

| Folder/File | Fungsi |
|---|---|
| `README.md` | Dokumentasi utama proyek |
| `.gitignore` | File yang tidak di-commit (`data/*.csv`, cache, dll.) |
| `requirements.txt` | Dependency Python |
| `data/` | Dataset dari Kaggle |
| `notebooks/` | Notebook eksplorasi & eksperimen |
| `scripts/` | Source code modular (`.py`, semua function/class memiliki docstring) |
| `models/` | Model tersimpan |
| `submissions/` | File submission CSV |
| `reports/figures/` | Gambar hasil analisis |

---

## 2. Cara menggunakan repository

### 2.1 Clone & instal dependency
```bash
git clone https://github.com/⟪USERNAME⟫/⟪NAMA-REPO⟫.git
cd ⟪NAMA-REPO⟫
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2.2 Unduh data dari Kaggle
Data **tidak** disimpan di GitHub (di-ignore lewat `.gitignore`). Gabung kompetisi dan terima aturannya di Kaggle terlebih dahulu, lalu:

```bash
# Opsi A — Kaggle CLI (butuh API token di ~/.kaggle/kaggle.json)
kaggle competitions download -c ⟪nama-kompetisi⟫ -p data/
unzip data/⟪nama-kompetisi⟫.zip -d data/

# Opsi B — manual: tab "Data" di halaman kompetisi → Download All → ekstrak ke folder data/
```
Pastikan `data/` berisi `train.csv`, `test.csv`, dan `sample_submission.csv`.

### 2.3 Menjalankan
**Lewat notebook** (dijalankan dari folder `notebooks/`):
```bash
cd notebooks
jupyter notebook         # jalankan berurutan (Run All): 01_eda → 02_modeling → 03_submission
```
- `01_eda.ipynb` → gambar EDA di `reports/figures/`
- `02_modeling.ipynb` → `models/pipeline_model_lgbm.pkl`, `models/cfg3.json` (bobot kelas), `feature_importance.png`, `confusion_matrix.png`
- `03_submission.ipynb` → `submissions/submission.csv`

**Lewat script** (dari root repository):
```bash
python -m scripts.train      # validasi → tuning bobot → latih akhir → simpan model + submission
python -m scripts.predict    # hanya prediksi dari model tersimpan -> submissions/submission.csv (setara 03_submission.ipynb)
```

### 2.4 Mengunggah ke Kaggle
Upload `submissions/submission.csv` (kolom `building_id`, `damage_grade` berisi angka 1–5) di tab *Submit Predictions*.

> Catatan: di Google Colab, unggah `train.csv` dan `test.csv` ke `data/`, dan jalankan notebook dari folder `notebooks/`.

---

## 3. Workflow analisis

```
Data mentah (train/test.csv)
   │
   ▼
[01_eda]  kualitas data · distribusi target · hubungan fitur ↔ kerusakan
   │
   ▼
[Fitur dasar]    umur di-clip 200 · luas & tinggi per lantai · kelompok lokasi dari building_id
   │
   ▼
[Fitur lokasi]   target encoding (berpenghalusan) · fitur tetangga umum & bersyarat — dihitung OUT-OF-FOLD (10 fold)
   │
   ▼
[Tipe kategori]  kolom kategorik + ID wilayah → dtype category (kategori dari data latih)
   │
   ▼
[LightGBM]       multikelas, early stopping pada 20% data validasi
   │
   ▼
[Tuning Macro F1] bobot per kelas dicari lewat coordinate ascent di data validasi
   │
   ▼
[Training akhir]  latih ulang pada seluruh train (1,1 × iterasi terbaik) → simpan model          (02_modeling)
   │
   ▼
[Submission]     muat model → prediksi test → validasi format → submission.csv                  (03_submission)
```

### Pipeline & ColumnTransformer
Seluruh preprocessing hingga model berada dalam satu objek `sklearn.pipeline.Pipeline` (`scripts/train.py::build_pipeline`):

| Langkah | Komponen | Peran |
|---|---|---|
| `features` | `ColumnTransformer([BaseFeatureBuilder, LocationFeatureBuilder])` | fitur dasar + fitur lokasi/tetangga |
| `encode` | `ColumnTransformer([CategoryEncoder])` | menandai kolom kategorik & wilayah sebagai `category` |
| `model` | `LGBMBoosterClassifier` | LightGBM (pembungkus `lightgbm.train`) |

Karena fitur lokasi memakai **label**, `LocationFeatureBuilder.fit_transform` mengembalikan fitur *out-of-fold* untuk data latih, sedangkan `transform` (validasi/test) memakai seluruh data latih sebagai referensi. Dengan begitu skor validasi jujur dan tidak ada kebocoran label.

---

## 4. Laporan interpretasi

### 4.1 Temuan dari EDA (angka dari data train, 533.465 bangunan)
- **Kelas tidak seimbang**: Grade 5 = 36,2%, Grade 4 = 24,1%, Grade 3 = 17,9%, Grade 2 = 11,4%, Grade 1 = 10,3%. Karena itu dipakai Macro F1 dan bobot kelas untuk menyetel keputusan.
- **Material superstruktur adalah faktor paling jelas.** Bangunan *stone* (`stone_flag`) 50,2% rusak total (Grade 5) dan *mud mortar–stone* 42,8%, sedangkan *RC engineered* hanya 0,7% Grade 5 dan 71,9% Grade 1. Material berikatan lumpur/tanpa semen jauh lebih rentan daripada beton bertulang/semen.
- **Fondasi** sejalan: fondasi *mud mortar–stone/brick* 41,7% Grade 5 (hanya 4,7% Grade 1), fondasi RC 59,5% Grade 1.
- **Jumlah lantai & umur**: rata-rata grade naik dari 2,99 (1 lantai) ke 4,09 (3 lantai), tetapi bangunan ≥4 lantai (jumlahnya sangat sedikit) kembali rendah (≈2,0–2,5) — kemungkinan karena bangunan tinggi lebih sering berupa beton bertulang. Rata-rata grade naik dengan umur (2,95 untuk 0–5 tahun → ≈4,06 untuk 41–80 tahun) lalu datar.
- **Lokasi sangat menentukan**: rata-rata grade per distrik berkisar 2,35 (distrik 31) sampai 4,68 (distrik 23) — sesuai pola guncangan yang bergantung jarak dari episentrum. Ini alasan dibuatnya fitur wilayah dan tetangga.
- Kualitas data: tidak ada nilai kosong, tidak ada `building_id` duplikat; umur memiliki nilai ekstrem (maks 999) sehingga di-*clip* di 200.

Gambar: `reports/figures/target_distribution.png`, `damage_by_superstructure.png`, `damage_by_foundation.png`, `damage_by_floors_age.png`, `damage_by_district.png`.

### 4.2 Performa model  *(isi setelah menjalankan `02_modeling.ipynb`)*
| Metrik (data validasi 20%) | Nilai |
|---|---|
| Macro F1 — `argmax` biasa | ⟪ISI: `val macroF1 raw`⟫ |
| Macro F1 — setelah bobot kelas | ⟪ISI: `val macroF1 tuned`⟫ |
| Akurasi | ⟪ISI: `acc`⟫ |
| Iterasi terbaik (early stopping) | ⟪ISI: `best`⟫ |
| Bobot kelas hasil tuning | ⟪ISI: `w`⟫ |

Interpretasi yang bisa ditulis (sesuaikan dengan angka Anda):
- Selisih Macro F1 *raw* vs *tuned* menunjukkan seberapa besar manfaat menggeser keputusan ke kelas minoritas (Grade 1–2); `argmax` condong ke kelas mayoritas (Grade 5/4).
- Akurasi biasanya lebih tinggi daripada Macro F1 karena kelas tidak seimbang dan kelas tengah (Grade 2–3) sulit dibedakan.

### 4.3 Confusion matrix & kesalahan  *(`reports/figures/confusion_matrix.png`)*
Tuliskan: kelas mana yang paling akurat (umumnya Grade 5 dan Grade 1 yang ekstrem), dan pasangan kelas yang paling sering tertukar (umumnya kelas bersebelahan, mis. 2↔3 dan 3↔4). Hal ini wajar karena `damage_grade` bersifat ordinal dan penilaian surveyor di batas antar-grade bersifat subjektif.

### 4.4 Fitur paling berpengaruh  *(`reports/figures/feature_importance.png`)*
Tuliskan 5 fitur teratas dari `imp.head(15)` dan artinya. Catat apakah fitur **lokasi/tetangga** (`nb*`, `te_*`, `c_*`) mendominasi dibanding fitur fisik bangunan (`superstructure`, `foundation_type`, `count_floors_pre_eq`, `age_building`). Jika ya, kesimpulannya: kerusakan sangat dipengaruhi lokasi dan kondisi bangunan di sekitarnya, selain material bangunan itu sendiri.

### 4.5 Keterbatasan
- Fitur tetangga memanfaatkan urutan `building_id` yang menyandikan lokasi/urutan survei; manfaatnya bergantung pada train dan test berasal dari sebaran wilayah yang sama (benar pada kompetisi ini), tetapi belum tentu untuk daerah baru.
- Bobot kelas disetel pada data validasi yang sama dengan yang dipakai early stopping, sehingga skor validasi sedikit optimistis.
- Tidak ada data kekuatan guncangan/jarak episentrum langsung; lokasi hanya ditangkap lewat ID wilayah dan tetangga.

### 4.6 Kesimpulan


---

## 5. Referensi
- Kompetisi Kaggle: Structural Damage Classification (⟪isi link⟫)
- Data asli: Nepal 2015 Gorkha Earthquake — CBS Nepal / Open Data Nepal
