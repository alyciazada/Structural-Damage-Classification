"""Pemuatan data, pembuatan label, fitur dasar, dan ColumnTransformer untuk tipe kategori."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer

from .config import CATS, GEO, DATA_DIR


def load_data(data_dir=DATA_DIR):
    """Baca ``train.csv`` dan ``test.csv``; tambahkan kolom ``y`` (0..4) pada train.

    Parameters
    ----------
    data_dir : path-like
        Folder berisi train.csv dan test.csv.

    Returns
    -------
    tuple of pd.DataFrame
        ``(train, test)``. Label ``'Grade 3'`` diubah menjadi angka 2 (geser 0..4,
        syarat LightGBM).
    """
    tr = pd.read_csv(f"{data_dir}/train.csv")
    te = pd.read_csv(f"{data_dir}/test.csv")
    tr["y"] = tr["damage_grade"].str[-1].astype(int) - 1
    return tr, te


def all_columns(X):
    """Selector kolom untuk ColumnTransformer: pilih seluruh kolom ``X``.

    Dibuat sebagai fungsi bernama (bukan lambda) supaya pipeline bisa di-pickle.
    """
    return list(X.columns)


class BaseFeatureBuilder(BaseEstimator, TransformerMixin):
    """Fitur dasar: buang ID/target, clip umur, rasio bangunan, dan kelompok lokasi dari ID.

    Parameters
    ----------
    all_ids : pd.Series, optional
        Seluruh ``building_id`` (train + test) untuk menghitung jumlah bangunan per
        kelompok. Tidak memakai label sehingga aman. Jika ``None`` dipakai ID dari data
        yang sedang ditransformasi.
    """

    def __init__(self, all_ids=None):
        """Simpan ``all_ids`` (lihat parameter kelas)."""
        self.all_ids = all_ids

    def fit(self, X, y=None):
        """Tidak ada yang dipelajari; mengembalikan ``self``."""
        return self

    def transform(self, d):
        """Hitung fitur dasar untuk DataFrame ``d`` dan kembalikan DataFrame baru."""
        allids = d["building_id"] if self.all_ids is None else self.all_ids
        X = d.drop(columns=["building_id", "damage_grade", "y"], errors="ignore").copy()
        X["age_building"] = X["age_building"].clip(upper=200)                       # umur ekstrem dipotong
        X["area_per_floor"] = X["plinth_area_sq_ft"] / X["count_floors_pre_eq"].clip(lower=1)
        X["height_per_floor"] = X["height_ft_pre_eq"] / X["count_floors_pre_eq"].clip(lower=1)
        X["id_loc"] = d["building_id"] // 1000     # bagian "lokasi" dari ID
        X["id_seq"] = d["building_id"] % 1000      # bagian "urutan" dari ID
        for nm, dv in [("g8", 10**8), ("g6", 10**6), ("g4", 10**4), ("g3", 10**3)]:
            g = d["building_id"] // dv
            X["n_" + nm] = g.map((allids // dv).value_counts())   # jumlah bangunan per kelompok
        X = X.reset_index(drop=True)
        self.names_out_ = list(X.columns)
        return X

    def get_feature_names_out(self, input_features=None):
        """Nama kolom keluaran (tersedia setelah ``transform`` pertama dipanggil)."""
        return np.array(self.names_out_)


class CategoryEncoder(BaseEstimator, TransformerMixin):
    """Ubah kolom tertentu menjadi dtype ``category`` dengan daftar kategori dari data latih.

    Kategori dipelajari saat ``fit`` sehingga data validasi/test memakai daftar yang
    sama (nilai tak dikenal menjadi NaN), persis seperti yang dibutuhkan LightGBM.
    Kolom lain dibiarkan dan urutan kolom dipertahankan.

    Parameters
    ----------
    columns : list of str
        Kolom yang dijadikan kategori.
    """

    def __init__(self, columns=None):
        """Simpan daftar kolom yang akan dijadikan kategori."""
        self.columns = columns

    def fit(self, X, y=None):
        """Simpan daftar kategori tiap kolom dari data latih."""
        self.categories_ = {c: X[c].astype("category").cat.categories for c in self.columns}
        self.names_out_ = list(X.columns)
        return self

    def transform(self, X):
        """Terapkan dtype category dengan kategori hasil ``fit``."""
        out = X.copy()
        for c in self.columns:
            out[c] = pd.Categorical(out[c], categories=self.categories_[c])
        return out

    def get_feature_names_out(self, input_features=None):
        """Nama kolom keluaran = nama kolom masukan."""
        return np.array(self.names_out_)


def build_type_encoder():
    """Buat ``ColumnTransformer`` yang menandai kolom kategorik + ID wilayah sebagai ``category``.

    Transformer diterapkan pada seluruh kolom (urutan kolom terjaga, penting karena
    LightGBM memilih fitur acak per pohon), tetapi hanya ``CATS + GEO`` yang diubah.
    """
    ct = ColumnTransformer(
        [("cat", CategoryEncoder(columns=CATS + GEO), all_columns)],
        remainder="drop", verbose_feature_names_out=False)
    return ct.set_output(transform="pandas")
