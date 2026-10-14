"""Fitur lokasi: target encoding, fitur tetangga (berdasarkan urutan ID), dan versi out-of-fold.

Semua statistik label dihitung dari *data referensi* ``R`` lalu dipetakan ke data ``Q``.
Saat melatih, ``Q`` dan ``R`` dipisah lewat skema out-of-fold supaya label sebuah baris
tidak ikut membentuk fiturnya sendiri (anti kebocoran).
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold

from .config import N_FOLDS, SEED_OOF

# kolom minimal yang dibutuhkan fitur lokasi (menjaga objek pipeline tetap kecil)
NEEDED_COLS = ["building_id", "ward_id", "superstructure", "count_floors_pre_eq",
               "foundation_type", "ground_floor_type"]


def keys(d):
    """Buat kunci kelompok lokasi dari ``building_id`` dan kolom lain.

    Parameters
    ----------
    d : pd.DataFrame
        Data bangunan (butuh ``building_id``, ``ward_id``, ``superstructure``,
        ``count_floors_pre_eq``).

    Returns
    -------
    dict
        Nama kelompok -> Series kunci (kasar ``g8`` sampai halus ``g3``, ``ward``, dan
        kombinasi lokasi x superstruktur / jumlah lantai).
    """
    b = d["building_id"]
    return {"g8": b // 10**8, "g6": b // 10**6, "g4": b // 10**4, "g3": b // 10**3,
            "ward": d["ward_id"],
            "g6sup": (b // 10**6).astype(str) + d["superstructure"],
            "g6fl": (b // 10**6).astype(str) + "_" + d["count_floors_pre_eq"].astype(str)}


def te_features(Q, R):
    """Target encoding berpenghalusan: proporsi tiap kelas kerusakan per kelompok.

    Rumus smoothing ``(jumlah + a*prior) / (n + a)`` dengan ``a = 5`` agar kelompok
    kecil tidak ekstrem.

    Parameters
    ----------
    Q : pd.DataFrame
        Data yang ingin diberi fitur.
    R : pd.DataFrame
        Data referensi yang memiliki kolom label ``y`` (0..4).

    Returns
    -------
    dict
        ``te_<kelompok>_<kelas>`` -> array float32 (5 kolom per kelompok).
    """
    out = {}
    prior = np.bincount(R.y.values, minlength=5) / len(R)    # proporsi global tiap kelas
    oh = np.eye(5, dtype=np.float32)[R.y.values]             # one-hot label
    kq, kr = keys(Q), keys(R)
    for nm in kq:
        s = pd.DataFrame(oh).groupby(kr[nm].values).sum()    # jumlah tiap kelas per kelompok
        n = s.sum(1)                                         # total bangunan per kelompok
        a = 5.0                                              # kekuatan smoothing
        s = (s.add(a * prior, axis=1)).div(n + a, axis=0)
        m = s.reindex(kq[nm].values).values                  # petakan ke baris Q
        m = np.where(np.isnan(m), prior, m)                  # kelompok tak dikenal -> prior global
        for c in range(5):
            out[f"te_{nm}_{c}"] = m[:, c].astype(np.float32)
    return out


def nb_features(Q, R):
    """Fitur tetangga umum pada urutan ``building_id`` data referensi.

    Menghitung rata-rata/proporsi grade pada jendela k = 3, 10, 30, 100 tetangga kiri
    dan kanan, grade tetangga langsung, jarak ID ke tetangga, dan kepadatan area.
    Jumlah kumulatif per kelas dipakai agar tiap jendela cukup satu pengurangan.

    Parameters
    ----------
    Q : pd.DataFrame
        Data yang diberi fitur.
    R : pd.DataFrame
        Data referensi berlabel ``y``.

    Returns
    -------
    dict
        Nama fitur -> array.
    """
    out = {}
    o = np.argsort(R.building_id.values)
    Rid = R.building_id.values[o]                            # ID referensi terurut
    Ry = R.y.values[o]                                       # label mengikuti urutan ID
    C = np.vstack([np.zeros((1, 5)), np.cumsum(np.eye(5)[Ry], 0)])
    n = len(Rid)
    pos = np.searchsorted(Rid, Q.building_id.values)         # posisi Q pada urutan referensi
    ordv = np.arange(1, 6)
    for k in (3, 10, 30, 100):
        lo = np.maximum(pos - k, 0)
        hi = np.minimum(pos + k, n)
        c = (hi - lo)[:, None]
        fr = (C[hi] - C[lo]) / c
        out[f"nb{k}_mean"] = (fr @ ordv).astype(np.float32)
        if k in (10, 30):
            for j in range(5):
                out[f"nb{k}_{j}"] = fr[:, j].astype(np.float32)
    l = np.clip(pos - 1, 0, n - 1)
    r = np.clip(pos, 0, n - 1)
    out["nb_left_y"] = Ry[l]
    out["nb_right_y"] = Ry[r]
    qid = Q.building_id.values
    out["nb_left_gap"] = np.log1p(np.abs(qid - Rid[l]))
    out["nb_right_gap"] = np.log1p(np.abs(Rid[r] - qid))
    out["nb_span10"] = np.log1p(Rid[np.minimum(pos + 9, n - 1)] - Rid[np.maximum(pos - 10, 0)])
    return out


def cond_nb_features(Q, R):
    """Fitur tetangga bersyarat: hanya tetangga dengan atribut bangunan yang sama.

    Atribut syarat: superstructure, foundation_type, count_floors_pre_eq,
    ground_floor_type. Dibandingkan "apel dengan apel" pada jendela k = 5 dan 20.

    Parameters
    ----------
    Q : pd.DataFrame
        Data yang diberi fitur.
    R : pd.DataFrame
        Data referensi berlabel ``y``.

    Returns
    -------
    dict
        ``c_<atribut>_<k>_<kelas>`` -> array float32 (NaN bila referensi < 5).
    """
    out = {}
    for att in ("superstructure", "foundation_type", "count_floors_pre_eq", "ground_floor_type"):
        qa = Q[att].values
        ra = R[att].values
        for v in np.unique(qa):
            mq = qa == v
            mr = ra == v
            if mr.sum() < 5:
                continue                                      # referensi terlalu sedikit -> NaN
            Rs = R.building_id.values[mr]
            ys = R.y.values[mr]
            oo = np.argsort(Rs)
            Rs = Rs[oo]; ys = ys[oo]
            Cs = np.vstack([np.zeros((1, 5)), np.cumsum(np.eye(5)[ys], 0)])
            ns = len(Rs)
            ps = np.searchsorted(Rs, Q.building_id.values[mq])
            for k in (5, 20):
                lo = np.maximum(ps - k, 0)
                hi = np.minimum(ps + k, ns)
                c = (hi - lo)[:, None]
                fr = (Cs[hi] - Cs[lo]) / c
                for j in range(5):
                    key = f"c_{att[:5]}_{k}_{j}"
                    if key not in out:
                        out[key] = np.full(len(Q), np.nan, dtype=np.float32)
                    out[key][mq] = fr[:, j]
    return out


def feats(Q, R):
    """Gabungkan semua fitur lokasi/tetangga untuk ``Q`` dengan referensi ``R``.

    Returns
    -------
    pd.DataFrame
        Target encoding + tetangga umum + tetangga bersyarat.
    """
    out = {}
    out.update(te_features(Q, R))
    out.update(nb_features(Q, R))
    out.update(cond_nb_features(Q, R))
    return pd.DataFrame(out)


def oof(S, nf=N_FOLDS):
    """Hitung fitur lokasi secara out-of-fold untuk seluruh ``S``.

    Data dibagi ``nf`` fold; fitur tiap fold dihitung hanya dari fold lainnya sehingga
    label sebuah baris tidak ikut membentuk fiturnya sendiri.

    Parameters
    ----------
    S : pd.DataFrame
        Data berlabel (punya kolom ``y``) dengan index 0..N-1.
    nf : int
        Jumlah fold.

    Returns
    -------
    pd.DataFrame
        Fitur lokasi, urutan baris sama dengan ``S``.
    """
    idx = np.arange(len(S))
    kf = KFold(nf, shuffle=True, random_state=SEED_OOF)
    chunks = []
    for a, b in kf.split(idx):                                # a = referensi, b = yang dihitung
        f = feats(S.iloc[b], S.iloc[a])
        f.index = b
        chunks.append(f)
    return pd.concat(chunks).sort_index()


class LocationFeatureBuilder(BaseEstimator, TransformerMixin):
    """Transformer sklearn untuk fitur lokasi/tetangga yang memakai label.

    * ``fit_transform(X, y)`` (dipakai saat melatih pipeline) mengembalikan fitur
      **out-of-fold** sehingga tidak ada kebocoran label.
    * ``transform(X)`` (data validasi/test) memakai seluruh data latih sebagai referensi.

    Catatan: jangan memanggil ``transform`` pada data latih yang sama -- itu membuat fitur
    dari labelnya sendiri; gunakan ``fit_transform``.
    """

    def fit(self, X, y):
        """Simpan data referensi (kolom minimal + label ``y``)."""
        R = X[NEEDED_COLS].reset_index(drop=True).copy()
        R["y"] = np.asarray(y)
        self.ref_ = R
        return self

    def fit_transform(self, X, y=None, **fit_params):
        """Fit lalu kembalikan fitur out-of-fold untuk data latih."""
        self.fit(X, y)
        F = oof(self.ref_)
        self.names_out_ = list(F.columns)
        return F

    def transform(self, X):
        """Fitur lokasi untuk data baru dengan referensi = data latih saat ``fit``."""
        F = feats(X[NEEDED_COLS].reset_index(drop=True), self.ref_)
        self.names_out_ = list(F.columns)
        return F

    def get_feature_names_out(self, input_features=None):
        """Nama kolom keluaran (tersedia setelah ``fit_transform``/``transform``)."""
        return np.array(self.names_out_)
