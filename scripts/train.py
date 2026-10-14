"""Pipeline lengkap (fitur -> tipe kategori -> LightGBM), validasi, tuning bobot kelas, dan training akhir."""
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .config import (EARLY_STOPPING, FINAL_ITER_FACTOR, LGB_PARAMS, MAX_ROUNDS, MODEL_DIR,
                     SEED_SPLIT)
from .feature_engineering import LocationFeatureBuilder
from .preprocess import BaseFeatureBuilder, all_columns, build_type_encoder


class LGBMBoosterClassifier(BaseEstimator, ClassifierMixin):
    """Pembungkus ``lightgbm.train`` agar bisa menjadi langkah terakhir ``Pipeline``.

    Parameters
    ----------
    params : dict
        Parameter LightGBM (objective multiclass, learning_rate, dst).
    num_boost_round : int
        Jumlah iterasi bila tidak memakai early stopping.
    early_stopping_rounds : int
        Dipakai hanya jika ``fit`` diberi ``eval_set``.
    """

    def __init__(self, params=None, num_boost_round=MAX_ROUNDS, early_stopping_rounds=EARLY_STOPPING):
        """Simpan hyper-parameter (lihat parameter kelas)."""
        self.params = params
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds

    def fit(self, X, y, eval_set=None):
        """Latih booster.

        Parameters
        ----------
        X : pd.DataFrame
            Fitur (kolom kategorik bertipe ``category``).
        y : array-like
            Label 0..4.
        eval_set : tuple (X_val, y_val), optional
            Jika diberikan, pelatihan dipantau dan dihentikan lebih awal
            (``best_iteration_`` tersimpan).
        """
        import lightgbm as lgb
        params = LGB_PARAMS if self.params is None else self.params
        dtrain = lgb.Dataset(X, np.asarray(y))
        if eval_set is not None:
            Xv, yv = eval_set
            self.booster_ = lgb.train(
                params, dtrain, self.num_boost_round,
                valid_sets=[lgb.Dataset(Xv, np.asarray(yv))],
                callbacks=[lgb.early_stopping(self.early_stopping_rounds), lgb.log_evaluation(50)])
            self.best_iteration_ = self.booster_.best_iteration
        else:
            self.booster_ = lgb.train(params, dtrain, self.num_boost_round)
            self.best_iteration_ = None
        self.classes_ = np.arange(params["num_class"])
        self.feature_names_ = list(X.columns)
        return self

    def predict_proba(self, X):
        """Probabilitas 5 kelas; memakai iterasi terbaik jika early stopping dipakai."""
        return self.booster_.predict(X, num_iteration=self.best_iteration_)

    def predict(self, X):
        """Kelas dengan probabilitas tertinggi (0..4)."""
        return self.predict_proba(X).argmax(1)

    def feature_importance(self):
        """Pentingnya fitur berdasarkan *gain*, urut menurun (``pd.Series``)."""
        imp = pd.Series(self.booster_.feature_importance("gain"), index=self.feature_names_)
        return imp.sort_values(ascending=False)


def build_pipeline(all_ids=None, num_boost_round=MAX_ROUNDS):
    """Susun Pipeline: ColumnTransformer fitur -> ColumnTransformer tipe kategori -> LightGBM.

    Parameters
    ----------
    all_ids : pd.Series, optional
        Seluruh ``building_id`` (train + test) untuk fitur jumlah bangunan per kelompok.
    num_boost_round : int
        Jumlah iterasi model.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Langkah: ``features`` (fitur dasar + fitur lokasi), ``encode`` (tipe kategori),
        ``model`` (LightGBM).
    """
    features = ColumnTransformer(
        [("base", BaseFeatureBuilder(all_ids=all_ids), all_columns),
         ("location", LocationFeatureBuilder(), all_columns)],
        verbose_feature_names_out=False).set_output(transform="pandas")
    return Pipeline([
        ("features", features),
        ("encode", build_type_encoder()),
        ("model", LGBMBoosterClassifier(num_boost_round=num_boost_round)),
    ])


def macro_f1_with_weights(proba, y_true, w):
    """Macro F1 bila probabilitas tiap kelas dikalikan bobot ``w``."""
    return f1_score(y_true, (proba * w).argmax(1), average="macro")


def tune_class_weights(proba, y_true, grid=np.linspace(0.5, 2.5, 21), rounds=3):
    """Cari bobot kelas yang memaksimalkan Macro F1 (coordinate ascent).

    ``argmax`` probabilitas mengoptimalkan akurasi, sedangkan Macro F1 menganggap tiap
    kelas sama penting; bobot per kelas dicari satu per satu pada data validasi.

    Parameters
    ----------
    proba : np.ndarray, shape (n, 5)
        Probabilitas validasi.
    y_true : array-like
        Label validasi (0..4).
    grid : array-like
        Nilai bobot yang dicoba untuk tiap kelas.
    rounds : int
        Banyaknya putaran penyempurnaan.

    Returns
    -------
    tuple
        ``(bobot, skor_terbaik)``.
    """
    w = np.ones(proba.shape[1])
    best = macro_f1_with_weights(proba, y_true, w)
    for _ in range(rounds):
        for k in range(proba.shape[1]):
            for v in grid:
                w2 = w.copy()
                w2[k] = v
                s = macro_f1_with_weights(proba, y_true, w2)
                if s > best:
                    best, w = s, w2
    return w, best


def run_validation(train, all_ids, test_size=0.2):
    """Tahap 1: ukur performa jujur pada 20% train yang disisihkan.

    Fitur latih dihitung out-of-fold; fitur validasi dihitung dari data latih saja
    (meniru kondisi test). Pelatihan memakai early stopping pada data validasi.

    Returns
    -------
    dict
        ``pipe`` (pipeline terlatih), ``best_iter``, ``proba`` & ``y_val`` validasi,
        ``weights`` & ``f1_tuned`` (hasil tuning), ``f1_raw``, ``acc``.
    """
    t = time.time()
    A, B = train_test_split(train, test_size=test_size, stratify=train.y, random_state=SEED_SPLIT)
    A, B = A.reset_index(drop=True), B.reset_index(drop=True)
    pipe = build_pipeline(all_ids)
    prep = pipe[:-1]                                   # semua langkah kecuali model
    XA = prep.fit_transform(A.drop(columns=["damage_grade", "y"]), A.y.values)
    XB = prep.transform(B.drop(columns=["damage_grade", "y"]))
    print("features built", time.time() - t, XA.shape)
    pipe.named_steps["model"].fit(XA, A.y.values, eval_set=(XB, B.y.values))
    bi = pipe.named_steps["model"].best_iteration_
    print("train time", time.time() - t, "best", bi)
    pv = pipe.named_steps["model"].predict_proba(XB)
    yv = B.y.values
    f1_raw = f1_score(yv, pv.argmax(1), average="macro")
    acc = (pv.argmax(1) == yv).mean()
    print("val macroF1 raw", f1_raw, "acc", acc)
    w, best = tune_class_weights(pv, yv)
    print("val macroF1 tuned", best, w)
    return dict(pipe=pipe, best_iter=int(bi), proba=pv, y_val=yv, weights=w,
                f1_tuned=best, f1_raw=f1_raw, acc=acc)


def train_final(train, all_ids, best_iter):
    """Tahap 2: latih ulang pipeline pada *seluruh* train dengan ``1,1 x best_iter`` iterasi."""
    t = time.time()
    Tr = train.reset_index(drop=True)
    pipe = build_pipeline(all_ids, num_boost_round=int(best_iter * FINAL_ITER_FACTOR))
    pipe.fit(Tr.drop(columns=["damage_grade", "y"]), Tr.y.values)
    print("final done", time.time() - t)
    return pipe


def save_artifacts(pipe, weights, best_iter, f1, model_dir=MODEL_DIR):
    """Simpan pipeline (``.pkl``) dan hasil tuning (``cfg3.json``) ke folder ``models/``."""
    joblib.dump(pipe, f"{model_dir}/pipeline_model_lgbm.pkl", compress=3)
    json.dump({"w": list(map(float, weights)), "iters": int(best_iter), "f1": float(f1)},
              open(f"{model_dir}/cfg3.json", "w"))


if __name__ == "__main__":
    from .preprocess import load_data
    from .config import SUBMISSION_DIR
    from .predict import make_submission
    tr, te = load_data()
    ids = pd.concat([tr.building_id, te.building_id], ignore_index=True)
    res = run_validation(tr, ids)
    final = train_final(tr, ids, res["best_iter"])
    save_artifacts(final, res["weights"], res["best_iter"], res["f1_tuned"])
    make_submission(final, te, res["weights"], SUBMISSION_DIR / "submission.csv")
