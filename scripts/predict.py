"""Prediksi data test dengan pipeline tersimpan dan penulisan ``submission.csv``."""
import argparse
import json

import joblib
import numpy as np
import pandas as pd

from .config import DATA_DIR, MODEL_DIR, SUBMISSION_DIR


def predict_grades(pipe, test, weights):
    """Prediksi ``damage_grade`` (1..5) untuk ``test``.

    Probabilitas 5 kelas dikalikan bobot kelas (hasil tuning Macro F1), lalu dipilih
    kelas terbesar dan digeser dari 0..4 ke 1..5.

    Returns
    -------
    tuple
        ``(pred, proba)`` -- label 1..5 dan probabilitas mentah.
    """
    proba = pipe.predict_proba(test.reset_index(drop=True))
    return (proba * np.asarray(weights)).argmax(1) + 1, proba


def make_submission(pipe, test, weights, path):
    """Tulis file submission (``building_id``, ``damage_grade`` berupa angka 1-5) ke ``path``."""
    pred, _ = predict_grades(pipe, test, weights)
    sub = pd.DataFrame({"building_id": test["building_id"], "damage_grade": pred})
    sub.to_csv(path, index=False)
    return sub


def main():
    """CLI: ``python -m scripts.predict`` memuat model di ``models/`` dan menulis submission."""
    ap = argparse.ArgumentParser(description="Buat submission dari model tersimpan.")
    ap.add_argument("--model", default=f"{MODEL_DIR}/pipeline_model_lgbm.pkl")
    ap.add_argument("--config", default=f"{MODEL_DIR}/cfg3.json")
    ap.add_argument("--test", default=f"{DATA_DIR}/test.csv")
    ap.add_argument("--out", default=f"{SUBMISSION_DIR}/submission.csv")
    a = ap.parse_args()
    pipe = joblib.load(a.model)
    w = json.load(open(a.config))["w"]
    sub = make_submission(pipe, pd.read_csv(a.test), w, a.out)
    print(sub.damage_grade.value_counts(normalize=True).sort_index())


if __name__ == "__main__":
    main()
