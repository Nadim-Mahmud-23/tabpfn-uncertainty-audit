#!/usr/bin/env python3
"""
Build a prediction cache: fit each (dataset, base model, seed) ONCE and save the
calibration/test probabilities, labels, and group arrays. Everything downstream
(per-group coverage, empty-set rates, randomized APS, the calibration-size sweep,
and the GBDT+T RQ1 rows) is then derived cheaply from this cache without refitting
the expensive TabPFN model.

Cache layout:  cache/{dataset}__{model}__seed{seed}.npz  with keys
    cal_probs, test_probs, ycal, yte, n_classes,
    cal_sex, cal_race, cal_age, cal_predclass,
    te_sex,  te_race,  te_age,  te_predclass
For xgboost/lightgbm additionally:  test_probs_T, cal_probs_T, temperature.

Run:  python scripts/build_cache.py
"""
from __future__ import annotations
import os
import sys
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

# --- secrets / SSL bootstrap (TabPFN needs these on this machine) ----------- #
import certifi  # noqa: E402
_env = ROOT / ".env"
if _env.exists():
    for _line in _env.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))
os.environ.setdefault("SSL_CERT_FILE", certifi.where())
os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())

import warnings  # noqa: E402
warnings.filterwarnings("ignore")

from src.data_loaders import dataset_registry          # noqa: E402
from src.models import build_model, TemperatureScaler  # noqa: E402
from src.pipeline import three_way_split               # noqa: E402

DATASETS = ["ACSPublicCoverage-CA", "ACSIncome-CA", "ACSEmployment-CA"]
BASE_MODELS = ["tabpfn", "xgboost", "lightgbm", "mlp"]
SEEDS = [0, 1, 2, 3, 4]
CACHE = ROOT / "cache"
CACHE.mkdir(exist_ok=True)


def main():
    reg = dataset_registry()
    import time
    t0 = time.time()
    for ds in DATASETS:
        loader_fn, kwargs, _ = reg[ds]
        for model_name in BASE_MODELS:
            for seed in SEEDS:
                out = CACHE / f"{ds}__{model_name}__seed{seed}.npz"
                if out.exists():
                    print(f"[skip] {out.name}")
                    continue
                data = loader_fn(seed=seed, **kwargs)
                X, y, sens, K = data["X"], data["y"], data["sensitive"], data["n_classes"]
                (Xtr, ytr, _), (Xcal, ycal, scal), (Xte, yte, ste) = \
                    three_way_split(X, y, sens, seed)

                model = build_model(model_name, seed)
                model.fit(Xtr, ytr)
                cal_probs = model.predict_proba(Xcal)
                test_probs = model.predict_proba(Xte)

                payload = dict(
                    cal_probs=cal_probs, test_probs=test_probs,
                    ycal=ycal, yte=yte, n_classes=K,
                    cal_predclass=cal_probs.argmax(1), te_predclass=test_probs.argmax(1),
                )
                for ax in ("sex", "race", "age"):
                    payload[f"cal_{ax}"] = np.asarray(scal[ax])
                    payload[f"te_{ax}"] = np.asarray(ste[ax])

                # GBDT temperature scaling for the RQ1 +T rows
                if model_name in ("xgboost", "lightgbm"):
                    ts = TemperatureScaler(model).fit_temperature(Xcal, ycal)
                    payload["cal_probs_T"] = ts.predict_proba(Xcal)
                    payload["test_probs_T"] = ts.predict_proba(Xte)
                    payload["temperature"] = np.array([ts.T])

                np.savez_compressed(out, **payload)
                print(f"[{time.time()-t0:7.1f}s] wrote {out.name}  "
                      f"acc={(test_probs.argmax(1)==yte).mean():.3f}")
    print(f"DONE in {time.time()-t0:.1f}s  ({len(list(CACHE.glob('*.npz')))} cells cached)")


if __name__ == "__main__":
    main()
