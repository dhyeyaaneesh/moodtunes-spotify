"""
MoodTunes - backend API + website server

    uvicorn app:app --reload          then open  http://127.0.0.1:8000

Needs the ./models folder, which is created by:  python train_model.py
"""
import json
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

import core  # noqa: F401  (the saved mood model needs core.SeededGMM)
from core import GENRE_MAP, GMM_FEATURES, MOODS, RETURN_FEATURES, audio_matrix, lookup_key, meta_matrix

ROOT = Path(__file__).parent
MODEL_DIR = ROOT / "models"

needed = ["mood_model.joblib", "estimate_model.joblib", "catalogue.joblib", "catalogue_P.npy",
          "tracks.joblib", "info.json"]
missing = [f for f in needed if not (MODEL_DIR / f).exists()]
if missing:
    raise RuntimeError("Model files not found: " + ", ".join(missing) +
                       "\nRun  python train_model.py  first (see README).")

MOOD = joblib.load(MODEL_DIR / "mood_model.joblib")
EST = joblib.load(MODEL_DIR / "estimate_model.joblib")
CAT = joblib.load(MODEL_DIR / "catalogue.joblib")
CAT_P = np.load(MODEL_DIR / "catalogue_P.npy")
TRACKS = joblib.load(MODEL_DIR / "tracks.joblib")
INFO = json.loads((MODEL_DIR / "info.json").read_text())

TRACK_INDEX = {k: i for i, k in enumerate(TRACKS["key"].values)}
GENRE_ROWS = {g: np.flatnonzero((CAT["g"] == g).values) for g in GENRE_MAP}
POP = CAT["pop"].to_numpy(dtype=float)
RNG = np.random.default_rng()

app = FastAPI(title="MoodTunes API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/")
def home():
    return FileResponse(ROOT / "moodtunes.html")


@app.get("/api/health")
def health():
    return {"status": "ok", "catalogue_rows": int(len(CAT))}


@app.get("/api/info")
def info():
    return INFO


# ---------------------------------------------------------------- recommendations
def pick_for_genre(mood_i, genre, k, excluded):
    """best-matching songs for one mood+genre: score = P(mood | audio) x popularity, then a weighted
    random draw from the top of the list so every shuffle is different"""
    idx = GENRE_ROWS.get(genre)
    if idx is None or len(idx) == 0:
        return []
    score = CAT_P[idx, mood_i] * (0.35 + 0.65 * np.sqrt(POP[idx] / 100.0))
    order = np.argsort(-score)
    for pool_size in (120, 400, len(order)):
        pool = idx[order[:pool_size]]
        pool_scores = score[order[:pool_size]]
        keep = np.array([CAT["id"].iat[i] not in excluded for i in pool])
        if keep.sum() >= k or pool_size >= len(order):
            break
    if keep.sum() < k:                       # everything seen: start again from the top
        keep = np.ones(len(pool), bool)
    cand, w = pool[keep], pool_scores[keep] ** 2 + 1e-9
    k = min(k, len(cand))
    return list(RNG.choice(cand, size=k, replace=False, p=w / w.sum()))


@app.get("/api/songs")
def songs(mood: str, genres: str, n: int = Query(15, ge=1, le=50), exclude: str = ""):
    if mood not in MOODS:
        raise HTTPException(400, f"mood must be one of {MOODS}")
    chosen = [g.strip() for g in genres.split(",") if g.strip()]
    chosen = list(dict.fromkeys(chosen))
    if not 1 <= len(chosen) <= 3 or any(g not in GENRE_MAP for g in chosen):
        raise HTTPException(400, f"send 1 to 3 genres from: {list(GENRE_MAP)}")
    excluded = set(exclude.split(",")) if exclude else set()
    per = -(-n // len(chosen))
    mi = MOODS.index(mood)
    rows = []
    for g in chosen:
        rows += pick_for_genre(mi, g, per, excluded)
    RNG.shuffle(rows)
    rows = rows[:n]
    return {"songs": [{"id": str(CAT["id"].iat[i]), "t": str(CAT["t"].iat[i]), "a": str(CAT["a"].iat[i]),
                       "b": int(CAT["b"].iat[i]), "dur": int(CAT["dur"].iat[i]), "_g": str(CAT["g"].iat[i])}
                      for i in rows]}


# ---------------------------------------------------------------- analyser
class AnalyzeRequest(BaseModel):
    features: Optional[dict] = None       # Spotify audio features, when Spotify returned them
    name: str = ""
    artists: str = ""
    genres: List[str] = []
    explicit: bool = False
    popularity: float = 50
    duration_ms: float = 210000


def mood_from_audio(f):
    X = audio_matrix({k: [f[k]] for k in GMM_FEATURES})
    p = MOOD["gmm"].predict_proba(MOOD["scaler"].transform(X))[0]
    top = int(p.argmax())
    return MOODS[top], int(round(100 * p[top])), {m: round(float(v), 4) for m, v in zip(MOODS, p)}


@app.post("/api/analyze")
def analyze(req: AnalyzeRequest):
    # 1) Spotify gave us the audio features -> mood model
    if req.features:
        missing = [k for k in GMM_FEATURES if k not in req.features or req.features[k] is None]
        if missing:
            raise HTTPException(400, f"features missing: {missing}")
        mood, conf, probs = mood_from_audio(req.features)
        return {"source": "audio", "mood": mood, "conf": conf, "probs": probs, "features": req.features}

    # 2) the song is in the training data -> use its real audio features
    i = TRACK_INDEX.get(lookup_key(req.name, req.artists))
    if i is not None:
        row = TRACKS.iloc[i]
        feats = {k: float(row[k]) for k in RETURN_FEATURES}
        feats["mode"] = int(feats["mode"])
        if req.duration_ms:
            feats["duration_ms"] = float(req.duration_ms)
        mood, conf, probs = mood_from_audio(feats)
        return {"source": "dataset", "mood": mood, "conf": conf, "probs": probs, "features": feats}

    # 3) otherwise estimate from genre / title / explicit / popularity / duration
    M = meta_matrix([" ".join(req.genres)], [req.name], [req.explicit], [req.popularity], [req.duration_ms],
                    EST["genre_cv"], EST["title_tf"])
    p = EST["clf"].predict_proba(M)[0]
    classes = EST["clf"].classes_
    probs = {MOODS[int(c)]: round(float(v), 4) for c, v in zip(classes, p)}
    mood = max(probs, key=probs.get)
    est = np.clip(EST["reg"].predict(M)[0], 0, 1)
    feats = {f: round(float(v), 3) for f, v in zip(EST["bar_features"], est)}
    return {"source": "estimate", "mood": mood, "conf": int(round(100 * probs[mood])), "probs": probs,
            "features": feats}
