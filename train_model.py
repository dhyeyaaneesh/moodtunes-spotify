"""
MoodTunes - trains every model and saves it into ./models

    python train_model.py                     (finds the CSV inside ./data automatically)
    python train_model.py path/to/file.csv    (or give the path yourself)

Expects the Kaggle "Spotify Tracks Dataset" (columns such as track_id, artists, track_name, popularity,
duration_ms, explicit, danceability, energy, loudness, mode, speechiness, acousticness,
instrumentalness, liveness, valence, tempo, track_genre).

Models trained
  1. Mood model      : 12-component Gaussian Mixture on 8 audio features (EM, seeded from Russell's
                       circumplex), gives P(mood | audio features); checked against Spotify's own mood tags
  2. Estimate model  : logistic regression + ridge regressions that guess mood and audio features from
                       genre / title / explicit / popularity / duration (used when Spotify returns no audio features)
  3. Song catalogue  : every track scored for every mood, used for the recommendations
"""
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, r2_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from core import (EDITORIAL_TAGS, GENRE_MAP, GMM_FEATURES, MOODS, SeededGMM, audio_matrix, genre_text,
                  lookup_key, meta_matrix, prototype_matrix)

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
SEED = 42
PRIOR_STRENGTH = 1.0      # how strongly each mood is anchored to its seed (0 = free EM)
BAR_FEATURES = ["valence", "energy", "danceability", "acousticness", "instrumentalness"]


def step(msg):
    print(f"\n=== {msg}", flush=True)
    return time.time()


def done(t0):
    print(f"    done in {time.time() - t0:.1f}s", flush=True)


def find_csv():
    if len(sys.argv) > 1:
        return Path(sys.argv[1])
    found = sorted(DATA_DIR.glob("*.csv")) if DATA_DIR.exists() else []
    if found:
        return found[0]
    sys.exit("\nERROR: no CSV found. Put the Kaggle file (e.g. dataset.csv) inside the 'data' folder, "
             "or run: python train_model.py path/to/file.csv\n")


def load_clean(path):
    df = pd.read_csv(path)
    raw_rows = len(df)
    df = df.rename(columns={"name": "track_name", "artist": "artists", "artist_name": "artists",
                            "id": "track_id", "genre": "track_genre"})
    need = ["track_name", "artists", "track_genre", "liveness"] + GMM_FEATURES
    missing = [c for c in need if c not in df.columns]
    if missing:
        sys.exit(f"\nERROR: the CSV is missing columns: {missing}\nFound: {list(df.columns)}\n"
                 "(MoodTunes needs the Kaggle 'Spotify Tracks Dataset', which has a track_genre column.)\n")
    for col, default in [("popularity", 50), ("duration_ms", 210000), ("mode", 1)]:
        if col not in df.columns:
            df[col] = default
    df["explicit"] = (df["explicit"].astype(str).str.lower().isin(["true", "1", "1.0"])
                      if "explicit" in df.columns else False)
    df = df.dropna(subset=need).copy()
    for c in ["valence", "energy", "danceability", "acousticness", "instrumentalness", "speechiness", "liveness"]:
        df[c] = df[c].astype(float).clip(0, 1)
    df = df[(df["tempo"] > 40) & (df["tempo"] < 250) & df["loudness"].between(-60, 5)]   # tempo 0 = broken rows
    df["track_genre"] = df["track_genre"].astype(str).str.strip().str.lower()
    df["track_name"] = df["track_name"].astype(str).str.strip()
    df["artists"] = df["artists"].astype(str).str.strip()
    df["popularity"] = df["popularity"].astype(float)
    df["key"] = [lookup_key(t, a) for t, a in zip(df["track_name"], df["artists"])]
    if "track_id" not in df.columns:
        df["track_id"] = df["key"]
    df = df.sort_values("popularity", ascending=False).reset_index(drop=True)
    print(f"    raw rows: {raw_rows:,}   usable rows: {len(df):,}   genre tags: {df['track_genre'].nunique()}")
    return df


# ---------------------------------------------------------------------------
# 1. mood model
# ---------------------------------------------------------------------------
def train_mood_model(trk, train_idx, test_idx):
    t0 = step("Mood model: Gaussian mixture on audio features (seeded EM)")
    X = audio_matrix(trk)
    scaler = StandardScaler().fit(X[train_idx])
    Z = scaler.transform(X)
    P0 = scaler.transform(prototype_matrix())

    gmm = SeededGMM(P0, prior_strength=PRIOR_STRENGTH).fit(Z[train_idx])
    print(f"    EM converged: {gmm.converged_}   iterations: {gmm.n_iter_}   (prior strength {PRIOR_STRENGTH})")
    ll_train, ll_test = gmm.score(Z[train_idx]), gmm.score(Z[test_idx])
    print(f"    avg log-likelihood  train={ll_train:.3f}  held-out={ll_test:.3f}")

    # did every mood keep its meaning after EM? (its centre must still be closest to its own seed)
    nearest = cdist(gmm.means_, P0).argmin(1)
    drifted = [MOODS[k] for k in range(len(MOODS)) if nearest[k] != k]
    if drifted:
        print(f"    WARNING: these moods drifted towards another mood's seed: {drifted}")
    else:
        print("    every mood kept its meaning (centre still closest to its own seed)")

    probs = gmm.predict_proba(Z)
    top = probs.argmax(1)
    summary = []
    print("\n    mood         share   valence energy  dance  acoust  tempo")
    for k, m in enumerate(MOODS):
        sel = top == k
        row = {"mood": m, "share": round(float(sel.mean()), 4)}
        for f in ["valence", "energy", "danceability", "acousticness", "tempo"]:
            row[f] = round(float(trk.loc[sel, f].mean()), 3) if sel.any() else None
        summary.append(row)
        if sel.any():
            print(f"    {m:12s} {sel.mean()*100:5.1f}%  {row['valence']:.2f}    {row['energy']:.2f}   "
                  f"{row['danceability']:.2f}   {row['acousticness']:.2f}   {row['tempo']:.0f}")
    joblib.dump({"scaler": scaler, "gmm": gmm}, MODEL_DIR / "mood_model.joblib", compress=3)
    done(t0)
    stats = {"converged": bool(gmm.converged_), "iterations": int(gmm.n_iter_),
             "loglik_train": round(float(ll_train), 4), "loglik_heldout": round(float(ll_test), 4),
             "prior_strength": PRIOR_STRENGTH, "drifted_moods": drifted, "mood_summary": summary}
    return probs, stats


def validate_with_tags(trk, probs):
    """AUC of 'tagged with X' vs 'not tagged', scored by the model's probability of the matching moods"""
    t0 = step("Validation: do Spotify's mood tags (happy, sad, sleep, ...) get matching moods?")
    mi = {m: i for i, m in enumerate(MOODS)}
    rows = []
    for tag, moods in EDITORIAL_TAGS.items():
        # a track can carry several genre tags, so look at every row with that tag
        ids = set(trk.attrs["all_rows"].loc[trk.attrs["all_rows"]["track_genre"] == tag, "track_id"])
        y = trk["track_id"].isin(ids).values
        if y.sum() < 150:
            continue
        score = probs[:, [mi[m] for m in moods]].sum(1)
        auc = roc_auc_score(y, score)
        rows.append({"tag": tag, "expected_moods": moods, "tracks": int(y.sum()), "auc": round(float(auc), 4)})
        print(f"    {tag:12s} n={int(y.sum()):5d}  AUC={auc:.3f}  (expects {', '.join(moods)})")
    if not rows:
        print("    (none of Spotify's mood tags found in this file - skipped)")
    done(t0)
    return rows


# ---------------------------------------------------------------------------
# 2. estimate model (no audio features available)
# ---------------------------------------------------------------------------
def train_estimate_model(trk, probs, train_idx, test_idx):
    t0 = step("Estimate model: genre + title + explicit + popularity + duration -> mood and audio features")
    genre_cv = CountVectorizer(binary=True, token_pattern=r"[a-z]+", lowercase=True)
    title_tf = TfidfVectorizer(max_features=5000, min_df=5, sublinear_tf=True, dtype=np.float32)
    genre_cv.fit([genre_text(g) for g in trk["track_genre"].iloc[train_idx]])
    title_tf.fit(trk["track_name"].iloc[train_idx])
    M = meta_matrix(trk["track_genre"], trk["track_name"], trk["explicit"], trk["popularity"],
                    trk["duration_ms"], genre_cv, title_tf)
    y_mood = probs.argmax(1)                       # teacher: the audio-feature mood model
    clf = LogisticRegression(C=1.0, max_iter=300).fit(M[train_idx], y_mood[train_idx])
    pred = clf.predict(M[test_idx])
    acc = accuracy_score(y_mood[test_idx], pred)
    f1 = f1_score(y_mood[test_idx], pred, average="macro")
    base = np.bincount(y_mood[train_idx]).max() / len(train_idx)
    print(f"    mood from metadata: accuracy={acc:.3f}  macro-F1={f1:.3f}   (always guess the commonest mood = {base:.3f})")

    Y = trk[BAR_FEATURES].values
    reg = Ridge(alpha=3.0).fit(M[train_idx], Y[train_idx])
    pr = np.clip(reg.predict(M[test_idx]), 0, 1)
    r2 = {f: round(float(r2_score(Y[test_idx][:, j], pr[:, j])), 4) for j, f in enumerate(BAR_FEATURES)}
    print("    audio features from metadata, test R2: " + "  ".join(f"{k}={v:.2f}" for k, v in r2.items()))
    joblib.dump({"genre_cv": genre_cv, "title_tf": title_tf, "clf": clf, "reg": reg, "bar_features": BAR_FEATURES},
                MODEL_DIR / "estimate_model.joblib", compress=3)
    done(t0)
    return {"mood_accuracy_vs_audio_model": round(float(acc), 4), "mood_macro_f1": round(float(f1), 4),
            "majority_baseline": round(float(base), 4), "feature_r2": r2}


# ---------------------------------------------------------------------------
# 3. catalogue for recommendations + lookup table for the analyser
# ---------------------------------------------------------------------------
def build_catalogue(df, trk, probs):
    t0 = step("Song catalogue: every track scored for every mood")
    pos = pd.Series(np.arange(len(trk)), index=trk["track_id"].values)
    parts, coverage = [], {}
    for ui, tags in GENRE_MAP.items():
        sub = df[df["track_genre"].isin(tags)]
        sub = sub[sub["track_id"].isin(pos.index)].sort_values("popularity", ascending=False)
        sub = sub.drop_duplicates("key")                    # same song re-released -> keep the most popular
        keep = sub[sub["popularity"] >= 5]
        sub = keep if len(keep) >= 200 else sub
        coverage[ui] = int(len(sub))
        sub = sub.assign(ui_genre=ui)
        parts.append(sub)
    cat = pd.concat(parts, ignore_index=True)
    P = probs[pos.loc[cat["track_id"].values].values].astype(np.float32)
    out = pd.DataFrame({"id": cat["track_id"].values, "t": cat["track_name"].values,
                        "a": cat["artists"].str.replace(";", ", ").values, "b": cat["tempo"].round().astype(int).values,
                        "dur": (cat["duration_ms"] / 1000).round().astype(int).values,
                        "pop": cat["popularity"].values, "g": cat["ui_genre"].values})
    joblib.dump(out, MODEL_DIR / "catalogue.joblib", compress=3)
    np.save(MODEL_DIR / "catalogue_P.npy", P)
    for g, n in coverage.items():
        print(f"    {g:12s} {n:6d} tracks" + ("   <-- NO TRACKS: this genre will return nothing" if n == 0 else ""))
    lookup = trk.drop_duplicates("key")[["key", "track_name", "artists"] + GMM_FEATURES + ["liveness", "mode", "duration_ms"]]
    joblib.dump(lookup.reset_index(drop=True), MODEL_DIR / "tracks.joblib", compress=3)
    done(t0)
    return coverage, int(len(out)), int(len(lookup))


def main():
    MODEL_DIR.mkdir(exist_ok=True)
    path = find_csv()
    t0 = step(f"Loading {path}")
    df = load_clean(path)
    trk = df.drop_duplicates("track_id").reset_index(drop=True)      # one row per track for the models
    trk.attrs["all_rows"] = df[["track_id", "track_genre"]]
    print(f"    unique tracks: {len(trk):,}")
    done(t0)

    idx = np.arange(len(trk))
    train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=SEED)

    probs, mood_stats = train_mood_model(trk, train_idx, test_idx)
    tag_rows = validate_with_tags(trk, probs)
    est_stats = train_estimate_model(trk, probs, train_idx, test_idx)
    coverage, n_cat, n_lookup = build_catalogue(df, trk, probs)

    info = {"source": path.name,
            "dataset": {"rows": int(len(df)), "unique_tracks": int(len(trk)), "genre_tags": int(df["track_genre"].nunique()),
                        "catalogue_rows": n_cat, "lookup_tracks": n_lookup},
            "moods": MOODS, "mood_model": mood_stats, "tag_validation": tag_rows,
            "estimate_model": est_stats, "genre_coverage": coverage,
            "notes": "There are no mood labels in the dataset. The mood model is a seeded Gaussian mixture "
                     "(unsupervised EM); it is validated against Spotify's own mood-flavoured genre tags."}
    (MODEL_DIR / "info.json").write_text(json.dumps(info, indent=1))
    print(f"\nAll done. Files written to {MODEL_DIR}")
    print("Next step:  uvicorn app:app --reload")


if __name__ == "__main__":
    main()
