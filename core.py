"""
Shared code for train_model.py and app.py
(moods, the audio features the mood model uses, genre mapping, text helpers).
"""
import re

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import logsumexp

MOODS = ["happy", "sad", "angry", "stressed", "studious", "nostalgic",
         "period", "chill", "romantic", "energetic", "melancholic", "confident"]

# audio features the mood model sees (Spotify audio features)
GMM_FEATURES = ["valence", "energy", "danceability", "acousticness",
                "instrumentalness", "tempo", "loudness", "speechiness"]
SQRT_FEATURES = ["instrumentalness", "speechiness"]       # very skewed -> square-root first
RETURN_FEATURES = GMM_FEATURES + ["liveness", "mode", "duration_ms"]

# Starting points ("seeds") for each mood in audio-feature space, following Russell's circumplex model
# (valence = pleasant/unpleasant, energy = calm/aroused). The model is NOT a lookup of these numbers:
# EM training moves every mood's centre and spread to fit the real tracks.
PROTOTYPES = {
    #              valence energy dance  acoust instrum tempo  loud   speech
    "happy":       (0.78, 0.72, 0.72, 0.10, 0.01, 118, -6.0, 0.05),
    "sad":         (0.22, 0.28, 0.30, 0.45, 0.02,  90, -11.0, 0.04),
    "angry":       (0.20, 0.85, 0.42, 0.04, 0.03, 140, -4.5, 0.08),
    "stressed":    (0.35, 0.78, 0.45, 0.06, 0.02, 135, -5.5, 0.06),
    "studious":    (0.45, 0.30, 0.28, 0.70, 0.75,  95, -16.0, 0.04),
    "nostalgic":   (0.50, 0.32, 0.42, 0.65, 0.12, 100, -10.0, 0.04),
    "period":      (0.40, 0.28, 0.30, 0.48, 0.05,  88, -12.0, 0.04),
    "chill":       (0.58, 0.32, 0.55, 0.35, 0.08, 100, -10.0, 0.05),
    "romantic":    (0.68, 0.38, 0.60, 0.40, 0.04,  98, -9.0, 0.04),
    "energetic":   (0.62, 0.84, 0.80, 0.06, 0.02, 130, -5.0, 0.07),
    "melancholic": (0.18, 0.24, 0.26, 0.52, 0.10,  85, -13.0, 0.04),
    "confident":   (0.72, 0.74, 0.78, 0.05, 0.01, 120, -6.0, 0.15),
}

# the 20 genres in the UI -> genre tags of the Kaggle "Spotify Tracks Dataset" (track_genre column)
GENRE_MAP = {
    "English Pop": ["pop", "power-pop", "synth-pop"],
    "Spanish": ["spanish"],
    "Hip-Hop": ["hip-hop"],
    "K-Pop": ["k-pop"],
    "R&B / Soul": ["r-n-b", "soul"],
    "Indie": ["indie", "indie-pop"],
    "Classical": ["classical", "opera"],
    "Rock": ["rock", "hard-rock", "rock-n-roll", "psych-rock"],
    "Metal": ["metal", "heavy-metal", "death-metal", "black-metal", "metalcore", "grindcore"],
    "Electronic": ["electronic", "edm", "house", "techno", "trance", "electro", "deep-house",
                   "progressive-house", "dubstep", "drum-and-bass"],
    "Latin": ["latin", "latino", "salsa", "reggaeton"],
    "Jazz": ["jazz"],
    "Alternative": ["alternative", "alt-rock", "grunge", "emo"],
    "Bollywood": ["indian"],
    "French": ["french"],
    "Afrobeats": ["afrobeat"],
    "Portuguese": ["brazil", "pagode", "forro", "sertanejo", "mpb", "samba"],
    "Turkish": ["turkish"],
    "German": ["german"],
    "J-Pop": ["j-pop", "j-idol", "j-rock", "j-dance", "anime"],
}

# Spotify's own mood-flavoured genre tags. They are NOT used for training; they are an independent
# check: do tracks tagged "sad" really get a high probability for sad-type moods?
EDITORIAL_TAGS = {
    "happy": ["happy"], "sad": ["sad", "melancholic"], "sleep": ["chill", "studious", "period"],
    "study": ["studious"], "chill": ["chill"], "romance": ["romantic"],
    "party": ["energetic", "confident", "happy"], "work-out": ["energetic", "angry"],
    "ambient": ["studious", "chill"], "piano": ["studious", "nostalgic", "melancholic", "sad"],
    "death-metal": ["angry"], "black-metal": ["angry"], "grindcore": ["angry"],
    "heavy-metal": ["angry", "stressed"], "hardcore": ["angry", "stressed"], "punk": ["angry", "energetic"],
}


def audio_matrix(rows):
    """rows: DataFrame (or dict of lists) with GMM_FEATURES -> (n, 8) array, skewed features square-rooted"""
    X = pd.DataFrame(rows)[GMM_FEATURES].astype(float).copy()
    for c in SQRT_FEATURES:
        X[c] = np.sqrt(np.clip(X[c], 0, None))
    return X.values


def prototype_matrix():
    return audio_matrix(pd.DataFrame([PROTOTYPES[m] for m in MOODS], columns=GMM_FEATURES))


def norm_title(s):
    s = str(s).lower()
    s = re.sub(r"\(.*?\)|\[.*?\]", " ", s)
    s = re.sub(r"\s+-\s+.*$", " ", s)
    out = re.sub(r"[\W_]+", "", s)
    return out or str(s).strip()


def norm_artist(s):
    first = re.split(r";|,|&| feat\.? | ft\.? | x ", str(s).lower())[0]
    return re.sub(r"[\W_]+", "", first)


def lookup_key(title, artists):
    return norm_title(title) + "|" + norm_artist(artists)


def genre_text(x):
    """'hip-hop' / ['dance pop', 'k-pop'] -> 'hip hop dance pop k pop' (same tokens on both sides)"""
    if isinstance(x, (list, tuple)):
        x = " ".join(map(str, x))
    return " ".join(re.findall(r"[a-z]+", str(x).lower()))


def meta_matrix(genre_texts, titles, explicit, popularity, duration_ms, genre_cv, title_tf):
    """features of the 'estimate' model, used when a track's audio features are not available"""
    G = genre_cv.transform([genre_text(g) for g in genre_texts])
    T = title_tf.transform([str(t) for t in titles])
    num = np.column_stack([np.asarray(explicit, float),
                           np.asarray(popularity, float) / 100.0,
                           np.clip(np.asarray(duration_ms, float) / 60000.0, 0, 10) / 10.0])
    return sparse.hstack([G, T, sparse.csr_matrix(num)]).tocsr()


class SeededGMM:
    """Diagonal Gaussian mixture trained with EM, where every component is one mood.

    Plain EM lets components wander (a component called "angry" can end up as "pop-rock"), so the
    M-step uses a Normal prior on each mean centred at that mood's seed (MAP-EM):
        mean_k = (sum_i r_ik * x_i + kappa * seed_k) / (N_k + kappa)
    kappa = prior_strength * N / K, so the data can move a mood's centre and always reshapes its spread
    and size, but a mood cannot lose its meaning. Weights and variances are learned freely.
    """

    def __init__(self, seeds, prior_strength=0.5, max_iter=200, tol=1e-5, reg_var=1e-2):
        self.seeds = np.asarray(seeds, float)
        self.prior_strength, self.max_iter, self.tol, self.reg_var = prior_strength, max_iter, tol, reg_var

    def _log_joint(self, Z):
        out = np.empty((len(Z), len(self.weights_)))
        for k in range(len(self.weights_)):
            out[:, k] = np.log(self.weights_[k]) - 0.5 * np.sum(
                np.log(2 * np.pi * self.vars_[k]) + (Z - self.means_[k]) ** 2 / self.vars_[k], axis=1)
        return out

    def fit(self, Z):
        n, K = len(Z), len(self.seeds)
        d = ((Z[:, None, :] - self.seeds[None]) ** 2).sum(2) if n * K * Z.shape[1] < 5e7 else \
            np.stack([((Z - s) ** 2).sum(1) for s in self.seeds], axis=1)
        assign = d.argmin(1)
        counts = np.bincount(assign, minlength=K)
        self.means_ = self.seeds.copy()
        self.vars_ = np.stack([np.maximum(Z[assign == k].var(0), self.reg_var) if counts[k] > 50
                               else np.maximum(Z.var(0), self.reg_var) for k in range(K)])
        w = np.maximum(counts / n, 0.01)
        self.weights_ = w / w.sum()
        kappa = self.prior_strength * n / K
        prev, self.n_iter_, self.converged_ = -np.inf, 0, False
        for it in range(self.max_iter):
            lj = self._log_joint(Z)
            lse = logsumexp(lj, axis=1)
            ll = float(lse.mean())
            R = np.exp(lj - lse[:, None])
            Nk = R.sum(0) + 1e-9
            w = np.maximum(Nk / n, 0.005)
            self.weights_ = w / w.sum()
            sx, sxx = R.T @ Z, R.T @ (Z ** 2)
            self.means_ = (sx + kappa * self.seeds) / (Nk[:, None] + kappa)
            var = sxx / Nk[:, None] - 2 * self.means_ * sx / Nk[:, None] + self.means_ ** 2
            self.vars_ = np.maximum(var, self.reg_var)
            self.n_iter_ = it + 1
            if abs(ll - prev) < self.tol:
                self.converged_ = True
                break
            prev = ll
        return self

    def predict_proba(self, Z):
        lj = self._log_joint(np.asarray(Z, float))
        return np.exp(lj - logsumexp(lj, axis=1)[:, None])

    def score(self, Z):
        return float(logsumexp(self._log_joint(np.asarray(Z, float)), axis=1).mean())


# ---------------------------------------------------------------------------------------------
# The dataset's genre tags are noisy (e.g. lots of Hindi/Tamil songs sit under "k-pop", "pop", "jazz").
# train_model.py therefore learns a language/region ("origin") model from the tags that are reliable,
# then uses it to clean every UI genre.
# ---------------------------------------------------------------------------------------------
ORIGIN_TAGS = {
    "indian": ["indian"],
    "turkish": ["turkish"],
    "french": ["french"],
    "german": ["german"],
    "spanish": ["spanish", "salsa", "tango"],
    "portuguese": ["brazil", "pagode", "forro", "sertanejo", "mpb", "samba"],
    "japanese": ["j-pop", "j-rock", "j-idol", "j-dance", "anime"],
    # tags that are (almost) always Western / English-language music
    "western": ["country", "blues", "bluegrass", "honky-tonk", "death-metal", "black-metal", "heavy-metal",
                "metal", "metalcore", "grindcore", "hardcore", "punk", "punk-rock", "rock-n-roll", "rockabilly",
                "hard-rock", "psych-rock", "grunge", "goth", "emo", "industrial", "ska", "reggae", "dancehall",
                "dub", "disco", "funk", "gospel", "show-tunes", "disney", "children", "kids", "comedy",
                "classical", "opera", "piano", "new-age", "trip-hop", "techno", "minimal-techno",
                "detroit-techno", "chicago-house", "drum-and-bass", "dubstep", "hardstyle", "british",
                "swedish"],
}
FOREIGN = {"indian", "turkish", "french", "german", "spanish", "portuguese", "japanese"}

# strict  = keep only tracks the origin model puts in this group (language-defined genres)
# exclude = drop tracks the origin model puts in one of these groups (style-defined genres)
UI_RULES = {g: ("exclude", FOREIGN) for g in GENRE_MAP}
UI_RULES.update({
    "Bollywood": ("strict", {"indian"}),
    "Turkish": ("strict", {"turkish"}),
    "French": ("strict", {"french"}),
    "German": ("strict", {"german"}),
    "Spanish": ("strict", {"spanish"}),
    "Portuguese": ("strict", {"portuguese"}),
    "J-Pop": ("strict", {"japanese"}),
    "Latin": ("exclude", {"indian", "turkish", "french", "german", "japanese"}),
    "K-Pop": ("exclude", {"indian", "turkish", "french", "german", "spanish", "portuguese"}),
})
