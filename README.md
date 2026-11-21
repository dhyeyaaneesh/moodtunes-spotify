# 🎵 MoodTunes — mood-based music discovery

Pick a mood and up to 3 genres and get songs that match, or search any song on Spotify and see what mood it carries. Recommendations and mood analysis now come from **models trained on the Kaggle Spotify tracks dataset** (audio features + genre tags for ~114,000 tracks) instead of hard-coded song lists and if/else rules.

## What is actually trained

| Feature on the site | Model |
|---|---|
| **Song recommendations** (mood + genres) | Every track is scored for all 12 moods by the mood model. For a mood and genre the app ranks tracks by `P(mood | audio features) × popularity` and draws a weighted random sample from the top of the list, so each shuffle is different and avoids songs already shown. |
| **Song analyser** (audio features available) | **Mood model:** a 12-component diagonal Gaussian mixture on 8 audio features (valence, energy, danceability, acousticness, instrumentalness, tempo, loudness, speechiness), trained with EM. Every component is one mood and starts at a seed point from Russell's circumplex model; a Normal prior on each mean keeps it from drifting away from its meaning. Output = probability of each mood. |
| **Song analyser** (Spotify returns no audio features) | The song is looked up in the dataset (title + artist) and its real audio features are used. If it is not there, an **estimate model** (logistic regression + ridge regressions on genre tokens, title words, explicit flag, popularity, duration) predicts the mood and the audio-feature bars. |

There are **no mood labels** in the dataset, so the mood model is unsupervised (seeded EM). It is checked against Spotify's own mood-flavoured genre tags (`happy`, `sad`, `sleep`, `study`, `chill`, `romance`, `party`, `work-out`, `ambient`, `piano`, `death-metal`, ...), which are never used for training: for each tag, how well does the probability of the matching moods separate tagged tracks from the rest (AUC)? The estimate model is scored against the mood model's own labels on held-out tracks.

## Project structure

```
moodtunes-spotify/
├── moodtunes.html        website (UI unchanged, now calls the API)
├── app.py                FastAPI backend: loads the trained models, serves the site
├── train_model.py        trains everything
├── core.py               shared code (moods, seeds, genre map, seeded Gaussian mixture)
├── requirements.txt
├── data/                 put the Kaggle CSV here (not pushed to git)
└── models/               created by train_model.py (not pushed to git)
```

## Run it

1. Download the Kaggle **Spotify Tracks Dataset** (CSV with a `track_genre` column) and put it in `data/`.
2. Install and train:

```bash
python -m venv venv
venv\Scripts\activate            # Windows   (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt
python train_model.py
```

3. Start the site and open <http://127.0.0.1:8000>:

```bash
uvicorn app:app --reload
```

Spotify login (PKCE) for the analyser works as before; add `http://127.0.0.1:8000` as a redirect URI in your Spotify app settings if you use the local server.

## Things to check after training

- `every mood kept its meaning` is printed (if a `WARNING: ... drifted` line appears, a mood's centre moved closer to another mood's seed).
- The `Validation` table: AUC well above 0.5 for the Spotify tags is a good sign.
- The `Song catalogue` table: every UI genre should have tracks. If one shows `0`, its genre tags in `GENRE_MAP` (`core.py`) do not exist in your file.

## Results

After training, copy the numbers from `models/info.json` (or the training printout) here: tag-validation AUCs, estimate-model accuracy and R².

## API

| Endpoint | Purpose |
|---|---|
| `GET /api/songs?mood=sad&genres=English Pop,Jazz&exclude=id1,id2` | 15 recommended songs |
| `POST /api/analyze` | Mood of a track (from audio features, a dataset lookup, or the estimate model) |
| `GET /api/info` | Training summary and validation numbers |

Interactive docs: <http://127.0.0.1:8000/docs>

## Limitations

- Mood is subjective and the dataset has no mood labels; several moods overlap in audio space (sad / melancholic / period, happy / energetic / confident), so the model separates them softly. Percentages are model probabilities, not accuracy.
- Spotify has restricted the audio-features endpoint for newer apps; that is why the analyser falls back to the dataset lookup and the estimate model.
- Genre buckets are approximate (for example Bollywood uses the dataset's `indian` tag).
