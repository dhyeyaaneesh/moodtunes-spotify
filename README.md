# MoodTunes

MoodTunes is a mood-based music discovery and analysis web app that lets you:
- Discover songs based on how you feel
- Analyze any song’s mood using real Spotify audio features
- Explore curated playlists across moods and genres

## Features

### Mood-Based Discovery
- Select your current mood
- Choose up to 3 genres
- Get curated song recommendations

### Song Mood Analyzer
- Search any song using Spotify
- Analyze real audio features:
  - Valence (positivity)
  - Energy
  - Danceability
  - Instrumentalness
- Get a predicted mood based on these features

### Spotify Integration
- OAuth login with Spotify
- Uses Spotify Web API for:
  - Search
  - Audio features
- Token stored locally (no backend)

### User System (Local)
- Login / Signup (stored in browser)
- Profile editing

---

## Tech Stack

- HTML, CSS, JavaScript
- Spotify Web API (OAuth + audio features)
- LocalStorage (for auth + user data)

---

## Privacy
- No server, no tracking
- Data stored locally
- Spotify handles API requests

---

## License
- MIT

---

This project uses the Spotify Web API but is not affiliated with Spotify.
