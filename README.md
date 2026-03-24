# FlipPhone Training

Trainiert einen Random-Forest-Klassifikator, um Skateboard-Tricks anhand von Beschleunigungs- und Gyroskopdaten zu erkennen. Stellt das trainierte Modell als REST API bereit.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Verwendung

### 1. Daten laden

```bash
python fetch_data.py --url https://your-server.com --key fp_yourAdminKey
```

Speichert das Dataset als `data/dataset.csv`.

### 2. Tricks auswaehlen

In `train.py` die Liste `SELECTED_TRICKS` anpassen, um festzulegen welche Tricks trainiert werden:

```python
SELECTED_TRICKS = [
    "Kickflip",
    "Treflip",
    "FS Shuvit",
]
```

### 3. Training starten

```bash
python train.py
```

Ausgabe:
- Classification Report (Precision, Recall, F1)
- Confusion Matrix
- Top 10 wichtigste Features
- Gespeichertes Modell unter `models/rf_model.pkl`

### 4. Prediction API starten

```bash
python server.py          # startet auf Port 8000
PORT=5000 python server.py  # custom Port
```

#### `POST /api/predict`

Request:
```json
{
  "samples": [
    {"t": 0.0, "ax": 1.2, "ay": -0.5, "az": 9.8, "gx": 0.1, "gy": -0.3, "gz": 0.02},
    {"t": 10.0, "ax": 1.5, "ay": -0.8, "az": 9.5, "gx": 0.5, "gy": -1.2, "gz": 0.1}
  ]
}
```

Response:
```json
{
  "trick": "Kickflip",
  "confidence": 0.92,
  "probabilities": {"Kickflip": 0.92, "Treflip": 0.05, "FS Shuvit": 0.03}
}
```

## Deployment

Push auf `main` deployt automatisch per GitHub Actions auf den Hetzner Server. Der Service wird unter `/opt/flipphone-training` ausgecheckt und via systemd als `flipphone-training` verwaltet.

## Projektstruktur

```
fetch_data.py       # Daten vom Server holen
train.py            # Feature-Extraktion & Training
server.py           # FastAPI Prediction API
requirements.txt    # Python-Abhaengigkeiten
data/               # CSV-Daten (gitignored)
models/             # Trainierte Modelle (gitignored)
.github/workflows/  # Auto-Deploy auf Hetzner
```
