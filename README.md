# FlipPhone Training

Trainiert einen Random-Forest-Klassifikator, um Skateboard-Tricks anhand von Beschleunigungs- und Gyroskopdaten zu erkennen.

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

## Projektstruktur

```
fetch_data.py       # Daten vom Server holen
train.py            # Feature-Extraktion & Training
requirements.txt    # Python-Abhaengigkeiten
data/               # CSV-Daten (gitignored)
models/             # Trainierte Modelle (gitignored)
```
