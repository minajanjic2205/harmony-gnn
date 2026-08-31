# Predviđanje harmonije iz melodije korišćenjem GNN
## Šta je do sada implementirano

### Faza 1 Preprocesiranje podataka (`dataset.py`)

- Učitavanje Nottingham Music Dataset-a iz `music21` corpus-a (ili lokalnih `.abc` fajlova)
- Parsovanje ABC notacije u **MIDI pitch klase** (0–11) i trajanja putem `music21`
- Enkodovanje akorda (dur/mol trijade) u celobrojne indekse sa globalnim rečnikom
- Deterministička podela: **trening 70% / validacija 15% / test 15%**
- Funkcija `provera_skupa_podataka()` koja ispisuje:
  - ukupan broj pesama i nota
  - rečnik akorda sa distribucijom učestalosti
  - distribuciju melodijskih pitch klasa
- Keširani izlaz u `podaci/obradeni/` (`.pkl` + `.json`)

### Faza 2  LSTM Baseline model (`baseline.py`)

- `MelodijaAkordDataset`: klizeći prozor (sliding window) dužine 16 nota → jedan akord kao cilj
- `LSTMPredvidjanjAkorda`: Embedding → stacked LSTM (2 sloja, 128 jedinica) → Dropout → Linear
- `Weighted CrossEntropyLoss`: neutrališe dominantnost klase `<NEPOZNAT>` (indeks 0)
- `ReduceLROnPlateau` raspored stope učenja + gradient clipping
- Automatsko čuvanje najboljeg checkpointa u `modeli/lstm_baseline_najbolji.pt`
- Evaluacija: **Chord Accuracy (CA)** i **Top-3 Accuracy** na test skupu
---

## Instalacija zavisnosti

```bash
pip install music21 torch torchvision torch-geometric
pip install mir_eval scikit-learn numpy pandas matplotlib
```

> **Napomena:** Za GPU podrška instalirajte PyTorch sa odgovarajućom CUDA verzijom.  
> Google Colab (besplatni GPU tier) je dovoljan za ovaj dataset.

---

## Kako pokrenuti

### 1. Preprocesiranje podataka

```bash
python dataset.py
```

Očekivani izlaz:
```
[INFO] Tražim Nottingham pesme u music21 corpus-u...
[INFO] Uspešno učitano 1034 pesama.
[INFO] Veličina rečnika akorda: 31 (uključuje '<NEPOZNAT>')
[INFO] Podela: trening=723, validacija=155, test=156
═══════════════════════════════════════════════════════════
  PROVERA INTEGRITETA — TRENING SKUP
...
```

Ako Nottingham nije u corpus-u, preuzmite ručno i pozovite sa lokalnom putanjom:

```python
from dataset import pripremi_skup_podataka
from pathlib import Path

trening, val, test, recnik = pripremi_skup_podataka(
    lokalni_abc_dir=Path("podaci/nottingham/ABC"),
    forsirati_ponovnu_obradu=True,
)
```

### 2. Treniranje LSTM baseline modela

```bash
python baseline.py
```

Očekivani izlaz:
```
 Epoha    Trening Gubitak     Val Gubitak     Val Tačnost
──────────────────────────────────────────────────────────
     1           2.8432          2.7901           14.32%
    ...
    30           1.2103          1.3420           52.18%
══════════════════════════════════════════════════════════
  REZULTATI NA TEST SKUPU (E1 — LSTM Baseline)
  Chord Accuracy (CA) : 51.87%
  Top-3 Accuracy      : 74.23%
```

### 3. Brza provera bez pokretanja treniranja

```python
from dataset import pripremi_skup_podataka, provera_skupa_podataka

trening, val, test, recnik = pripremi_skup_podataka()
provera_skupa_podataka(trening + val + test, recnik, naziv_skupa="ceo skup")
```

---

## Struktura projekta

```
projekat/
├── dataset.py              # Faza 1: preprocesiranje
├── baseline.py             # Faza 2: LSTM baseline (E1)
├── podaci/
│   ├── nottingham/         # sirovi ABC fajlovi (opcionalno)
│   └── obradeni/
│       ├── trening.pkl
│       ├── validacija.pkl
│       ├── test.pkl
│       └── recnik_akorada.json
└── modeli/
    └── lstm_baseline_najbolji.pt
```

---

## Sledeći koraci — Faza 3: Konstrukcija tonalnog grafa

Prema sekciji 4.2 projektnog predloga, potrebno je implementirati heterogeni neusmeren graf  
`G = (V, E, W)` u novom fajlu `graf.py`.

### 3.1 Čvorovi grafa

```python
# Tip A: Čvorovi nota: 12 hromatskih pitch klasa
cvorovi_nota = list(range(12))  # C=0, C#=1, ..., B=11

# Tip B : Čvorovi akorda: svi akordi iz recnik_akorada
#          (isključiti NEPOZNAT_AKORD sa indeksom 0)
cvorovi_akorda = [naziv for naziv, idx in recnik_akorada.items() if idx > 0]
```

### 3.2 Grane: Tip 1: Teorijska pripadnost note akordu

Binarna grana nota↔akord ako nota ulazi u sastav trijade (koristiti `music21.harmony`):

```python
from music21 import harmony

def note_u_akordu(naziv_akordu: str) -> list[int]:
    """Vraća pitch klase nota koje čine triadni akord."""
    cs = harmony.ChordSymbol(naziv_akordu)
    return [p.midi % 12 for p in cs.pitches]
```

### 3.3 Grane: Tip 2: Kvintni krug (teorijska blizina)

```python
# Kvintni krug: svakih 7 polustepeni u smeru kazaljke
KVINTNI_KRUG = [0, 7, 2, 9, 4, 11, 6, 1, 8, 3, 10, 5]  # C G D A E B F# C# G# D# A# F

def rastojanje_na_kvintu(akord_a: str, akord_b: str) -> int:
    """Vraća broj koraka između dva tonaliteta na kvintnom krugu (0–6)."""
    ...

# Težina grane Tipa 2: w = 1 / (1 + d)
```

### 3.4 Grane: Tip 3: Statistička ko-pojava

```python
# Iz trening podataka: p(akord | pitch_klasa) = broj ko-pojava / ukupno pojava note
def izracunaj_ko_pojave(trening: list[dict]) -> dict[tuple, float]:
    """Vraća normalizovanu uslovnu verovatnoću p(akord | nota) iz trening skupa."""
    ...
```

### 3.5 Težinska funkcija (ključni hiperparametar)

```python
def izracunaj_tezinu(
    teorijska_vrednost: float,
    statisticka_vrednost: float,
    alfa: float,  # α ∈ [0, 1], α + β = 1
) -> float:
    return alfa * teorijska_vrednost + (1 - alfa) * statisticka_vrednost
```

### 3.6 PyTorch Geometric reprezentacija

Graf se konvertuje u `torch_geometric.data.HeteroData` objekat sa:
- `node_types`: `['nota', 'akord']`  
- `edge_types`: `[('nota', 'pripada', 'akord'), ('akord', 'blizina', 'akord'), ('nota', 'kopojava', 'akord')]`
- `edge_attr`: tenzori težina za svaki tip grane

### 3.7 Validacija grafa pre treniranja

Proveriti:
- Da li svaka pitch klasa ima barem jednu granu ka nekom akordu (Tip 1)?
- Da li Cmaj↔Gmaj grana postoji sa očekivanom težinom `w = 1/(1+1) = 0.5`?
- Da li je graf simetričan (neusmeren)?

```bash
# Pokrenuti validaciju:
python graf.py --validacija
```

---

## Metrike evaluacije (podsećanje)

| Metrika | Implementacija | Cilj |
|---------|---------------|------|
| Chord Accuracy (CA) | `baseline.py → evaluiraj_na_test_skupu()` | E1 referentna tačka |
| Top-3 Accuracy | isto | dodatni uvid |
| Harmonska sličnost | `mir_eval.chord` | greška C→Am ≠ C→F# |
| α/β grid search | `grid_search.py` (Faza 4) | optimalni balans teorija/statistika |