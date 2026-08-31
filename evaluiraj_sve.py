"""
evaluiraj_sve.py — Brza evaluacija svih već istreniranih modela
Učitava postojeće modele i evaluira ih bez ponovnog treniranja.
"""

import json
import pickle
import matplotlib.pyplot as plt
from pathlib import Path

import torch

from config import VREDNOSTI_ALFA, PUTANJA_OBRADENIH, PUTANJA_GRAFA, PUTANJA_MODELA, PUTANJA_REZULTATA
from gnn_model import GNNPredvidjanjAkorda, evaluiraj_gnn, napravi_mapu

PUTANJA_REZULTATA.mkdir(parents=True, exist_ok=True)

# Učitavamo podatke
with open(PUTANJA_OBRADENIH / "test.pkl", "rb") as f:
    test = pickle.load(f)
with open(PUTANJA_OBRADENIH / "recnik_akorada.json", "r") as f:
    recnik_akorada = json.load(f)
with open(PUTANJA_GRAFA / "indeksi_akorada.json", "r") as f:
    indeksi_akorada = json.load(f)

rezultati = []

for alfa in VREDNOSTI_ALFA:
    # Tražimo model fajl
    putanja = PUTANJA_MODELA / f"gnn_GS_alfa{alfa:.2f}_alfa{alfa:.2f}.pt"
    if not putanja.exists():
        putanja = PUTANJA_MODELA / f"gnn_GS_alfa{alfa:.2f}.pt"
    if not putanja.exists():
        print(f"[PRESKAČEM] Model za α={alfa:.2f} nije pronađen.")
        continue

    # Učitavamo graf
    putanja_grafa = PUTANJA_GRAFA / f"tonalni_graf_alfa{alfa:.2f}.pt"
    if not putanja_grafa.exists():
        print(f"[PRESKAČEM] Graf za α={alfa:.2f} nije pronađen.")
        continue

    graf = torch.load(putanja_grafa, weights_only=False)

    # Učitavamo model
    checkpoint = torch.load(putanja, weights_only=False)
    model = GNNPredvidjanjAkorda(len(indeksi_akorada))
    model.load_state_dict(checkpoint["stanje"])
    model.eval()

    # Evaluacija
    print(f"[INFO] Evaluiram α={alfa:.2f}...")
    metrike = evaluiraj_gnn(model, test, graf, indeksi_akorada, recnik_akorada)

    rezultati.append({
        "alfa": alfa,
        "beta": round(1 - alfa, 1),
        "chord_accuracy": metrike["chord_accuracy"],
        "top3_accuracy": metrike["top3_accuracy"],
    })

# Štampamo tabelu
print("\n" + "═" * 55)
print(f"  {'Alfa':>6}  {'Beta':>6}  {'Chord Accuracy':>16}  {'Top-3':>8}")
print("─" * 55)

lstm_ca = 0.9019
najbolji = max(rezultati, key=lambda r: r["chord_accuracy"])

for r in rezultati:
    marker = " ← BEST" if r["alfa"] == najbolji["alfa"] else ""
    print(f"  {r['alfa']:>6.2f}  {r['beta']:>6.2f}  "
          f"{r['chord_accuracy']:>15.2%}  {r['top3_accuracy']:>7.2%}{marker}")

print("═" * 55)
print(f"\n  Optimalna alfa : {najbolji['alfa']:.2f}")
print(f"  Chord Accuracy : {najbolji['chord_accuracy']:.2%}")
print(f"  LSTM Baseline  : {lstm_ca:.2%}")
print(f"  Poboljšanje    : {(najbolji['chord_accuracy'] - lstm_ca):+.2%}")

# Čuvamo JSON
with open(PUTANJA_REZULTATA / "grid_search_rezultati.json", "w") as f:
    json.dump(rezultati, f, indent=2)
print(f"\n[INFO] Rezultati sačuvani u rezultati/grid_search_rezultati.json")