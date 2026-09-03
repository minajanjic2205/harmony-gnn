"""
baseline_visestruko.py — LSTM baseline sa višestrukim treniranjem
Autor: Mina Janjić | Predmet: Računarstvo
Opis: Trenira LSTM model 10 puta sa fiksnim, dokumentovanim seedovima,
      računa prosek i standardnu devijaciju za fer poređenje sa GNN modelom.

Namenjeno za pokretanje LOKALNO (na običnom računaru) dok se GNN
grid search paralelno izvršava na Petnica klasteru.
"""

import json
import pickle
import random
from pathlib import Path
from statistics import mean, stdev

import numpy as np
import torch

from config import PUTANJA_OBRADENIH, PUTANJA_REZULTATA
from baseline import treniraj_model, evaluiraj_na_test_skupu

PUTANJA_REZULTATA.mkdir(parents=True, exist_ok=True)


# Fiksna, dokumentovana lista seedova — ISTA kao u grid_search_visestruko.py,
# radi direktnog poređenja LSTM naspram GNN modela.


SEEDOVI = [23, 155, 22, 7, 42, 100, 8, 250, 99, 12]


def postavi_seed(seed: int) -> None:
    """Postavlja seed za sve izvore nasumičnosti radi reproduktivnosti."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



# GLAVNA FUNKCIJA


def pokreni_visestruko_lstm(trening, validacija, test, recnik_akorada, seedovi=SEEDOVI):
    """
    Trenira LSTM model `len(seedovi)` puta, evaluira svaki put na test skupu,
    i vraća listu rezultata (chord_accuracy, top3_accuracy) po seedu.
    """
    ca_rezultati = []
    top3_rezultati = []

    print("\n" + "═" * 60)
    print(f"  LSTM BASELINE — VIŠESTRUKO TRENIRANJE")
    print(f"  Broj ponavljanja: {len(seedovi)}")
    print(f"  Seedovi: {seedovi}")
    print("═" * 60)

    for i, seed in enumerate(seedovi, start=1):
        print(f"\n{'─'*60}")
        print(f"  [{i}/{len(seedovi)}] Seed = {seed}")
        print(f"{'─'*60}")

        postavi_seed(seed)

        model = treniraj_model(
            trening, validacija, recnik_akorada,
            sacuvati_model=False,  # ne trebaju nam svi checkpointi, samo rezultati
        )

        rezultati = evaluiraj_na_test_skupu(model, test, recnik_akorada)

        ca_rezultati.append(rezultati["chord_accuracy"])
        top3_rezultati.append(rezultati["top3_accuracy"])

        print(f"  → Seed {seed}: CA={rezultati['chord_accuracy']:.2%}  "
              f"Top3={rezultati['top3_accuracy']:.2%}")

        # Čuvamo posle svakog seeda (u slučaju prekida)
        privremeno = {
            "seedovi_zavrseni": seedovi[:i],
            "ca_svi": ca_rezultati,
            "top3_svi": top3_rezultati,
        }
        with open(PUTANJA_REZULTATA / "lstm_visestruko_privremeno.json", "w") as f:
            json.dump(privremeno, f, indent=2)

    return ca_rezultati, top3_rezultati


def prikazi_rezultate(ca_rezultati: list[float], top3_rezultati: list[float], seedovi=SEEDOVI) -> None:
    """Ispisuje finalnu statistiku i čuva je u JSON."""

    prosek_ca = mean(ca_rezultati)
    std_ca = stdev(ca_rezultati) if len(ca_rezultati) > 1 else 0.0
    prosek_top3 = mean(top3_rezultati)
    std_top3 = stdev(top3_rezultati) if len(top3_rezultati) > 1 else 0.0

    print("\n" + "═" * 60)
    print("  FINALNI REZULTATI — LSTM BASELINE (10 treniranja)")
    print("═" * 60)
    print(f"\n  Pojedinačni rezultati (Chord Accuracy):")
    for seed, ca in zip(seedovi, ca_rezultati):
        print(f"    Seed {seed:>4}: {ca:.2%}")

    print(f"\n  ═══ PROSEK ═══")
    print(f"  Chord Accuracy : {prosek_ca:.2%} ± {std_ca:.2%}")
    print(f"  Top-3 Accuracy : {prosek_top3:.2%} ± {std_top3:.2%}")
    print("═" * 60)

    rezultat_finalni = {
        "model": "LSTM Baseline (E1)",
        "broj_ponavljanja": len(seedovi),
        "seedovi": seedovi,
        "ca_prosek": prosek_ca,
        "ca_std": std_ca,
        "ca_svi": ca_rezultati,
        "top3_prosek": prosek_top3,
        "top3_std": std_top3,
        "top3_svi": top3_rezultati,
    }

    with open(PUTANJA_REZULTATA / "lstm_visestruko.json", "w") as f:
        json.dump(rezultat_finalni, f, indent=2)

    print(f"\n[INFO] Rezultati sačuvani u: rezultati/lstm_visestruko.json")



# POKRETANJE


if __name__ == "__main__":
    with open(PUTANJA_OBRADENIH / "trening.pkl", "rb") as f:
        trening = pickle.load(f)
    with open(PUTANJA_OBRADENIH / "validacija.pkl", "rb") as f:
        validacija = pickle.load(f)
    with open(PUTANJA_OBRADENIH / "test.pkl", "rb") as f:
        test = pickle.load(f)
    with open(PUTANJA_OBRADENIH / "recnik_akorada.json", "r") as f:
        recnik_akorada = json.load(f)

    print(f"[INFO] Uređaj: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    ca_rezultati, top3_rezultati = pokreni_visestruko_lstm(
        trening, validacija, test, recnik_akorada,
    )

    prikazi_rezultate(ca_rezultati, top3_rezultati)