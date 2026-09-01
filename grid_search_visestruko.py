"""
grid_search_visestruko.py grid search sa višestrukim treniranjem (statistička pouzdanost)
Opis: Za svaku vrednost alfe trenira model N puta sa različitim seedovima,
      računa prosek i standardnu devijaciju. Rešava problem stohastičnosti
      pojedinačnog treniranja (jedno treniranje ≠ pouzdan zaključak).

Napomena: Namenjeno za pokretanje na Petnica klasteru (GPU) zbog brzine.
"""

import json
import pickle
import random
from pathlib import Path
from statistics import mean, stdev

import numpy as np
import torch
import matplotlib.pyplot as plt

from config import PUTANJA_OBRADENIH, PUTANJA_GRAFA, PUTANJA_REZULTATA
from gnn_model import treniraj_gnn, evaluiraj_gnn
from graf import izgradi_hetero_graf, sacuvaj_graf

PUTANJA_REZULTATA.mkdir(parents=True, exist_ok=True)


# KONFIGURACIJA


VREDNOSTI_ALFA = [round(a * 0.1, 1) for a in range(11)]  # 0.0 do 1.0
BROJ_PONAVLJANJA = 10   # koliko puta treniramo svaku alfu (menjaj na 50 kasnije)

# Fiksna, dokumentovana lista seedova — dogovoreno sa mentorkom.
# Isti seedovi se koriste kroz ceo projekat radi reproduktivnosti:
# pokretanje sa istim seedom uvek daje identičan rezultat, što omogućava
# proveru i poređenje rezultata u budućnosti.
SEEDOVI = [23, 155, 22, 7, 42, 100, 8, 250, 99, 12]


def postavi_seed(seed: int) -> None:
    """Postavlja seed za sve izvore nasumičnosti radi reproduktivnosti."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



# GLAVNA FUNKCIJA


def pokreni_visestruki_grid_search(
    trening, validacija, test, recnik_akorada,
    vrednosti_alfa=VREDNOSTI_ALFA,
    broj_ponavljanja=BROJ_PONAVLJANJA,
):
    """
    Za svaku alfu trenira model `broj_ponavljanja` puta sa različitim seedovima.
    Vraća listu rečnika sa prosekom i standardnom devijacijom po alfi.
    """
    rezultati_finalni = []

    print("\n" + "═" * 65)
    print(f"  GRID SEARCH SA VIŠESTRUKIM TRENIRANJEM")
    print(f"  Vrednosti alfe: {vrednosti_alfa}")
    print(f"  Ponavljanja po alfi: {broj_ponavljanja}")
    print(f"  Ukupno treniranja: {len(vrednosti_alfa) * broj_ponavljanja}")
    print("═" * 65)

    # Gradimo grafove jednom za svaku alfu (graf ne zavisi od seeda)
    grafovi = {}
    indeksi_akorada_svi = {}

    for alfa in vrednosti_alfa:
        putanja_grafa = PUTANJA_GRAFA / f"tonalni_graf_alfa{alfa:.2f}.pt"
        if putanja_grafa.exists():
            graf = torch.load(putanja_grafa, weights_only=False)
            with open(PUTANJA_GRAFA / "indeksi_akorada.json", "r") as f:
                indeksi_akorada = json.load(f)
        else:
            graf, indeksi_akorada = izgradi_hetero_graf(
                recnik_akorada=recnik_akorada, trening=trening, alfa=alfa,
            )
            sacuvaj_graf(graf, indeksi_akorada, alfa)
        grafovi[alfa] = graf
        indeksi_akorada_svi[alfa] = indeksi_akorada

    # Za svaku alfu, treniramo N puta
    for alfa in vrednosti_alfa:
        print(f"\n{'─'*65}")
        print(f"  ALFA = {alfa:.2f}  (β = {1-alfa:.2f})")
        print(f"{'─'*65}")

        graf = grafovi[alfa]
        indeksi_akorada = indeksi_akorada_svi[alfa]

        ca_rezultati = []
        top3_rezultati = []

        for i, seed in enumerate(SEEDOVI[:broj_ponavljanja], start=1):
            print(f"\n  [{i}/{broj_ponavljanja}] Seed={seed}...")
            postavi_seed(seed)

            model = treniraj_gnn(
                trening, validacija, graf, indeksi_akorada, recnik_akorada,
                alfa=alfa, naziv=f"MULTI_alfa{alfa:.2f}_seed{seed}",
            )
            metrike = evaluiraj_gnn(model, test, graf, indeksi_akorada, recnik_akorada)

            ca_rezultati.append(metrike["chord_accuracy"])
            top3_rezultati.append(metrike["top3_accuracy"])

            print(f"  → CA={metrike['chord_accuracy']:.2%}  Top3={metrike['top3_accuracy']:.2%}")

        # Statistika za ovu alfu
        prosek_ca = mean(ca_rezultati)
        std_ca = stdev(ca_rezultati) if len(ca_rezultati) > 1 else 0.0
        prosek_top3 = mean(top3_rezultati)
        std_top3 = stdev(top3_rezultati) if len(top3_rezultati) > 1 else 0.0

        print(f"\n  ═══ REZULTAT ZA α={alfa:.2f} ═══")
        print(f"  Chord Accuracy: {prosek_ca:.2%} ± {std_ca:.2%}")
        print(f"  Top-3 Accuracy: {prosek_top3:.2%} ± {std_top3:.2%}")

        rezultati_finalni.append({
            "alfa": alfa,
            "beta": round(1 - alfa, 2),
            "ca_prosek": prosek_ca,
            "ca_std": std_ca,
            "ca_svi": ca_rezultati,
            "top3_prosek": prosek_top3,
            "top3_std": std_top3,
            "top3_svi": top3_rezultati,
        })

        # Čuvamo posle svake alfe (u slučaju prekida)
        with open(PUTANJA_REZULTATA / "grid_search_visestruko.json", "w") as f:
            json.dump(rezultati_finalni, f, indent=2)

    return rezultati_finalni


def prikazi_finalne_rezultate(rezultati: list[dict]) -> None:
    """Ispisuje tabelu i crta grafikon sa error barovima."""

    print("\n" + "═" * 70)
    print(f"  {'Alfa':>6}  {'CA prosek':>12}  {'CA std':>10}  {'Top3 prosek':>12}")
    print("─" * 70)

    najbolji = max(rezultati, key=lambda r: r["ca_prosek"])

    for r in rezultati:
        marker = " ← BEST" if r["alfa"] == najbolji["alfa"] else ""
        print(f"  {r['alfa']:>6.2f}  {r['ca_prosek']*100:>11.2f}%  "
              f"±{r['ca_std']*100:>8.2f}%  {r['top3_prosek']*100:>11.2f}%{marker}")

    print("═" * 70)
    print(f"\n  Optimalna alfa: {najbolji['alfa']:.2f}")
    print(f"  CA: {najbolji['ca_prosek']:.2%} ± {najbolji['ca_std']:.2%}")

    # Grafikon sa error barovima
    alfas = [r["alfa"] for r in rezultati]
    ca_proseci = [r["ca_prosek"] * 100 for r in rezultati]
    ca_stdovi = [r["ca_std"] * 100 for r in rezultati]

    fig, ax = plt.subplots(figsize=(11, 6))

    ax.errorbar(alfas, ca_proseci, yerr=ca_stdovi, fmt="o-",
                color="#2196F3", linewidth=2.5, markersize=8,
                capsize=5, capthick=1.5, label="GNN Chord Accuracy (prosek ± std)")

    ax.axhline(y=90.19, color="#F44336", linestyle="--",
               linewidth=2, label="LSTM Baseline (90.19%)")

    ax.scatter([najbolji["alfa"]], [najbolji["ca_prosek"]*100],
               color="#4CAF50", s=150, zorder=5)

    ax.set_xlabel("Alfa (α) — balans teorija ↔ statistika", fontsize=12)
    ax.set_ylabel("Chord Accuracy (%)", fontsize=12)
    ax.set_title(f"Uticaj α na tačnost — prosek od {BROJ_PONAVLJANJA} treniranja po alfi\n"
                 "Error bar = standardna devijacija", fontsize=13)
    ax.set_xticks(alfas)
    ax.set_xticklabels([f"{a:.1f}" for a in alfas])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(PUTANJA_REZULTATA / "09_grid_search_visestruko.png", dpi=150)
    plt.close()
    print(f"\n[INFO] Grafikon sačuvan: rezultati/09_grid_search_visestruko.png")



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

    rezultati = pokreni_visestruki_grid_search(
        trening, validacija, test, recnik_akorada,
        broj_ponavljanja=BROJ_PONAVLJANJA,
    )

    prikazi_finalne_rezultate(rezultati)

    print(f"\n[INFO] Svi rezultati sačuvani u: rezultati/grid_search_visestruko.json")