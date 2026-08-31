"""
Generisanje svih grafika za rad
"""

import json
import pickle
from pathlib import Path
from collections import Counter

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from config import PUTANJA_REZULTATA, PUTANJA_OBRADENIH

PUTANJA_REZULTATA.mkdir(parents=True, exist_ok=True)

# Boje za konzistentnost kroz sve grafike
BOJA_LSTM   = "#F44336"   # crvena
BOJA_GNN    = "#2196F3"   # plava
BOJA_OPT    = "#4CAF50"   # zelena
BOJA_MUTED  = "#90A4AE"   # siva


# PODACI


# Grid search rezultati
alfas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
ca    = [89.68, 90.19, 89.44, 90.57, 90.00, 89.87, 90.11, 89.44, 89.84, 90.46, 89.98]
top3  = [98.58, 98.76, 98.47, 98.90, 98.71, 98.79, 98.71, 97.96, 98.76, 98.79, 98.52]

lstm_ca   = 90.19
lstm_top3 = 99.03
gnn_ca    = 90.57   # α=0.30
gnn_top3  = 98.90

# Učitavamo pesme za analizu dataseta
with open(PUTANJA_OBRADENIH / "trening.pkl", "rb") as f:
    trening = pickle.load(f)
with open(PUTANJA_OBRADENIH / "validacija.pkl", "rb") as f:
    validacija = pickle.load(f)
with open(PUTANJA_OBRADENIH / "test.pkl", "rb") as f:
    test = pickle.load(f)
with open(PUTANJA_OBRADENIH / "recnik_akorada.json", "r") as f:
    recnik = json.load(f)

sve_pesme = trening + validacija + test
inv_recnik = {v: k for k, v in recnik.items()}



# GRAFIK 1: Grid search — uticaj alfe na tačnost


def grafik_grid_search():
    fig, ax = plt.subplots(figsize=(11, 6))

    ax.plot(alfas, ca, "o-", color=BOJA_GNN, linewidth=2.5,
            markersize=8, label="GNN Chord Accuracy", zorder=3)

    ax.axhline(y=lstm_ca, color=BOJA_LSTM, linestyle="--",
               linewidth=2, label=f"LSTM Baseline ({lstm_ca:.2f}%)", zorder=2)

    # Optimum
    ax.scatter([0.3], [90.57], color=BOJA_OPT, s=150, zorder=5)
    ax.annotate(
        f"Optimum α=0.30\n(90.57%)",
        xy=(0.3, 90.57), xytext=(0.42, 90.45),
        fontsize=10, color=BOJA_OPT,
        arrowprops=dict(arrowstyle="->", color=BOJA_OPT, lw=1.5),
    )

    # Zona ispod baseline
    ax.fill_between(alfas, ca, lstm_ca,
                    where=[c < lstm_ca for c in ca],
                    alpha=0.15, color=BOJA_LSTM, label="Ispod baseline")
    ax.fill_between(alfas, ca, lstm_ca,
                    where=[c >= lstm_ca for c in ca],
                    alpha=0.15, color=BOJA_GNN, label="Iznad baseline")

    ax.set_xlabel("Alfa (α) — balans teorija ↔ statistika", fontsize=12)
    ax.set_ylabel("Chord Accuracy (%)", fontsize=12)
    ax.set_title("Uticaj parametra α na tačnost predviđanja akorda\n"
                 "α=0.0: čisto statistički  |  α=1.0: čisto teorijski", fontsize=13)
    ax.set_xticks(alfas)
    ax.set_xticklabels([f"{a:.1f}" for a in alfas])
    ax.set_ylim(89.0, 91.0)
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(PUTANJA_REZULTATA / "01_grid_search.png", dpi=150)
    plt.close()
    print("[INFO] Sačuvan: 01_grid_search.png")



# GRAFIK 2: Poređenje modela — bar chart


def grafik_poredjenje_modela():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    modeli = ["LSTM\nBaseline (E1)", "GNN α=0.0\n(čisto statistički)",
              "GNN α=0.3\n(optimum, E3)", "GNN α=1.0\n(čisto teorijski)"]
    ca_vrednosti = [lstm_ca, 89.68, gnn_ca, 89.98]
    top3_vrednosti = [lstm_top3, 98.58, gnn_top3, 98.52]
    boje = [BOJA_LSTM, BOJA_MUTED, BOJA_OPT, BOJA_MUTED]

    # Chord Accuracy
    bars1 = ax1.bar(modeli, ca_vrednosti, color=boje, edgecolor="white",
                    linewidth=1.5, width=0.6)
    ax1.set_ylim(89.0, 91.2)
    ax1.set_ylabel("Chord Accuracy (%)", fontsize=11)
    ax1.set_title("Chord Accuracy — poređenje modela", fontsize=12)
    ax1.axhline(y=lstm_ca, color=BOJA_LSTM, linestyle="--", alpha=0.5, linewidth=1)
    for bar, val in zip(bars1, ca_vrednosti):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
                 f"{val:.2f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax1.grid(True, alpha=0.3, axis="y")

    # Top-3 Accuracy
    bars2 = ax2.bar(modeli, top3_vrednosti, color=boje, edgecolor="white",
                    linewidth=1.5, width=0.6)
    ax2.set_ylim(97.5, 99.5)
    ax2.set_ylabel("Top-3 Accuracy (%)", fontsize=11)
    ax2.set_title("Top-3 Accuracy — poređenje modela", fontsize=12)
    for bar, val in zip(bars2, top3_vrednosti):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
                 f"{val:.2f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax2.grid(True, alpha=0.3, axis="y")

    plt.suptitle("Poređenje performansi modela na test skupu", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(PUTANJA_REZULTATA / "02_poredjenje_modela.png", dpi=150)
    plt.close()
    print("[INFO] Sačuvan: 02_poredjenje_modela.png")



# GRAFIK 3: Distribucija akorda u datasetu


def grafik_distribucija_akorda():
    brojac = Counter()
    for pesma in sve_pesme:
        for idx in pesma.get("akordi_enkodovani", []):
            if idx > 0:
                naziv = inv_recnik.get(idx, "?")
                if naziv != "<NEPOZNAT>":
                    brojac[naziv] += 1

    top_akordi = brojac.most_common(15)
    nazivi = [a[0] for a in top_akordi]
    brojevi = [a[1] for a in top_akordi]

    # Dur/mol boje
    boje = [BOJA_GNN if "maj" in n else BOJA_LSTM for n in nazivi]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(nazivi, brojevi, color=boje, edgecolor="white", linewidth=1.2)

    ax.set_xlabel("Akord", fontsize=12)
    ax.set_ylabel("Broj pojavljivanja", fontsize=12)
    ax.set_title("Distribucija akorda u Nottingham datasetu (top 15)", fontsize=13)

    # Legenda
    dur_patch = mpatches.Patch(color=BOJA_GNN, label="Dur akordi (major)")
    mol_patch = mpatches.Patch(color=BOJA_LSTM, label="Mol akordi (minor)")
    ax.legend(handles=[dur_patch, mol_patch], fontsize=10)

    for bar, val in zip(bars, brojevi):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                str(val), ha="center", va="bottom", fontsize=8)

    ax.grid(True, alpha=0.3, axis="y")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(PUTANJA_REZULTATA / "03_distribucija_akorda.png", dpi=150)
    plt.close()
    print("[INFO] Sačuvan: 03_distribucija_akorda.png")



# GRAFIK 4: Distribucija melodijskih nota


def grafik_distribucija_nota():
    NAZIVI_NOTA = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    brojac_nota = Counter()

    for pesma in sve_pesme:
        for pk, _ in pesma["melodija_pitch_klase"]:
            if 0 <= pk <= 11:
                brojac_nota[NAZIVI_NOTA[pk]] += 1

    brojevi = [brojac_nota.get(n, 0) for n in NAZIVI_NOTA]
    ukupno = sum(brojevi)
    procenti = [b / ukupno * 100 for b in brojevi]

    # Crne i bele dirke
    crne = {"C#", "D#", "F#", "G#", "A#"}
    boje = [BOJA_MUTED if n in crne else BOJA_GNN for n in NAZIVI_NOTA]

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(NAZIVI_NOTA, procenti, color=boje, edgecolor="white", linewidth=1.2)

    ax.set_xlabel("Nota (pitch klasa)", fontsize=12)
    ax.set_ylabel("Učestalost (%)", fontsize=12)
    ax.set_title("Distribucija melodijskih nota u Nottingham datasetu", fontsize=13)

    bela_patch = mpatches.Patch(color=BOJA_GNN, label="Bele dirke")
    crna_patch = mpatches.Patch(color=BOJA_MUTED, label="Crne dirke")
    ax.legend(handles=[bela_patch, crna_patch], fontsize=10)

    for bar, val in zip(bars, procenti):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=8)

    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(PUTANJA_REZULTATA / "04_distribucija_nota.png", dpi=150)
    plt.close()
    print("[INFO] Sačuvan: 04_distribucija_nota.png")



# GRAFIK 5: Top-3 accuracy po alfa vrednostima


def grafik_top3():
    fig, ax = plt.subplots(figsize=(11, 5))

    ax.plot(alfas, top3, "s-", color=BOJA_OPT, linewidth=2.5,
            markersize=8, label="GNN Top-3 Accuracy")
    ax.axhline(y=lstm_top3, color=BOJA_LSTM, linestyle="--",
               linewidth=2, label=f"LSTM Baseline ({lstm_top3:.2f}%)")

    ax.set_xlabel("Alfa (α)", fontsize=12)
    ax.set_ylabel("Top-3 Accuracy (%)", fontsize=12)
    ax.set_title("Top-3 Accuracy po vrednostima α\n"
                 "(da li je tačan akord među 3 najverovantnija?)", fontsize=13)
    ax.set_xticks(alfas)
    ax.set_xticklabels([f"{a:.1f}" for a in alfas])
    ax.set_ylim(97.5, 99.5)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(PUTANJA_REZULTATA / "05_top3_accuracy.png", dpi=150)
    plt.close()
    print("[INFO] Sačuvan: 05_top3_accuracy.png")



# GRAFIK 6: Dužina pesama u datasetu


def grafik_duzine_pesama():
    duzine = [len(p["melodija_pitch_klase"]) for p in sve_pesme]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(duzine, bins=30, color=BOJA_GNN, edgecolor="white",
            linewidth=0.8, alpha=0.85)

    ax.axvline(x=np.mean(duzine), color=BOJA_LSTM, linestyle="--",
               linewidth=2, label=f"Prosek: {np.mean(duzine):.1f} nota")
    ax.axvline(x=np.median(duzine), color=BOJA_OPT, linestyle="-.",
               linewidth=2, label=f"Medijana: {np.median(duzine):.1f} nota")

    ax.set_xlabel("Broj nota u pesmi", fontsize=12)
    ax.set_ylabel("Broj pesama", fontsize=12)
    ax.set_title("Distribucija dužina pesama u Nottingham datasetu", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(PUTANJA_REZULTATA / "06_duzine_pesama.png", dpi=150)
    plt.close()
    print("[INFO] Sačuvan: 06_duzine_pesama.png")



# POKRETANJE SVEGA


if __name__ == "__main__":
    print("Generišem grafike...\n")
    grafik_grid_search()
    grafik_poredjenje_modela()
    grafik_distribucija_akorda()
    grafik_distribucija_nota()
    grafik_top3()
    grafik_duzine_pesama()
    print(f"\n✓ Svih 6 grafika sačuvano u: rezultati/")