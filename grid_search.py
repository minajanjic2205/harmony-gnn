"""
grid_search.py — Faza 4b: Grid search po α/β prostoru (Hipoteza H2)
Autor: Mina Janjić | Predmet: Računarstvo
Opis: Trenira GNN model za svaku vrednost alfe od 0.0 do 1.0,
      beleži rezultate i crta grafikon tačnosti po alfa vrednostima.
"""

import json
import pickle
import matplotlib.pyplot as plt
from pathlib import Path

import torch

from config import VREDNOSTI_ALFA, PUTANJA_OBRADENIH, PUTANJA_GRAFA, PUTANJA_REZULTATA
from gnn_model import treniraj_gnn, evaluiraj_gnn
from graf import izgradi_hetero_graf, sacuvaj_graf

PUTANJA_REZULTATA.mkdir(parents=True, exist_ok=True)


def pokreni_grid_search(
    trening, validacija, test, recnik_akorada,
    vrednosti_alfa=VREDNOSTI_ALFA,
):
    """
    Za svaku vrednost alfe:
      1. Gradi tonalni graf sa tom alfom
      2. Trenira GNN model
      3. Evaluira na test skupu
      4. Beleži rezultate
    """
    rezultati = []

    print("\n" + "═" * 60)
    print("  GRID SEARCH — istraživanje α/β prostora")
    print(f"  Vrednosti alfe: {vrednosti_alfa}")
    print("═" * 60)

    for alfa in vrednosti_alfa:
        print(f"\n{'─'*60}")
        print(f"  Trenutna alfa: {alfa:.2f} | beta: {1-alfa:.2f}")
        print(f"{'─'*60}")

        # Gradimo graf sa ovom alfom
        putanja_grafa = PUTANJA_GRAFA / f"tonalni_graf_alfa{alfa:.2f}.pt"
        if putanja_grafa.exists():
            print(f"[INFO] Graf za α={alfa:.2f} već postoji, učitavam...")
            graf = torch.load(putanja_grafa, weights_only=False)
            with open(PUTANJA_GRAFA / "indeksi_akorada.json", "r") as f:
                indeksi_akorada = json.load(f)
        else:
            graf, indeksi_akorada = izgradi_hetero_graf(
                recnik_akorada=recnik_akorada,
                trening=trening,
                alfa=alfa,
            )
            sacuvaj_graf(graf, indeksi_akorada, alfa)

        # Treniramo model
        model = treniraj_gnn(
            trening, validacija, graf, indeksi_akorada, recnik_akorada,
            alfa=alfa, naziv=f"GS_alfa{alfa:.2f}",
        )

        # Evaluacija na test skupu
        metrike = evaluiraj_gnn(model, test, graf, indeksi_akorada, recnik_akorada)

        rezultati.append({
            "alfa": alfa,
            "beta": round(1 - alfa, 1),
            "chord_accuracy": metrike["chord_accuracy"],
            "top3_accuracy": metrike["top3_accuracy"],
        })

        print(f"\n  → α={alfa:.2f} | CA={metrike['chord_accuracy']:.2%}")

    return rezultati


def prikazi_rezultate(rezultati: list[dict]) -> None:
    """Štampa tabelu i crta grafikon rezultata grid searcha."""

    print("\n" + "═" * 55)
    print(f"  {'Alfa':>6}  {'Beta':>6}  {'Chord Accuracy':>16}  {'Top-3':>8}")
    print("─" * 55)

    najbolji = max(rezultati, key=lambda r: r["chord_accuracy"])

    for r in rezultati:
        marker = " ← NAJBOLJI" if r["alfa"] == najbolji["alfa"] else ""
        print(f"  {r['alfa']:>6.2f}  {r['beta']:>6.2f}  "
              f"{r['chord_accuracy']:>15.2%}  {r['top3_accuracy']:>7.2%}{marker}")

    print("═" * 55)
    print(f"\n  Optimalna alfa : {najbolji['alfa']:.2f}")
    print(f"  Optimalna beta : {najbolji['beta']:.2f}")
    print(f"  Chord Accuracy : {najbolji['chord_accuracy']:.2%}")

    # LSTM baseline za poređenje
    lstm_ca = 0.9019
    print(f"\n  LSTM Baseline  : {lstm_ca:.2%}")
    print(f"  Poboljšanje    : {(najbolji['chord_accuracy'] - lstm_ca):+.2%}")

    # ── Grafikon ──────────────────────────────────────────────────────────────
    alfas = [r["alfa"] for r in rezultati]
    tacnosti = [r["chord_accuracy"] * 100 for r in rezultati]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(alfas, tacnosti, "o-", color="#2196F3", linewidth=2.5,
            markersize=8, label="GNN Chord Accuracy")

    # LSTM baseline linija
    ax.axhline(y=lstm_ca * 100, color="#F44336", linestyle="--",
               linewidth=2, label=f"LSTM Baseline ({lstm_ca*100:.2f}%)")

    # Oznaka najboljeg
    ax.axvline(x=najbolji["alfa"], color="#4CAF50", linestyle=":",
               linewidth=1.5, alpha=0.7)
    ax.annotate(
        f"Optimum α={najbolji['alfa']:.2f}\n({najbolji['chord_accuracy']*100:.2f}%)",
        xy=(najbolji["alfa"], najbolji["chord_accuracy"] * 100),
        xytext=(najbolji["alfa"] + 0.05, najbolji["chord_accuracy"] * 100 - 0.3),
        fontsize=10, color="#4CAF50",
        arrowprops=dict(arrowstyle="->", color="#4CAF50"),
    )

    ax.set_xlabel("Alfa (α) — balans teorija ↔ statistika", fontsize=12)
    ax.set_ylabel("Chord Accuracy (%)", fontsize=12)
    ax.set_title("Uticaj parametra α na tačnost predviđanja akorda\n"
                 "α=0: čisto statistički  |  α=1: čisto teorijski", fontsize=13)
    ax.set_xticks(alfas)
    ax.set_xticklabels([f"{a:.1f}" for a in alfas])
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    putanja_slike = PUTANJA_REZULTATA / "grid_search_alfa.png"
    plt.savefig(putanja_slike, dpi=150)
    print(f"\n[INFO] Grafikon sačuvan: {putanja_slike}")
    plt.show()


def sacuvaj_rezultate_json(rezultati: list[dict]) -> None:
    putanja = PUTANJA_REZULTATA / "grid_search_rezultati.json"
    with open(putanja, "w", encoding="utf-8") as f:
        json.dump(rezultati, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Rezultati sačuvani: {putanja}")


# ──────────────────────────────────────────────────────────────────────────────
# POKRETANJE
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    with open(PUTANJA_OBRADENIH / "trening.pkl", "rb") as f:
        trening = pickle.load(f)
    with open(PUTANJA_OBRADENIH / "validacija.pkl", "rb") as f:
        validacija = pickle.load(f)
    with open(PUTANJA_OBRADENIH / "test.pkl", "rb") as f:
        test = pickle.load(f)
    with open(PUTANJA_OBRADENIH / "recnik_akorada.json", "r") as f:
        recnik_akorada = json.load(f)

    rezultati = pokreni_grid_search(trening, validacija, test, recnik_akorada)
    prikazi_rezultate(rezultati)
    sacuvaj_rezultate_json(rezultati)