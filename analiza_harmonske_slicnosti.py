"""
analiza_harmonske_slicnosti.py-Analiza kvaliteta grešaka (Chord Distance)

Opis: Implementira metriku iz sekcije 5.1 projektnog predloga — "Harmonska
      sličnost (Chord Distance)": greška C→Am nije ista kao greška C→F#.

      Umesto da meri SAMO da li je model pogodio tačan akord (Chord
      Accuracy), ova skripta meri KOLIKO OZBILJNA je greška kada se
      desi, na osnovu broja zajedničkih nota između predviđenog i
      tačnog akorda trijade.

      Hipoteza: čak i ako GNN i LSTM imaju sličnu Chord Accuracy,
      GNN bi mogao da pravi ,,pametnije", muzikalnije greške (bliže
      tačnom akordu) jer koristi tonalni graf, dok LSTM greši
      "nasumičnije" jer uči čisto statistički iz podataka.
Koristi već istrenirane modele: lstm_baseline_najbolji.pt i
gnn_E3_alfa0.50.pt.
"""

import json
import pickle
from pathlib import Path
from collections import defaultdict

import torch
import matplotlib.pyplot as plt

from config import PUTANJA_OBRADENIH, PUTANJA_GRAFA, PUTANJA_MODELA, PUTANJA_REZULTATA
from graf import note_u_akordu
from dataset import NEPOZNAT_AKORD

import baseline as bl
import gnn_model as gm

PUTANJA_REZULTATA.mkdir(parents=True, exist_ok=True)

ALFA_GNN = 0.5  # koji GNN checkpoint koristimo (mora postojati na disku)



# METRIKA  Chord Distance (broj zajedničkih nota između trijada)


def chord_distance(naziv_a: str, naziv_b: str) -> int:
    """
    Vraća distancu između dva akorda na osnovu broja ZAJEDNIČKIH nota
    u njihovim trijadama.

    0 = isti akord (3 zajedničke note)
    1 = dele 2 note (npr. Cmaj i Amin dele C, E → distanca 1)
    2 = dele 1 notu
    3 = ne dele nijednu (najveća moguća greška, npr. Cmaj i F#maj)

    Ovo direktno implementira "Harmonsku sličnost (Chord Distance)"
    iz sekcije 5.1 projektnog predloga.
    """
    if naziv_a == naziv_b:
        return 0

    note_a = set(note_u_akordu(naziv_a))
    note_b = set(note_u_akordu(naziv_b))

    if not note_a or not note_b:
        return 3  # nepoznat akord → maksimalna kazna

    zajednicke = len(note_a & note_b)
    return 3 - zajednicke


# UČITAVANJE PODATAKA


with open(PUTANJA_OBRADENIH / "test.pkl", "rb") as f:
    test = pickle.load(f)
with open(PUTANJA_OBRADENIH / "recnik_akorada.json", "r") as f:
    recnik_akorada = json.load(f)
with open(PUTANJA_GRAFA / "indeksi_akorada.json", "r") as f:
    indeksi_akorada = json.load(f)

inv_recnik = {v: k for k, v in recnik_akorada.items()}
inv_indeksi = {v: k for k, v in indeksi_akorada.items()}



# PRIKUPLJANJE PREDIKCIJA — LSTM


def prikupi_predikcije_lstm() -> list:
    """Vraća listu (tačan_naziv, predviđen_naziv) parova za LSTM model."""
    checkpoint = torch.load(
        PUTANJA_MODELA / "lstm_baseline_najbolji.pt", weights_only=False
    )
    model = bl.LSTMPredvidjanjAkorda(broj_akorda=len(recnik_akorada))
    model.load_state_dict(checkpoint["stanje_modela"])
    model.eval()

    loader = bl.napravi_loader(
        test, bl.PODRAZUMEVANI_HP["velicina_prozora"],
        bl.PODRAZUMEVANI_HP["velicina_serije"], mesati=False,
    )

    parovi = []
    with torch.no_grad():
        for ulaz, cilj in loader:
            logiti, _ = model(ulaz)
            pred = logiti.argmax(dim=1)
            for t, p in zip(cilj.tolist(), pred.tolist()):
                naziv_t = inv_recnik.get(t, NEPOZNAT_AKORD)
                naziv_p = inv_recnik.get(p, NEPOZNAT_AKORD)
                if naziv_t != NEPOZNAT_AKORD:
                    parovi.append((naziv_t, naziv_p))
    return parovi


# PRIKUPLJANJE PREDIKCIJA — GNN


def prikupi_predikcije_gnn() -> list:
    """Vraća listu (tačan_naziv, predviđen_naziv) parova za GNN model."""
    putanja = PUTANJA_MODELA / f"gnn_E3_alfa{ALFA_GNN:.2f}.pt"
    checkpoint = torch.load(putanja, weights_only=False)

    model = gm.GNNPredvidjanjAkorda(broj_akorada=len(indeksi_akorada))
    model.load_state_dict(checkpoint["stanje"])
    model.eval()

    graf = torch.load(
        PUTANJA_GRAFA / f"tonalni_graf_alfa{ALFA_GNN:.2f}.pt", weights_only=False
    )
    x_nota = graf["nota"].x
    edge_na = graf["nota", "pripada", "akord"].edge_index
    edge_na_w = graf["nota", "pripada", "akord"].edge_attr
    edge_aa = graf["akord", "blizina", "akord"].edge_index
    edge_aa_w = graf["akord", "blizina", "akord"].edge_attr

    mapa = gm.napravi_mapu(recnik_akorada, indeksi_akorada)
    loader = gm.napravi_loader(test, mapa, mesati=False)

    parovi = []
    with torch.no_grad():
        for mel, cilj in loader:
            logiti = model(mel, x_nota, edge_na, edge_na_w, edge_aa, edge_aa_w)
            pred = logiti.argmax(dim=1)
            for t, p in zip(cilj.tolist(), pred.tolist()):
                naziv_t = inv_indeksi.get(t, NEPOZNAT_AKORD)
                naziv_p = inv_indeksi.get(p, NEPOZNAT_AKORD)
                parovi.append((naziv_t, naziv_p))
    return parovi



# ANALIZA


def analiziraj(parovi: list, naziv_modela: str) -> dict:
    """
    Računa statistiku Chord Distance za dati model.
    Vraća rečnik sa prosečnom distancom (sve/samo greške) i raspodelu.
    """
    sve_distance = []
    distance_gresaka = []
    tacno = 0

    for tacan, pred in parovi:
        d = chord_distance(tacan, pred)
        sve_distance.append(d)
        if d == 0:
            tacno += 1
        else:
            distance_gresaka.append(d)

    ukupno = len(parovi)
    prosek_sve = sum(sve_distance) / ukupno if ukupno else 0
    prosek_gresaka = (
        sum(distance_gresaka) / len(distance_gresaka) if distance_gresaka else 0
    )

    # Raspodela distanci grešaka (1, 2, ili 3 zajedničke note manje)
    raspodela = defaultdict(int)
    for d in distance_gresaka:
        raspodela[d] += 1

    print(f"\n{'='*55}")
    print(f"  {naziv_modela}")
    print(f"{'='*55}")
    print(f"  Ukupno uzoraka        : {ukupno}")
    print(f"  Tačno pogođeno        : {tacno} ({tacno/ukupno:.2%})")
    print(f"  Broj grešaka          : {len(distance_gresaka)}")
    print(f"  Prosečna distanca (sve): {prosek_sve:.3f}")
    print(f"  Prosečna distanca (samo greške): {prosek_gresaka:.3f}")
    print(f"\n  Raspodela grešaka po distanci:")
    for d in [1, 2, 3]:
        broj = raspodela.get(d, 0)
        procenat = broj / len(distance_gresaka) * 100 if distance_gresaka else 0
        opis = {1: "dele 2 note (bliska greška)",
                2: "dele 1 notu (srednja greška)",
                3: "ne dele nijednu (daleka greška)"}[d]
        print(f"    Distanca {d} ({opis}): {broj} ({procenat:.1f}%)")

    return {
        "model": naziv_modela,
        "ukupno": ukupno,
        "tacno": tacno,
        "chord_accuracy": tacno / ukupno if ukupno else 0,
        "prosek_distanca_sve": prosek_sve,
        "prosek_distanca_greske": prosek_gresaka,
        "raspodela_gresaka": dict(raspodela),
    }


def nacrtaj_poredjenje(rezultat_lstm: dict, rezultat_gnn: dict) -> None:
    """Crta grafikon poređenja prosečne distance greške."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))

    # Levo: prosečna distanca greške
    modeli = ["LSTM\nBaseline", f"GNN\nalfa={ALFA_GNN:.1f}"]
    proseci = [rezultat_lstm["prosek_distanca_greske"], rezultat_gnn["prosek_distanca_greske"]]
    boje = ["#F44336", "#2196F3"]

    bars = ax1.bar(modeli, proseci, color=boje, edgecolor="white", linewidth=1.5, width=0.5)
    ax1.set_ylabel("Prosecna Chord Distance (samo greske)", fontsize=11)
    ax1.set_title("Koliko su greske 'ozbiljne'?\n(0=iste note, 3=nema zajednickih nota)", fontsize=12)
    ax1.set_ylim(0, 3)
    for bar, val in zip(bars, proseci):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax1.grid(True, alpha=0.3, axis="y")

    # Desno: raspodela grešaka po distanci (grupisan bar, %)
    kategorije = ["Distanca 1\n(bliska)", "Distanca 2\n(srednja)", "Distanca 3\n(daleka)"]
    lstm_gresaka_ukupno = sum(rezultat_lstm["raspodela_gresaka"].values())
    gnn_gresaka_ukupno = sum(rezultat_gnn["raspodela_gresaka"].values())

    lstm_pct = [rezultat_lstm["raspodela_gresaka"].get(d, 0) / lstm_gresaka_ukupno * 100
                if lstm_gresaka_ukupno else 0 for d in [1, 2, 3]]
    gnn_pct = [rezultat_gnn["raspodela_gresaka"].get(d, 0) / gnn_gresaka_ukupno * 100
               if gnn_gresaka_ukupno else 0 for d in [1, 2, 3]]

    x = range(len(kategorije))
    sirina = 0.35
    ax2.bar([i - sirina/2 for i in x], lstm_pct, sirina, label="LSTM", color="#F44336")
    ax2.bar([i + sirina/2 for i in x], gnn_pct, sirina, label=f"GNN alfa={ALFA_GNN:.1f}", color="#2196F3")
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(kategorije)
    ax2.set_ylabel("% od svih greaka", fontsize=11)
    ax2.set_title("Raspodela gresaka po ozbiljnosti", fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(PUTANJA_REZULTATA / "10_harmonska_slicnost.png", dpi=150)
    plt.close()
    print(f"\n[INFO] Grafikon sacuvan: rezultati/10_harmonska_slicnost.png")



# POKRETANJE


if __name__ == "__main__":
    print("[INFO] Prikupljam predikcije LSTM modela...")
    parovi_lstm = prikupi_predikcije_lstm()

    print("[INFO] Prikupljam predikcije GNN modela...")
    parovi_gnn = prikupi_predikcije_gnn()

    rezultat_lstm = analiziraj(parovi_lstm, "LSTM Baseline (E1)")
    rezultat_gnn = analiziraj(parovi_gnn, f"GNN (alfa={ALFA_GNN:.1f})")

    nacrtaj_poredjenje(rezultat_lstm, rezultat_gnn)

    # Cuvamo rezultate
    with open(PUTANJA_REZULTATA / "harmonska_slicnost.json", "w", encoding="utf-8") as f:
        json.dump({"lstm": rezultat_lstm, "gnn": rezultat_gnn}, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 55)
    print("  ZAKLJUCAK")
    print("=" * 55)
    razlika = rezultat_lstm["prosek_distanca_greske"] - rezultat_gnn["prosek_distanca_greske"]
    if razlika > 0.05:
        print(f"  GNN pravi ZNACAJNO manje ozbiljne greske od LSTM-a")
        print(f"  (razlika u proseku: {razlika:.3f})")
    elif razlika < -0.05:
        print(f"  LSTM pravi manje ozbiljne greske od GNN-a")
        print(f"  (razlika u proseku: {-razlika:.3f})")
    else:
        print(f"  Nema znacajne razlike u ozbiljnosti gresaka izmedju modela")
        print(f"  (razlika u proseku: {razlika:.3f} - unutar suma)")
    print("=" * 55)