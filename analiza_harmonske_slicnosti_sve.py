"""
analiza_harmonske_slicnosti_sve.py — Chord Distance kroz SVE seedove i SVE alfe

Opis: Proširenje analiza_harmonske_slicnosti.py sa jednog para modela
      (LSTM vs GNN alfa=0.5) na kompletnu statistiku:

        - 11 vrednosti alfe x 10 seedova = 110 GNN checkpointa
        - 10 LSTM checkpointa (lstm_MULTI_seed{seed}.pt)

      Cilj: proveriti da li je nalaz "GNN pravi manje ozbiljne greske"
      (iz jednog poredjenja alfa=0.5) DOSLEDAN obrazac kroz sve seedove
      i sve alfe, ili je bio slucajnost tog jednog treniranja.

      Ovo je ista logika kao u grid_search_visestruko.py (10 treniranja
      po alfi -> mean +/- std), samo primenjena na Chord Distance metriku
      umesto na Chord Accuracy.

VAZNA NAPOMENA O METODOLOGIJI:
      Graf (tonalni_graf_alfaX.pt) NE zavisi od seeda — isti graf se
      koristi za svih 10 treniranja jedne alfe (seed utice samo na
      inicijalizaciju tezina mreze i redosled mesanja podataka).
      Zato se graf i test loader za datu alfu ucitavaju SAMO JEDNOM,
      a zatim se kroz njih provlaci svih 10 checkpointa te alfe.
"""

import json
import pickle
from pathlib import Path
from collections import defaultdict
from statistics import mean, stdev
from typing import Optional

import torch
import matplotlib.pyplot as plt

from config import PUTANJA_OBRADENIH, PUTANJA_GRAFA, PUTANJA_MODELA, PUTANJA_REZULTATA, VREDNOSTI_ALFA
from graf import note_u_akordu
from dataset import NEPOZNAT_AKORD

import baseline as bl
import gnn_model as gm

PUTANJA_REZULTATA.mkdir(parents=True, exist_ok=True)

# Isti seedovi kao u grid_search_visestruko.py i baseline_visestruko.py —
# MORA biti identicna lista, inace poredjenje LSTM naspram GNN nije fer.
SEEDOVI = [23, 155, 22, 7, 42, 100, 8, 250, 99, 12]



# METRIKA — Chord Distance (identicno kao u analiza_harmonske_slicnosti.py)


def chord_distance(naziv_a: str, naziv_b: str) -> int:
    """
    0 = isti akord, 1 = dele 2 note, 2 = dele 1 notu, 3 = ne dele nijednu.
    """
    if naziv_a == naziv_b:
        return 0

    note_a = set(note_u_akordu(naziv_a))
    note_b = set(note_u_akordu(naziv_b))

    if not note_a or not note_b:
        return 3

    zajednicke = len(note_a & note_b)
    return 3 - zajednicke


def analiziraj_parove(parovi: list) -> dict:
    """
    Racuna Chord Distance statistiku za jednu listu (tacan, predvidjen) parova
    — odnosno za JEDAN checkpoint (jedan seed, jedna alfa).
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
    return {
        "ukupno": ukupno,
        "chord_accuracy": tacno / ukupno if ukupno else 0.0,
        "prosek_distanca_sve": sum(sve_distance) / ukupno if ukupno else 0.0,
        "prosek_distanca_greske": (
            sum(distance_gresaka) / len(distance_gresaka) if distance_gresaka else 0.0
        ),
        "broj_gresaka": len(distance_gresaka),
    }



# UCITAVANJE PODATAKA (zajednicko za sve)


with open(PUTANJA_OBRADENIH / "test.pkl", "rb") as f:
    test = pickle.load(f)
with open(PUTANJA_OBRADENIH / "recnik_akorada.json", "r") as f:
    recnik_akorada = json.load(f)
with open(PUTANJA_GRAFA / "indeksi_akorada.json", "r") as f:
    indeksi_akorada = json.load(f)

inv_recnik = {v: k for k, v in recnik_akorada.items()}
inv_indeksi = {v: k for k, v in indeksi_akorada.items()}



# PREDIKCIJE — LSTM (jedan checkpoint po pozivu)


def prikupi_predikcije_lstm(seed: int) -> Optional[list]:
    """Vraca (tacan, predvidjen) parove za lstm_MULTI_seed{seed}.pt, ili None ako fajl ne postoji."""
    putanja = PUTANJA_MODELA / f"lstm_MULTI_seed{seed}.pt"
    if not putanja.exists():
        return None

    checkpoint = torch.load(putanja, weights_only=False, map_location='cpu')
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



# PREDIKCIJE — GNN (graf/loader se prave JEDNOM po alfi, checkpoint po seedu)


def ucitaj_graf_i_loader_za_alfu(alfa: float):
    """
    Ucitava graf za datu alfu i pravi test loader.
    Ovo se poziva SAMO JEDNOM po alfi (ne po seedu) jer graf ne zavisi od seeda.
    Vraca (graf_tenzori, loader, mapa) ili None ako graf ne postoji.
    """
    putanja_grafa = PUTANJA_GRAFA / f"tonalni_graf_alfa{alfa:.2f}.pt"
    if not putanja_grafa.exists():
        return None

    graf = torch.load(putanja_grafa, weights_only=False, map_location='cpu')
    x_nota = graf["nota"].x
    edge_na = graf["nota", "pripada", "akord"].edge_index
    edge_na_w = graf["nota", "pripada", "akord"].edge_attr
    edge_aa = graf["akord", "blizina", "akord"].edge_index
    edge_aa_w = graf["akord", "blizina", "akord"].edge_attr

    mapa = gm.napravi_mapu(recnik_akorada, indeksi_akorada)
    loader = gm.napravi_loader(test, mapa, mesati=False)

    graf_tenzori = (x_nota, edge_na, edge_na_w, edge_aa, edge_aa_w)
    return graf_tenzori, loader


def prikupi_predikcije_gnn(alfa: float, seed: int, graf_tenzori, loader) -> Optional[list]:
    """Vraca (tacan, predvidjen) parove za dati (alfa, seed) checkpoint, ili None ako ne postoji."""
    putanja = PUTANJA_MODELA / f"gnn_MULTI_alfa{alfa:.2f}_seed{seed}_alfa{alfa:.2f}.pt"
    if not putanja.exists():
        return None

    checkpoint = torch.load(putanja, weights_only=False, map_location='cpu')
    model = gm.GNNPredvidjanjAkorda(broj_akorada=len(indeksi_akorada))
    model.load_state_dict(checkpoint["stanje"])
    model.eval()

    x_nota, edge_na, edge_na_w, edge_aa, edge_aa_w = graf_tenzori

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



# GLAVNA PETLJA — GNN: svih 11 alfi x 10 seedova


def obradi_sve_gnn(vrednosti_alfa=VREDNOSTI_ALFA, seedovi=SEEDOVI) -> dict:
    """
    Za svaku alfu, provlaci svih 10 seedova kroz analizu i agregira
    (mean +/- std preko seedova). Vraca rezultat po alfi.
    """
    rezultati_po_alfi = {}

    for alfa in vrednosti_alfa:
        print(f"\n{'='*60}")
        print(f"  ALFA = {alfa:.2f}")
        print(f"{'='*60}")

        podaci_grafa = ucitaj_graf_i_loader_za_alfu(alfa)
        if podaci_grafa is None:
            print(f"  [PRESKACEM] Graf za alfa={alfa:.2f} ne postoji na disku.")
            continue
        graf_tenzori, loader = podaci_grafa

        ca_svi, dist_sve_svi, dist_greske_svi = [], [], []
        nedostaju = []

        for seed in seedovi:
            parovi = prikupi_predikcije_gnn(alfa, seed, graf_tenzori, loader)
            if parovi is None:
                nedostaju.append(seed)
                continue

            rez = analiziraj_parove(parovi)
            ca_svi.append(rez["chord_accuracy"])
            dist_sve_svi.append(rez["prosek_distanca_sve"])
            dist_greske_svi.append(rez["prosek_distanca_greske"])

            print(f"  Seed {seed:>4}: CA={rez['chord_accuracy']:.2%}  "
                  f"dist(sve)={rez['prosek_distanca_sve']:.3f}  "
                  f"dist(greske)={rez['prosek_distanca_greske']:.3f}")

        if nedostaju:
            print(f"  [UPOZORENJE] Nedostaju checkpointi za seedove: {nedostaju}")

        if not ca_svi:
            print(f"  [PRESKACEM] Nijedan checkpoint za alfa={alfa:.2f} nije pronadjen.")
            continue

        rezultati_po_alfi[alfa] = {
            "alfa": alfa,
            "broj_seedova_pronadjenih": len(ca_svi),
            "seedovi_nedostaju": nedostaju,
            "ca_prosek": mean(ca_svi),
            "ca_std": stdev(ca_svi) if len(ca_svi) > 1 else 0.0,
            "dist_sve_prosek": mean(dist_sve_svi),
            "dist_sve_std": stdev(dist_sve_svi) if len(dist_sve_svi) > 1 else 0.0,
            "dist_greske_prosek": mean(dist_greske_svi),
            "dist_greske_std": stdev(dist_greske_svi) if len(dist_greske_svi) > 1 else 0.0,
            "dist_greske_svi_seedovi": dist_greske_svi,
        }

        print(f"\n  --> alfa={alfa:.2f}: dist(greske) = "
              f"{mean(dist_greske_svi):.3f} +/- "
              f"{(stdev(dist_greske_svi) if len(dist_greske_svi) > 1 else 0.0):.3f} "
              f"(n={len(ca_svi)} seedova)")

    return rezultati_po_alfi



# GLAVNA PETLJA — LSTM: 10 seedova


def obradi_sve_lstm(seedovi=SEEDOVI) -> dict:
    """Provlaci svih 10 LSTM checkpointa i agregira (mean +/- std)."""
    print(f"\n{'='*60}")
    print(f"  LSTM BASELINE — svi seedovi")
    print(f"{'='*60}")

    ca_svi, dist_sve_svi, dist_greske_svi = [], [], []
    nedostaju = []

    for seed in seedovi:
        parovi = prikupi_predikcije_lstm(seed)
        if parovi is None:
            nedostaju.append(seed)
            continue

        rez = analiziraj_parove(parovi)
        ca_svi.append(rez["chord_accuracy"])
        dist_sve_svi.append(rez["prosek_distanca_sve"])
        dist_greske_svi.append(rez["prosek_distanca_greske"])

        print(f"  Seed {seed:>4}: CA={rez['chord_accuracy']:.2%}  "
              f"dist(sve)={rez['prosek_distanca_sve']:.3f}  "
              f"dist(greske)={rez['prosek_distanca_greske']:.3f}")

    if nedostaju:
        print(f"  [UPOZORENJE] Nedostaju LSTM checkpointi za seedove: {nedostaju}")
        print(f"  (Ako baseline_visestruko.py jos nije zavrsio, pokreni ga prvo.)")

    if not ca_svi:
        print("  [GRESKA] Nijedan LSTM checkpoint nije pronadjen. Ne mogu da nastavim.")
        return None

    rezultat = {
        "broj_seedova_pronadjenih": len(ca_svi),
        "seedovi_nedostaju": nedostaju,
        "ca_prosek": mean(ca_svi),
        "ca_std": stdev(ca_svi) if len(ca_svi) > 1 else 0.0,
        "dist_sve_prosek": mean(dist_sve_svi),
        "dist_sve_std": stdev(dist_sve_svi) if len(dist_sve_svi) > 1 else 0.0,
        "dist_greske_prosek": mean(dist_greske_svi),
        "dist_greske_std": stdev(dist_greske_svi) if len(dist_greske_svi) > 1 else 0.0,
        "dist_greske_svi_seedovi": dist_greske_svi,
    }

    print(f"\n  --> LSTM: dist(greske) = {rezultat['dist_greske_prosek']:.3f} +/- "
          f"{rezultat['dist_greske_std']:.3f} (n={len(ca_svi)} seedova)")

    return rezultat



# VIZUALIZACIJA


def nacrtaj_poredjenje(rezultati_gnn: dict, rezultat_lstm: dict) -> None:
    """
    Grafikon: x = alfa, y = prosecna Chord Distance (samo greske),
    error bar = std preko 10 seedova. Horizontalna linija/pojas = LSTM prosek +/- std.
    """
    alfe_sortirane = sorted(rezultati_gnn.keys())
    proseci = [rezultati_gnn[a]["dist_greske_prosek"] for a in alfe_sortirane]
    stdovi = [rezultati_gnn[a]["dist_greske_std"] for a in alfe_sortirane]

    fig, ax = plt.subplots(figsize=(11, 6.5))

    ax.errorbar(
        alfe_sortirane, proseci, yerr=stdovi, fmt="o-",
        color="#2196F3", linewidth=2.5, markersize=8,
        capsize=5, capthick=1.5,
        label="GNN — prosecna Chord Distance greske (mean ± std, 10 seedova)",
    )

    if rezultat_lstm is not None:
        lstm_prosek = rezultat_lstm["dist_greske_prosek"]
        lstm_std = rezultat_lstm["dist_greske_std"]

        ax.axhline(y=lstm_prosek, color="#F44336", linestyle="--", linewidth=2,
                   label=f"LSTM Baseline (prosek = {lstm_prosek:.3f})")
        ax.axhspan(lstm_prosek - lstm_std, lstm_prosek + lstm_std,
                   color="#F44336", alpha=0.12,
                   label=f"LSTM ± std ({lstm_std:.3f})")

    ax.set_xlabel("Alfa (α) — balans teorija ↔ statistika", fontsize=12)
    ax.set_ylabel("Prosecna Chord Distance (samo greske)", fontsize=12)
    ax.set_title(
        "Da li GNN pravi manje ozbiljne greske od LSTM-a?\n"
        "Kroz sve vrednosti α, agregirano preko 10 seedova po tacki",
        fontsize=13,
    )
    ax.set_xticks(alfe_sortirane)
    ax.set_xticklabels([f"{a:.1f}" for a in alfe_sortirane])
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    putanja_slike = PUTANJA_REZULTATA / "11_harmonska_slicnost_sve.png"
    plt.savefig(putanja_slike, dpi=150)
    plt.close()
    print(f"\n[INFO] Grafikon sacuvan: {putanja_slike}")



# ZAKLJUCAK — da li je obrazac dosledan kroz sve alfe


def proveri_doslednost(rezultati_gnn: dict, rezultat_lstm: dict) -> None:
    """
    Za svaku alfu proverava da li je GNN prosecna distanca greske MANJA
    od LSTM proseka (unutar/van jednog std-a), i ispisuje koliko od
    ukupnog broja alfi pokazuje taj obrazac.
    """
    if rezultat_lstm is None:
        print("\n[UPOZORENJE] Nema LSTM rezultata — ne mogu da proverim doslednost obrasca.")
        return

    lstm_prosek = rezultat_lstm["dist_greske_prosek"]

    print("\n" + "=" * 65)
    print("  PROVERA DOSLEDNOSTI: 'GNN pravi manje ozbiljne greske'")
    print("=" * 65)
    print(f"  LSTM prosek (10 seedova): {lstm_prosek:.3f} ± {rezultat_lstm['dist_greske_std']:.3f}\n")

    broj_boljih = 0
    broj_ukupno = 0

    for alfa in sorted(rezultati_gnn.keys()):
        r = rezultati_gnn[alfa]
        razlika = lstm_prosek - r["dist_greske_prosek"]  # pozitivno = GNN bolji (manja distanca)
        bolji = razlika > 0
        broj_ukupno += 1
        if bolji:
            broj_boljih += 1

        oznaka = "GNN bolji" if bolji else "LSTM bolji ili isto"
        print(f"  alfa={alfa:.2f}: GNN={r['dist_greske_prosek']:.3f}±{r['dist_greske_std']:.3f}  "
              f"razlika={razlika:+.3f}  ({oznaka})")

    print("\n" + "-" * 65)
    print(f"  GNN je bio bolji (manja distanca greske) u {broj_boljih}/{broj_ukupno} vrednosti alfe.")
    if broj_boljih == broj_ukupno:
        print("  ZAKLJUCAK: obrazac je POTPUNO DOSLEDAN kroz sve alfe.")
    elif broj_boljih >= broj_ukupno * 0.7:
        print("  ZAKLJUCAK: obrazac je VECINSKI DOSLEDAN, ali ne kod svih alfi.")
    elif broj_boljih <= broj_ukupno * 0.3:
        print("  ZAKLJUCAK: obrazac se NE potvrdjuje — LSTM je vecinom bolji ili isti.")
    else:
        print("  ZAKLJUCAK: MESOVIT rezultat — nema jasnog obrasca kroz alfe.")
    print("=" * 65)



# POKRETANJE


if __name__ == "__main__":
    print("[INFO] Obradjujem LSTM checkpointe (10 seedova)...")
    rezultat_lstm = obradi_sve_lstm()

    print("\n[INFO] Obradjujem GNN checkpointe (11 alfi x 10 seedova = do 110)...")
    rezultati_gnn = obradi_sve_gnn()

    if not rezultati_gnn:
        print("\n[GRESKA] Nijedan GNN rezultat nije pronadjen. Proveri da li su "
              "checkpointi preneti sa klastera u lokalni folder modeli/.")
    else:
        nacrtaj_poredjenje(rezultati_gnn, rezultat_lstm)
        proveri_doslednost(rezultati_gnn, rezultat_lstm)

        # Cuvamo sve u JSON (uklonimo alfa kljuceve u string radi JSON kompatibilnosti)
        izlaz = {
            "lstm": rezultat_lstm,
            "gnn_po_alfi": {f"{a:.2f}": v for a, v in rezultati_gnn.items()},
        }
        with open(PUTANJA_REZULTATA / "harmonska_slicnost_sve.json", "w", encoding="utf-8") as f:
            json.dump(izlaz, f, indent=2, ensure_ascii=False)
        print(f"\n[INFO] Svi rezultati sacuvani u: "
              f"{PUTANJA_REZULTATA / 'harmonska_slicnost_sve.json'}")