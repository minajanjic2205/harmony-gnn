"""
graf.py  faza 3: konstrukcija heterogenog tonalnog grafa
Opis: Implementacija grafa G = (V, E, W) prema sekciji 4.2 predloga projekta.
      Tri vrste grana: teorijska pripadnost, kvintni krug, statistička ko-pojava.
      Težinska funkcija: w(t, a) = α × teorija + β × statistika, α + β = 1.
"""

import json
import pickle
from pathlib import Path
from collections import defaultdict
from typing import Optional

import torch
from torch_geometric.data import HeteroData


# KONSTANTE


PUTANJA_OBRADENIH = Path("podaci/obradeni")
PUTANJA_GRAFA = Path("podaci/graf")

# 12 hromatskih nota (pitch klase 0–11)
NAZIVI_NOTA = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Kvintni krug — redosled tonaliteta u smeru kazaljke na satu
# Svaki sledeći je kvinta više (7 polustepeni)
KVINTNI_KRUG = [0, 7, 2, 9, 4, 11, 6, 1, 8, 3, 10, 5]
# C  G  D  A  E  B  F# C# G# D# A# F

# Trijade — koje note čine svaki tip akorda
# Durska trijada: osnovna, terca (4 polustepena), kvinta (7 polustepena)
# Molska trijada: osnovna, terca (3 polustepena), kvinta (7 polustepena)
INTERVALI_DUR = [0, 4, 7]
INTERVALI_MOL = [0, 3, 7]



# POMOĆNE FUNKCIJE — MUZIČKA TEORIJA


def naziv_u_koren_i_kvalitet(naziv_akorda: str) -> Optional[tuple[int, str]]:
    """
    Parsuje naziv akorda (npr. 'Dmaj', 'F#min', 'B-maj') u
    (pitch_klasa_korena, kvalitet).
    Vraća None ako naziv nije prepoznat.
    """
    # Mapiranje između notnih oznaka i pitch klasa
    mapa_nota = {
        "C": 0, "C#": 1, "Db": 1,
        "D": 2, "D#": 3, "Eb": 3, "E-": 3,
        "E": 4, "Fb": 4,
        "F": 5, "F#": 6, "Gb": 6,
        "G": 7, "G#": 8, "Ab": 8,
        "A": 9, "A#": 10, "Bb": 10, "B-": 10,
        "B": 11, "Cb": 11,
    }

    naziv = naziv_akorda.strip()

    # Određujemo sufiks (maj/min)
    if naziv.endswith("maj"):
        kvalitet = "maj"
        koren_str = naziv[:-3]
    elif naziv.endswith("min"):
        kvalitet = "min"
        koren_str = naziv[:-3]
    else:
        return None

    # Tražimo koren u mapi
    pk = mapa_nota.get(koren_str)
    if pk is None:
        return None

    return pk, kvalitet


def note_u_akordu(naziv_akorda: str) -> list[int]:
    """
    Vraća listu pitch klasa nota koje čine zadati akord.
    Npr. 'Cmaj' → [0, 4, 7], 'Amin' → [9, 0, 4]
    """
    rezultat = naziv_u_koren_i_kvalitet(naziv_akorda)
    if rezultat is None:
        return []

    koren, kvalitet = rezultat
    intervali = INTERVALI_DUR if kvalitet == "maj" else INTERVALI_MOL
    return [(koren + interval) % 12 for interval in intervali]


def rastojanje_na_kvintu(pk_a: int, pk_b: int) -> int:
    """
    Vraća minimalni broj koraka između dva tonaliteta na kvintnom krugu (0–6).
    Maksimum je 6 jer je krug simetričan.
    """
    if pk_a not in KVINTNI_KRUG or pk_b not in KVINTNI_KRUG:
        return 6  # nepoznat tonalitet — maksimalna udaljenost

    poz_a = KVINTNI_KRUG.index(pk_a)
    poz_b = KVINTNI_KRUG.index(pk_b)
    razlika = abs(poz_a - poz_b)
    return min(razlika, 12 - razlika)  # kraći put na krugu



# TIP 1 — TEORIJSKA PRIPADNOST NOTE AKORDU


def izgradi_teorijske_grane(
    indeksi_akorada: dict[str, int],
) -> tuple[list[tuple[int, int]], list[float]]:
    """
    Tip 1: Binarna grana nota↔akord ako nota ulazi u sastav trijade.
    Težina je uvek 1.0 (binarna relacija iz teorije muzike).

    Vraća:
        lista_grana : [(indeks_note, indeks_akorda), ...]
        tezine      : [1.0, 1.0, ...]
    """
    grane = []
    tezine = []

    for naziv_akorda, idx_akorda in indeksi_akorada.items():
        note_trijade = note_u_akordu(naziv_akorda)
        for pk_note in note_trijade:
            grane.append((pk_note, idx_akorda))
            tezine.append(1.0)

    print(f"[INFO] Tip 1 grane (teorijska pripadnost): {len(grane)}")
    return grane, tezine



# TIP 2 — KVINTNI KRUG (TONALNA BLIZINA IZMEĐU AKORADA)


def izgradi_kvintne_grane(
    indeksi_akorada: dict[str, int],
) -> tuple[list[tuple[int, int]], list[float]]:
    """
    Tip 2: Grana između dva akorda čiji su koreni bliski na kvintnom krugu.
    Težina: w = 1 / (1 + d), gde je d broj koraka na krugu.
    Maksimalno rastojanje za koje dodajemo granu: 3 koraka.

    Vraća:
        lista_grana : [(indeks_akorda_a, indeks_akorda_b), ...]
        tezine      : [w, ...]
    """
    MAX_RASTOJANJE = 3  # akordi dalje od 3 koraka nisu harmonski srodni

    nazivi = list(indeksi_akorada.keys())
    grane = []
    tezine = []

    for i in range(len(nazivi)):
        for j in range(i + 1, len(nazivi)):
            naziv_a = nazivi[i]
            naziv_b = nazivi[j]

            rez_a = naziv_u_koren_i_kvalitet(naziv_a)
            rez_b = naziv_u_koren_i_kvalitet(naziv_b)

            if rez_a is None or rez_b is None:
                continue

            pk_a, _ = rez_a
            pk_b, _ = rez_b

            d = rastojanje_na_kvintu(pk_a, pk_b)

            if d <= MAX_RASTOJANJE:
                idx_a = indeksi_akorada[naziv_a]
                idx_b = indeksi_akorada[naziv_b]
                tezina = 1.0 / (1.0 + d)

                # Neusmeren graf — dodajemo u oba smera
                grane.append((idx_a, idx_b))
                grane.append((idx_b, idx_a))
                tezine.append(tezina)
                tezine.append(tezina)

    print(f"[INFO] Tip 2 grane (kvintni krug): {len(grane)}")
    return grane, tezine



# TIP 3 — STATISTIČKA KO-POJAVA IZ TRENING PODATAKA


def izracunaj_ko_pojave(
    trening: list[dict],
    indeksi_akorada: dict[str, int],
) -> dict[tuple[int, int], float]:
    """
    Tip 3: p(akord | nota) iz trening podataka.
    p(akord | nota) = broj ko-pojava(nota, akord) / ukupno pojava(nota)

    Vraća rečnik {(pitch_klasa, indeks_akorda): verovatnoća}
    """
    # Brojimo ko-pojave: koliko puta se nota pk javlja uz akord a
    brojac_kopojava: dict[tuple[int, int], int] = defaultdict(int)
    # Ukupno pojava svake note
    brojac_note: dict[int, int] = defaultdict(int)

    # Inverzni rečnik: indeks → naziv
    inv_recnik = {v: k for k, v in indeksi_akorada.items()}

    for pesma in trening:
        melodija = pesma["melodija_pitch_klase"]
        akordi = pesma.get("akordi_enkodovani", [])

        for (pk, _), idx_akorda in zip(melodija, akordi):
            if pk < 0:
                continue  # preskačemo pauze
            if idx_akorda == 0:
                continue  # preskačemo NEPOZNAT_AKORD

            brojac_kopojava[(pk, idx_akorda)] += 1
            brojac_note[pk] += 1

    # Normalizacija → uslovne verovatnoće
    ko_pojave: dict[tuple[int, int], float] = {}
    for (pk, idx_akorda), broj in brojac_kopojava.items():
        ukupno_note = brojac_note[pk]
        if ukupno_note > 0:
            ko_pojave[(pk, idx_akorda)] = broj / ukupno_note

    print(f"[INFO] Statistička ko-pojava: {len(ko_pojave)} parova (nota, akord)")
    return ko_pojave


def izgradi_statisticke_grane(
    ko_pojave: dict[tuple[int, int], float],
    prag: float = 0.01,
) -> tuple[list[tuple[int, int]], list[float]]:
    """
    Tip 3: Grana nota↔akord sa težinom = p(akord | nota).
    Dodajemo samo grane gde je verovatnoća iznad praga (filtrišemo šum).

    Vraća:
        lista_grana : [(pitch_klasa, indeks_akorda), ...]
        tezine      : [p(akord|nota), ...]
    """
    grane = []
    tezine = []

    for (pk, idx_akorda), verovatnoca in ko_pojave.items():
        if verovatnoca >= prag:
            grane.append((pk, idx_akorda))
            tezine.append(verovatnoca)

    print(f"[INFO] Tip 3 grane (statistička ko-pojava, prag={prag}): {len(grane)}")
    return grane, tezine



# TEŽINSKA FUNKCIJA — KOMBINACIJA TEORIJE I STATISTIKE


def kombinuj_tezine(
    teorijske_grane: list[tuple[int, int]],
    teorijske_tezine: list[float],
    statisticke_grane: list[tuple[int, int]],
    statisticke_tezine: list[float],
    alfa: float,
) -> tuple[list[tuple[int, int]], list[float]]:
    """
    Kombinuje teorijske i statističke grane u jedinstven skup grana.
    Težinska funkcija: w(t, a) = α × teorija + (1-α) × statistika

    Za grane koje postoje u oba izvora — kombinuje težine.
    Za grane koje postoje samo u jednom — koristi samo tu komponentu.

    Parametri:
        alfa : float ∈ [0, 1]
               alfa=1.0 → čisto teorijski
               alfa=0.0 → čisto statistički
               alfa=0.5 → podjednako (hipoteza H2: optimum je negde između)
    """
    beta = 1.0 - alfa

    # Rečnik: grana → (teorijska_vrednost, statisticka_vrednost)
    mapa_grana: dict[tuple[int, int], list[float]] = {}

    for grana, tezina in zip(teorijske_grane, teorijske_tezine):
        if grana not in mapa_grana:
            mapa_grana[grana] = [0.0, 0.0]
        mapa_grana[grana][0] = tezina  # teorijska komponenta

    for grana, tezina in zip(statisticke_grane, statisticke_tezine):
        if grana not in mapa_grana:
            mapa_grana[grana] = [0.0, 0.0]
        mapa_grana[grana][1] = tezina  # statistička komponenta

    # Kombinovanje
    kombinovane_grane = []
    kombinovane_tezine = []

    for grana, (t_vrednost, s_vrednost) in mapa_grana.items():
        kombinovana_tezina = alfa * t_vrednost + beta * s_vrednost
        kombinovane_grane.append(grana)
        kombinovane_tezine.append(kombinovana_tezina)

    print(f"[INFO] Kombinovane grane (α={alfa:.2f}): {len(kombinovane_grane)}")
    return kombinovane_grane, kombinovane_tezine



# KONSTRUKCIJA PyTorch Geometric HeteroData GRAFA


def izgradi_hetero_graf(
    recnik_akorada: dict[str, int],
    trening: list[dict],
    alfa: float = 0.5,
) -> HeteroData:
    """
    Gradi heterogeni graf G = (V, E, W) spreman za PyTorch Geometric.

    Tipovi čvorova:
        'nota'  : 12 čvorova (pitch klase 0–11)
        'akord' : N čvorova (iz rečnika, bez NEPOZNAT_AKORD)

    Tipovi grana:
        ('nota',  'pripada',  'akord') — Tip 1 + Tip 3 kombinovano
        ('akord', 'blizina',  'akord') — Tip 2 (kvintni krug)
        ('akord', 'pripada_inv', 'nota') — inverz Tipa 1 (za propagaciju u oba smera)

    Parametri:
        recnik_akorada : {naziv: indeks}
        trening        : lista pesama za računanje statistike
        alfa           : težinski parametar (α ∈ [0, 1])

    Vraća:
        PyTorch Geometric HeteroData objekat
    """
    # Filtriramo NEPOZNAT_AKORD (indeks 0) iz rečnika
    indeksi_akorada = {
        naziv: idx - 1  # re-indeksiramo od 0
        for naziv, idx in recnik_akorada.items()
        if idx > 0
    }
    broj_akorada = len(indeksi_akorada)
    broj_nota = 12

    print(f"\n[INFO] Gradim graf sa {broj_nota} čvorova-nota i "
          f"{broj_akorada} čvorova-akorda...")

    #  Tip 1: Teorijske grane
    t1_grane, t1_tezine = izgradi_teorijske_grane(indeksi_akorada)

    #  Tip 2: Kvintni krug 
    t2_grane, t2_tezine = izgradi_kvintne_grane(indeksi_akorada)

    # Tip 3: Statistička ko-pojava 
    ko_pojave = izracunaj_ko_pojave(trening, recnik_akorada)

    # Mapiramo originalne indekse akorda na re-indeksirane
    ko_pojave_reindeksirane = {}
    for (pk, orig_idx), vred in ko_pojave.items():
        naziv = next(
            (n for n, i in recnik_akorada.items() if i == orig_idx), None
        )
        if naziv and naziv in indeksi_akorada:
            novi_idx = indeksi_akorada[naziv]
            ko_pojave_reindeksirane[(pk, novi_idx)] = vred

    t3_grane, t3_tezine = izgradi_statisticke_grane(ko_pojave_reindeksirane)

    #Kombinovanje Tipa 1 i Tipa 3 
    komb_grane, komb_tezine = kombinuj_tezine(
        t1_grane, t1_tezine,
        t3_grane, t3_tezine,
        alfa=alfa,
    )

    
    graf = HeteroData()

    # Čvorovi — koristimo one-hot enkodovanje kao početne osobine
    graf["nota"].x = torch.eye(broj_nota, dtype=torch.float)
    graf["akord"].x = torch.eye(broj_akorada, dtype=torch.float)

    # Grane Tip 1+3: nota → akord
    if komb_grane:
        src = torch.tensor([g[0] for g in komb_grane], dtype=torch.long)
        dst = torch.tensor([g[1] for g in komb_grane], dtype=torch.long)
        graf["nota", "pripada", "akord"].edge_index = torch.stack([src, dst])
        graf["nota", "pripada", "akord"].edge_attr = torch.tensor(
            komb_tezine, dtype=torch.float
        ).unsqueeze(1)

        # Inverz: akord → nota (za propagaciju u oba smera)
        graf["akord", "sadrzi", "nota"].edge_index = torch.stack([dst, src])
        graf["akord", "sadrzi", "nota"].edge_attr = torch.tensor(
            komb_tezine, dtype=torch.float
        ).unsqueeze(1)

    # Grane Tip 2: akord ↔ akord (kvintni krug)
    if t2_grane:
        src2 = torch.tensor([g[0] for g in t2_grane], dtype=torch.long)
        dst2 = torch.tensor([g[1] for g in t2_grane], dtype=torch.long)
        graf["akord", "blizina", "akord"].edge_index = torch.stack([src2, dst2])
        graf["akord", "blizina", "akord"].edge_attr = torch.tensor(
            t2_tezine, dtype=torch.float
        ).unsqueeze(1)

    print(f"[INFO] Graf uspešno konstruisan!")
    return graf, indeksi_akorada



# VALIDACIJA GRAFA


def validiraj_graf(
    graf: HeteroData,
    indeksi_akorada: dict[str, int],
) -> None:
    """
    Proverava ispravnost konstruisanog grafa.
    Ispisuje detaljan izveštaj.
    """
    print("\n" + "═" * 60)
    print("  VALIDACIJA TONALNOG GRAFA")
    print("═" * 60)

    # Broj čvorova
    print(f"  Čvorovi-note  : {graf['nota'].x.shape[0]}")
    print(f"  Čvorovi-akordi: {graf['akord'].x.shape[0]}")

    # Grane nota→akord
    if ("nota", "pripada", "akord") in graf.edge_types:
        ei = graf["nota", "pripada", "akord"].edge_index
        ea = graf["nota", "pripada", "akord"].edge_attr
        print(f"\n  Grane nota→akord (Tip1+3):")
        print(f"    Broj grana : {ei.shape[1]}")
        print(f"    Min težina : {ea.min().item():.4f}")
        print(f"    Max težina : {ea.max().item():.4f}")
        print(f"    Avg težina : {ea.mean().item():.4f}")

    # Grane akord↔akord
    if ("akord", "blizina", "akord") in graf.edge_types:
        ei2 = graf["akord", "blizina", "akord"].edge_index
        ea2 = graf["akord", "blizina", "akord"].edge_attr
        print(f"\n  Grane akord↔akord (Tip 2 — kvintni krug):")
        print(f"    Broj grana : {ei2.shape[1]}")
        print(f"    Min težina : {ea2.min().item():.4f}")
        print(f"    Max težina : {ea2.max().item():.4f}")

    # Provera Cmaj↔Gmaj (treba da postoji, d=1, w=0.5)
    if "Cmaj" in indeksi_akorada and "Gmaj" in indeksi_akorada:
        idx_c = indeksi_akorada["Cmaj"]
        idx_g = indeksi_akorada["Gmaj"]
        if ("akord", "blizina", "akord") in graf.edge_types:
            ei2 = graf["akord", "blizina", "akord"].edge_index
            ea2 = graf["akord", "blizina", "akord"].edge_attr
            maska = (ei2[0] == idx_c) & (ei2[1] == idx_g)
            if maska.any():
                tezina = ea2[maska].item()
                print(f"\n  ✓ Cmaj↔Gmaj grana postoji, težina={tezina:.4f} "
                      f"(očekivano ~0.5000)")
            else:
                print(f"\n  ✗ UPOZORENJE: Cmaj↔Gmaj grana ne postoji!")

    # Provera da svaka nota ima bar jednu granu
    if ("nota", "pripada", "akord") in graf.edge_types:
        ei = graf["nota", "pripada", "akord"].edge_index
        note_sa_granama = set(ei[0].tolist())
        print(f"\n  Note sa bar jednom granom: {len(note_sa_granama)}/12")
        if len(note_sa_granama) < 12:
            note_bez = [i for i in range(12) if i not in note_sa_granama]
            print(f"  ✗ Note bez grana: {[NAZIVI_NOTA[i] for i in note_bez]}")
        else:
            print(f"  ✓ Sve note imaju bar jednu granu ka akordu")

    print("═" * 60 + "\n")



# ČUVANJE I UČITAVANJE GRAFA


def sacuvaj_graf(
    graf: HeteroData,
    indeksi_akorada: dict[str, int],
    alfa: float,
    putanja: Path = PUTANJA_GRAFA,
) -> None:
    """Čuva graf i metapodatke na disk."""
    putanja.mkdir(parents=True, exist_ok=True)
    torch.save(graf, putanja / f"tonalni_graf_alfa{alfa:.2f}.pt")
    with open(putanja / "indeksi_akorada.json", "w", encoding="utf-8") as f:
        json.dump(indeksi_akorada, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Graf sačuvan u: {putanja}")


def ucitaj_graf(
    alfa: float,
    putanja: Path = PUTANJA_GRAFA,
) -> tuple[HeteroData, dict[str, int]]:
    """Učitava sačuvani graf sa diska."""
    graf = torch.load(putanja / f"tonalni_graf_alfa{alfa:.2f}.pt", weights_only=False)
    with open(putanja / "indeksi_akorada.json", "r", encoding="utf-8") as f:
        indeksi_akorada = json.load(f)
    print(f"[INFO] Graf učitan iz: {putanja}")
    return graf, indeksi_akorada



# GLAVNA FUNKCIJA


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Faza 3: Konstrukcija tonalnog grafa"
    )
    parser.add_argument(
        "--alfa", type=float, default=0.5,
        help="Težinski parametar α ∈ [0,1] (default: 0.5)"
    )
    parser.add_argument(
        "--validacija", action="store_true",
        help="Pokreni samo validaciju bez ponovne konstrukcije"
    )
    args = parser.parse_args()

    # Učitavamo obrađene podatke
    with open(PUTANJA_OBRADENIH / "trening.pkl", "rb") as f:
        trening = pickle.load(f)
    with open(PUTANJA_OBRADENIH / "recnik_akorada.json", "r", encoding="utf-8") as f:
        recnik_akorada = json.load(f)

    if args.validacija and (PUTANJA_GRAFA / f"tonalni_graf_alfa{args.alfa:.2f}.pt").exists():
        graf, indeksi_akorada = ucitaj_graf(args.alfa)
    else:
        # Konstrukcija grafa
        graf, indeksi_akorada = izgradi_hetero_graf(
            recnik_akorada=recnik_akorada,
            trening=trening,
            alfa=args.alfa,
        )
        sacuvaj_graf(graf, indeksi_akorada, args.alfa)

    # Validacija
    validiraj_graf(graf, indeksi_akorada)

    print("Tipovi grana u grafu:")
    for tip in graf.edge_types:
        print(f"  {tip}")