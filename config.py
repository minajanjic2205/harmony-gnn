"""
config.py — Centralizovani hiperparametri projekta
Autor: Mina Janjić | Predmet: Računarstvo
Opis: Sve vrednosti koje se mogu menjati su ovde — ne treba dirati ostale fajlove.
"""

from pathlib import Path


# AKTIVNI DATASET


# Menjaj ovde kad prebacuješ između datasetova: "nottingham" ili "openewld".
# SVE putanje ispod (podaci, modeli, rezultati) automatski se granaju
# prema ovoj vrednosti, tako da se rezultati različitih dataseta NIKAD ne
# mešaju niti jedan prepisuje drugi u istom folderu.
AKTIVNI_DATASET = "openewld"

# Koji nivo pojednostavljivanja rečnika akorada koristimo:
#   "osnovno"    — Faza 1: dur/mol (+umanjeni/uvećani za ne-Nottingham) trijade
#   "septakordi" — Faza 2: dodaje dominant7/major7/minor7, vidi akord_u_oznaku()
# Menjanje ove vrednosti NE prepisuje rezultate druge faze — svaka ima
# svoj odvojen podfolder, da bi mogle da se porede jedna naspram druge.
PROSIRENJE_AKORADA = "septakordi"

# Koju varijantu strukture tonalnog grafa koristimo:
#   "standardna" — originalna podešavanja (kvintno rastojanje 3, prag 0.01)
#   "pojacana"   — čistiji signal: kraće kvintno rastojanje (samo direktni
#                  susedi), stroži prag ko-pojave (filtrira statistički šum)
# Isto kao PROSIRENJE_AKORADA — svaka varijanta ima svoj odvojen podfolder.
STRUKTURA_GRAFA = "pojacana"

_STRUKTURE_GRAFA = {
    "standardna": {"maks_kvintno_rastojanje": 3, "prag_kopojave": 0.01},
    "pojacana":   {"maks_kvintno_rastojanje": 1, "prag_kopojave": 0.05},
}


# PUTANJE


PUTANJA_ABC             = Path("podaci/nottingham/ABC")   # sirovi ABC fajlovi, samo za Nottingham
PUTANJA_OPENEWLD_SIROVI = Path("podaci/openewld/sirovi")  # sirovi .mxl fajlovi, samo za OpenEWLD

PUTANJA_OBRADENIH   = Path(f"podaci/{AKTIVNI_DATASET}/{PROSIRENJE_AKORADA}/obradeni")
PUTANJA_GRAFA       = Path(f"podaci/{AKTIVNI_DATASET}/{PROSIRENJE_AKORADA}/{STRUKTURA_GRAFA}/graf")
PUTANJA_MODELA      = Path(f"modeli/{AKTIVNI_DATASET}/{PROSIRENJE_AKORADA}/{STRUKTURA_GRAFA}")
PUTANJA_REZULTATA   = Path(f"rezultati/{AKTIVNI_DATASET}/{PROSIRENJE_AKORADA}/{STRUKTURA_GRAFA}")


# PREPROCESIRANJE


OMJER_TRENINGA      = 0.70
OMJER_VALIDACIJE    = 0.15
# test = 1 - trening - validacija = 0.15


# LSTM BASELINE (E1)


LSTM_HP = {
    "velicina_ugradnje"  : 32,
    "skrivene_jedinice"  : 128,
    "broj_lstm_slojeva"  : 2,
    "dropout_stopa"      : 0.3,
    "velicina_prozora"   : 16,
    "velicina_serije"    : 64,
    "stopa_ucenja"       : 1e-3,
    "broj_epoha"         : 30,
    "klip_gradijenta"    : 1.0,
}


# TONALNI GRAF


# Maksimalno rastojanje na kvintnom krugu za koje dodajemo granu
MAKS_KVINTNO_RASTOJANJE = _STRUKTURE_GRAFA[STRUKTURA_GRAFA]["maks_kvintno_rastojanje"]

# Minimalna statistička verovatnoća za dodavanje grane Tipa 3
PRAG_KOPOJAVE = _STRUKTURE_GRAFA[STRUKTURA_GRAFA]["prag_kopojave"]


# GNN MODEL (E2 i E3)


# Koju vrstu propagacionog sloja koristi GNN:
#   "gcn" — originalna arhitektura: FIKSNA tezina grane (alfa*teorija+beta*statistika)
#   "gat" — Graph Attention: MODEL UCI koliko paznje da posveti svakom
#           susedu, uz fiksnu tezinu grane kao dodatni signal (hibridno)
# NAPOMENA: ovo NE utice na sam graf (isti tonalni_graf_alfaX.pt fajlovi
# se koriste za oba) — utice samo na to KAKO GNN sloj cita taj graf.
# Da se GCN i GAT rezultati ne mesaju, checkpoint fajlovi i rezultati
# imaju ARHITEKTURA ugradjen u ime fajla (vidi grid_search_visestruko.py).
ARHITEKTURA = "gcn"


GNN_HP = {
    "skrivene_dimenzije" : 128,
    "broj_gcn_slojeva"   : 2,
    "dropout_stopa"      : 0.3,
    "velicina_prozora"   : 16,
    "velicina_serije"    : 64,
    "stopa_ucenja"       : 1e-3,
    "broj_epoha"         : 30,
    "klip_gradijenta"    : 1.0,
}


# GRID SEARCH (istraživanje α/β prostora — Hipoteza H2)


# Vrednosti alfe koje se ispituju: 0.0, 0.1, 0.2, ..., 1.0
VREDNOSTI_ALFA = [round(a * 0.1, 1) for a in range(11)]