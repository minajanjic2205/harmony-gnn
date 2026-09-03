"""
config.py — Centralizovani hiperparametri projekta
Autor: Mina Janjić | Predmet: Računarstvo
Opis: Sve vrednosti koje se mogu menjati su ovde — ne treba dirati ostale fajlove.
"""

from pathlib import Path


# PUTANJE


PUTANJA_ABC         = Path("podaci/nottingham/ABC")
PUTANJA_OBRADENIH   = Path("podaci/obradeni")
PUTANJA_GRAFA       = Path("podaci/graf")
PUTANJA_MODELA      = Path("modeli")
PUTANJA_REZULTATA   = Path("rezultati")


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
MAKS_KVINTNO_RASTOJANJE = 3

# Minimalna statistička verovatnoća za dodavanje grane Tipa 3
PRAG_KOPOJAVE = 0.01


# GNN MODEL (E2 i E3)


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