"""
dataset.py Faza 1: Učitavanje i preprocesiranje Nottingham Music Dataset-a
Opis: Parsovanje ABC notacije u MIDI pitch klase i trajanja za melodiju,
      enkodovanje akorda u celobrojne klase, provera integriteta skupa podataka.
"""

import os
import re
import json
import pickle
import urllib.request
import zipfile
from pathlib import Path
from collections import Counter
from typing import Optional

import music21
from music21 import corpus, converter, note, chord, stream


# konstanty


PUTANJA_PODATAKA = Path("podaci/nottingham")
PUTANJA_OBRADENIH = Path("podaci/obradeni")
URL_NOTTINGHAM = (
    "https://raw.githubusercontent.com/jukedeck/nottingham-dataset"
    "/master/ABC/"
)

# Sve 12 hromatskih nota (pitch klase 0–11)
SVE_NOTE = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Standardni akordi (dur + mol za svih 12 tonaliteta)
SVE_VRSTE_AKORADA = ["major", "minor"]
RECNIK_AKORADA: dict[str, int] = {}  # popunjava se tokom obrade

# Sentinel vrednost za nepoznat akord
NEPOZNAT_AKORD = "<NEPOZNAT>"



# pomoćne funkcije


def pitch_u_klasu(visina_tona: str) -> int:
    """Pretvara ime note (npr. 'C#4') u pitch klasu (0–11)."""
    try:
        midi_broj = music21.pitch.Pitch(visina_tona).midi
        return midi_broj % 12
    except Exception:
        return -1  # neprepoznata nota


def akord_u_oznaku(m21_akord: chord.Chord) -> str:
    """
    Pretvara music21 akord objekat u string oznaku oblika 'Cmaj' / 'Am'.
    Pokriva dur i mol trijade. Ostale vrste vraća kao '<NEPOZNAT>'.
    """
    try:
        koren = m21_akord.root().name  # npr. 'C', 'F#'
        kvalitet = m21_akord.quality   # 'major', 'minor', 'diminished', ...
        if kvalitet == "major":
            return f"{koren}maj"
        elif kvalitet == "minor":
            return f"{koren}min"
        else:
            return NEPOZNAT_AKORD
    except Exception:
        return NEPOZNAT_AKORD


def izgradi_recnik_akorada(lista_akorada: list[str]) -> dict[str, int]:
    """
    Gradi rečnik akord→indeks iz liste svih pronađenih akorda.
    Nepoznati akord uvek dobija indeks 0.
    """
    jedinstveni = sorted(set(a for a in lista_akorada if a != NEPOZNAT_AKORD))
    recnik = {NEPOZNAT_AKORD: 0}
    for i, naziv in enumerate(jedinstveni, start=1):
        recnik[naziv] = i
    return recnik



# parsovanje abc fajlova


def parsiraj_abc_pesmu(abc_tekst: str) -> Optional[dict]:
    """
    Parsuje jednu ABC pesmu koristeći music21.
    Vraća rečnik sa:
      - 'melodija_pitch_klase': lista (pitch_klasa, trajanje_u_cetvrtinama)
      - 'akordi_tekst': lista string oznaka akorada po poziciji nota
    Vraća None ako parsovanje ne uspe.
    """
    try:
        partitura = converter.parse(abc_tekst, format="abc")
    except Exception as greska:
        return None

    melodija_pitch_klase = []
    akordi_tekst = []

    # Prolazimo kroz sve delove partiture
    for deo in partitura.parts:
        for element in deo.flatten().notesAndRests:
            trajanje = float(element.duration.quarterLength)

            if isinstance(element, note.Note):
                pk = element.pitch.midi % 12
                melodija_pitch_klase.append((pk, trajanje))
                akordi_tekst.append(NEPOZNAT_AKORD)  # placeholder

            elif isinstance(element, chord.Chord):
                # Uzimamo najvišu notu akorda kao melodijsku notu
                visine = element.sortAscending().pitches
                if not visine:
                    continue  # preskačemo akord bez nota
                pk = visine[-1].midi % 12
                melodija_pitch_klase.append((pk, trajanje))
                oznaka = akord_u_oznaku(element)
                akordi_tekst.append(oznaka)

            elif isinstance(element, note.Rest):
                melodija_pitch_klase.append((-1, trajanje))  # -1 = pauza
                akordi_tekst.append(NEPOZNAT_AKORD)

    if not melodija_pitch_klase:
        return None

    return {
        "melodija_pitch_klase": melodija_pitch_klase,
        "akordi_tekst": akordi_tekst,
    }



# ucitavanje skupa podataka


def ucitaj_nottingham_iz_music21() -> list[dict]:
    """
    Učitava Nottingham folk pesme koje su ugrađene u music21 corpus.
    Vraća listu parsovanih pesama.
    """
    print("[INFO] Tražim Nottingham pesme u music21 corpus-u...")
    putanje = corpus.getComposer("nottingham")

    if not putanje:
        print("[UPOZORENJE] Nottingham nije pronađen u corpus-u. "
              "Instalirajte: pip install music21[corpus]")
        return []

    pesme = []
    for i, putanja in enumerate(putanje):
        try:
            partitura = corpus.parse(putanja)
            melodija_pc = []
            akordi_tekst = []

            for deo in partitura.parts:
                for el in deo.flatten().notesAndRests:
                    trajanje = float(el.duration.quarterLength)
                    if isinstance(el, note.Note):
                        melodija_pc.append((el.pitch.midi % 12, trajanje))
                        akordi_tekst.append(NEPOZNAT_AKORD)
                    elif isinstance(el, chord.Chord):
                        pk = el.sortAscending().pitches[-1].midi % 12
                        melodija_pc.append((pk, trajanje))
                        akordi_tekst.append(akord_u_oznaku(el))
                    elif isinstance(el, note.Rest):
                        melodija_pc.append((-1, trajanje))
                        akordi_tekst.append(NEPOZNAT_AKORD)

            if melodija_pc:
                pesme.append({
                    "id": i,
                    "naziv": str(putanja),
                    "melodija_pitch_klase": melodija_pc,
                    "akordi_tekst": akordi_tekst,
                })
        except Exception as e:
            print(f"[GREŠKA] Ne mogu da parsem {putanja}: {e}")
            continue

        if (i + 1) % 50 == 0:
            print(f"  → Obrađeno {i + 1}/{len(putanje)} pesama...")

    print(f"[INFO] Uspešno učitano {len(pesme)} pesama.")
    return pesme


def ucitaj_abc_iz_direktorijuma(putanja_dir: Path) -> list[dict]:
    """
    Alternativno: učitava sve .abc fajlove iz lokalnog direktorijuma.
    Korisno ako je Nottingham dataset ručno preuzet.
    """
    abc_fajlovi = list(putanja_dir.glob("**/*.abc"))
    print(f"[INFO] Pronađeno {len(abc_fajlovi)} ABC fajlova u {putanja_dir}")

    pesme = []
    for fajl in abc_fajlovi:
        tekst = fajl.read_text(encoding="utf-8", errors="ignore")

        # Razdvajamo pesme unutar jednog fajla (počinju sa 'X:')
        blokovi = re.split(r"\nX:", tekst)
        for j, blok in enumerate(blokovi):
            prefiks = "" if j == 0 else "X:"
            rezultat = parsiraj_abc_pesmu(prefiks + blok)
            if rezultat:
                rezultat["id"] = f"{fajl.stem}_{j}"
                rezultat["naziv"] = fajl.name
                pesme.append(rezultat)

    print(f"[INFO] Uspešno parsovano {len(pesme)} pesama.")
    return pesme



# enkodovanje i podela skupa podataka   


def enkoduj_akorde(pesme: list[dict]) -> tuple[list[dict], dict[str, int]]:
    """
    Kreira globalni rečnik akorda i enkoduje sve akorde u celobrojne indekse.
    Vraća ažuriranu listu pesama i rečnik akord→indeks.
    """
    svi_akordi = []
    for pesma in pesme:
        svi_akordi.extend(pesma["akordi_tekst"])

    recnik = izgradi_recnik_akorada(svi_akordi)
    print(f"[INFO] Veličina rečnika akorda: {len(recnik)} "
          f"(uključuje '{NEPOZNAT_AKORD}')")

    for pesma in pesme:
        pesma["akordi_enkodovani"] = [
            recnik.get(a, 0) for a in pesma["akordi_tekst"]
        ]

    return pesme, recnik


def podeli_skup_podataka(
    pesme: list[dict],
    omjer_treninga: float = 0.70,
    omjer_validacije: float = 0.15,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Deli pesme na trening (70%), validacioni (15%) i test skup (15%).
    Podela je deterministička (bez mešanja) radi reproduktivnosti.
    """
    ukupno = len(pesme)
    kraj_treninga = int(ukupno * omjer_treninga)
    kraj_validacije = kraj_treninga + int(ukupno * omjer_validacije)

    trening = pesme[:kraj_treninga]
    validacija = pesme[kraj_treninga:kraj_validacije]
    test = pesme[kraj_validacije:]

    print(f"[INFO] Podela: trening={len(trening)}, "
          f"validacija={len(validacija)}, test={len(test)}")
    return trening, validacija, test



# provera integriteta skupa podataka


def provera_skupa_podataka(
    pesme: list[dict],
    recnik_akorada: dict[str, int],
    naziv_skupa: str = "ceo skup",
) -> None:
    """
    Štampa sažetak skupa podataka: ukupan broj pesama,
    rečnik akorda, distribucija klasa i prosečna dužina sekvenci.
    """
    print("\n" + "═" * 60)
    print(f"  PROVERA INTEGRITETA — {naziv_skupa.upper()}")
    print("═" * 60)
    print(f"  Ukupno pesama     : {len(pesme)}")

    ukupno_nota = sum(len(p["melodija_pitch_klase"]) for p in pesme)
    prosecna_duzina = ukupno_nota / len(pesme) if pesme else 0
    print(f"  Ukupno nota       : {ukupno_nota}")
    print(f"  Prosečna dužina   : {prosecna_duzina:.1f} nota/pesmi")

    # Distribucija akorda (top 15)
    brojac_akorada: Counter = Counter()
    for pesma in pesme:
        for akord_id in pesma.get("akordi_enkodovani", []):
            # Tražimo ime iz rečnika
            ime = next(
                (k for k, v in recnik_akorada.items() if v == akord_id),
                NEPOZNAT_AKORD,
            )
            if ime != NEPOZNAT_AKORD:
                brojac_akorada[ime] += 1

    print(f"\n  Veličina rečnika akorda: {len(recnik_akorada)}")
    print(f"  Akordi po učestalosti (top 15):")
    for naziv, broj in brojac_akorada.most_common(15):
        print(f"    {naziv:<12} : {broj}")

    # Distribucija pitch klasa u melodiji
    brojac_nota: Counter = Counter()
    for pesma in pesme:
        for pk, _ in pesma["melodija_pitch_klase"]:
            if pk >= 0:
                brojac_nota[SVE_NOTE[pk]] += 1

    print(f"\n  Distribucija melodijskih nota (pitch klase):")
    for ime_note in SVE_NOTE:
        br = brojac_nota.get(ime_note, 0)
        print(f"    {ime_note:<3} : {br}")

    print("═" * 60 + "\n")


# 
# cuvanje i ucitavanje obrađenih podataka

def sacuvaj_obradene_podatke(
    trening: list[dict],
    validacija: list[dict],
    test: list[dict],
    recnik: dict[str, int],
    izlazna_putanja: Path = PUTANJA_OBRADENIH,
) -> None:
    """Čuva obrađene skupove podataka i rečnik akorda na disk."""
    izlazna_putanja.mkdir(parents=True, exist_ok=True)

    with open(izlazna_putanja / "trening.pkl", "wb") as f:
        pickle.dump(trening, f)
    with open(izlazna_putanja / "validacija.pkl", "wb") as f:
        pickle.dump(validacija, f)
    with open(izlazna_putanja / "test.pkl", "wb") as f:
        pickle.dump(test, f)
    with open(izlazna_putanja / "recnik_akorada.json", "w", encoding="utf-8") as f:
        json.dump(recnik, f, ensure_ascii=False, indent=2)

    print(f"[INFO] Podaci sačuvani u: {izlazna_putanja}")


def ucitaj_obradene_podatke(
    putanja: Path = PUTANJA_OBRADENIH,
) -> tuple[list[dict], list[dict], list[dict], dict[str, int]]:
    """Učitava prethodno obrađene skupove podataka i rečnik sa diska."""
    with open(putanja / "trening.pkl", "rb") as f:
        trening = pickle.load(f)
    with open(putanja / "validacija.pkl", "rb") as f:
        validacija = pickle.load(f)
    with open(putanja / "test.pkl", "rb") as f:
        test = pickle.load(f)
    with open(putanja / "recnik_akorada.json", "r", encoding="utf-8") as f:
        recnik = json.load(f)

    print(f"[INFO] Obrađeni podaci učitani iz: {putanja}")
    return trening, validacija, test, recnik



# glavna funkcija


def pripremi_skup_podataka(
    lokalni_abc_dir: Optional[Path] = None,
    forsirati_ponovnu_obradu: bool = False,
) -> tuple[list[dict], list[dict], list[dict], dict[str, int]]:
    """
    Glavna ulazna tačka za Fazu 1.
    1. Pokušava da učita već obrađene podatke (keš).
    2. Ako ne postoje ili je forsirana ponovna obrada, učitava iz izvora.
    3. Enkoduje akorde, deli skup i čuva na disk.

    Parametri:
        lokalni_abc_dir: putanja do lokalnih ABC fajlova (opcionalno)
        forsirati_ponovnu_obradu: ako True, ignoriše keš

    Vraća:
        (trening, validacija, test, recnik_akorada)
    """
    kesh_postoji = (PUTANJA_OBRADENIH / "trening.pkl").exists()

    if kesh_postoji and not forsirati_ponovnu_obradu:
        print("[INFO] Pronađen keš obrađenih podataka. Učitavam...")
        return ucitaj_obradene_podatke()

    # Izvor podataka
    if lokalni_abc_dir and lokalni_abc_dir.exists():
        pesme = ucitaj_abc_iz_direktorijuma(lokalni_abc_dir)
    else:
        pesme = ucitaj_nottingham_iz_music21()

    if not pesme:
        raise RuntimeError(
            "[GREŠKA] Nije moguće učitati ni jednu pesmu. "
            "Proverite instalaciju music21 corpus-a ili putanju do ABC fajlova."
        )

    # Enkodovanje i podela
    pesme, recnik = enkoduj_akorde(pesme)
    trening, validacija, test = podeli_skup_podataka(pesme)

    # Provera integriteta na trening skupu
    provera_skupa_podataka(trening, recnik, naziv_skupa="trening skup")

    # Čuvanje
    sacuvaj_obradene_podatke(trening, validacija, test, recnik)

    return trening, validacija, test, recnik



# pokretanje


if __name__ == "__main__":
    trening, validacija, test, recnik = pripremi_skup_podataka(
        lokalni_abc_dir=Path("podaci/nottingham/ABC"),
        forsirati_ponovnu_obradu=True,
    )

    print("Primer prve pesme iz trening skupa:")
    primer = trening[0]
    print(f"  Naziv     : {primer.get('naziv', 'N/A')}")
    print(f"  Br. nota  : {len(primer['melodija_pitch_klase'])}")
    print(f"  Prve note (pk, trajanje): {primer['melodija_pitch_klase'][:8]}")
    print(f"  Prvih 8 akorada (indeksi): {primer['akordi_enkodovani'][:8]}")