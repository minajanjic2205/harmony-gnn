"""
baseline.py - faza 2: LSTM Baseline Model (Eksperiment E1)
Opis: PyTorch LSTM arhitektura koja predviđa akorde iz melodijskog niza
      pitch klasa. Obuhvata DataLoader, petlju za treniranje i evaluaciju.
"""

import json
import pickle
import random
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# KONSTANTE I PODRAZUMEVANE VREDNOSTI

PUTANJA_OBRADENIH = Path("podaci/obradeni")
PUTANJA_MODELA = Path("modeli")

# Pitch klase: 0–11 + 12 za pauzu (-1 se mapira na 12)
BROJ_PITCH_KLASA = 13  # 0–11 note + 12 = pauza

PODRAZUMEVANI_HP = {
    "velicina_ugradnje": 32,      # dimenzija embedding sloja
    "skrivene_jedinice": 128,     # broj LSTM skrivenih jedinica
    "broj_lstm_slojeva": 2,       # dubina LSTM-a
    "dropout_stopa": 0.3,         # stopa regularizacije
    "velicina_prozora": 16,       # broj nota u jednom ulaznom uzorku
    "velicina_serije": 64,        # veličina mini-batch-a
    "stopa_ucenja": 1e-3,         # learning rate
    "broj_epoha": 30,             # ukupan broj epoha treniranja
    "klip_gradijenta": 1.0,       # gradient clipping vrednost
}



# SKUP PODATAKA (PyTorch Dataset)


class MelodijaAkordDataset(Dataset):
    """
    PyTorch Dataset koji konvertuje pesme u (ulaz, cilj) parove.

    Pristup klizećeg prozora (sliding window):
      - Ulaz: sekvenca od `velicina_prozora` pitch klasa
      - Cilj: akord koji odgovara POSLEDNJOJ noti u prozoru
    """

    def __init__(
        self,
        pesme: list[dict],
        velicina_prozora: int = PODRAZUMEVANI_HP["velicina_prozora"],
    ) -> None:
        self.prozori: list[tuple[list[int], int]] = []
        self.velicina_prozora = velicina_prozora

        for pesma in pesme:
            pitch_klase = [
                pk if pk >= 0 else 12  # -1 (pauza) → 12
                for pk, _ in pesma["melodija_pitch_klase"]
            ]
            akordi = pesma.get("akordi_enkodovani", [])

            if len(pitch_klase) < velicina_prozora:
                continue

            for pocetak in range(len(pitch_klase) - velicina_prozora):
                kraj = pocetak + velicina_prozora
                prozor_ulaz = pitch_klase[pocetak:kraj]
                ciljni_akord = akordi[kraj - 1]  # akord poslednje note u prozoru

                # Preskačemo prozore sa nepoznatim (0) akordima kao ciljem
                if ciljni_akord == 0:
                    continue

                self.prozori.append((prozor_ulaz, ciljni_akord))

    def __len__(self) -> int:
        return len(self.prozori)

    def __getitem__(self, indeks: int) -> tuple[torch.Tensor, torch.Tensor]:
        ulaz, cilj = self.prozori[indeks]
        return (
            torch.tensor(ulaz, dtype=torch.long),
            torch.tensor(cilj, dtype=torch.long),
        )


def napravi_loader(
    pesme: list[dict],
    velicina_prozora: int,
    velicina_serije: int,
    mesati: bool = True,
) -> DataLoader:
    """Kreira DataLoader iz liste pesama."""
    skup = MelodijaAkordDataset(pesme, velicina_prozora)
    print(f"[INFO] Kreirano {len(skup)} uzoraka (prozori dužine {velicina_prozora})")
    return DataLoader(skup, batch_size=velicina_serije, shuffle=mesati)


# LSTM ARHITEKTURA


class LSTMPredvidjanjAkorda(nn.Module):
    """
    LSTM model za predviđanje akorda iz melodije (Eksperiment E1).

    Arhitektura:
        1. Embedding sloj: pitch_klasa → vektor dimenzije `velicina_ugradnje`
        2. LSTM: `broj_lstm_slojeva` slojeva sa `skrivene_jedinice` jedinica
        3. Dropout: regularizacija između LSTM i linearnog sloja
        4. Linearni sloj: skriveno stanje → distribucija nad `broj_akorda` klasa
    """

    def __init__(
        self,
        broj_akorda: int,
        velicina_ugradnje: int = PODRAZUMEVANI_HP["velicina_ugradnje"],
        skrivene_jedinice: int = PODRAZUMEVANI_HP["skrivene_jedinice"],
        broj_lstm_slojeva: int = PODRAZUMEVANI_HP["broj_lstm_slojeva"],
        dropout_stopa: float = PODRAZUMEVANI_HP["dropout_stopa"],
    ) -> None:
        super().__init__()

        self.broj_akorda = broj_akorda
        self.skrivene_jedinice = skrivene_jedinice
        self.broj_lstm_slojeva = broj_lstm_slojeva

        # 1. Embedding za pitch klase (0–12)
        self.ugradnja = nn.Embedding(
            num_embeddings=BROJ_PITCH_KLASA,
            embedding_dim=velicina_ugradnje,
            padding_idx=12,  # pauza ne doprinosi gradijentima
        )

        # 2. LSTM enkoder
        self.lstm = nn.LSTM(
            input_size=velicina_ugradnje,
            hidden_size=skrivene_jedinice,
            num_layers=broj_lstm_slojeva,
            batch_first=True,          # (serija, sekvenca, osobine)
            dropout=dropout_stopa if broj_lstm_slojeva > 1 else 0.0,
        )

        # 3. Regularizacija
        self.dropout = nn.Dropout(dropout_stopa)

        # 4. Klasifikacioni sloj
        self.fc = nn.Linear(skrivene_jedinice, broj_akorda)

    def forward(
        self,
        x: torch.Tensor,                          # (serija, duzina_sekvence)
        skriveno: Optional[tuple] = None,
    ) -> tuple[torch.Tensor, tuple]:
        """
        Prolazak unapred.

        Parametri:
            x        : tenzor celobrojnih pitch klasa (serija, duzina_sekvence)
            skriveno : (h_n, c_n) prethodno LSTM stanje (opcionalno)

        Vraća:
            logiti  : (serija, broj_akorda) — nenormalizovane verovatnoće
            skriveno: ažurirano LSTM stanje
        """
        # (serija, duzina_sekvence) → (serija, duzina_sekvence, velicina_ugradnje)
        ugradnuto = self.ugradnja(x)

        # (serija, duzina_sekvence, velicina_ugradnje) → (serija, duzina_sekvence, skrivene_jedinice)
        izlaz_lstm, skriveno = self.lstm(ugradnuto, skriveno)

        # Uzimamo samo poslednje vremensko stanje: (serija, skrivene_jedinice)
        poslednje_stanje = izlaz_lstm[:, -1, :]

        # Regularizacija
        poslednje_stanje = self.dropout(poslednje_stanje)

        # Logiti: (serija, broj_akorda)
        logiti = self.fc(poslednje_stanje)

        return logiti, skriveno

    def inicijalizuj_skriveno(
        self, velicina_serije: int, uredjaj: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Vraća nulto inicijalizovano LSTM skriveno stanje."""
        h_0 = torch.zeros(
            self.broj_lstm_slojeva, velicina_serije, self.skrivene_jedinice,
            device=uredjaj,
        )
        c_0 = torch.zeros_like(h_0)
        return h_0, c_0



# PETLJA ZA TRENIRANJE


def treniraj_jednu_epohu(
    model: LSTMPredvidjanjAkorda,
    loader: DataLoader,
    optimizator: optim.Optimizer,
    kriterijum: nn.CrossEntropyLoss,
    uredjaj: torch.device,
    klip_gradijenta: float,
) -> float:
    """
    Trenira model jednu epohu.
    Vraća prosečan gubitak po seriji.
    """
    model.train()
    ukupni_gubitak = 0.0

    for serija_ulaz, serija_cilj in loader:
        serija_ulaz = serija_ulaz.to(uredjaj)
        serija_cilj = serija_cilj.to(uredjaj)

        optimizator.zero_grad()

        logiti, _ = model(serija_ulaz)
        gubitak = kriterijum(logiti, serija_cilj)

        gubitak.backward()

        # Gradient clipping — sprečava eksplodiranje gradijenata u LSTM-u
        nn.utils.clip_grad_norm_(model.parameters(), klip_gradijenta)

        optimizator.step()
        ukupni_gubitak += gubitak.item()

    return ukupni_gubitak / len(loader)


@torch.no_grad()
def evaluiraj(
    model: LSTMPredvidjanjAkorda,
    loader: DataLoader,
    kriterijum: nn.CrossEntropyLoss,
    uredjaj: torch.device,
) -> tuple[float, float]:
    """
    Evaluira model na zadatom skupu.
    Vraća (prosecni_gubitak, tacnost_akorda).
    """
    model.eval()
    ukupni_gubitak = 0.0
    tacno = 0
    ukupno = 0

    for serija_ulaz, serija_cilj in loader:
        serija_ulaz = serija_ulaz.to(uredjaj)
        serija_cilj = serija_cilj.to(uredjaj)

        logiti, _ = model(serija_ulaz)
        gubitak = kriterijum(logiti, serija_cilj)
        ukupni_gubitak += gubitak.item()

        predvidjanja = logiti.argmax(dim=1)
        tacno += (predvidjanja == serija_cilj).sum().item()
        ukupno += serija_cilj.size(0)

    tacnost = tacno / ukupno if ukupno > 0 else 0.0
    return ukupni_gubitak / len(loader), tacnost


def treniraj_model(
    trening: list[dict],
    validacija: list[dict],
    recnik_akorada: dict[str, int],
    hp: dict = PODRAZUMEVANI_HP,
    sacuvati_model: bool = True,
    naziv: str = "lstm_baseline_najbolji",
) -> LSTMPredvidjanjAkorda:
    """
    Kompletna petlja treniranja za E1 LSTM Baseline.

    Parametri:
        trening         : lista pesama za treniranje
        validacija      : lista pesama za validaciju
        recnik_akorada  : mapiranje akord→indeks
        hp              : hiperparametri (koristite PODRAZUMEVANI_HP ili custom)
        sacuvati_model  : da li čuvati checkpoint sa najboljim modelom

    Vraća:
        Istrenirani LSTMPredvidjanjAkorda model.
    """
    uredjaj = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Koristi se uređaj: {uredjaj}")

    # ── Loaderi ──────────────────────────────────────────────────────────────
    trening_loader = napravi_loader(
        trening,
        hp["velicina_prozora"],
        hp["velicina_serije"],
        mesati=True,
    )
    validacioni_loader = napravi_loader(
        validacija,
        hp["velicina_prozora"],
        hp["velicina_serije"],
        mesati=False,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    broj_akorda = len(recnik_akorada)
    model = LSTMPredvidjanjAkorda(
        broj_akorda=broj_akorda,
        velicina_ugradnje=hp["velicina_ugradnje"],
        skrivene_jedinice=hp["skrivene_jedinice"],
        broj_lstm_slojeva=hp["broj_lstm_slojeva"],
        dropout_stopa=hp["dropout_stopa"],
    ).to(uredjaj)

    ukupno_parametara = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Ukupno parametara modela: {ukupno_parametara:,}")

    # ── Optimizator i gubitak ─────────────────────────────────────────────────
    # Weighted CrossEntropy: težina 0 za NEPOZNAT_AKORD klasu (indeks 0)
    tezine_klasa = torch.ones(broj_akorda, device=uredjaj)
    tezine_klasa[0] = 0.0  # NEPOZNAT_AKORD ne penalizujemo

    kriterijum = nn.CrossEntropyLoss(weight=tezine_klasa)
    optimizator = optim.Adam(model.parameters(), lr=hp["stopa_ucenja"])

    # ── Raspored stope učenja ─────────────────────────────────────────────────
    rasporedivac = optim.lr_scheduler.ReduceLROnPlateau(
        optimizator, mode="min", patience=3, factor=0.5
    )

    # ── Petlja treniranja ─────────────────────────────────────────────────────
    PUTANJA_MODELA.mkdir(parents=True, exist_ok=True)
    putanja_checkpointa = PUTANJA_MODELA / f"{naziv}.pt"

    najbolja_val_tacnost = 0.0
    istorija: list[dict] = []

    print("\n" + "─" * 60)
    print(f"{'Epoha':>6}  {'Trening Gubitak':>16}  "
          f"{'Val Gubitak':>12}  {'Val Tačnost':>12}")
    print("─" * 60)

    for epoha in range(1, hp["broj_epoha"] + 1):
        tr_gubitak = treniraj_jednu_epohu(
            model, trening_loader, optimizator, kriterijum,
            uredjaj, hp["klip_gradijenta"],
        )
        val_gubitak, val_tacnost = evaluiraj(
            model, validacioni_loader, kriterijum, uredjaj,
        )
        rasporedivac.step(val_gubitak)

        print(f"{epoha:>6}  {tr_gubitak:>16.4f}  "
              f"{val_gubitak:>12.4f}  {val_tacnost:>11.2%}")

        istorija.append({
            "epoha": epoha,
            "trening_gubitak": tr_gubitak,
            "val_gubitak": val_gubitak,
            "val_tacnost": val_tacnost,
        })

        # Čuvamo checkpoint najboljeg modela
        if val_tacnost > najbolja_val_tacnost:
            najbolja_val_tacnost = val_tacnost
            if sacuvati_model:
                torch.save({
                    "epoha": epoha,
                    "stanje_modela": model.state_dict(),
                    "stanje_optimizatora": optimizator.state_dict(),
                    "val_tacnost": val_tacnost,
                    "recnik_akorada": recnik_akorada,
                    "hp": hp,
                }, putanja_checkpointa)
                print(f"  → Checkpoint sačuvan (val tačnost: {val_tacnost:.2%})")

    print("─" * 60)
    print(f"[INFO] Treniranje završeno. Najbolja val. tačnost: {najbolja_val_tacnost:.2%}")

    return model



# EVALUACIJA NA TEST SKUPU


def evaluiraj_na_test_skupu(
    model: LSTMPredvidjanjAkorda,
    test: list[dict],
    recnik_akorada: dict[str, int],
    hp: dict = PODRAZUMEVANI_HP,
) -> dict:
    """
    Evaluira istrenirani model na test skupu.
    Vraća rečnik sa chord_accuracy i top3_accuracy metrikama.
    """
    uredjaj = next(model.parameters()).device
    test_loader = napravi_loader(
        test, hp["velicina_prozora"], hp["velicina_serije"], mesati=False,
    )

    model.eval()
    tacno_top1 = 0
    tacno_top3 = 0
    ukupno = 0

    tezine = torch.ones(len(recnik_akorada), device=uredjaj)
    tezine[0] = 0.0
    kriterijum = nn.CrossEntropyLoss(weight=tezine)

    with torch.no_grad():
        for serija_ulaz, serija_cilj in test_loader:
            serija_ulaz = serija_ulaz.to(uredjaj)
            serija_cilj = serija_cilj.to(uredjaj)

            logiti, _ = model(serija_ulaz)

            # Top-1 tačnost
            pred_top1 = logiti.argmax(dim=1)
            tacno_top1 += (pred_top1 == serija_cilj).sum().item()

            # Top-3 tačnost
            _, pred_top3 = logiti.topk(k=min(3, logiti.size(1)), dim=1)
            tacno_top3 += (
                pred_top3 == serija_cilj.unsqueeze(1)
            ).any(dim=1).sum().item()

            ukupno += serija_cilj.size(0)

    chord_accuracy = tacno_top1 / ukupno if ukupno > 0 else 0.0
    top3_accuracy = tacno_top3 / ukupno if ukupno > 0 else 0.0

    rezultati = {
        "chord_accuracy": chord_accuracy,
        "top3_accuracy": top3_accuracy,
        "ukupno_uzoraka": ukupno,
    }

    print("\n" + "═" * 40)
    print("  REZULTATI NA TEST SKUPU (E1 — LSTM Baseline)")
    print("═" * 40)
    print(f"  Chord Accuracy (CA) : {chord_accuracy:.2%}")
    print(f"  Top-3 Accuracy      : {top3_accuracy:.2%}")
    print(f"  Ukupno test uzoraka : {ukupno}")
    print("═" * 40 + "\n")

    return rezultati



# POKRETANJE


if __name__ == "__main__":
    import pickle

    # Učitavamo obrađene podatke (mora prethodno pokrenuti dataset.py)
    with open(PUTANJA_OBRADENIH / "trening.pkl", "rb") as f:
        trening = pickle.load(f)
    with open(PUTANJA_OBRADENIH / "validacija.pkl", "rb") as f:
        validacija = pickle.load(f)
    with open(PUTANJA_OBRADENIH / "test.pkl", "rb") as f:
        test = pickle.load(f)
    with open(PUTANJA_OBRADENIH / "recnik_akorada.json", "r") as f:
        recnik = json.load(f)

    print(f"[INFO] Veličina rečnika: {len(recnik)} akorda")

    # Treniranje LSTM baseline modela
    model = treniraj_model(trening, validacija, recnik)

    # Evaluacija na test skupu
    evaluiraj_na_test_skupu(model, test, recnik)