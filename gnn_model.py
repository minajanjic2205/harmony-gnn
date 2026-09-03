"""
gnn_model.py faza 4: GNN Model za predviđanje akorda (E2 i E3)
Opis: Graph Convolutional Network koji koristi tonalni graf iz graf.py.
      Jednostavna, pouzdana arhitektura:
        1. Embedding melodije - LSTM - melodijski vektor
        2. GCN propagacija nad grafom - obogaćeni vektori akorada
        3. Linearni sloj: melodijski vektor → logiti nad akordima
"""

import json
import pickle
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import HeteroData

from config import GNN_HP, PUTANJA_OBRADENIH, PUTANJA_GRAFA, PUTANJA_MODELA


# DATASET


class MelodijaGrafDataset(Dataset):
    """
    Klizeći prozor: ulaz = 16 pitch klasa, cilj = graf-indeks akorda.
    """
    def __init__(
        self,
        pesme: list[dict],
        mapa_orig_u_graf: dict[int, int],
        velicina_prozora: int = GNN_HP["velicina_prozora"],
    ) -> None:
        self.prozori: list[tuple[list[int], int]] = []

        for pesma in pesme:
            pitch_klase = [
                pk if pk >= 0 else 12
                for pk, _ in pesma["melodija_pitch_klase"]
            ]
            akordi = pesma.get("akordi_enkodovani", [])

            if len(pitch_klase) < velicina_prozora:
                continue

            for poc in range(len(pitch_klase) - velicina_prozora):
                kraj = poc + velicina_prozora
                orig_akord = akordi[kraj - 1]
                if orig_akord not in mapa_orig_u_graf:
                    continue
                self.prozori.append((
                    pitch_klase[poc:kraj],
                    mapa_orig_u_graf[orig_akord],
                ))

    def __len__(self) -> int:
        return len(self.prozori)

    def __getitem__(self, idx: int):
        ulaz, cilj = self.prozori[idx]
        return torch.tensor(ulaz, dtype=torch.long), torch.tensor(cilj, dtype=torch.long)


def napravi_mapu(recnik_akorada: dict[str, int], indeksi_akorada: dict[str, int]) -> dict[int, int]:
    """Pravi mapiranje: originalni indeks akorda → graf indeks akorda."""
    mapa = {}
    for naziv, graf_idx in indeksi_akorada.items():
        orig_idx = recnik_akorada.get(naziv)
        if orig_idx is not None:
            mapa[orig_idx] = graf_idx
    return mapa


def napravi_loader(pesme, mapa, mesati=True) -> DataLoader:
    skup = MelodijaGrafDataset(pesme, mapa)
    print(f"[INFO] Dataset: {len(skup)} uzoraka")
    return DataLoader(skup, batch_size=GNN_HP["velicina_serije"], shuffle=mesati)



# GCN SLOJ (rucna implementacija — pouzdanija od hetero wrappera)

class GCNSloj(nn.Module):
    """
    Jedan sloj grafovske konvolucije — SA TEŽINAMA GRANA.

    Svaki čvor agregira poruke od svojih suseda, PONDERISANE težinom
    veze (edge_attr). Jača veza (npr. w=0.85) više utiče na rezultat
    od slabije veze (npr. w=0.10) — umesto da se sve tretiraju jednako.

    Ovo je ispravka bug-a: prethodna verzija je pravila prost
    (nesponderisan) prosek suseda i potpuno ignorisala edge_attr,
    zbog čega parametar α iz tonalnog grafa nije imao efekta na model.
    """
    def __init__(self, ulazna_dim: int, izlazna_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(ulazna_dim * 2, izlazna_dim)
        self.aktivacija = nn.ReLU()

    def forward(
        self,
        x_izvor: torch.Tensor,      # osobine izvornih čvorova (broj_izvora, dim)
        x_cilj: torch.Tensor,       # osobine ciljnih čvorova (broj_ciljeva, dim)
        edge_index: torch.Tensor,   # (2, broj_grana)
        edge_weight: torch.Tensor,  # (broj_grana, 1) — težina svake grane
    ) -> torch.Tensor:
        src, dst = edge_index[0], edge_index[1]

        # Poruke od izvornih čvorova, PONDERISANE težinom njihove grane.
        # edge_weight ima oblik (broj_grana, 1) pa se automatski
        # broadcast-uje preko svih dim kolona poruke.
        poruke = x_izvor[src] * edge_weight  # (broj_grana, dim)

        # Ponderisano sabiranje poruka po ciljnom čvoru
        agregat = torch.zeros_like(x_cilj)
        agregat.scatter_add_(0, dst.unsqueeze(1).expand_as(poruke), poruke)

        # Normalizacija sa SUMOM TEŽINA (ne brojem suseda!) — ovo je
        # ponderisani prosek: čvor sa jednim jakim susedom (w=0.9) treba
        # da bude drugačiji od čvora sa jednim slabim susedom (w=0.1).
        suma_tezina = torch.zeros(x_cilj.shape[0], 1, device=x_cilj.device)
        suma_tezina.scatter_add_(0, dst.unsqueeze(1), edge_weight)
        suma_tezina = suma_tezina.clamp(min=1e-8)  # sprečava deljenje nulom
        agregat = agregat / suma_tezina

        # Konkatenacija sa sopstvenom reprezentacijom i transformacija
        kombinovano = torch.cat([x_cilj, agregat], dim=-1)
        return self.aktivacija(self.linear(kombinovano))


# GNN ARHITEKTURA


class GNNPredvidjanjAkorda(nn.Module):
    """
    GNN model za predviđanje akorda.

    Tok podataka:
      melodija (16 nota) → Embedding → LSTM → melodijski_vektor (dim,)
      graf → GCN slojevi → repr_akorada (N_akorada, dim)
      melodijski_vektor @ repr_akorada.T → logiti (N_akorada,)
    """

    def __init__(
        self,
        broj_akorada: int,
        dim: int = GNN_HP["skrivene_dimenzije"],
        dropout: float = GNN_HP["dropout_stopa"],
    ) -> None:
        super().__init__()

        self.broj_akorada = broj_akorada
        self.dim = dim

        # 1. Melodijski enkoder
        self.ugradnja = nn.Embedding(13, dim, padding_idx=12)
        self.lstm = nn.LSTM(dim, dim, num_layers=2, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

        # 2. Graf enkoder — čvorovi akorada počinju kao naučivi vektori
        self.repr_akorada = nn.Parameter(torch.randn(broj_akorada, dim) * 0.1)

        # GCN slojevi: nota→akord i akord↔akord propagacija
        self.gcn_nota_akord = nn.ModuleList([
            GCNSloj(dim, dim) for _ in range(GNN_HP["broj_gcn_slojeva"])
        ])
        self.gcn_akord_akord = nn.ModuleList([
            GCNSloj(dim, dim) for _ in range(GNN_HP["broj_gcn_slojeva"])
        ])

        # Embedding za čvorove nota u grafu (fiksni one-hot → dim)
        self.proj_nota = nn.Linear(12, dim)

        # 3. Projekcija melodijskog vektora
        self.proj_mel = nn.Linear(dim, dim)

    def forward(
        self,
        melodija: torch.Tensor,        # (serija, 16)
        x_nota: torch.Tensor,          # (12, 12) one-hot
        edge_index_na: torch.Tensor,   # nota→akord grane
        edge_weight_na: torch.Tensor,  # nota→akord težine (α×teorija+β×statistika)
        edge_index_aa: torch.Tensor,   # akord↔akord grane
        edge_weight_aa: torch.Tensor,  # akord↔akord težine (kvintni krug)
    ) -> torch.Tensor:

        # ── Melodijski enkoder ────────────────────────────────────────────────
        ugr = self.ugradnja(melodija)             # (serija, 16, dim)
        lstm_izlaz, _ = self.lstm(ugr)
        mel_vektor = self.dropout(lstm_izlaz[:, -1, :])   # (serija, dim)
        mel_vektor = self.proj_mel(mel_vektor)    # (serija, dim)

        # ── Graf enkoder ──────────────────────────────────────────────────────
        h_nota = self.proj_nota(x_nota)           # (12, dim)
        h_akord = self.repr_akorada               # (N, dim)

        for gcn_na, gcn_aa in zip(self.gcn_nota_akord, self.gcn_akord_akord):
            # Poruke od nota ka akordima — PONDERISANE tonalnim težinama
            h_akord = gcn_na(h_nota, h_akord, edge_index_na, edge_weight_na)
            # Poruke između akorada (kvintni krug) — PONDERISANE
            h_akord = gcn_aa(h_akord, h_akord, edge_index_aa, edge_weight_aa)

        # ── Skorovanje: skalarni proizvod melodije i svakog akorda ───────────
        # mel_vektor: (serija, dim), h_akord: (N, dim)
        logiti = mel_vektor @ h_akord.T           # (serija, N)

        return logiti


# TRENIRANJE

def treniraj_gnn(
    trening, validacija, graf, indeksi_akorada, recnik_akorada,
    alfa=0.5, hp=GNN_HP, naziv="E3",
):
    uredjaj = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[INFO] Treniranje {naziv} | α={alfa:.2f} | {uredjaj}")

    mapa = napravi_mapu(recnik_akorada, indeksi_akorada)
    trening_loader = napravi_loader(trening, mapa, mesati=True)
    val_loader = napravi_loader(validacija, mapa, mesati=False)

    broj_akorada = len(indeksi_akorada)
    model = GNNPredvidjanjAkorda(broj_akorada).to(uredjaj)
    print(f"[INFO] Parametara: {sum(p.numel() for p in model.parameters()):,}")

    # Graf tenzori
    x_nota = graf["nota"].x.to(uredjaj)
    edge_na = graf["nota", "pripada", "akord"].edge_index.to(uredjaj)
    edge_na_w = graf["nota", "pripada", "akord"].edge_attr.to(uredjaj)
    edge_aa = graf["akord", "blizina", "akord"].edge_index.to(uredjaj)
    edge_aa_w = graf["akord", "blizina", "akord"].edge_attr.to(uredjaj)

    kriterijum = nn.CrossEntropyLoss()
    optimizator = optim.Adam(model.parameters(), lr=hp["stopa_ucenja"])
    rasporedivac = optim.lr_scheduler.ReduceLROnPlateau(optimizator, patience=3, factor=0.5)

    PUTANJA_MODELA.mkdir(parents=True, exist_ok=True)
    putanja_ckpt = PUTANJA_MODELA / f"gnn_{naziv}_alfa{alfa:.2f}.pt"
    najbolja = 0.0

    print(f"\n{'Epoha':>6}  {'Tr. Gubitak':>12}  {'Val Gubitak':>12}  {'Val Tačnost':>12}")
    print("─" * 52)

    for epoha in range(1, hp["broj_epoha"] + 1):
        # Treniranje
        model.train()
        tr_ukupno = 0.0
        for mel, cilj in trening_loader:
            mel, cilj = mel.to(uredjaj), cilj.to(uredjaj)
            optimizator.zero_grad()
            logiti = model(mel, x_nota, edge_na, edge_na_w, edge_aa, edge_aa_w)
            gubitak = kriterijum(logiti, cilj)
            gubitak.backward()
            nn.utils.clip_grad_norm_(model.parameters(), hp["klip_gradijenta"])
            optimizator.step()
            tr_ukupno += gubitak.item()

        tr_g = tr_ukupno / len(trening_loader)

        # Validacija
        model.eval()
        val_ukupno, tacno, ukupno = 0.0, 0, 0
        with torch.no_grad():
            for mel, cilj in val_loader:
                mel, cilj = mel.to(uredjaj), cilj.to(uredjaj)
                logiti = model(mel, x_nota, edge_na, edge_na_w, edge_aa, edge_aa_w)
                val_ukupno += kriterijum(logiti, cilj).item()
                tacno += (logiti.argmax(1) == cilj).sum().item()
                ukupno += cilj.size(0)

        val_g = val_ukupno / len(val_loader)
        val_t = tacno / ukupno
        rasporedivac.step(val_g)

        print(f"{epoha:>6}  {tr_g:>12.4f}  {val_g:>12.4f}  {val_t:>11.2%}")

        if val_t > najbolja:
            najbolja = val_t
            torch.save({"stanje": model.state_dict(), "alfa": alfa, "val_tacnost": val_t}, putanja_ckpt)
            print(f"  → Checkpoint sačuvan ({val_t:.2%})")

    print(f"\n[INFO] Završeno. Najbolja val. tačnost: {najbolja:.2%}")
    return model


@torch.no_grad()
def evaluiraj_gnn(model, test, graf, indeksi_akorada, recnik_akorada, hp=GNN_HP):
    uredjaj = next(model.parameters()).device
    mapa = napravi_mapu(recnik_akorada, indeksi_akorada)
    test_loader = napravi_loader(test, mapa, mesati=False)

    x_nota = graf["nota"].x.to(uredjaj)
    edge_na = graf["nota", "pripada", "akord"].edge_index.to(uredjaj)
    edge_na_w = graf["nota", "pripada", "akord"].edge_attr.to(uredjaj)
    edge_aa = graf["akord", "blizina", "akord"].edge_index.to(uredjaj)
    edge_aa_w = graf["akord", "blizina", "akord"].edge_attr.to(uredjaj)

    model.eval()
    tacno_top1, tacno_top3, ukupno = 0, 0, 0

    for mel, cilj in test_loader:
        mel, cilj = mel.to(uredjaj), cilj.to(uredjaj)
        logiti = model(mel, x_nota, edge_na, edge_na_w, edge_aa, edge_aa_w)
        tacno_top1 += (logiti.argmax(1) == cilj).sum().item()
        _, top3 = logiti.topk(k=min(3, logiti.size(1)), dim=1)
        tacno_top3 += (top3 == cilj.unsqueeze(1)).any(1).sum().item()
        ukupno += cilj.size(0)

    ca = tacno_top1 / ukupno
    top3 = tacno_top3 / ukupno

    print("\n" + "═" * 45)
    print("  REZULTATI NA TEST SKUPU")
    print("═" * 45)
    print(f"  Chord Accuracy : {ca:.2%}")
    print(f"  Top-3 Accuracy : {top3:.2%}")
    print(f"  Test uzoraka   : {ukupno}")
    print("═" * 45)

    return {"chord_accuracy": ca, "top3_accuracy": top3}



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

    ALFA = 0.5
    graf = torch.load(PUTANJA_GRAFA / f"tonalni_graf_alfa{ALFA:.2f}.pt", weights_only=False)
    with open(PUTANJA_GRAFA / "indeksi_akorada.json", "r") as f:
        indeksi_akorada = json.load(f)

    print(f"[INFO] Graf učitan | {len(indeksi_akorada)} akorada")

    model = treniraj_gnn(trening, validacija, graf, indeksi_akorada, recnik_akorada, alfa=ALFA)

    print("\n" + "═" * 45)
    print("  REZULTATI NA TEST SKUPU (E3 — GNN α=0.5)")
    print("═" * 45)
    evaluiraj_gnn(model, test, graf, indeksi_akorada, recnik_akorada)