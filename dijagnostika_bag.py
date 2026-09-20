"""
dijagnostika_bag.py — Provera da li GNN stvarno koristi graf, ili se
efektivno ponaša isto kao LSTM (sto bi ukazivalo na skriveni bag).

Tri nezavisne provere:

  PROVERA 1 — Da li se SAMI GRAFOVI za razlicite alfe uopste razlikuju?
              (najbrza, najosnovnija provera — ako grafovi za alfa=0.0
              i alfa=1.0 imaju identicne edge_attr tezine, graf.py ima
              bag jos pre nego sto GNN uopste pocne da trenira)

  PROVERA 2 — Da li GNN modeli TRENIRANI na alfa=0.0 i alfa=1.0 (isti
              seed) nauce RAZLICITE reprezentacije akorada?
              (ako su repr_akorada tezine posle treniranja skoro
              identicne uprkos razlicitim grafovima, model ne uspeva
              da iskoristi razliku u grafu tokom treniranja)

  PROVERA 3 — Da li su GNN i LSTM PREDIKCIJE na istim test uzorcima
              sumnjivo identicne (znak da GNN efektivno radi isto sto
              i LSTM), ili se razlikuju (dva razlicita modela, koji
              slucajno postizu slicnu UKUPNU tacnost)?
"""

import json
import pickle

import torch
import torch.nn.functional as F

from config import PUTANJA_OBRADENIH, PUTANJA_GRAFA, PUTANJA_MODELA
from dataset import NEPOZNAT_AKORD

import baseline as bl
import gnn_model as gm


# Seed koji koristimo za provere 2 i 3 (bilo koji vec istreniran seed)
SEED_ZA_PROVERU = 23

# Dve alfe koje porede najvise (ekstremi: cisto statisticki vs cisto teorijski)
ALFA_A = 0.00
ALFA_B = 1.00



# PROVERA 1 — da li se grafovi razlikuju


def provera_1_grafovi_razlicito():
    print("\n" + "=" * 70)
    print("  PROVERA 1: Da li se grafovi za razlicite alfe stvarno razlikuju?")
    print("=" * 70)

    graf_a = torch.load(
        PUTANJA_GRAFA / f"tonalni_graf_alfa{ALFA_A:.2f}.pt",
        weights_only=False, map_location="cpu",
    )
    graf_b = torch.load(
        PUTANJA_GRAFA / f"tonalni_graf_alfa{ALFA_B:.2f}.pt",
        weights_only=False, map_location="cpu",
    )

    ea_a = graf_a["nota", "pripada", "akord"].edge_attr
    ea_b = graf_b["nota", "pripada", "akord"].edge_attr

    if ea_a.shape != ea_b.shape:
        print(f"  [UPOZORENJE] Grafovi imaju RAZLICIT broj grana "
              f"({ea_a.shape[0]} vs {ea_b.shape[0]}) — mogli su se graditi "
              f"nad razlicitim recnicima akorada. Poredim ipak prve zajednicke.")
        n = min(ea_a.shape[0], ea_b.shape[0])
        ea_a, ea_b = ea_a[:n], ea_b[:n]

    razlika = (ea_a - ea_b).abs()
    print(f"  Prosecna apsolutna razlika tezina grana : {razlika.mean().item():.6f}")
    print(f"  Maksimalna razlika                       : {razlika.max().item():.6f}")
    print(f"  Broj grana sa IDENTICNOM tezinom (<1e-6) : "
          f"{(razlika < 1e-6).sum().item()} / {razlika.numel()}")

    if razlika.mean().item() < 1e-6:
        print("\n  [CRVENA ZASTAVICA] Grafovi za alfa=0.00 i alfa=1.00 su "
              "PRAKTICNO IDENTICNI — graf.py verovatno ima bag, alfa se ne "
              "primenjuje ispravno pri gradnji grafa.")
        return False
    else:
        print("\n  [OK] Grafovi se stvarno razlikuju — graf.py ispravno "
              "primenjuje alfa pri gradnji tezina grana.")
        return True



# PROVERA 2 — da li GNN naucene reprezentacije akorada zavise od alfe


def provera_2_naucene_reprezentacije():
    print("\n" + "=" * 70)
    print("  PROVERA 2: Da li GNN modeli (alfa=0.00 vs alfa=1.00) nauce")
    print("             RAZLICITE reprezentacije akorada?")
    print("=" * 70)

    with open(PUTANJA_GRAFA / "indeksi_akorada.json", "r", encoding="utf-8") as f:
        indeksi_akorada = json.load(f)
    broj_akorada = len(indeksi_akorada)

    putanja_a = PUTANJA_MODELA / f"gnn_MULTI_alfa{ALFA_A:.2f}_seed{SEED_ZA_PROVERU}_alfa{ALFA_A:.2f}.pt"
    putanja_b = PUTANJA_MODELA / f"gnn_MULTI_alfa{ALFA_B:.2f}_seed{SEED_ZA_PROVERU}_alfa{ALFA_B:.2f}.pt"

    if not putanja_a.exists() or not putanja_b.exists():
        print(f"  [PRESKACEM] Nedostaje checkpoint ({putanja_a.name} ili {putanja_b.name})")
        return None

    ck_a = torch.load(putanja_a, weights_only=False, map_location="cpu")
    ck_b = torch.load(putanja_b, weights_only=False, map_location="cpu")

    model_a = gm.GNNPredvidjanjAkorda(broj_akorada)
    model_b = gm.GNNPredvidjanjAkorda(broj_akorada)
    model_a.load_state_dict(ck_a["stanje"])
    model_b.load_state_dict(ck_b["stanje"])

    repr_a = model_a.repr_akorada.detach()  # (broj_akorada, dim) — POCETNE, naucive reprezentacije
    repr_b = model_b.repr_akorada.detach()

    # Kosinusna slicnost po akordu (1.0 = identicno, 0.0 = ortogonalno)
    kosinus = F.cosine_similarity(repr_a, repr_b, dim=1)
    print(f"  Prosecna kosinusna slicnost repr_akorada : {kosinus.mean().item():.4f}")
    print(f"  Min / Max kosinusna slicnost              : "
          f"{kosinus.min().item():.4f} / {kosinus.max().item():.4f}")

    # Poredimo i sa slicnoscu dva NEZAVISNO, NASUMICNO inicijalizovana
    # modela (bez treniranja) — to nam je "kontrolna" vrednost koliko bi
    # slicnost trebalo da bude AKO modeli nemaju nikakve veze jedan s drugim.
    torch.manual_seed(0)
    kontrolni_a = torch.randn(broj_akorada, model_a.dim) * 0.1
    torch.manual_seed(1)
    kontrolni_b = torch.randn(broj_akorada, model_a.dim) * 0.1
    kosinus_kontrolni = F.cosine_similarity(kontrolni_a, kontrolni_b, dim=1)
    print(f"  (Kontrola: slicnost DVA NASUMICNA vektora : "
          f"{kosinus_kontrolni.mean().item():.4f})")

    if kosinus.mean().item() > 0.95:
        print("\n  [CRVENA ZASTAVICA] Reprezentacije su GOTOVO IDENTICNE "
              "uprkos treniranju na potpuno razlicitim grafovima (alfa=0 "
              "vs alfa=1) — model verovatno ne koristi graf efektivno.")
        return False
    else:
        print("\n  [OK] Reprezentacije se primetno razlikuju izmedju "
              "alfa=0.00 i alfa=1.00 treniranja — graf ima merljiv uticaj "
              "na ono sto model nauci.")
        return True



# PROVERA 3 — da li su GNN i LSTM predikcije sumnjivo identicne


def provera_3_predikcije_poredjenje():
    print("\n" + "=" * 70)
    print("  PROVERA 3: Da li su GNN i LSTM predikcije sumnjivo identicne?")
    print("=" * 70)

    with open(PUTANJA_OBRADENIH / "test.pkl", "rb") as f:
        test = pickle.load(f)
    with open(PUTANJA_OBRADENIH / "recnik_akorada.json", "r", encoding="utf-8") as f:
        recnik_akorada = json.load(f)
    with open(PUTANJA_GRAFA / "indeksi_akorada.json", "r", encoding="utf-8") as f:
        indeksi_akorada = json.load(f)

    inv_recnik = {v: k for k, v in recnik_akorada.items()}
    inv_indeksi = {v: k for k, v in indeksi_akorada.items()}

    # --- LSTM predikcije ---
    putanja_lstm = PUTANJA_MODELA / f"lstm_MULTI_seed{SEED_ZA_PROVERU}.pt"
    if not putanja_lstm.exists():
        print(f"  [PRESKACEM] Nedostaje LSTM checkpoint: {putanja_lstm.name}")
        return None

    ck_lstm = torch.load(putanja_lstm, weights_only=False, map_location="cpu")
    model_lstm = bl.LSTMPredvidjanjAkorda(broj_akorda=len(recnik_akorada))
    model_lstm.load_state_dict(ck_lstm["stanje_modela"])
    model_lstm.eval()

    loader_lstm = bl.napravi_loader(
        test, bl.PODRAZUMEVANI_HP["velicina_prozora"],
        bl.PODRAZUMEVANI_HP["velicina_serije"], mesati=False,
    )

    lstm_predikcije = []
    lstm_tacni = []
    with torch.no_grad():
        for ulaz, cilj in loader_lstm:
            logiti, _ = model_lstm(ulaz)
            pred = logiti.argmax(dim=1)
            for t, p in zip(cilj.tolist(), pred.tolist()):
                naziv_t = inv_recnik.get(t, NEPOZNAT_AKORD)
                naziv_p = inv_recnik.get(p, NEPOZNAT_AKORD)
                if naziv_t != NEPOZNAT_AKORD:
                    lstm_tacni.append(naziv_t)
                    lstm_predikcije.append(naziv_p)

    # --- GNN predikcije (koristimo alfa=0.50 kao reprezentativnu sredinu) ---
    alfa_gnn = 0.50
    putanja_gnn = PUTANJA_MODELA / f"gnn_MULTI_alfa{alfa_gnn:.2f}_seed{SEED_ZA_PROVERU}_alfa{alfa_gnn:.2f}.pt"
    if not putanja_gnn.exists():
        print(f"  [PRESKACEM] Nedostaje GNN checkpoint: {putanja_gnn.name}")
        return None

    graf = torch.load(
        PUTANJA_GRAFA / f"tonalni_graf_alfa{alfa_gnn:.2f}.pt",
        weights_only=False, map_location="cpu",
    )
    x_nota = graf["nota"].x
    edge_na = graf["nota", "pripada", "akord"].edge_index
    edge_na_w = graf["nota", "pripada", "akord"].edge_attr
    edge_aa = graf["akord", "blizina", "akord"].edge_index
    edge_aa_w = graf["akord", "blizina", "akord"].edge_attr

    ck_gnn = torch.load(putanja_gnn, weights_only=False, map_location="cpu")
    model_gnn = gm.GNNPredvidjanjAkorda(len(indeksi_akorada))
    model_gnn.load_state_dict(ck_gnn["stanje"])
    model_gnn.eval()

    mapa = gm.napravi_mapu(recnik_akorada, indeksi_akorada)
    loader_gnn = gm.napravi_loader(test, mapa, mesati=False)

    gnn_predikcije = []
    gnn_tacni = []
    with torch.no_grad():
        for mel, cilj in loader_gnn:
            logiti = model_gnn(mel, x_nota, edge_na, edge_na_w, edge_aa, edge_aa_w)
            pred = logiti.argmax(dim=1)
            for t, p in zip(cilj.tolist(), pred.tolist()):
                gnn_tacni.append(inv_indeksi.get(t, NEPOZNAT_AKORD))
                gnn_predikcije.append(inv_indeksi.get(p, NEPOZNAT_AKORD))

    # --- Poredjenje ---
    # NAPOMENA: LSTM i GNN loaderi mogu imati blago razlicit broj uzoraka
    # (LSTM preskace nepoznate ciljeve unutar loader-a; GNN preskace tokom
    # mapiranja originalnih indeksa u graf indekse) — poredimo prvih N
    # zajednickih uzoraka po redosledu.
    n = min(len(lstm_predikcije), len(gnn_predikcije))
    print(f"  LSTM uzoraka: {len(lstm_predikcije)}  |  GNN uzoraka: {len(gnn_predikcije)}  "
          f"|  Poredim prvih: {n}")

    isti_akord_predvidjen = sum(
        1 for i in range(n) if lstm_predikcije[i] == gnn_predikcije[i]
    )
    procenat_isti = isti_akord_predvidjen / n * 100

    print(f"\n  Model se SLAZE (isti predvidjen akord) u: {procenat_isti:.1f}% uzoraka")

    # Kontrola: kolika bi slaganja bila da su predikcije potpuno nezavisne
    # i nasumicne (na osnovu broja jedinstvenih akorada u predikcijama)?
    jedinstveni_akordi = len(set(lstm_predikcije[:n]) | set(gnn_predikcije[:n]))
    ocekivano_nasumicno = 100 / jedinstveni_akordi if jedinstveni_akordi else 0
    print(f"  (Kontrola: ocekivano slaganje da su predikcije POTPUNO "
          f"nasumicne i nezavisne: ~{ocekivano_nasumicno:.1f}%)")

    if procenat_isti > 90:
        print("\n  [CRVENA ZASTAVICA] GNN i LSTM se slazu u preko 90% predikcija "
              "— sumnjivo visoko, moguce da GNN efektivno kopira LSTM ponasanje "
              "ili da oba modela kolapsiraju na isti, dominantan akord.")
    elif procenat_isti < ocekivano_nasumicno * 1.5:
        print("\n  [NEOCEKIVANO] Modeli se slazu jedva vise nego cisto nasumicno "
              "— ovo bi znacilo da bar jedan model nije nista naucio.")
    else:
        print("\n  [OK] Modeli se slazu znacajno vise nego nasumicno (oba su "
              "naucila nesto smisleno), ali NE toliko da izgleda kao da je "
              "jedan model kopija drugog — konzistentno sa dva RAZLICITA "
              "modela koja nezavisno postizu slicnu ukupnu tacnost.")

    # Bonus: da li oba modela FAVORIZUJU isti, dominantan akord?
    from collections import Counter
    naj_lstm = Counter(lstm_predikcije[:n]).most_common(3)
    naj_gnn = Counter(gnn_predikcije[:n]).most_common(3)
    print(f"\n  Najcesce predvidjani akordi — LSTM: {naj_lstm}")
    print(f"  Najcesce predvidjani akordi — GNN : {naj_gnn}")



# POKRETANJE


if __name__ == "__main__":
    rez1 = provera_1_grafovi_razlicito()
    rez2 = provera_2_naucene_reprezentacije()
    provera_3_predikcije_poredjenje()

    print("\n" + "=" * 70)
    print("  ZAKLJUCAK")
    print("=" * 70)
    if rez1 is False or rez2 is False:
        print("  Postoje CRVENE ZASTAVICE — verovatno POSTOJI bag koji sprecava")
        print("  GNN da stvarno iskoristi tonalni graf. Treba dalje istraziti")
        print("  gnn_model.py (GCN agregacija) i graf.py (gradnja tezina).")
    else:
        print("  Nema crvenih zastavica — graf se ispravno gradi RAZLICITO po")
        print("  alfi, i model uci RAZLICITE reprezentacije u zavisnosti od")
        print("  alfe. Slican ukupan Chord Accuracy izmedju GNN i LSTM je")
        print("  verovatno STVARAN nalaz (oba pristupa dostizu slican plafon")
        print("  na ovom zadatku), NE simptom bag-a.")
    print("=" * 70)