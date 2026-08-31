import json, pickle, torch
from gnn_model import treniraj_gnn, evaluiraj_gnn
from config import PUTANJA_OBRADENIH, PUTANJA_GRAFA

with open(PUTANJA_OBRADENIH / "trening.pkl", "rb") as f:
    trening = pickle.load(f)
with open(PUTANJA_OBRADENIH / "validacija.pkl", "rb") as f:
    validacija = pickle.load(f)
with open(PUTANJA_OBRADENIH / "test.pkl", "rb") as f:
    test = pickle.load(f)
with open(PUTANJA_OBRADENIH / "recnik_akorada.json") as f:
    recnik = json.load(f)
with open(PUTANJA_GRAFA / "indeksi_akorada.json") as f:
    indeksi = json.load(f)

graf = torch.load(PUTANJA_GRAFA / "tonalni_graf_alfa0.30.pt", weights_only=False)

model = treniraj_gnn(trening, validacija, graf, indeksi, recnik, alfa=0.3, naziv="FINAL")
evaluiraj_gnn(model, test, graf, indeksi, recnik)