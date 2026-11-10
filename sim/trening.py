#!/usr/bin/env python3
# =============================================================================
#  trening.py  -  uczenie quadrupeda chodzenia (RL, algorytm PPO)
#
#  Uzycie (z glownego folderu repo):
#      python sim/trening.py                 # od poczatku, poziom 0
#      python sim/trening.py --wznow         # kontynuuj od ostatniego zapisu
#      python sim/trening.py --srodowiska 4  # ile symulacji naraz
#
#  CURRICULUM: zaczyna od latwych komend (powoli do przodu). Gdy robot
#  dobrze sledzi komende i rzadko sie przewraca, przechodzi na nastepny
#  poziom z srodowisko.py (POZIOMY). Po kazdym poziomie zapisuje siec do
#  folderu modele/ (np. modele/poziom_0.zip) - obejrzysz ja w pokaz.py.
#
#  W terminalu co chwile widac:
#    - ile prob konczy sie upadkiem, sredni czas proby,
#    - TABELKE NAGROD: ile punktow na probe dal kazdy skladnik z nagrody.py.
#      To glowne narzedzie do strojenia nagrod - widac, ktora kara "zjada"
#      nagrode i co robot faktycznie optymalizuje.
#  Przerwac mozna Ctrl+C - postep sie zapisze.
#
#  CZAS: uczenie idzie na procesorze. Pierwsze sensowne kroki zwykle po
#  kilku-kilkunastu milionach krokow (godziny). Siec jest mala, karta
#  graficzna przy PPO prawie nic nie daje.
# =============================================================================

import argparse
import os
import signal
import subprocess
import sys
from collections import deque

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import srodowisko                                     # noqa: E402
from srodowisko import POZIOMY, Chodzenie            # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FOLDER = os.path.join(REPO, "modele")

KROKI_MAX = 50_000_000        # gorny limit krokow calego uczenia
OKNO = 200                    # z ilu ostatnich prob liczyc statystyki
CO_ILE_RAPORT = 2000          # co ile krokow (na srodowisko) wypisac postep

# warunki awansu na nastepny poziom komend
AWANS_SLEDZENIE = 0.75        # srednia wartosc "predkosc_xy" (0..1) na krok (stanie daje ok. 0,3)
AWANS_MAKS_UPADKI = 0.10      # najwyzej tyle prob moze konczyc sie upadkiem


class Postep(BaseCallback):
    """Wypisuje statystyki, tabelke nagrod i pilnuje curriculum."""

    def __init__(self, poziom=0):
        super().__init__()
        self.poziom = poziom
        self.upadki = deque(maxlen=OKNO)
        self.czasy = deque(maxlen=OKNO)
        self.skladniki = {}
        self.licznik = 0

    def _on_training_start(self):
        self.training_env.env_method("ustaw_poziom", self.poziom)

    def _on_step(self):
        for info in self.locals["infos"]:
            if "nagrody" in info:
                self.upadki.append(info["upadek"])
                self.czasy.append(info["czas"])
                for k, v in info["nagrody"].items():
                    self.skladniki.setdefault(k, deque(maxlen=OKNO)).append((v, info["czas"]))
        self.licznik += 1
        if self.licznik % 20000 == 0:
            zapisz(self.model, os.path.join(FOLDER, "ostatni.zip"))
        if self.licznik % CO_ILE_RAPORT == 0 and self.upadki:
            self._raport()
            self._curriculum()
        return True

    def _raport(self):
        upadki, czas = np.mean(self.upadki), np.mean(self.czasy)
        print(f"\n[{self.num_timesteps:>11,} krokow]  poziom {self.poziom}  "
              f"upadki: {upadki*100:4.0f}%  sredni czas proby: {czas:4.1f} s")
        print("   skladnik              pkt/probe    pkt/s")
        suma = 0.0
        for k, wartosci in sorted(self.skladniki.items(), key=lambda kv: -np.mean([v for v, _ in kv[1]])):
            na_probe = np.mean([v for v, _ in wartosci])
            na_s = np.sum([v for v, _ in wartosci]) / max(1e-9, np.sum([t for _, t in wartosci]))
            suma += na_probe
            print(f"   {k:<20s} {na_probe:10.1f} {na_s:8.2f}")
        print(f"   {'RAZEM':<20s} {suma:10.1f}")

    def _sledzenie(self):
        """Srednia wartosc skladnika predkosc_xy na krok (0..1)."""
        from nagrody import WAGI
        if "predkosc_xy" not in self.skladniki or not WAGI.get("predkosc_xy"):
            return 0.0
        w = self.skladniki["predkosc_xy"]
        na_s = np.sum([v for v, _ in w]) / max(1e-9, np.sum([t for _, t in w]))
        return na_s * srodowisko.DT_STEROWANIA / WAGI["predkosc_xy"]

    def _curriculum(self):
        if len(self.upadki) < OKNO:
            return
        sled = self._sledzenie()
        if sled >= AWANS_SLEDZENIE and np.mean(self.upadki) <= AWANS_MAKS_UPADKI:
            nazwa = os.path.join(FOLDER, f"poziom_{self.poziom}")
            self.model.save(nazwa)
            print(f"\n*** POZIOM {self.poziom} ZALICZONY (sledzenie {sled:.2f}) - zapisano {nazwa}.zip ***\n")
            if self.poziom + 1 < len(POZIOMY):
                self.poziom += 1
                self.training_env.env_method("ustaw_poziom", self.poziom)
                self.upadki.clear()
                self.czasy.clear()
                self.skladniki.clear()
                with open(os.path.join(FOLDER, "poziom.txt"), "w") as f:
                    f.write(str(self.poziom))


def zapisz(model, plik):
    """Zapis przez plik tymczasowy - przerwany zapis nie psuje poprzedniego."""
    model.save(plik + ".tmp.zip")
    os.replace(plik + ".tmp.zip", plik)


def stworz_srodowisko(i):
    def _f():
        return Chodzenie(seed=i)
    return _f


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--wznow", action="store_true", help="kontynuuj od ostatniego zapisu")
    p.add_argument("--srodowiska", type=int, default=max(2, (os.cpu_count() or 4) - 1),
                   help="ile symulacji naraz (domyslnie: liczba rdzeni - 1)")
    p.add_argument("--kroki", type=int, default=KROKI_MAX, help="ile krokow uczenia")
    args = p.parse_args()

    subprocess.run([sys.executable, os.path.join(REPO, "sim", "robot.py")], check=True,
                   stdout=subprocess.DEVNULL)               # opis robota -> model/quadruped.xml
    os.makedirs(FOLDER, exist_ok=True)
    env = VecMonitor(SubprocVecEnv([stworz_srodowisko(i) for i in range(args.srodowiska)]))

    ostatni = os.path.join(FOLDER, "ostatni.zip")
    poziom = 0
    if args.wznow and os.path.exists(ostatni):
        model = PPO.load(ostatni, env=env, device="cpu")
        plik_poziomu = os.path.join(FOLDER, "poziom.txt")
        if os.path.exists(plik_poziomu):
            poziom = int(open(plik_poziomu).read())
        print(f"Wznawiam od {ostatni}, poziom {poziom}")
    else:
        model = PPO("MlpPolicy", env, device="cpu", verbose=0,
                    n_steps=1024, batch_size=4096, n_epochs=5,
                    learning_rate=3e-4, gamma=0.99, gae_lambda=0.95,
                    ent_coef=0.003, clip_range=0.2,
                    policy_kwargs=dict(net_arch=[256, 256]))

    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    print(f"Uczenie: {args.srodowiska} symulacji naraz. Ctrl+C = przerwij i zapisz.")
    try:
        model.learn(total_timesteps=args.kroki, callback=Postep(poziom),
                    reset_num_timesteps=not args.wznow)
    except KeyboardInterrupt:
        print("\nPrzerwano.")
    finally:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        zapisz(model, ostatni)
        print(f"Zapisano {os.path.relpath(ostatni, REPO)}  (kontynuacja: python sim/trening.py --wznow)")
        env.close()
