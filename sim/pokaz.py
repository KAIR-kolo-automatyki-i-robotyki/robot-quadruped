#!/usr/bin/env python3
# =============================================================================
#  pokaz.py  -  ogladanie quadrupeda w viewerze (z siecia albo bez)
#
#  Uzycie (z glownego folderu repo):
#      python sim/pokaz.py                          # ostatni zapis z modele/
#      python sim/pokaz.py modele/poziom_0.zip      # wybrana siec
#      python sim/pokaz.py --bez-sieci              # sama poza "stoi" (test modelu)
#      python sim/pokaz.py --vx 0.2                 # stala komenda: 0,2 m/s do przodu
#      python sim/pokaz.py --pewnie                 # siec bez losowosci
#
#  KLAWISZE (kliknij najpierw w okno):
#      strzalki gora / dol     komenda predkosci do przodu +/- 0,05 m/s
#      strzalki lewo / prawo   komenda obrotu +/- 0,2 rad/s
#      Enter                   komenda = stoj
#      Ctrl + P                zrzut ekranu -> folder zrzuty/ (zrzut_001.png, 002...)
#
#  Po kazdej probie terminal pokazuje, ile punktow dal kazdy skladnik
#  nagrody (jak w trening.py). Zamknij okno, zeby skonczyc.
# =============================================================================

import argparse
import os
import subprocess
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import srodowisko                                     # noqa: E402
from srodowisko import DT_STEROWANIA, Chodzenie      # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GORA, DOL, LEWO, PRAWO, ENTER = 265, 264, 263, 262, 257

p = argparse.ArgumentParser()
p.add_argument("plik", nargs="?", default=os.path.join(REPO, "modele", "ostatni.zip"))
p.add_argument("--bez-sieci", action="store_true", help="akcja 0 = robot tylko stoi")
p.add_argument("--pewnie", action="store_true", help="siec bez losowosci")
p.add_argument("--losowanie", action="store_true", help="wlacz losowanie swiata (popychanie itp.)")
p.add_argument("--vx", type=float, default=None)
p.add_argument("--vy", type=float, default=0.0)
p.add_argument("--wz", type=float, default=0.0)
p.add_argument("--poziom", type=int, default=len(srodowisko.POZIOMY) - 1)
args = p.parse_args()

subprocess.run([sys.executable, os.path.join(REPO, "sim", "robot.py")], check=True,
               stdout=subprocess.DEVNULL)

siec = None
if not args.bez_sieci:
    from stable_baselines3 import PPO
    siec = PPO.load(args.plik, device="cpu")
    print(f"Siec: {os.path.relpath(args.plik, REPO)}")

env = Chodzenie(poziom=args.poziom, losowanie=args.losowanie)
komenda = None if args.vx is None else np.array([args.vx, args.vy, args.wz])
if komenda is not None:
    srodowisko.ZMIANA_KOMENDY = 0                    # nie losuj komendy w trakcie proby


def klawisz(kod):
    global komenda
    k = env.komenda.copy() if komenda is None else komenda
    if kod == GORA:    k[0] += 0.05
    elif kod == DOL:   k[0] -= 0.05
    elif kod == LEWO:  k[2] += 0.2
    elif kod == PRAWO: k[2] -= 0.2
    elif kod == ENTER: k[:] = 0
    else:
        return
    komenda = k
    srodowisko.ZMIANA_KOMENDY = 0
    print(f"komenda: vx {k[0]:+.2f} m/s, vy {k[1]:+.2f} m/s, obrot {k[2]:+.1f} rad/s")


# --- zrzuty ekranu ---------------------------------------------------------------
# Viewer zawsze zapisuje zrzut jako screenshot.png (w folderze, z ktorego
# uruchomiono program) i nadpisuje poprzedni. Petla co chwile sprawdza, czy
# taki plik sie pojawil, i przenosi go do zrzuty/ pod kolejnym numerem.
FOLDER_ZRZUTOW = "zrzuty"


def pilnuj_zrzutow():
    plik = "screenshot.png"
    if not os.path.exists(plik) or time.time() - os.path.getmtime(plik) < 0.3:
        return                                       # brak albo jeszcze sie zapisuje
    os.makedirs(FOLDER_ZRZUTOW, exist_ok=True)
    nr = 1
    while os.path.exists(os.path.join(FOLDER_ZRZUTOW, f"zrzut_{nr:03d}.png")):
        nr += 1
    cel = os.path.join(FOLDER_ZRZUTOW, f"zrzut_{nr:03d}.png")
    os.replace(plik, cel)
    print(f"Zrzut ekranu zapisany: {cel}")


obs, _ = env.reset()
krok = 0
with mujoco.viewer.launch_passive(env.model, env.data, key_callback=klawisz) as viewer:
    with viewer.lock():
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = env.korpus
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 1.2, 135, -20
    proba, nagroda = 1, 0.0
    while viewer.is_running():
        t0 = time.time()
        if komenda is not None:
            env.komenda = komenda
        if siec is None:
            akcja = np.zeros(12)
        else:
            akcja, _ = siec.predict(obs, deterministic=args.pewnie)
        with viewer.lock():
            obs, r, koniec, limit, info = env.step(akcja)
        nagroda += r
        if hasattr(viewer, "set_texts"):
            k = env.komenda
            viewer.set_texts([(None, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
                               f"komenda vx {k[0]:+.2f}  vy {k[1]:+.2f}  obrot {k[2]:+.1f}\n"
                               f"predkosc robota {(env._R().T @ env.data.qvel[0:3])[0]:+.2f} m/s",
                               None)])
        viewer.sync()
        krok += 1
        if krok % 10 == 0:                           # ok. 5 razy na sekunde
            pilnuj_zrzutow()
        if koniec or limit:
            print(f"\nproba {proba}: {'UPADEK' if info['upadek'] else 'koniec czasu'} "
                  f"po {info['czas']:.1f} s, nagroda {nagroda:.0f}")
            for nazwa, v in sorted(info["nagrody"].items(), key=lambda kv: -kv[1]):
                print(f"   {nazwa:<20s} {v:8.1f}")
            time.sleep(1.0)
            with viewer.lock():
                obs, _ = env.reset()
            proba, nagroda = proba + 1, 0.0
        time.sleep(max(0.0, DT_STEROWANIA - (time.time() - t0)))
