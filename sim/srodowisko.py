#!/usr/bin/env python3
# =============================================================================
#  srodowisko.py  -  zadanie "chodz z zadana predkoscia" do uczenia (RL)
#
#  Tego pliku nie uruchamia sie bezposrednio - korzystaja z niego
#  trening.py (uczenie) i pokaz.py (ogladanie nauczonego robota).
#
#  JAK TO DZIALA
#    Co 20 ms (50 razy na sekunde) "mozg" robota (siec neuronowa):
#      1. dostaje OBSERWACJE - to, co robot czuje: IMU (gdzie jest dol,
#         obroty), predkosc korpusu, katy i predkosci 12 przegubow,
#         poprzednia akcje i KOMENDE (z jaka predkoscia ma isc),
#      2. wybiera AKCJE - 12 liczb -1..1 = przesuniecie kazdego przegubu
#         od pozy "stoi" (akcja 0 = robot stoi),
#      3. dostaje NAGRODE - punkty z nagrody.py.
#    Proba (epizod) konczy sie upadkiem albo po CZAS_PROBY sekund.
#
#  Co tu ustawiasz: czas proby, sile akcji, zakresy komend (poziomy
#  trudnosci), warunki upadku i "losowanie swiata" (tarcie, masa,
#  popychanie) - zeby siec nie nauczyla sie jednej idealnej symulacji.
#  NAGRODY I KARY sa w osobnym pliku: nagrody.py.
# =============================================================================

import math
import os
from types import SimpleNamespace

import gymnasium as gym
import mujoco
import numpy as np

import nagrody
from kinematyka import NOGI, stawy

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLIK_MODELU = os.path.join(REPO, "model", "quadruped.xml")    # generowany przez robot.py

DT_STEROWANIA = 0.02       # [s] co ile siec podejmuje decyzje (50 Hz)
CZAS_PROBY = 10.0          # [s] maks. dlugosc jednej proby
SKALA_AKCJI = 0.4          # [rad] akcja 1.0 = przegub obrocony o tyle od pozy "stoi"
WYGLADZANIE = 1.0          # 0..1; mniej = plynniej (polecenie dochodzi stopniowo)

# --- KOMENDY: z jaka predkoscia robot ma isc --------------------------------------
# Kolejne POZIOMY trudnosci (curriculum). trening.py przechodzi na nastepny,
# gdy robot dobrze sobie radzi. Zakresy: (min, max), losowane w kazdej probie.
#   vx = przod/tyl [m/s], vy = bokiem [m/s], wz = obrot [rad/s]
POZIOMY = [
    dict(vx=(0.10, 0.25), vy=(0.0, 0.0),   wz=(0.0, 0.0)),
    dict(vx=(-0.10, 0.25), vy=(-0.05, 0.05), wz=(-0.3, 0.3)),
    dict(vx=(-0.20, 0.40), vy=(-0.10, 0.10), wz=(-0.6, 0.6)),
]
SZANSA_STOJ = 0.1          # w tylu procentach prob komenda = stoj w miejscu
ZMIANA_KOMENDY = 5.0       # [s] co ile losowac nowa komende w trakcie proby

# --- UPADEK (koniec proby) ---------------------------------------------------------
MAKS_PRZECHYL = math.radians(45)
MIN_WYSOKOSC = 0.08        # [m] korpus nizej = upadek

# --- LOSOWANIE SWIATA (domain randomization) ----------------------------------------
# Kazda proba ma troche inny swiat - siec uczy sie chodu, ktory dziala tez
# na prawdziwym robocie. Ustaw None, zeby wylaczyc dana rzecz.
LOSOWANIE = dict(
    tarcie=(0.5, 1.25),          # tarcie stop o podloze
    dodatkowa_masa=(-0.2, 0.5),  # [kg] dolozona do korpusu (np. bateria, elektronika)
    popchniecie=(4.0, 0.3),      # (co ile sekund, najwieksza predkosc kopniecia [m/s])
    szum_obserwacji=0.01,        # szum "czujnikow"
)


class Chodzenie(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, poziom=0, losowanie=True, seed=None):
        m = self.model = mujoco.MjModel.from_xml_path(PLIK_MODELU)
        self.data = mujoco.MjData(m)
        self.n_krokow = round(DT_STEROWANIA / m.opt.timestep)
        self.losowanie = losowanie
        self._seed = seed
        self.poziom = poziom

        self.korpus = m.body("base_link").id
        self.podloga = m.geom("podloga").id
        self.stopy_geom = np.array([m.geom(f"stopa_{n}").id for n in NOGI])
        self.stopy_site = [m.site(f"stopa_{n}").id for n in NOGI]
        self.korpus_geomy = {i for i in range(m.ngeom) if m.geom_bodyid[i] == self.korpus}

        nazwy = [s for n in NOGI for s in stawy(n)]
        self.akt = np.array([m.actuator(s).id for s in nazwy])
        self.q_adr = np.array([m.joint(s).qposadr[0] for s in nazwy])
        self.v_adr = np.array([m.joint(s).dofadr[0] for s in nazwy])
        self.q_min, self.q_max = m.jnt_range[[m.joint(s).id for s in nazwy]].T

        self.key = m.key("stoi").id
        self.q_stoi = m.key_qpos[self.key][self.q_adr].copy()
        self.z_cel = m.key_qpos[self.key][2] - 0.005
        self.masa_korpusu = m.body_mass[self.korpus]
        self.tarcie_stop = m.geom_friction[self.stopy_geom].copy()

        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(12,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(48,), dtype=np.float32)

    # --- curriculum ------------------------------------------------------------------
    def ustaw_poziom(self, poziom):
        self.poziom = min(int(poziom), len(POZIOMY) - 1)

    def _losuj_komende(self):
        if self.np_random.random() < SZANSA_STOJ:
            return np.zeros(3)
        z = POZIOMY[self.poziom]
        return np.array([self.np_random.uniform(*z[k]) for k in ("vx", "vy", "wz")])

    # --- obserwacja i stan ---------------------------------------------------------------
    def _R(self):
        return self.data.xmat[self.korpus].reshape(3, 3)

    def _obserwacja(self):
        d, R = self.data, self._R()
        obs = np.concatenate([
            R.T @ np.array([0, 0, -1.0]),            # gdzie jest dol (IMU)
            d.qvel[3:6] * 0.25,                      # obroty korpusu
            R.T @ d.qvel[0:3],                       # predkosc korpusu
            d.qpos[self.q_adr] - self.q_stoi,        # katy przegubow (wzgl. pozy stoi)
            d.qvel[self.v_adr] * 0.05,               # predkosci przegubow
            self.poprzednia_akcja,
            self.komenda,
        ])
        if self.losowanie and LOSOWANIE.get("szum_obserwacji"):
            obs += self.np_random.normal(0, LOSOWANIE["szum_obserwacji"], obs.shape)
        return obs.astype(np.float32)

    def _stan(self, akcja, upadek, kolizje):
        d, m, R = self.data, self.model, self._R()
        v_stop = np.zeros((4, 3))
        tmp = np.zeros(6)
        for i, sid in enumerate(self.stopy_site):
            mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_SITE, sid, tmp, 0)
            v_stop[i] = tmp[3:]
        return SimpleNamespace(
            v=R.T @ d.qvel[0:3], w=d.qvel[3:6].copy(), g=R.T @ np.array([0, 0, -1.0]),
            z=d.qpos[2], z_cel=self.z_cel, komenda=self.komenda,
            q=d.qpos[self.q_adr], dq=d.qvel[self.v_adr], q_stoi=self.q_stoi,
            q_min=self.q_min, q_max=self.q_max,
            akcja=akcja, poprzednia_akcja=self.poprzednia_akcja,
            moment=d.actuator_force[self.akt],
            kontakt=self.kontakt, pierwszy_kontakt=self.pierwszy_kontakt,
            czas_lotu=self.czas_lotu_przy_kontakcie, v_stop=v_stop,
            kolizje=kolizje, upadek=upadek,
        )

    def _kontakty(self):
        """Ktore stopy dotykaja ziemi, ile innych czesci dotyka, czy korpus dotyka."""
        d = self.data
        stopy = np.zeros(4, dtype=bool)
        inne, korpus = 0, False
        for c in d.contact[:d.ncon]:
            if self.podloga not in (c.geom1, c.geom2):
                continue
            g = c.geom2 if c.geom1 == self.podloga else c.geom1
            idx = np.flatnonzero(self.stopy_geom == g)
            if idx.size:
                stopy[idx[0]] = True
            else:
                inne += 1
                korpus |= g in self.korpus_geomy
        return stopy, inne, korpus

    # --- API Gymnasium --------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        if seed is None and self._seed is not None:
            seed, self._seed = self._seed, None
        super().reset(seed=seed)
        m, d = self.model, self.data
        mujoco.mj_resetDataKeyframe(m, d, self.key)

        if self.losowanie:
            L, r = LOSOWANIE, self.np_random
            if L.get("tarcie"):
                m.geom_friction[self.stopy_geom, 0] = r.uniform(*L["tarcie"])
            if L.get("dodatkowa_masa"):
                m.body_mass[self.korpus] = self.masa_korpusu + r.uniform(*L["dodatkowa_masa"])
            d.qpos[self.q_adr] += r.uniform(-0.05, 0.05, 12)      # troche inna poza startowa
            d.qpos[3:7] = [1, r.uniform(-0.02, 0.02), r.uniform(-0.02, 0.02), r.uniform(-0.1, 0.1)]
            d.qpos[3:7] /= np.linalg.norm(d.qpos[3:7])
        mujoco.mj_forward(m, d)

        self.komenda = self._losuj_komende()
        self.poprzednia_akcja = np.zeros(12)
        self.ctrl_cel = d.ctrl[self.akt].copy()
        self.kontakt = np.ones(4, dtype=bool)
        self.pierwszy_kontakt = np.zeros(4, dtype=bool)
        self.czas_lotu = np.zeros(4)
        self.czas_lotu_przy_kontakcie = np.zeros(4)
        self.sumy = {k: 0.0 for k, w in nagrody.WAGI.items() if w}
        self.t = 0.0
        self.krok = 0
        self.nastepne_popchniecie = self._kiedy_popchniecie()
        return self._obserwacja(), {}

    def _kiedy_popchniecie(self):
        p = LOSOWANIE.get("popchniecie") if self.losowanie else None
        return self.np_random.uniform(0.5, 1.5) * p[0] if p else math.inf

    def step(self, akcja):
        akcja = np.clip(np.asarray(akcja, dtype=np.float64), -1, 1)
        m, d = self.model, self.data

        cel = np.clip(self.q_stoi + SKALA_AKCJI * akcja, self.q_min, self.q_max)
        self.ctrl_cel += WYGLADZANIE * (cel - self.ctrl_cel)
        d.ctrl[self.akt] = self.ctrl_cel
        for _ in range(self.n_krokow):
            mujoco.mj_step(m, d)
        self.t += DT_STEROWANIA
        self.krok += 1

        # popychanie i zmiana komendy w trakcie proby
        if self.t >= self.nastepne_popchniecie:
            v = LOSOWANIE["popchniecie"][1]
            d.qvel[0:2] += self.np_random.uniform(-v, v, 2)
            self.nastepne_popchniecie = self.t + self._kiedy_popchniecie()
        if ZMIANA_KOMENDY and self.krok % round(ZMIANA_KOMENDY / DT_STEROWANIA) == 0:
            self.komenda = self._losuj_komende()

        # kontakty stop i czas lotu (do skladnika "czas_w_powietrzu")
        stopy, inne, korpus_na_ziemi = self._kontakty()
        self.pierwszy_kontakt = stopy & ~self.kontakt
        self.czas_lotu += DT_STEROWANIA
        self.czas_lotu_przy_kontakcie = self.czas_lotu * self.pierwszy_kontakt
        self.czas_lotu[stopy] = 0.0
        self.kontakt = stopy

        # upadek
        g_z = self._R()[2, 2]
        upadek = bool(g_z < math.cos(MAKS_PRZECHYL) or korpus_na_ziemi
                      or d.qpos[2] < MIN_WYSOKOSC or not np.isfinite(d.qpos).all())

        nagroda, czesci = nagrody.policz(self._stan(akcja, upadek, inne))
        for k, v in czesci.items():
            self.sumy[k] += v
        self.poprzednia_akcja = akcja

        limit = self.t >= CZAS_PROBY - 1e-9
        info = {}
        if upadek or limit:
            info = {"nagrody": dict(self.sumy), "upadek": upadek, "czas": self.t,
                    "poziom": self.poziom}
        return self._obserwacja(), float(nagroda), upadek, bool(limit and not upadek), info
