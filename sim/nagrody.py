#!/usr/bin/env python3
# =============================================================================
#  nagrody.py  -  SYSTEM NAGROD I KAR (tu najczesciej sie grzebie!)
#
#  Siec uczy sie robic to, za co dostaje punkty. Kazdy "skladnik" ponizej
#  to jedna rzecz, ktora oceniamy co krok sterowania (50 razy na sekunde):
#
#      nagroda_w_kroku = SUMA( WAGA[skladnik] * wartosc_skladnika )
#
#  JAK ZARZADZAC
#    - wylaczyc skladnik:   waga = 0
#    - wzmocnic / oslabic:  zmien wage (dodatnia = nagroda, ujemna = kara)
#    - nowy skladnik:       dopisz funkcje na dole (def moj_skladnik(s): ...)
#                           i dodaj jej nazwe z waga do WAGI
#    - sprawdzic, co dziala: trening.py co chwile wypisuje tabelke - ile
#      punktow na probe dal kazdy skladnik. Jesli jedna kara "zjada"
#      wszystko (robot stoi w miejscu, bo ruch sie nie oplaca), zmniejsz ja.
#
#  ZASADA KCIUKA: nagroda za glowny cel (tu: jazda z zadana predkoscia)
#  powinna dawac wiecej punktow niz wszystkie kary razem, inaczej robot
#  nauczy sie "nic nie robic". Zmieniaj JEDNA wage naraz i patrz w tabelke.
# =============================================================================

import numpy as np

# =============================================================================
# 1. WAGI  (nazwa skladnika -> waga)
# =============================================================================
WAGI = {
    # --- NAGRODY: to, czego chcemy ---------------------------------------------
    "predkosc_xy":      2.0,     # idz z predkoscia z komendy (przod / bok)
    "obrot_z":          0.5,     # obracaj sie z predkoscia z komendy
    "czas_w_powietrzu": 1.0,     # stawiaj prawdziwe kroki (stopa w gorze ok. 0,25 s)
    "zycie":            0.5,     # punkt za kazdy krok bez upadku

    # --- KARY: to, czego nie chcemy ----------------------------------------------
    "predkosc_z":      -2.0,     # podskakiwanie korpusu
    "kolysanie":       -0.05,    # bujanie korpusem na boki / przod-tyl
    "pochylenie":      -2.0,     # korpus krzywo (nie poziomo)
    "wysokosc":       -20.0,     # korpus za nisko / za wysoko wzgledem pozy "stoi"
    "moment":          -0.001,   # zuzycie energii (sila silnikow)
    "zmiana_akcji":    -0.02,    # szarpanie - nagle zmiany polecen
    "limity_przegubow": -2.0,    # dojezdzanie do krancow zakresu przegubow
    "kolizje":         -1.0,     # cos innego niz stopa dotyka ziemi
    "poza":            -0.05,    # nogi daleko od pozy "stoi" (porzadek w ruchach)
    "slizganie_stop":  -0.1,     # stopa na ziemi, a jedzie po podlodze
    "upadek":         -10.0,     # jednorazowo, gdy proba konczy sie upadkiem
}

# =============================================================================
# 2. PARAMETRY SKLADNIKOW
# =============================================================================
SIGMA_PREDKOSCI = 0.02     # tolerancja sledzenia predkosci (mniej = surowiej);
                           # przy 0.02 stanie w miejscu przy komendzie 0,15 m/s daje tylko 0,32 pkt
CZAS_KROKU_CEL = 0.25      # [s] ile stopa ma byc w powietrzu przy kroku
LIMIT_PRZEGUBU = 0.9       # kara, gdy przegub przekroczy 90% swojego zakresu


# =============================================================================
# 3. SKLADNIKI
#    Kazda funkcja dostaje `s` - stan robota w tym kroku (patrz srodowisko.py,
#    funkcja _stan). Najwazniejsze pola:
#      s.v        predkosc korpusu [vx, vy, vz] w ukladzie robota [m/s]
#      s.w        predkosc katowa korpusu [wx, wy, wz] [rad/s]
#      s.g        "gdzie jest dol" w ukladzie robota (jak IMU); prosto = [0, 0, -1]
#      s.z        wysokosc korpusu [m],  s.z_cel  wysokosc w pozie "stoi"
#      s.komenda  [vx, vy, wz] - zadana predkosc
#      s.q, s.dq  katy i predkosci przegubow;  s.q_stoi  katy pozy "stoi"
#      s.q_min, s.q_max  zakresy przegubow
#      s.akcja, s.poprzednia_akcja   polecenia sieci (-1..1)
#      s.moment   momenty silnikow [Nm]
#      s.kontakt  [4 x True/False] czy stopa FL, FR, RL, RR dotyka ziemi
#      s.pierwszy_kontakt  stopa wlasnie postawiona (po locie)
#      s.czas_lotu         ile ta stopa byla w powietrzu [s]
#      s.v_stop   predkosc stop [4 x 3] w ukladzie swiata
#      s.kolizje  ile innych czesci niz stopy dotyka ziemi
#      s.upadek   True w kroku, w ktorym robot sie przewrocil
# =============================================================================
def predkosc_xy(s):
    blad = np.sum((s.komenda[:2] - s.v[:2]) ** 2)
    return np.exp(-blad / SIGMA_PREDKOSCI)            # 1 = idealnie, 0 = zle


def obrot_z(s):
    return np.exp(-(s.komenda[2] - s.w[2]) ** 2 / SIGMA_PREDKOSCI)


def czas_w_powietrzu(s):
    """Przy postawieniu stopy: + gdy lot trwal dluzej niz CZAS_KROKU_CEL,
    - gdy krocej (drobienie). Nie nagradza, gdy komenda to "stoj"."""
    if np.linalg.norm(s.komenda[:2]) < 0.05:
        return 0.0
    return float(np.sum((s.czas_lotu - CZAS_KROKU_CEL) * s.pierwszy_kontakt))


def zycie(s):
    return 1.0


def predkosc_z(s):
    return s.v[2] ** 2


def kolysanie(s):
    return np.sum(s.w[:2] ** 2)


def pochylenie(s):
    return np.sum(s.g[:2] ** 2)                       # 0 = poziomo


def wysokosc(s):
    return (s.z - s.z_cel) ** 2


def moment(s):
    return np.sum(s.moment ** 2)


def zmiana_akcji(s):
    return np.sum((s.akcja - s.poprzednia_akcja) ** 2)


def limity_przegubow(s):
    srodek = (s.q_max + s.q_min) / 2
    pol = (s.q_max - s.q_min) / 2 * LIMIT_PRZEGUBU
    za_daleko = np.abs(s.q - srodek) - pol
    return np.sum(np.clip(za_daleko, 0, None))


def kolizje(s):
    return float(s.kolizje)


def poza(s):
    return np.sum((s.q - s.q_stoi) ** 2)


def slizganie_stop(s):
    return float(np.sum(np.sum(s.v_stop[:, :2] ** 2, axis=1) * s.kontakt))


def upadek(s):
    return 1.0 if s.upadek else 0.0


# =============================================================================
# 4. LICZENIE (tego nie trzeba ruszac)
# =============================================================================
def policz(s):
    """Zwraca (suma, {skladnik: wazona_wartosc}) - pomija skladniki z waga 0."""
    czesci = {}
    for nazwa, waga in WAGI.items():
        if waga:
            czesci[nazwa] = waga * float(globals()[nazwa](s))
    return sum(czesci.values()), czesci


def sprawdz():
    brak = [n for n in WAGI if not callable(globals().get(n))]
    if brak:
        raise NameError(f"W WAGI sa skladniki bez funkcji: {brak}")


sprawdz()
