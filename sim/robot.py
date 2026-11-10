#!/usr/bin/env python3
# =============================================================================
#  robot.py  -  OPIS SYMULACJI QUADRUPEDA (URDF z CAD-a + nasze ustawienia)
#
#  Uruchom (z glownego folderu repo):
#      python sim/robot.py        -> powstaje model/quadruped.xml + tabelka
#                                    + szybki test, czy robot stoi
#  (srodowisko.py, trening.py i pokaz.py robia to automatycznie)
#
#  Ksztalty, polaczenia i masy czesci bierzemy z URDF (model/robot.urdf).
#  TUTAJ ustawiasz wszystko, czego w URDF nie ma albo co chcesz zmienic:
#  silniki, stopy, tarcie, poze startowa, teren. Zmieniasz liczby na gorze
#  pliku - nie trzeba ruszac XML-a.
#
#  JEDNOSTKI: metry, radiany (chyba ze napisano [st]), kg, Nm, rad/s.
# =============================================================================

import math
import os
import sys

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kinematyka import NOGI, poza_stojaca, stawy, stopa     # noqa: E402
from urdf_mujoco import wczytaj_urdf                         # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLIK_URDF = os.path.join(REPO, "model", "robot.urdf")
PLIK_MODELU = os.path.join(REPO, "model", "quadruped.xml")

# =============================================================================
# 1. USTAWIENIA  (zmieniaj smialo)
# =============================================================================
KROK_CZASU = 0.002              # [s] krok fizyki (0.002 = 500 razy na sekunde)

# silniki (serwa) - takie same we wszystkich 12 przegubach
NAPED = dict(
    moment_max=3.0,             # [Nm] najwiekszy moment serwa
    predkosc_max=6.0,           # [rad/s] najwieksza predkosc serwa
    sztywnosc=25.0,             # [Nm/rad] jak mocno serwo trzyma zadany kat (kp)
)
ARMATURA = 0.01                 # bezwladnosc wirnika z przekladnia (uspokaja drgania)

# stopy: kulka zamiast siatki z CAD-a (kulka daje duzo stabilniejszy kontakt)
STOPA = dict(promien=0.012, tarcie=1.0)

# masa: None = jak w URDF; liczba = przeskaluj wszystkie masy do tej wartosci [kg]
# (URDF ma 3,0 kg, a Config.py z oryginalnego kodu ~1 kg - zwazcie robota!)
MASA_CALKOWITA = None

# poza startowa "stoi": stopy pod biodrami, tyle pod osia bioder [m]
WYSOKOSC_STANIA = 0.20
PRZESUNIECIE_STOP = 0.0         # [m] > 0 = stopy bardziej do przodu
ROZSTAW_STOP = 0.0              # [m] > 0 = stopy szerzej na boki

# teren: "plaski" albo "nierowny" (losowe garby o podanej wysokosci)
TEREN = dict(typ="plaski", wysokosc_garbow=0.02, rozmiar=8.0)

# czy czesci robota moga sie zderzac ze soba (False = szybciej i bez zakleszczen)
KOLIZJE_WLASNE = False


# =============================================================================
# 2. BUDOWA MODELU
# =============================================================================
def zbuduj(teren=None):
    """Zwraca (spec, model). teren nadpisuje TEREN (uzywa tego srodowisko)."""
    teren = teren or TEREN
    spec, korpus = wczytaj_urdf(PLIK_URDF)
    spec.option.timestep = KROK_CZASU
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.compiler.autolimits = True

    # --- kolizje: tylko robot <-> otoczenie ---------------------------------
    for g in spec.geoms:
        if g.contype or g.conaffinity:                 # geomy kolizyjne (nie wizualne)
            g.contype, g.conaffinity = (1, 1) if KOLIZJE_WLASNE else (1, 0)

    # --- stopy: kulka zamiast siatki ------------------------------------------
    for n in NOGI:
        b = spec.body(stopa(n))
        for g in list(b.geoms):
            if g.contype:                              # stara siatka kolizyjna stopy
                spec.delete(g)
        b.add_geom(name=f"stopa_{n}", type=mujoco.mjtGeom.mjGEOM_SPHERE,
                   size=[STOPA["promien"], 0, 0], contype=1,
                   conaffinity=1 if KOLIZJE_WLASNE else 0,
                   friction=[STOPA["tarcie"], 0.005, 0.0001], rgba=[0.2, 0.2, 0.2, 1],
                   mass=0)
        b.add_site(name=f"stopa_{n}", size=[0.005, 0, 0])

    # --- czujnik IMU w srodku korpusu (jak na prawdziwym robocie) --------------
    spec.body(korpus).add_site(name="imu", size=[0.01, 0, 0])
    for typ, nazwa in [(mujoco.mjtSensor.mjSENS_GYRO, "zyroskop"),
                       (mujoco.mjtSensor.mjSENS_ACCELEROMETER, "akcelerometr"),
                       (mujoco.mjtSensor.mjSENS_FRAMEQUAT, "orientacja")]:
        spec.add_sensor(name=nazwa, type=typ, objtype=mujoco.mjtObj.mjOBJ_SITE, objname="imu")

    # --- silniki ----------------------------------------------------------------
    tlumienie = NAPED["moment_max"] / NAPED["predkosc_max"]   # moment maleje z predkoscia
    for n in NOGI:
        for s in stawy(n):
            j = spec.joint(s)
            j.damping = [tlumienie, 0, 0]
            j.armature = ARMATURA
            j.actfrcrange = [-NAPED["moment_max"], NAPED["moment_max"]]
            a = spec.add_actuator(name=s, target=s, trntype=mujoco.mjtTrn.mjTRN_JOINT)
            a.set_to_position(kp=NAPED["sztywnosc"])
            a.ctrlrange = j.range
            a.ctrllimited = 1
            a.forcerange = [-NAPED["moment_max"], NAPED["moment_max"]]
            a.forcelimited = 1

    # --- otoczenie ----------------------------------------------------------------
    swiat = spec.worldbody
    swiat.add_light(pos=[0, 0, 3], dir=[0, 0, -1], diffuse=[0.6, 0.6, 0.6])
    tex = spec.add_texture(name="kratka", type=mujoco.mjtTexture.mjTEXTURE_2D,
                           builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                           rgb1=[0.82, 0.82, 0.85], rgb2=[0.72, 0.72, 0.76], width=300, height=300)
    mat = spec.add_material(name="podloga", texrepeat=[8, 8], reflectance=0.1)
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = tex.name
    if teren["typ"] == "nierowny":
        _nierowny_teren(spec, teren)
    else:
        swiat.add_geom(name="podloga", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[10, 10, 0.1],
                       material="podloga", contype=1, conaffinity=1)

    model = spec.compile()

    # --- masa -----------------------------------------------------------------------
    if MASA_CALKOWITA:
        skala = MASA_CALKOWITA / model.body_mass.sum()
        for b in spec.bodies:
            if b.name != "world":
                b.mass *= skala
                b.inertia = [i * skala for i in b.inertia]
        model = spec.compile()

    # --- poza "stoi" ------------------------------------------------------------------
    q, blad = poza_stojaca(model, WYSOKOSC_STANIA, PRZESUNIECIE_STOP, ROZSTAW_STOP)
    if blad > 0.005:
        print(f"UWAGA: stopy nie siegaja do celu (blad {blad*1000:.0f} mm) - "
              f"zmniejsz WYSOKOSC_STANIA albo przesuniecia.")
    q[0:3] = [0, 0, WYSOKOSC_STANIA + STOPA["promien"] + 0.005]
    q[3:7] = [1, 0, 0, 0]
    ctrl = np.array([q[model.joint(model.actuator(i).trnid[0]).qposadr[0]] for i in range(model.nu)])
    spec.add_key(name="stoi", qpos=q.tolist(), ctrl=ctrl.tolist())
    model = spec.compile()
    return spec, model


def _nierowny_teren(spec, teren):
    """Heightfield z losowymi, gladkimi garbami (staly seed = zawsze ten sam teren)."""
    n = 128
    rng = np.random.default_rng(0)
    h = rng.random((n, n))
    for _ in range(3):                                  # wygladzanie
        h = (h + np.roll(h, 1, 0) + np.roll(h, -1, 0) + np.roll(h, 1, 1) + np.roll(h, -1, 1)) / 5
    h = (h - h.min()) / (h.max() - h.min())
    c = n // 2
    h[c - 6:c + 6, c - 6:c + 6] = 0                     # plaskie miejsce na start
    r = teren["rozmiar"] / 2
    spec.add_hfield(name="teren", nrow=n, ncol=n, size=[r, r, teren["wysokosc_garbow"], 0.05],
                    userdata=h.flatten().tolist())
    spec.worldbody.add_geom(name="podloga", type=mujoco.mjtGeom.mjGEOM_HFIELD, hfieldname="teren",
                            material="podloga", contype=1, conaffinity=1)


# =============================================================================
# 3. PODSUMOWANIE I TEST STANIA
# =============================================================================
def podsumowanie(model):
    print(f"\nMODEL z {os.path.relpath(PLIK_URDF, REPO)}")
    print(f"  masa robota: {model.body_mass.sum():.2f} kg, przeguby napedzane: {model.nu}, "
          f"krok fizyki: {model.opt.timestep*1000:.0f} ms")
    print(f"  silniki: {NAPED['moment_max']} Nm, {NAPED['predkosc_max']} rad/s, kp {NAPED['sztywnosc']}")
    key = model.key("stoi")
    print(f"  poza 'stoi' (wysokosc {WYSOKOSC_STANIA*100:.0f} cm), katy [st]:")
    print("            biodro   udo   kolano")
    for n in NOGI:
        k = [math.degrees(key.qpos[model.joint(s).qposadr[0]]) for s in stawy(n)]
        print(f"    {n}:   {k[0]:7.1f} {k[1]:6.1f} {k[2]:8.1f}")


def test_stania(model, sekundy=3.0):
    """Puszcza symulacje z poza 'stoi'. Zwraca True, gdy robot stoi."""
    d = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, d, model.key("stoi").id)
    while d.time < sekundy:
        mujoco.mj_step(model, d)
    w, x, y, z = d.qpos[3:7]
    przechyl = math.degrees(math.acos(np.clip(1 - 2 * (x * x + y * y), -1, 1)))
    stopy = {model.geom(f"stopa_{n}").id for n in NOGI}
    podloga = model.geom("podloga").id
    inne = [c for c in d.contact[:d.ncon] if podloga in (c.geom1, c.geom2)
            and not ({c.geom1, c.geom2} & stopy)]
    ok = przechyl < 10 and not inne
    print(f"  test stania ({sekundy:.0f} s): wysokosc korpusu {d.qpos[2]*100:.1f} cm, "
          f"przechyl {przechyl:.1f} st -> {'STOI' if ok else 'NIE STOI'}"
          + ("" if not inne else " (cos innego niz stopy dotyka ziemi)"))
    return ok


if __name__ == "__main__":
    spec, model = zbuduj()
    with open(PLIK_MODELU, "w") as f:
        f.write(spec.to_xml())
    podsumowanie(model)
    test_stania(model)
    print(f"Zapisano {os.path.relpath(PLIK_MODELU, REPO)}"
          f"  (podglad: python -m mujoco.viewer --mjcf={os.path.relpath(PLIK_MODELU, REPO)})")
