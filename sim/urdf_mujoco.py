#!/usr/bin/env python3
# =============================================================================
#  urdf_mujoco.py  -  wczytanie URDF z CAD-a do MuJoCo (MjSpec)
#
#  Tego pliku nie trzeba zmieniac ani uruchamiac. Korzysta z niego robot.py.
#  Robi to samo co przygotuj_robota.py, ale zwraca model w pamieci
#  (MjSpec), zeby robot.py mogl go dalej przerobic.
#
#  Co poprawia w URDF:
#    - proste makra xacro (<xacro:property> i ${...}),
#    - pliki .dae -> .stl (MuJoCo nie czyta .dae; potrzebny pakiet trimesh),
#    - sciezki "package://..." -> sama nazwa pliku,
#    - dopisuje <mujoco><compiler .../></mujoco> z folderem siatek,
#    - dodaje korpusowi swobodny przegub (freejoint), zeby nie byl
#      przyspawany do swiata (tak domyslnie robi importer URDF w MuJoCo).
# =============================================================================

import glob
import os
import re
import sys
import xml.etree.ElementTree as ET

import mujoco


def _blad(tekst):
    sys.exit("BLAD: " + tekst)


def wczytaj_urdf(plik_urdf):
    """Zwraca (spec, nazwa_korpusu).

    Czesci przyczepione na sztywno (np. stopy FL_link4) zostaja osobnymi
    brylami - dzieki temu mozna sie do nich odwolac po nazwie.
    """
    tekst = open(plik_urdf, encoding="utf-8").read()

    # 0. makra xacro: <xacro:property name="X" value="V"/> i ${X}
    for nazwa, wartosc in re.findall(
            r'<xacro:property\s+name="([^"]+)"\s+value="([^"]*)"\s*/>', tekst):
        tekst = tekst.replace("${" + nazwa + "}", wartosc)
    if "${" in tekst:
        _blad("w URDF zostaly makra xacro (${...}), ktorych nie umiem zamienic.")
    robot = ET.fromstring(tekst.encode("utf-8"))
    for e in [e for e in robot if "property" in e.tag]:
        robot.remove(e)

    # 1. korzen = link, ktory nie jest dzieckiem zadnego przegubu
    linki = [l.get("name") for l in robot.findall("link")]
    dzieci = {j.find("child").get("link") for j in robot.findall("joint")}
    korzen = [l for l in linki if l not in dzieci][0]

    # 2. siatki: .dae -> .stl, package:// -> nazwa pliku
    folder = os.path.dirname(os.path.abspath(plik_urdf))
    folder_siatek = None
    for siatka in robot.iter("mesh"):
        nazwa = os.path.basename(siatka.get("filename"))
        znalezione = glob.glob(os.path.join(folder, "**", nazwa), recursive=True)
        if not znalezione:
            _blad(f"nie ma pliku {nazwa} obok URDF (wielkosc liter ma znaczenie).")
        if nazwa.lower().endswith(".dae"):
            nowa = nazwa[:-4] + "_z_dae.stl"
            nowy_plik = os.path.join(os.path.dirname(znalezione[0]), nowa)
            if not os.path.exists(nowy_plik):
                try:
                    import trimesh
                except ImportError:
                    _blad("URDF uzywa plikow .dae - zainstaluj: pip install trimesh pycollada")
                trimesh.load(znalezione[0], force="mesh").export(nowy_plik)
            nazwa = nowa
        siatka.set("filename", nazwa)
        if folder_siatek is None:
            folder_siatek = os.path.relpath(os.path.dirname(znalezione[0]), folder) + "/"

    for stary in robot.findall("mujoco"):
        robot.remove(stary)
    znacznik = ET.Element("mujoco")
    ET.SubElement(znacznik, "compiler", meshdir=folder_siatek or "./",
                  balanceinertia="true", discardvisual="false",
                  fusestatic="false")
    robot.insert(0, znacznik)

    tmp = os.path.join(folder, "_tymczasowy.urdf")
    ET.ElementTree(robot).write(tmp)
    try:
        spec = mujoco.MjSpec.from_file(tmp)
    finally:
        os.remove(tmp)
    # 2. korpus luzem (6 stopni swobody) - inaczej bylby przyspawany do swiata
    spec.body(korzen).add_freejoint(name="korpus")
    return spec, korzen
