"""
Zamienia nasz URDF na plik MuJoCo (robot.xml) gotowy do testu:
- odpina robota od świata (może spaść i chodzić),
- dodaje podłogę i światło,
- dodaje silnik pozycyjny do każdego przegubu,
- sam poprawia ścieżki do plików STL i dopisuje znacznik <mujoco>.
Użycie:  python sim/przygotuj_robota.py model/robot.urdf
"""
import sys, os, glob
import xml.etree.ElementTree as ET
import mujoco

# ====== USTAWIENIA, KTÓRE MOŻNA ZMIENIAĆ ======
WYSOKOSC_STARTU = 0.4   # na jakiej wysokości (w metrach) robot startuje
KP = 20                 # "siła" silników: drgania -> zmniejsz, siada -> zwiększ
# ==============================================

plik_urdf = sys.argv[1] if len(sys.argv) > 1 else "robot.urdf"

# 1. Znajdź główną część robota (link, który nie jest "dzieckiem" żadnego przegubu)
drzewo = ET.parse(plik_urdf)
robot = drzewo.getroot()
linki = [l.get("name") for l in robot.findall("link")]
dzieci = {j.find("child").get("link") for j in robot.findall("joint")}
korzen = [l for l in linki if l not in dzieci][0]
print("Główna część robota (korpus):", korzen)

# 2. Dodaj "pływający" przegub, żeby robot nie był przyczepiony do świata
if not any(j.get("type") == "floating" for j in robot.findall("joint")):
    ET.SubElement(robot, "link", name="swiat_pomocniczy")
    j = ET.SubElement(robot, "joint", name="wolny", type="floating")
    ET.SubElement(j, "parent", link="swiat_pomocniczy")
    ET.SubElement(j, "child", link=korzen)
# 3. Popraw ścieżki do plików STL ("package://..." -> sama nazwa pliku)
#    i dopisz znacznik <mujoco>, który mówi, w jakim folderze leżą siatki
folder_urdf = os.path.dirname(os.path.abspath(plik_urdf))
folder_siatek = None
for siatka in robot.iter("mesh"):
    nazwa = os.path.basename(siatka.get("filename"))
    siatka.set("filename", nazwa)
    if folder_siatek is None:
        znalezione = glob.glob(os.path.join(folder_urdf, "**", nazwa), recursive=True)
        if not znalezione:
            sys.exit(f"BŁĄD: nie znalazłem pliku {nazwa} w folderze z URDF. "
                     "Sprawdź nazwę (Linux rozróżnia wielkie i małe litery: .STL to nie .stl).")
        folder_siatek = os.path.relpath(os.path.dirname(znalezione[0]), folder_urdf) + "/"
for stary in robot.findall("mujoco"):
    robot.remove(stary)
znacznik = ET.Element("mujoco")
ET.SubElement(znacznik, "compiler", meshdir=folder_siatek or "./",
              balanceinertia="true", discardvisual="false")
robot.insert(0, znacznik)
print("Folder z siatkami:", folder_siatek)

tmp = os.path.join(folder_urdf, "_tymczasowy.urdf")
drzewo.write(tmp)

# 4. Wczytaj do MuJoCo i dołóż podłogę, światło i silniki
try:
    spec = mujoco.MjSpec.from_file(tmp)
finally:
    os.remove(tmp)
spec.worldbody.add_light(pos=[0, 0, 3], dir=[0, 0, -1])
spec.worldbody.add_geom(name="podloga", type=mujoco.mjtGeom.mjGEOM_PLANE,
                        size=[5, 5, 0.1], rgba=[0.8, 0.8, 0.8, 1])
spec.body(korzen).pos = [0, 0, WYSOKOSC_STARTU]

for przegub in spec.joints:
    if przegub.type in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
        silnik = spec.add_actuator(name="silnik_" + przegub.name, target=przegub.name,
                                   trntype=mujoco.mjtTrn.mjTRN_JOINT)
        silnik.set_to_position(kp=KP)
        zakres = list(przegub.range)
        if zakres[0] >= zakres[1]:       # przegub bez limitu w URDF
            zakres = [-1.57, 1.57]
        silnik.ctrlrange = zakres
        silnik.ctrllimited = 1
        print("  silnik dla przegubu:", przegub.name, "zakres:", [round(float(z), 2) for z in zakres])

model = spec.compile()
plik_wyjsciowy = os.path.join(folder_urdf, "robot.xml")
with open(plik_wyjsciowy, "w") as f:
    f.write(spec.to_xml())
print(f"\nGotowe! Zapisano {plik_wyjsciowy} ({model.nu} silników, masa robota: {sum(model.body_mass):.2f} kg)")
