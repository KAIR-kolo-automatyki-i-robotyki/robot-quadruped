# Robot czworonożny KAIR – kontekst projektu dla Claude Code

Koło naukowe KAIR (Koło Automatyki i Robotyki, PWr) buduje czworonożnego robota na bazie projektu
**Quadruped II** ([Printables](https://www.printables.com/model/1451339-quadruped-ii/files)).
Cel tego repo: symulacja robota w **MuJoCo**, potem nauka chodu (RL) i przeniesienie na prawdziwego robota.

Instrukcja instalacji krok po kroku (dla zespołu): https://claude.ai/artifact/YJxWJxYD8qXG3th7q5TSUB

## Jak ze mną rozmawiać

- Po polsku, prostym językiem. Część zespołu nie jest od software'u (np. osoba od mechaniki).
- Komendy podawaj w osobnych blokach, po kolei, z jednym zdaniem „co powinieneś zobaczyć”.
- Przed większymi zmianami (wiele plików, zmiana struktury, `git push`) najpierw zapytaj.
- Nie zakładaj znajomości ROS-a, Linuksa ani gita – tłumacz skrótowo, gdy pojawia się coś nowego.

## Środowisko

- Windows + **WSL, Ubuntu 26.04**, Python 3.14, venv w `~/robot_projekt/venv` (włączany automatycznie z `.bashrc`).
- Biblioteki i wersje: `requirements.txt` (mujoco 3.17.0, gymnasium 1.4.0, stable-baselines3 2.9.2,
  torch 2.15.0, numpy 2.5.4, trimesh 5.2.0, pycollada 0.9.3). Wersje zmieniamy tylko razem.
- Jedna osoba w zespole ma kartę **AMD** – wszystko musi działać na CPU (MuJoCo liczy na CPU, trening RL też).
- Ostrzeżenie pip o `launch-ros` (zainstalowany ROS 2) można ignorować.
- Repo: `KAIR-kolo-automatyki-i-robotyki/robot-quadruped` (prywatne, organizacja KAIR na GitHubie).

## Struktura repo

| Folder | Zawartość |
| --- | --- |
| `model/` | `robot.urdf` + `meshes/` (z oryginalnego kodu Quadruped II), generowany `robot.xml` |
| `sim/` | symulacja i uczenie chodu – opis plikow w `sim/README.md` |
| `rl/` | trening chodu – jeszcze pusty |
| `hardware/` | kod do prawdziwego robota, `stl_druk/` = pliki do druku 3D (nie do symulacji) |
| `docs/` | notatki, dane silników |

Komendy (z glownego folderu repo):
```
python sim/robot.py                 # URDF + ustawienia -> model/quadruped.xml, test stania
python sim/pokaz.py --bez-sieci     # podglad pozy "stoi"
python sim/trening.py [--wznow]     # uczenie PPO z curriculum, tabelka nagrod
python sim/pokaz.py [--vx 0.15]     # podglad nauczonej sieci (strzalki zmieniaja komende)
```

## Uklad plikow symulacji (wzorowany na projekcie inzynierskim Lumi)

- `sim/robot.py` – jedyne miejsce na ustawienia modelu (silniki, stopy-kulki, masa, poza "stoi"
  liczona z IK, teren plaski/nierowny, IMU). Generuje XML, nie edytujemy XML recznie.
- `sim/nagrody.py` – slownik `WAGI` + jedna funkcja na skladnik; waga 0 = wylaczony.
  Funkcje dostaja stan `s` (pola opisane w pliku). Nowy skladnik = nowa funkcja + wpis w WAGI.
- `sim/srodowisko.py` – Gymnasium env `Chodzenie` (50 Hz, akcja = przesuniecie od pozy "stoi",
  obserwacja 48 liczb, komendy vx/vy/wz z POZIOMY, LOSOWANIE swiata).
- `sim/trening.py` – PPO (SB3), curriculum po POZIOMY, tabelka pkt/probe na skladnik.
- `sim/kinematyka.py` – numeryczna IK na modelu MuJoCo (stopy pod biodrami itd.).
- `sim/urdf_mujoco.py` – wczytanie URDF (xacro, .dae, package://, freejoint, fusestatic=false).

## Co robi `sim/przygotuj_robota.py`

Zamienia proste makra xacro (`<xacro:property>` + `${...}`), konwertuje `.dae` → `.stl` (trimesh;
pliki `*_z_dae.stl` są w `.gitignore`), usuwa `package://` ze ścieżek, dopisuje `<mujoco><compiler meshdir
balanceinertia discardvisual=false>`, dodaje pływający przegub (robot nie jest przyczepiony do świata),
podłogę, światło i silnik pozycyjny (`kp=20`) na każdy przegub z zakresem z URDF.
Uwaga: importer URDF w MuJoCo „wtapia” korzeń w świat – dlatego skrypt dodaje przegub `floating`.

## Aktualny stan (2026-11-10)

- Model Quadruped II wczytuje się do MuJoCo: 12 przegubów `{FL,FR,RL,RR}_theta{1,2,3}`, masa 3,0 kg.
  Zakresy: theta1 ±0,79 rad, theta2 i theta3 ±3,0 rad. `effort` = 10 (makro xacro).
- W pozycji "0" z CAD-a udo jest poziomo do tylu i korpus lezy. Rozwiazane: `robot.py` liczy poze
  "stoi" z IK (wysokosc 20 cm: udo ok. 47 st, kolano ok. -6 st, kolana do tylu) i test stania przechodzi.
- Srodowisko RL dziala (stanie przy akcji 0 przezywa 10 s z popychaniem). Trening jeszcze nie byl
  puszczony na dluzej - pierwsze strojenie wag nagrod przed nami.
- Masa w URDF (3,0 kg) nie zgadza się z `Config.py` (ok. 1 kg) – do sprawdzenia z prawdziwym robotem.

## Oryginalny kod Quadruped II (ROS1, poza tym repo)

Z ZIP-a „0001-1000 Quadruped_II - code” (Printables). Najważniejsze:
- `Quadruped_ws/src/Quadruped_description/urdf/quadruped.urdf` – źródło naszego URDF (eksport z SolidWorksa);
  obok starsza wersja `quadruped copy.urdf` z innymi masami.
- `Quadruped_ws/src/Quadruped_control/src/quadruped_control/` – **gotowa kinematyka i chód**
  (styl Stanford Pupper): `Kinematics.py` (IK nogi i 4 nóg, FK), `Gaits.py`, `Controller.py`,
  `StanceController.py`, `SwingLegController.py`, `Config.py`.
- Wymiary z `Config.py`: L1 = 0,05162 m, L2 = 0,130 m, L3 = 0,13814 m; LEG_FB = 0,11165 m,
  LEG_LR = 0,061 m; domyślna wysokość stania `default_z_ref` = −0,25 m (stara wartość −0,16).
- Jest też pakiet `Quadruped_gazebo` (symulacja w Gazebo na ROS1).

## Własna kinematyka zespołu (notatki ręczne, 1 noga)

Q1 (biodro) z widoku z przodu: A = √(x² + y² − L1²), atan2.
Q3 i Q2 z widoku z boku, **poprawione wzory** (sprawdzone numerycznie):
```
D  = cos Q3 = (A² + z² − L2² − L3²) / (2·L2·L3)
Q3 = atan2(±√(1 − D²), D)            # znak wybrany pod nasze kolano
Q2 = atan2(z, A) − atan2(L3·sin Q3, L2 + L3·cos Q3)
```
(W oryginalnych notatkach był odwrócony znak w D i brakowało L2 w drugim atan2.)
Warto porównać z `Kinematics.py` i z konwencją zer/znaków przegubów w URDF.

## Otwarte decyzje i pomysły na później

- **Gazebo czy MuJoCo:** Gazebo działa z gotowym kodem ROS1; MuJoCo – stabilniejszy kontakt stóp,
  szybsze (RL), działa na AMD. Rozważany kompromis: oba, do różnych celów.
- PyTorch w wersji CPU (mniejsza instalacja niż domyślna z CUDA).
- Ulepszenia skryptu: ostrzeżenia o zerowych masach / milimetrach, siatki w wielu folderach,
  kąty do stania, tłumienie i `armature` silników zbliżone do prawdziwych serw.
- Dane serw (moment, prędkość) od ekipy sprzętu → sprawdzić `effort` w URDF.

## Zasady pracy z gitem

`git pull` przed pracą; po pracy `git add`, `git commit -m "..."`, `git push`. Małe commity, opisy po polsku.
Każda ekipa pracuje w swoim folderze. Bez haseł i plików > 100 MB.
