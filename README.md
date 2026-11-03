# Robot czworonożny – symulacja w MuJoCo

Uczymy naszego czworonoga chodzić w symulatorze MuJoCo metodą uczenia ze wzmocnieniem (RL), bez ręcznego liczenia kinematyki.

**Instrukcja instalacji krok po kroku:** https://claude.ai/artifact/YJxWJxYD8qXG3th7q5TSUB

## Co gdzie leży

| Folder | Co tam jest | Kto się zajmuje |
| --- | --- | --- |
| `model/` | URDF + pliki STL z CAD-a, wygenerowany `robot.xml` | ekipa modelu |
| `sim/` | skrypty symulacji (przygotowanie modelu, test stania) | ekipa symulacji |
| `rl/` | trening chodu (Gymnasium + Stable-Baselines3) | ekipa RL |
| `hardware/` | kod do prawdziwego robota | ekipa sprzętu |
| `docs/` | notatki, dane silników, zdjęcia | wszyscy |

## Szybki start

```
pip install -r requirements.txt
python sim/przygotuj_robota.py model/robot.urdf
python sim/test_stania.py model/robot.xml
python -m mujoco.viewer --mjcf=model/robot.xml
```

## Zasady

1. Przed pracą: `git pull`. Po pracy: `git add .`, `git commit -m "co zmieniłem"`, `git push`.
2. Małe commity z jasnym opisem, po polsku jest OK.
3. Pracujemy każdy w swoim folderze. Jeśli trzeba zmienić cudzy plik, najpierw pytamy.
4. Nie wrzucamy haseł, folderu `venv/` ani plików powyżej 100 MB.
5. Wersje bibliotek zmieniamy tylko razem, w `requirements.txt`.
6. Zadania prowadzimy w zakładce **Issues**: jedno zadanie = jeden wpis z przypisaną osobą.
