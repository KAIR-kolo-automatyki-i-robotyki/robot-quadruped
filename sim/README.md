# sim – symulacja i uczenie chodu quadrupeda

Wszystkie komendy uruchamiasz z **głównego folderu repo** (`~/robot_projekt`).

| Plik | Co robi | Czy go zmieniasz |
| --- | --- | --- |
| `robot.py` | URDF z CAD-a + nasze ustawienia (silniki, stopy, masa, poza „stoi”, teren) → `model/quadruped.xml`. Na koniec sprawdza, czy robot stoi. | **tak** – ustawienia na górze pliku |
| `nagrody.py` | **System nagród i kar.** Tabela `WAGI` + po jednej funkcji na każdy składnik. | **tak** – najczęściej |
| `srodowisko.py` | Zadanie „chodź z zadaną prędkością”: obserwacje, akcje, komendy (poziomy trudności), upadek, losowanie świata. | czasem – ustawienia na górze |
| `trening.py` | Uczenie PPO z curriculum. Co chwilę wypisuje tabelkę: ile punktów dał każdy składnik nagrody. | rzadko |
| `pokaz.py` | Podgląd w viewerze: nauczona sieć albo sama poza „stoi”. Komendę zmieniasz strzałkami. | nie |
| `kinematyka.py` | Kinematyka odwrotna (stopy → kąty przegubów), liczona numerycznie na modelu. | nie |
| `urdf_mujoco.py` | Wczytanie URDF do MuJoCo (makra xacro, pliki .dae, ścieżki). | nie |
| `przygotuj_robota.py`, `test_stania.py` | Pierwsze, prostsze wersje (z instrukcji). Zostają dla porządku. | nie |

## Typowa praca

```
python sim/robot.py                 # zbuduj model i sprawdź, czy stoi
python sim/pokaz.py --bez-sieci     # obejrzyj robota w pozie "stoi"
python sim/trening.py               # ucz (Ctrl+C = przerwij i zapisz)
python sim/trening.py --wznow       # ucz dalej
python sim/pokaz.py                 # obejrzyj ostatnią sieć
python sim/pokaz.py --vx 0.15       # ...ze stałą komendą 0,15 m/s
```

## Strojenie nagród – w skrócie

1. Puść trening i patrz na tabelkę `pkt/probe` dla każdego składnika.
2. Zmieniaj **jedną** wagę naraz w `nagrody.py` (0 = wyłączona).
3. Nagroda za cel (`predkosc_xy`) powinna przeważać nad sumą kar – inaczej robot nauczy się stać w miejscu.
4. Po zmianie wag zacznij uczenie od nowa (`python sim/trening.py`, bez `--wznow`) albo kontynuuj, jeśli zmiana jest mała.

Nowy składnik: dopisz funkcję `def moj_skladnik(s): ...` w `nagrody.py` i dodaj `"moj_skladnik": waga` do `WAGI`. Dostępne pola stanu `s` są opisane nad funkcjami.
