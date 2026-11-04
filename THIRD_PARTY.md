# Licencje zewnętrzne

Licencja MIT w pliku `LICENSE` dotyczy tylko kodu napisanego przez KAIR (np. `sim/`, `rl/`).
Części repo pochodzą z innych źródeł i obowiązują je ich własne licencje:

| Co | Skąd | Licencja |
| --- | --- | --- |
| `model/` (URDF i pliki 3D do symulacji) | projekt Quadruped II, autor TomKnox, pakiet `Quadruped_description` | w kodzie oznaczony jako BSD, bez pełnego tekstu licencji – do potwierdzenia u autora |
| `hardware/stl_druk/` (pliki do druku) | [Quadruped II na Printables](https://www.printables.com/model/1451339-quadruped-ii), autor TomKnox | licencja ze strony modelu na Printables – do sprawdzenia; wersja na MakerWorld ma „Standard Digital File License” |
| kod sterowania z oryginalnego projektu (kinematyka, chód), jeśli zostanie tu skopiowany | Quadruped II, oparte na Stanford Pupper | MIT, Copyright (c) 2020 Stanford Student Robotics – przy kopiowaniu trzeba zachować tę informację |

Dopóki licencje plików 3D nie są potwierdzone, repo pozostaje **prywatne** i nie udostępniamy tych plików publicznie.
