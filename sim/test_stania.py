"""
Sprawdza, czy robot stoi: puszcza symulację na 5 sekund bez okna
i co sekundę wypisuje wysokość i pochylenie korpusu.
Użycie:  python sim/test_stania.py model/robot.xml
"""
import sys
import numpy as np
import mujoco

model = mujoco.MjModel.from_xml_path(sys.argv[1] if len(sys.argv) > 1 else "robot.xml")
data = mujoco.MjData(model)
data.ctrl[:] = 0          # wszystkie silniki trzymają kąt 0 (pozycja z modelu CAD)

def pochylenie_korpusu():
    w, x, y, z = data.qpos[3:7]
    gora_z = 1 - 2 * (x * x + y * y)          # jak bardzo korpus "patrzy" w górę
    return np.degrees(np.arccos(np.clip(gora_z, -1, 1)))

print("czas [s] | wysokość korpusu [m] | pochylenie [°]")
for sekunda in range(6):
    while data.time < sekunda:
        mujoco.mj_step(model, data)
    print(f"{sekunda:8d} | {data.qpos[2]:21.3f} | {pochylenie_korpusu():14.1f}")

if pochylenie_korpusu() > 45:
    print("\nWYNIK: robot się przewrócił.")
elif data.qpos[2] < 0.05:
    print("\nWYNIK: korpus leży na ziemi - nogi się złożyły (zwiększ KP).")
elif np.abs(data.qvel[:3]).max() > 0.05:
    print("\nWYNIK: robot nie upadł, ale wciąż się rusza albo drga.")
else:
    print("\nWYNIK: robot stoi stabilnie!")
