#!/usr/bin/env python3
# =============================================================================
#  kinematyka.py  -  gdzie postawic stopy -> jakie katy przegubow
#
#  Tego pliku nie trzeba uruchamiac. Korzysta z niego robot.py (poza
#  "stoi") i mozna go uzyc wszedzie, gdzie trzeba policzyc katy nog.
#
#  Kinematyka odwrotna liczona NUMERYCZNIE na modelu MuJoCo: program
#  probuje katow, patrzy, gdzie wypada stopa, i poprawia katy, az stopa
#  trafi w cel (metoda "tlumionych najmniejszych kwadratow"). Zaleta: nie
#  trzeba znac konwencji zer i kierunkow przegubow z CAD-a - liczy sie
#  dokladnie to, co jest w modelu.
#
#  Wzory analityczne dla jednej nogi (Q1 z widoku z przodu, Q2 i Q3 z boku)
#  sa w CLAUDE.md - mozna nimi sprawdzac wyniki tej funkcji.
# =============================================================================

import mujoco
import numpy as np

NOGI = ["FL", "FR", "RL", "RR"]          # przod-lewa, przod-prawa, tyl-lewa, tyl-prawa


def stawy(noga):
    """Nazwy przegubow nogi: biodro (na boki), udo, kolano."""
    return [f"{noga}_theta1", f"{noga}_theta2", f"{noga}_theta3"]


def stopa(noga):
    return f"{noga}_link4"


def biodro(noga):
    """Bryla, od ktorej liczymy "stopa pod biodrem"."""
    return f"{noga}_link2"


def ik(model, cele, poczatek=None, iteracje=200, tlumienie=0.01):
    """Kinematyka odwrotna dla wszystkich nog naraz.

    cele: {noga: (x, y, z)} - polozenie stopy [m] w ukladzie korpusu
          (korpus w punkcie 0, prosto).
    poczatek: wektor qpos, od ktorego zaczac (wybiera "galaz" rozwiazania,
              np. kolano do przodu albo do tylu).
    Zwraca (qpos, najwiekszy_blad_w_metrach).
    """
    d = mujoco.MjData(model)
    if poczatek is not None:
        d.qpos[:] = poczatek
    d.qpos[0:7] = [0, 0, 0, 1, 0, 0, 0]          # korpus w zerze, prosto
    blad_max = 0.0
    for noga, cel in cele.items():
        q_adr = [model.joint(s).qposadr[0] for s in stawy(noga)]
        v_adr = [model.joint(s).dofadr[0] for s in stawy(noga)]
        zakres = np.array([model.joint(s).range for s in stawy(noga)])
        b = model.body(stopa(noga)).id
        jac = np.zeros((3, model.nv))
        for _ in range(iteracje):
            mujoco.mj_kinematics(model, d)
            mujoco.mj_comPos(model, d)
            e = np.asarray(cel) - d.xpos[b]
            if np.linalg.norm(e) < 1e-5:
                break
            mujoco.mj_jac(model, d, jac, None, d.xpos[b], b)
            J = jac[:, v_adr]
            dq = J.T @ np.linalg.solve(J @ J.T + tlumienie ** 2 * np.eye(3), e)
            d.qpos[q_adr] = np.clip(d.qpos[q_adr] + dq, zakres[:, 0], zakres[:, 1])
        mujoco.mj_kinematics(model, d)
        blad_max = max(blad_max, float(np.linalg.norm(np.asarray(cel) - d.xpos[b])))
    return d.qpos.copy(), blad_max


def pozycja_bioder(model):
    """Polozenie bioder (x, y, z) w pozie zerowej - stad liczymy "stopa pod biodrem"."""
    d = mujoco.MjData(model)
    d.qpos[0:7] = [0, 0, 0, 1, 0, 0, 0]
    mujoco.mj_kinematics(model, d)
    return {n: d.xpos[model.body(biodro(n)).id].copy() for n in NOGI}


def poza_stojaca(model, wysokosc, przesuniecie_x=0.0, rozstaw_y=0.0, poczatek=None):
    """Katy przegubow, przy ktorych stopy sa pod biodrami, `wysokosc` [m]
    pod osia bioder. przesuniecie_x > 0 przesuwa stopy do przodu,
    rozstaw_y > 0 rozsuwa je na boki."""
    biodra = pozycja_bioder(model)
    cele = {}
    for n, p in biodra.items():
        strona = 1 if p[1] > 0 else -1
        cele[n] = (p[0] + przesuniecie_x, p[1] + strona * rozstaw_y, p[2] - wysokosc)
    return ik(model, cele, poczatek)
