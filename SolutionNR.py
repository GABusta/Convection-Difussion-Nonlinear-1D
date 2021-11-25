import numpy as np
import sympy as sym
from MRK_Matrix import K_Matrix_P_NR, MR_Matrix_P_NR

def NR(y, dt, tp, km, tol, Nnod):
    OP = np.zeros((Nnod, tp))  # Solution tita
    OP[0:Nnod-1, 0] = 1.0      # Initial condition
    k_iter = 0
    err = np.zeros((km, tp))
    OK = OP[:, 0]
    M, R = MR_Matrix_P_NR(y, Nnod)

    for i in range(1, tp-1, 1):
        error = 1.0
        k_iter += 1
        while (error >= tol) and (k_iter < km):
            K = K_Matrix_P_NR(y, dt, tp, OK, Nnod)
            RR = dt * R + np.dot(M, OP[:, i-1])   # R_NR
            FF = np.dot(M + dt * K, OK)           # F_NR
            KT = Kij_tang(y, OK, dt, Nnod)        # KT - tangent

            FF[Nnod - 1], RR[Nnod - 1] = 0.0, 0.0
            d_tita = np.dot(np.linalg.inv(KT), RR-FF)

            OK22 = OK + d_tita
            error = np.linalg.norm(OK22 - OK)/np.linalg.norm(OK22)
            err[k_iter, i] = error
            OK = OK22
        OP[:, i] = OK[:]
    return err, k_iter, OP

# -------------------------------------------
# -------- " KT " - Tangent Matrix  ---------
# -------------------------------------------
def Kij_tang(y, OK, dt, Nnod):
    r = sym.Symbol("r")
    # ---- Shape functions ----
    h1, h2 = 0.5 * (1 - r), 0.5 * (1 + r)
    H = [h1, h2]
    dH = [H[0].diff(r), H[1].diff(r)]

    KT = np.zeros((Nnod, Nnod))
    for i in range(0, Nnod-1, 1):
        # ---- Jacobian ----
        x = H[0] * y[i] + H[1] * y[i + 1]
        dx_dr = x.diff(r)
        Je = dx_dr

        # ---- " M " matrix ----
        mm = [[H[0]*H[0]*Je, H[0]*H[1]*Je] , [H[1]*H[0]*Je, H[1]*H[1]*Je]]
        m1 = [[mm[0][0].integrate((r, -1, 1)), mm[0][1].integrate((r, -1, 1))],
              [mm[1][0].integrate((r, -1, 1)), mm[1][1].integrate((r, -1, 1))]]

        # ---- " K " matrix ----
        Be = [dH[0]/Je, dH[1]/Je]
        kk = [[Be[0]*Be[0]*Je, Be[0]*Be[1]*Je] , [Be[1]*Be[0]*Je, Be[1]*Be[1]*Je]]
        tm = (OK[i] + OK[i + 1]) * 0.5
        alfa = 0.5 * (tm**2 + 1)
        k1 = [[alfa * kk[0][0].integrate((r, -1, 1)), alfa * kk[0][1].integrate((r, -1, 1))] ,
              [alfa * kk[1][0].integrate((r, -1, 1)), alfa * kk[1][1].integrate((r, -1, 1))]]
        # dK(tita) / dtita_r
        k11 = [[tm * kk[0][0].integrate((r, -1, 1)), tm * kk[0][1].integrate((r, -1, 1))] ,
              [tm * kk[1][0].integrate((r, -1, 1)), tm * kk[1][1].integrate((r, -1, 1))]]

        # ---- " KT " matrix ----
        der = [[1.0, 0.0] , [0.0, 1.0]]
        tita = [[OK[i], OK[i]] , [OK[i + 1], OK[i + 1]]]
        kt_ij = np.dot(m1, der) + dt * np.dot(k1, der) + dt * np.dot(k11, tita)

        KT[i][i] += kt_ij[0][0]
        KT[i][i + 1] += kt_ij[0][1]
        KT[i + 1][i] += kt_ij[1][0]
        KT[i + 1][i + 1] += kt_ij[1][1]
    # ---- BCs ----
    KT[Nnod - 1, :], KT[:, Nnod - 1] = 0.0, 0.0
    KT[Nnod - 1, Nnod - 1] = 1.0
    return KT