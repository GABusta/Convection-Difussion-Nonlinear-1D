import numpy as np
import sympy as sym

# --------------------------------------------------------------------------
def Matrix(y, N):
    r = sym.Symbol("r")
    # ----- Shape Functions -----
    h1, h2 = 0.5*(1 - r), 0.5*(1 + r)
    H = [h1, h2]
    dH = [H[0].diff(r), H[1].diff(r)]

    K, M = np.zeros((N, N)), np.zeros((N, N))
    R = np.zeros((N))
    for i in range(0, N-1, 1):

        # ---- Elemental Jacobian ----
        x = H[0]*y[i] + H[1]*y[i+1]
        dx_dr = x.diff(r)
        Je = dx_dr

        # ---- " M " matrix ----
        # M = integral ( Transpose(H)*H*Je )
        mm = [[H[0]*H[0]*Je, H[0]*H[1]*Je], [H[1]*H[0]*Je, H[1]*H[1]*Je]]
        M[i][i] += mm[0][0].integrate((r, -1.0, 1.0))
        M[i][i + 1] += mm[0][1].integrate((r, -1.0, 1.0))
        M[i + 1][i] += mm[1][0].integrate((r, -1.0, 1.0))
        M[i + 1][i + 1] += mm[1][1].integrate((r, -1.0, 1.0))

        # ---- " K " matrix ----
        # K = integral ( Transpose(Be)*Be*Je )
        Be = [dH[0]/Je, dH[1]/Je]
        kk = [[Be[0] * Be[0] * Je, Be[0] * Be[1] * Je],
              [Be[1] * Be[0] * Je, Be[1] * Be[1] * Je]]
        K[i][i] += kk[0][0].integrate((r, -1.0, 1.0))
        K[i][i + 1] += kk[0][1].integrate((r, -1.0, 1.0))
        K[i + 1][i] += kk[1][0].integrate((r, -1.0, 1.0))
        K[i + 1][i + 1] += kk[1][1].integrate((r, -1.0, 1.0))

    return M, R, K

# --------------------------------------------------------------------------
def K_Matrix_P_NR(y, dt, p, OK, N):
    r = sym.Symbol("r")
    # ----- Shape Functions -----
    h1, h2 = 0.5*(1 - r), 0.5*(1 + r)
    H = [h1, h2]
    dH = [H[0].diff(r), H[1].diff(r)]

    K = np.zeros((N,N))
    for i in range(0, N-1, 1):

        # ---- Elemental Jacobian ----
        x = H[0]*y[i] + H[1]*y[i+1]
        dx_dr = x.diff(r)
        Je = dx_dr

        # ---- " K " matrix ----
        # K = integral ( Transpose(Be)*Be*Je )
        Be = [dH[0]/Je, dH[1]/Je]
        kk = [[Be[0] * Be[0] * Je, Be[0] * Be[1] * Je],
              [Be[1] * Be[0] * Je, Be[1] * Be[1] * Je]]
        tm = (OK[i] + OK[i + 1])*0.5
        alfa = 0.5 * (tm**2 + 1.0)
        K[i][i] += alfa * kk[0][0].integrate((r, -1.0, 1.0))
        K[i][i + 1] += alfa * kk[0][1].integrate((r, -1.0, 1.0))
        K[i + 1][i] += alfa * kk[1][0].integrate((r, -1.0, 1.0))
        K[i + 1][i + 1] += alfa * kk[1][1].integrate((r, -1.0, 1.0))

    return K


# --------------------------------------------------------------------------
def MR_Matrix_P_NR(y, N):
    r = sym.Symbol("r")
    # ----- Shape Functions -----
    h1, h2 = 0.5*(1 - r), 0.5*(1 + r)
    H = [h1, h2]
    dH = [H[0].diff(r), H[1].diff(r)]

    M = np.zeros((N,N))
    R = np.zeros((N))
    for i in range(0, N-1, 1):

        # ---- Elemental Jacobian ----
        x = H[0]*y[i] + H[1]*y[i+1]
        dx_dr = x.diff(r)
        Je = dx_dr

        # ---- " M " matrix ----
        # M = integral ( Transpose(H)*H*Je )
        mm = [[H[0]*H[0]*Je, H[0]*H[1]*Je], [H[1]*H[0]*Je, H[1]*H[1]*Je]]
        M[i][i] += mm[0][0].integrate((r, -1.0, 1.0))
        M[i][i + 1] += mm[0][1].integrate((r, -1.0, 1.0))
        M[i + 1][i] += mm[1][0].integrate((r, -1.0, 1.0))
        M[i + 1][i + 1] += mm[1][1].integrate((r, -1.0, 1.0))

    return M, R