import numpy as np
from MRK_Matrix import K_Matrix_P_NR, MR_Matrix_P_NR

def Picard(y, dt, p, km, tol, Nnod):
    OP = np.zeros((Nnod, p))   # Solution tita
    OP[0: Nnod-1, 0] = 1.0     # Initial condition
    k_iter = 0                 # Number of iterations
    err = np.zeros((km, p))
    OK = OP[:, 0]
    M, R = MR_Matrix_P_NR(y, Nnod)

    for i in range(1, p, 1):
        error = 1.0
        k_iter += 1
        while (error >= tol) and (k_iter < km):
            K = K_Matrix_P_NR(y, dt, p, OK, Nnod)
            RR = dt * R + np.dot(M, OP[:, i - 1])
            FF = M + dt*K
            # --- BCs ----
            FF[Nnod - 1, :], FF[:, Nnod - 1], FF[Nnod - 1, Nnod - 1] = 0.0, 0.0, 1.0
            RR[Nnod - 1] = 0.0
            # --- solution ----
            OK22 = np.dot(np.linalg.inv(FF), RR)
            # --- error ----
            error = np.linalg.norm(OK22 - OK) / np.linalg.norm(OK22)
            err[k_iter, i] = error
            OK = OK22
        OP[:, i] = OK[:]

    return err, k_iter, OP