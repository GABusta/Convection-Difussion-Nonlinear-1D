import numpy as np

def Exe3(tp, M, R, K, Nnod, dt):
    OP = np.zeros((Nnod, tp))
    OP[0:Nnod-1, 0] = 1.0
    for i in range(1, Nnod-1, 1):
        RR = dt * R + np.dot(M, OP[:, i - 1])
        FF = M + dt * K
        FF[Nnod - 1, :], FF[:, Nnod - 1] = 0.0, 0.0
        FF[Nnod - 1, Nnod - 1] = 1.0

        OK = np.dot(np.linalg.inv(FF), RR)
        OP[:, i] = OK[:]
    return OP