# -------------------------------------------------------------
# ---------              author: GABusta              ---------
# ----------  " Nonlinear Convection - difussion " ------------
# -------------------------------------------------------------
import numpy as np
from MRK_Matrix import Matrix
from SolutionPicard import Picard
from SolutionNR import NR
from SolutionExe3 import Exe3
from Graphics import plotting

N, L, = 10, 1.0               # Nº , [m]
y = np.linspace(0.0, L, N+1)
dy, dt, Nnod = L/N, 0.1, N+1  # [m], [s], Nº of nodes
tp, km, tol = 20, 100, 1E-10   # time steps, Nº, Tolerance

# ------- Exe. 2 - PICARD ---------------
err1, k_iter1, OP1 = Picard(y, dt, tp, km, tol, Nnod)
print("example 1 - Done !")

# ------- Exe. 2 - Newton-Raphson -------
err2, k_iter2, OP2 = NR(y, dt, tp, km, tol, Nnod)
print("example 2 - Done !")

# ------- Exe. 3 ------------------------
M, R, K = Matrix(y, Nnod)
OP3 = Exe3(tp, M, R, K, Nnod, dt)
print("example 3 - Done !")

# ------- Plot's ------------------------
a = plotting(OP1, OP2, OP3, tp)



