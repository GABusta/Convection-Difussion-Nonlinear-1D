import numpy as np
import sympy as sym
#print(np.dot(A, np.transpose(B)))

r = sym.Symbol("r")
h1, h2 = 0.5 * (1 - r), 0.5 * (1 + r)
H = [h1, h2]

x = H[0] * 0 + H[1] * 0.1
dx_dr = x.diff(r)
Je = dx_dr
mm = [[H[0] * H[0] * Je, H[0] * H[1] * Je], [H[1] * H[0] * Je, H[1] * H[1] * Je]]
m1 = mm[0][0].integrate((r, -1, 1))
print(m1)