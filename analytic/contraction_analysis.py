# Analysis of contaction space

from sympy import Symbol, Matrix
from sympy import diff, sin, cos

from sympy import init_session

init_session()

Ta = Symbol("T_a")
Tb = Symbol("T_c")
Tc = Symbol("T_b")
Td = Symbol("T_d")

fa = Symbol("f_a")
fb = Symbol("f_b")
fc = Symbol("f_c")
fd = Symbol("f_d")

Theta = Matrix([[Ta, Tb], [Tc, Td]])

f = Matrix([[fa, fb], [fc, fd]])

prod = Theta * f * Theta ** (-1)
prod_simple = simplify(prod)
