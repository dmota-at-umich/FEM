# -*- coding: utf-8 -*-
"""
Reddy, Introduction to the Finite Element Method (2a ed.), sección 3.3.3
Ejemplo 3.4: pila de concreto de un puente (barra de sección variable)
====================================================================
Probado para calfem-python 3.6.13 y Python 3.9.

Problema (x medido hacia abajo desde la parte superior, 0 <= x <= 2 m):

    -d/dx [ E·A(x) du/dx ] = f(x),       A(x) = (1 + x)/4  [m²]
                                          f(x) = 6.25(1 + x) [kN/m] (peso propio)
    [E·A du/dx]_{x=0} = -5 kN  (carga del puente: 20 kN/m² × 0.5 m × 0.5 m)
    u(2) = 0                    (empotrada en la base)

    E = 28×10⁶ kN/m²

Elementos lineales (Reddy, ec. 3.125):
    K^e = E/(4h_e)·[1 + (x_e + x_{e+1})/2]·[[1, -1], [-1, 1]]
    f^e = 6.25·h_e/2·( {1, 1} + 1/3·{x_{e+1} + 2x_e, 2x_{e+1} + x_e} )

K^e es exactamente la rigidez de un resorte con k = E·A_prom/h_e, porque A(x)
es lineal; por eso se calcula con spring1e de CALFEM. El vector f^e (carga
distribuida lineal) no tiene función propia en CALFEM y se calcula aquí.

Solución exacta (Reddy, ec. 3.127):
    u(x) = (1/E)·[56.25 - 6.25(1 + x)² - 7.5·ln((1 + x)/3)]

Requisitos:  pip install calfem-python numpy matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
import calfem.core as cfc

# ---------------------------------------------------------------------------
# 1. Datos del problema (unidades: kN, m)
# ---------------------------------------------------------------------------
L = 2.0           # altura de la pila [m]
E = 28e6          # módulo de elasticidad del concreto [kN/m²]
F_sup = 5.0       # carga puntual en la parte superior [kN] (positiva hacia abajo)


def area(x):
    """Área de la sección transversal A(x) = (0.5 + 0.5x)·0.5 [m²]."""
    return 0.25 * (1.0 + x)


def u_exacta(x):
    return (56.25 - 6.25 * (1 + x)**2 - 7.5 * np.log((1 + x) / 3.0)) / E


def sigma_exacta(x):
    # σ = E du/dx
    return -12.5 * (1 + x) - 7.5 / (1 + x)


# ---------------------------------------------------------------------------
# 2. Funciones de elemento
# ---------------------------------------------------------------------------
def barra_rigidez(ex):
    """K^e de Reddy (3.125) con spring1e de CALFEM: k = E·A_prom/h."""
    h = ex[1] - ex[0]
    A_prom = 0.5 * (area(ex[0]) + area(ex[1]))   # exacto para A(x) lineal
    return cfc.spring1e(E * A_prom / h)


def barra_carga(ex):
    """f^e de Reddy (3.125) para f(x) = 6.25(1 + x) con funciones lineales."""
    xa, xb = ex
    h = xb - xa
    return 6.25 * h / 2.0 * (np.array([1.0, 1.0])
                             + np.array([xb + 2 * xa, 2 * xb + xa]) / 3.0)


# ---------------------------------------------------------------------------
# 3. Solución por elementos finitos para una malla de nel elementos
# ---------------------------------------------------------------------------
def resolver(nel, mostrar=False):
    nnod = nel + 1
    x = np.linspace(0.0, L, nnod)

    Edof = np.array([[e + 1, e + 2] for e in range(nel)])   # base 1
    Ex = np.array([[x[e], x[e + 1]] for e in range(nel)])

    K = np.zeros((nnod, nnod))
    f = np.zeros((nnod, 1))
    for e in range(nel):
        Ke = barra_rigidez(Ex[e])
        fe = barra_carga(Ex[e]).reshape(2, 1)
        cfc.assem(Edof[e], K, Ke, f, fe)

    if mostrar:
        np.set_printoptions(precision=3, suppress=True)
        print("Matriz ensamblada K / E:")
        print(K / E)
        print("Vector de cargas distribuidas f [kN]:")
        print(f.ravel())

    # Carga puntual en la parte superior (Q_1^1 = 5 kN)
    f[0, 0] += F_sup

    # Condición de frontera: u = 0 en la base (último nodo)
    bc = np.array([nnod])
    a, r = cfc.solveq(K, f, bc)
    a = np.asarray(a).ravel()
    r = np.asarray(r).ravel()

    # Esfuerzo (constante en cada elemento): σ = E (u2 - u1)/h
    sig = np.array([E * (a[e + 1] - a[e]) / (x[e + 1] - x[e]) for e in range(nel)])
    return x, a, r, sig


# ---------------------------------------------------------------------------
# 4. Malla de dos elementos (la del libro)
# ---------------------------------------------------------------------------
print("=== Malla de 2 elementos (h1 = h2 = 1 m) ===")
x2, a2, r2, s2 = resolver(2, mostrar=True)
print(f"\nU1 = {a2[0]:.4e} m   (Reddy: 2.111e-06 m,  exacta u(0) = {u_exacta(0):.4e} m)")
print(f"U2 = {a2[1]:.4e} m   (Reddy: 1.238e-06 m,  exacta u(1) = {u_exacta(1):.4e} m)")
print(f"Reacción en la base Q_2^2 = {r2[-1]:.3f} kN   (Reddy: -30 kN)")
print(f"Esfuerzo en la base σ = Q/A = {r2[-1]/area(L):.3f} kN/m²   (Reddy: -40 kN/m²)")
print(f"Esfuerzo FEM del elemento inferior = {s2[-1]:.3f} kN/m²")

# ---------------------------------------------------------------------------
# 5. Convergencia al refinar la malla
# ---------------------------------------------------------------------------
print("\n=== Convergencia ===")
print(f"{'nel':>4} {'u(0) [m]':>12} {'error u(0) [%]':>15} {'u(1) [m]':>12}"
      f" {'error u(1) [%]':>15} {'reacción [kN]':>14}")
for nel in [2, 4, 8, 16, 32]:
    x, a, r, sig = resolver(nel)
    i1 = np.argmin(abs(x - 1.0))
    e0 = 100 * (a[0] - u_exacta(0)) / u_exacta(0)
    e1 = 100 * (a[i1] - u_exacta(1)) / u_exacta(1)
    print(f"{nel:>4} {a[0]:12.4e} {e0:15.4f} {a[i1]:12.4e} {e1:15.4f} {r[-1]:14.3f}")

# ---------------------------------------------------------------------------
# 6. Gráficas: desplazamiento y esfuerzo (2 y 4 elementos vs. exacta)
# ---------------------------------------------------------------------------
xx = np.linspace(0, L, 300)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

ax1.plot(u_exacta(xx) * 1e6, xx, "k-", lw=1.2, label="Exacta (3.127)")
for nel, mk in [(2, "o--"), (4, "s--")]:
    x, a, r, sig = resolver(nel)
    ax1.plot(a * 1e6, x, mk, ms=5, label=f"CALFEM, {nel} elementos")
ax1.set_xlabel("u(x) [×10⁻⁶ m]")
ax1.set_ylabel("x [m] (hacia abajo)")
ax1.invert_yaxis()
ax1.set_title("Desplazamiento axial")
ax1.grid(alpha=0.3)
ax1.legend(fontsize=8)

ax2.plot(sigma_exacta(xx), xx, "k-", lw=1.2, label="Exacta")
for nel in [2, 4]:
    x, a, r, sig = resolver(nel)
    xs = np.repeat(x, 2)[1:-1]          # escalones: σ constante por elemento
    ax2.plot(np.repeat(sig, 2), xs, "-", label=f"CALFEM, {nel} elementos")
ax2.set_xlabel("σ(x) [kN/m²]")
ax2.invert_yaxis()
ax2.set_title("Esfuerzo axial (compresión < 0)")
ax2.grid(alpha=0.3)
ax2.legend(fontsize=8)

fig.suptitle("Reddy, Ejemplo 3.4: pila de concreto")
fig.tight_layout()
fig.savefig("pila_concreto.png", dpi=150)

plt.show()
