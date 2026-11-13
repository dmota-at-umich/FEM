# -*- coding: utf-8 -*-
"""
Oscilaciones de una viga simplemente apoyada con CALFEM para Python
===================================================================

1) Análisis modal: frecuencias y modos naturales (problema de valores propios
   K·φ = ω²·M·φ) y comparación con la solución analítica de Euler-Bernoulli:
        ω_n = (nπ/L)² · sqrt(E·I / m)
2) Respuesta en el tiempo: la viga se deforma estáticamente con una carga
   puntual P en el centro y se suelta en t = 0 (vibración libre). Se integra
   con el método de Newmark (aceleración media) y se compara con la serie
   analítica por superposición modal.

"""

import numpy as np
import matplotlib.pyplot as plt
import calfem.core as cfc

# ---------------------------------------------------------------------------
# 1. Datos del problema (unidades SI)
# ---------------------------------------------------------------------------
L = 10.0          # longitud de la viga [m]
E = 210e9         # módulo de Young [Pa] (acero)
rho = 7850.0      # densidad [kg/m³]
b, h = 0.20, 0.30 # sección rectangular: base y altura [m]
A = b * h                 # área [m²]
I = b * h**3 / 12.0       # momento de inercia [m⁴]
m = rho * A               # masa por unidad de longitud [kg/m]

nel = 20          # número de elementos (usar un número par: nodo central)
P = 10e3          # carga puntual central para la deformación inicial [N]
zeta = 0.0        # amortiguamiento modal (0 = sin amortiguamiento)
n_modos = 6       # modos a reportar

# ---------------------------------------------------------------------------
# 2. Malla: nodos, grados de libertad (u, v, θ) y topología
# ---------------------------------------------------------------------------
nnod = nel + 1
x = np.linspace(0.0, L, nnod)
ndof = 3 * nnod

# Dof[i] = [u, v, θ] del nodo i (numeración base 1, como en CALFEM)
Dof = np.arange(1, ndof + 1).reshape(nnod, 3)

Edof = np.zeros((nel, 6), dtype=int)
Ex = np.zeros((nel, 2))
Ey = np.zeros((nel, 2))
for e in range(nel):
    Edof[e, :] = np.hstack((Dof[e], Dof[e + 1]))
    Ex[e, :] = [x[e], x[e + 1]]
    Ey[e, :] = [0.0, 0.0]

# ---------------------------------------------------------------------------
# 3. Ensamble de rigidez K (beam2e de CALFEM) y masa M (masa consistente)
# ---------------------------------------------------------------------------
def beam2_masa(ex, ey, m):
    """Matriz de masa consistente de un elemento viga 2D (u, v, θ por nodo),
    equivalente a la Me de beam2d de CALFEM para MATLAB."""
    dx, dy = ex[1] - ex[0], ey[1] - ey[0]
    Le = np.hypot(dx, dy)
    c, s = dx / Le, dy / Le
    Mle = m * Le / 420.0 * np.array([
        [140,      0,          0,  70,        0,          0],
        [  0,    156,     22*Le,    0,       54,     -13*Le],
        [  0,  22*Le,  4*Le**2,    0,    13*Le,  -3*Le**2],
        [ 70,      0,          0, 140,        0,          0],
        [  0,     54,     13*Le,    0,      156,     -22*Le],
        [  0, -13*Le, -3*Le**2,    0,   -22*Le,   4*Le**2]])
    G = np.array([[ c, s, 0,  0, 0, 0],
                  [-s, c, 0,  0, 0, 0],
                  [ 0, 0, 1,  0, 0, 0],
                  [ 0, 0, 0,  c, s, 0],
                  [ 0, 0, 0, -s, c, 0],
                  [ 0, 0, 0,  0, 0, 1]])
    return G.T @ Mle @ G


K = np.zeros((ndof, ndof))
M = np.zeros((ndof, ndof))
ep = [E, A, I]
for e in range(nel):
    Ke = cfc.beam2e(Ex[e], Ey[e], ep)
    Me = beam2_masa(Ex[e], Ey[e], m)
    cfc.assem(Edof[e], K, Ke)
    cfc.assem(Edof[e], M, Me)

# ---------------------------------------------------------------------------
# 4. Condiciones de frontera: apoyo fijo (u, v) a la izquierda,
#    apoyo deslizante (v) a la derecha
# ---------------------------------------------------------------------------
bc = np.array([Dof[0, 0], Dof[0, 1], Dof[-1, 1]])   # base 1

# ---------------------------------------------------------------------------
# 5. Análisis modal
# ---------------------------------------------------------------------------
lam, X = cfc.eigen(K, M, bc)          # lam = ω², X = modos (normalizados a M)
lam = np.real(lam)
X = np.real(X)

# Separar modos de flexión de los axiales: un modo es de flexión si
# su energía cinética está dominada por los dof verticales (v, θ)
dof_u = Dof[:, 0] - 1
es_flexion = []
for k in range(X.shape[1]):
    phi = X[:, k]
    total = phi @ M @ phi
    axial = phi[dof_u] @ M[np.ix_(dof_u, dof_u)] @ phi[dof_u]
    es_flexion.append(axial / total < 0.5)
idx_flex = np.where(es_flexion)[0]

omega_fem = np.sqrt(lam[idx_flex])
modos = X[:, idx_flex]

n = np.arange(1, n_modos + 1)
omega_an = (n * np.pi / L) ** 2 * np.sqrt(E * I / m)

print("Modos de flexión de la viga simplemente apoyada")
print(f"{'n':>3} {'f FEM [Hz]':>12} {'f analítica [Hz]':>17} {'error [%]':>10}")
for k in range(n_modos):
    f_fem = omega_fem[k] / (2 * np.pi)
    f_an = omega_an[k] / (2 * np.pi)
    print(f"{k+1:>3} {f_fem:12.4f} {f_an:17.4f} {100*(f_fem-f_an)/f_an:10.4f}")

# Gráfica de los primeros 4 modos (desplazamiento vertical v)
dof_v = Dof[:, 1] - 1
xx = np.linspace(0, L, 200)
fig1, axs = plt.subplots(4, 1, figsize=(8, 8), sharex=True)
for k in range(4):
    v = modos[dof_v, k]
    v = v / np.max(np.abs(v))
    # alinear signo con el modo analítico sin(kπx/L)
    ref = np.sin((k + 1) * np.pi * x / L)
    v *= np.sign(v @ ref)
    axs[k].plot(xx, np.sin((k + 1) * np.pi * xx / L), "k--", lw=1, label="Analítico")
    axs[k].plot(x, v, "o-", ms=3, label="CALFEM")
    axs[k].axhline(0, color="gray", lw=0.5)
    axs[k].set_ylabel(f"Modo {k+1}")
    axs[k].set_title(f"f = {omega_fem[k]/(2*np.pi):.2f} Hz", fontsize=9, loc="right")
axs[0].legend(loc="upper right", fontsize=8)
axs[-1].set_xlabel("x [m]")
fig1.suptitle("Modos de vibración: viga simplemente apoyada")
fig1.tight_layout()
fig1.savefig("modos_viga.png", dpi=150)

# ---------------------------------------------------------------------------
# 6. Condición inicial: deflexión estática bajo P en el centro
# ---------------------------------------------------------------------------
nodo_c = nel // 2
f = np.zeros((ndof, 1))
f[Dof[nodo_c, 1] - 1, 0] = -P
a0, r0 = cfc.solveq(K, f, bc)
a0 = np.asarray(a0).ravel()

w_estatica = P * L**3 / (48 * E * I)
print(f"\nDeflexión estática al centro: FEM = {abs(a0[Dof[nodo_c,1]-1])*1e3:.4f} mm,"
      f" analítica PL³/48EI = {w_estatica*1e3:.4f} mm")

# ---------------------------------------------------------------------------
# 7. Integración en el tiempo (Newmark, aceleración media) — vibración libre
# ---------------------------------------------------------------------------
libres = np.setdiff1d(np.arange(ndof), bc - 1)
Kf = K[np.ix_(libres, libres)]
Mf = M[np.ix_(libres, libres)]

# Amortiguamiento de Rayleigh C = αM + βK ajustado a ζ en los modos 1 y 2
w1, w2 = omega_fem[0], omega_fem[1]
alfa = 2 * zeta * w1 * w2 / (w1 + w2)
beta = 2 * zeta / (w1 + w2)
Cf = alfa * Mf + beta * Kf

T1 = 2 * np.pi / omega_fem[0]
dt = T1 / 200.0                 # paso: 200 pasos por periodo fundamental
t_final = 3 * T1
nt = int(np.ceil(t_final / dt)) + 1
t = np.arange(nt) * dt

gam, bet = 0.5, 0.25
d = a0[libres].copy()
v = np.zeros_like(d)
acc = np.linalg.solve(Mf, -Cf @ v - Kf @ d)

Keff = Kf + gam / (bet * dt) * Cf + 1.0 / (bet * dt**2) * Mf
Keff_inv = np.linalg.inv(Keff)

i_c = np.where(libres == Dof[nodo_c, 1] - 1)[0][0]
w_c = np.zeros(nt)
w_c[0] = d[i_c]
for k in range(1, nt):
    rhs = (Mf @ (d / (bet * dt**2) + v / (bet * dt) + (0.5 / bet - 1) * acc)
           + Cf @ (gam / (bet * dt) * d + (gam / bet - 1) * v
                   + dt * (gam / (2 * bet) - 1) * acc))
    d_new = Keff_inv @ rhs
    acc_new = (d_new - d) / (bet * dt**2) - v / (bet * dt) - (0.5 / bet - 1) * acc
    v = v + dt * ((1 - gam) * acc + gam * acc_new)
    d, acc = d_new, acc_new
    w_c[k] = d[i_c]

# Solución analítica (sin amortiguamiento) por serie modal:
# w(L/2, t) = -Σ 2PL³/(π⁴EI n⁴) · sin²(nπ/2) · cos(ω_n t)
w_an = np.zeros(nt)
for nn in range(1, 60, 2):
    wn = (nn * np.pi / L) ** 2 * np.sqrt(E * I / m)
    w_an += -2 * P * L**3 / (np.pi**4 * E * I * nn**4) * np.cos(wn * t)

fig2, ax = plt.subplots(figsize=(9, 4))
ax.plot(t, w_c * 1e3, label="CALFEM + Newmark")
if zeta == 0:
    ax.plot(t, w_an * 1e3, "k--", lw=1, label="Analítica (serie modal)")
ax.set_xlabel("t [s]")
ax.set_ylabel("Desplazamiento al centro [mm]")
ax.set_title(f"Vibración libre tras soltar la carga central (ζ = {zeta})")
ax.grid(alpha=0.3)
ax.legend()
fig2.tight_layout()
fig2.savefig("respuesta_tiempo_viga.png", dpi=150)

plt.show()
