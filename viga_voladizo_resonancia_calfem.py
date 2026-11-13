# -*- coding: utf-8 -*-
"""
Oscilaciones de una viga en voladizo con CALFEM para Python
===========================================================
Probado para calfem-python 3.6.13 y Python 3.9.

1) Análisis modal: frecuencias y modos naturales (K·φ = ω²·M·φ) y comparación
   con la solución analítica de Euler-Bernoulli para el voladizo:
        ω_n = (β_n·L)² · sqrt(E·I / (m·L⁴))
   con β_n·L = 1.8751, 4.6941, 7.8548, 10.9955, ...  (raíces de 1 + cos·cosh = 0)
2) Respuesta en el tiempo: el extremo libre se deforma con una carga puntual P
   y se suelta en t = 0 (vibración libre). Se integra con Newmark (aceleración
   media) y se compara con la serie analítica por superposición modal.

"""

import numpy as np
import matplotlib.pyplot as plt
import calfem.core as cfc

# ---------------------------------------------------------------------------
# 1. Datos del problema (unidades SI)
# ---------------------------------------------------------------------------
L = 3.0           # longitud del voladizo [m]
E = 210e9         # módulo de Young [Pa] (acero)
rho = 7850.0      # densidad [kg/m³]
b, h = 0.10, 0.15 # sección rectangular: base y altura [m]
A = b * h                 # área [m²]
I = b * h**3 / 12.0       # momento de inercia [m⁴]
m = rho * A               # masa por unidad de longitud [kg/m]

nel = 20          # número de elementos
P = 5e3           # carga puntual en el extremo libre para la deformación inicial [N]
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
# 4. Condiciones de frontera: empotramiento en x = 0 (u = v = θ = 0),
#    extremo x = L libre
# ---------------------------------------------------------------------------
bc = np.array(Dof[0, :])   # base 1

# ---------------------------------------------------------------------------
# 5. Análisis modal
# ---------------------------------------------------------------------------
lam, X = cfc.eigen(K, M, bc)          # lam = ω², X = modos
lam = np.real(np.asarray(lam)).ravel()
X = np.real(np.asarray(X))

# Separar modos de flexión de los axiales (energía cinética en dof u)
dof_u = Dof[:, 0] - 1
es_flexion = []
for k in range(X.shape[1]):
    phi = X[:, k]
    total = phi @ M @ phi
    axial = phi[dof_u] @ M[np.ix_(dof_u, dof_u)] @ phi[dof_u]
    es_flexion.append(axial / total < 0.5)
idx_flex = np.where(es_flexion)[0]
idx_flex = idx_flex[np.argsort(lam[idx_flex])]

omega_fem = np.sqrt(lam[idx_flex])
modos = X[:, idx_flex]

# Solución analítica del voladizo
betaL = np.array([1.875104069, 4.694091133, 7.854757438,
                  10.99554073, 14.13716839, 17.27875953])
omega_an = betaL**2 * np.sqrt(E * I / (m * L**4))


def modo_analitico(xx, bL):
    """Forma modal analítica del voladizo empotrado-libre."""
    bx = bL * xx / L
    sig = (np.cosh(bL) + np.cos(bL)) / (np.sinh(bL) + np.sin(bL))
    return np.cosh(bx) - np.cos(bx) - sig * (np.sinh(bx) - np.sin(bx))


print("Modos de flexión de la viga en voladizo")
print(f"{'n':>3} {'f FEM [Hz]':>12} {'f analítica [Hz]':>17} {'error [%]':>10}")
for k in range(n_modos):
    f_fem = omega_fem[k] / (2 * np.pi)
    f_an = omega_an[k] / (2 * np.pi)
    print(f"{k+1:>3} {f_fem:12.4f} {f_an:17.4f} {100*(f_fem-f_an)/f_an:10.4f}")

# Gráfica de los primeros 4 modos (desplazamiento vertical v)
dof_v = Dof[:, 1] - 1
xx = np.linspace(0, L, 300)
fig1, axs = plt.subplots(4, 1, figsize=(8, 8), sharex=True)
for k in range(4):
    ref = modo_analitico(xx, betaL[k])
    ref = ref / np.max(np.abs(ref))
    v = modos[dof_v, k]
    v = v / np.max(np.abs(v))
    v *= np.sign(v[-1]) * np.sign(ref[-1])   # mismo signo en el extremo libre
    axs[k].plot(xx, ref, "k--", lw=1, label="Analítico")
    axs[k].plot(x, v, "o-", ms=3, label="CALFEM")
    axs[k].axhline(0, color="gray", lw=0.5)
    axs[k].set_ylabel(f"Modo {k+1}")
    axs[k].set_title(f"f = {omega_fem[k]/(2*np.pi):.2f} Hz", fontsize=9, loc="right")
axs[0].legend(loc="upper left", fontsize=8)
axs[-1].set_xlabel("x [m]")
fig1.suptitle("Modos de vibración: viga en voladizo")
fig1.tight_layout()
fig1.savefig("modos_voladizo.png", dpi=150)

# ---------------------------------------------------------------------------
# 6. Condición inicial: deflexión estática bajo P en el extremo libre
# ---------------------------------------------------------------------------
dof_punta = Dof[-1, 1] - 1          # dof vertical del extremo libre (base 0)
f = np.zeros((ndof, 1))
f[dof_punta, 0] = -P
a0, r0 = cfc.solveq(K, f, bc)
a0 = np.asarray(a0).ravel()

w_estatica = P * L**3 / (3 * E * I)
print(f"\nDeflexión estática en la punta: FEM = {abs(a0[dof_punta])*1e3:.4f} mm,"
      f" analítica PL³/3EI = {w_estatica*1e3:.4f} mm")

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
dt = T1 / 400.0                 # paso: 400 pasos por periodo fundamental
t_final = 3 * T1
nt = int(np.ceil(t_final / dt)) + 1
t = np.arange(nt) * dt

gam, bet = 0.5, 0.25
d = a0[libres].copy()
v = np.zeros_like(d)
acc = np.linalg.solve(Mf, -Cf @ v - Kf @ d)

Keff = Kf + gam / (bet * dt) * Cf + 1.0 / (bet * dt**2) * Mf
Keff_inv = np.linalg.inv(Keff)

i_p = np.where(libres == dof_punta)[0][0]
w_p = np.zeros(nt)
w_p[0] = d[i_p]
for k in range(1, nt):
    rhs = (Mf @ (d / (bet * dt**2) + v / (bet * dt) + (0.5 / bet - 1) * acc)
           + Cf @ (gam / (bet * dt) * d + (gam / bet - 1) * v
                   + dt * (gam / (2 * bet) - 1) * acc))
    d_new = Keff_inv @ rhs
    acc_new = (d_new - d) / (bet * dt**2) - v / (bet * dt) - (0.5 / bet - 1) * acc
    v = v + dt * ((1 - gam) * acc + gam * acc_new)
    d, acc = d_new, acc_new
    w_p[k] = d[i_p]


# Solución analítica (sin amortiguamiento) por superposición modal:
# w(x,t) = Σ q_n φ_n(x) cos(ω_n t),  q_n = ∫ w0 φ_n dx / ∫ φ_n² dx
def integrar(y, xs):
    return np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(xs))


xs = np.linspace(0, L, 4001)
w0 = -P * xs**2 * (3 * L - xs) / (6 * E * I)     # deflexión estática inicial
w_an = np.zeros(nt)
for k in range(len(betaL)):
    phi = modo_analitico(xs, betaL[k])
    q = integrar(w0 * phi, xs) / integrar(phi**2, xs)
    w_an += q * phi[-1] * np.cos(omega_an[k] * t)

fig2, ax = plt.subplots(figsize=(9, 4))
ax.plot(t, w_p * 1e3, label="CALFEM + Newmark")
if zeta == 0:
    ax.plot(t, w_an * 1e3, "k--", lw=1, label="Analítica (6 modos)")
ax.set_xlabel("t [s]")
ax.set_ylabel("Desplazamiento en la punta [mm]")
ax.set_title(f"Vibración libre del voladizo tras soltar la carga (ζ = {zeta})")
ax.grid(alpha=0.3)
ax.legend()
fig2.tight_layout()
fig2.savefig("respuesta_tiempo_voladizo.png", dpi=150)

# ---------------------------------------------------------------------------
# 8. Carga armónica en la punta: F(t) = F0·sin(Ω·t)
# ---------------------------------------------------------------------------
F0 = 1e3            # amplitud de la carga [N]
zeta_arm = 0.02     # amortiguamiento para esta sección (necesario en resonancia)
razones = [0.5, 1.0]   # Ω/ω1 a simular en el tiempo (1.0 = resonancia)

# Rayleigh ajustado a zeta_arm en los modos 1 y 2
a_R = 2 * zeta_arm * w1 * w2 / (w1 + w2)
b_R = 2 * zeta_arm / (w1 + w2)
Ch = a_R * Mf + b_R * Kf

# 8a. Respuesta en frecuencia (estado estacionario): (K - Ω²M + iΩC)·X = F
fF = np.zeros(len(libres))
fF[i_p] = F0
f_barrido = np.linspace(0.5, 1.15 * omega_fem[3] / (2 * np.pi), 4000)  # [Hz]
X_fem = np.zeros(len(f_barrido))
for j, fj in enumerate(f_barrido):
    Om = 2 * np.pi * fj
    Xc = np.linalg.solve(Kf - Om**2 * Mf + 1j * Om * Ch, fF)
    X_fem[j] = abs(Xc[i_p])

# Analítica: superposición modal con modos exactos normalizados a la masa
# X(Ω) = Σ φn(L)² F0 / (ωn² - Ω² + 2i ζn ωn Ω),  ζn = a/(2ωn) + b·ωn/2
X_an = np.zeros(len(f_barrido), dtype=complex)
for k in range(len(betaL)):
    phi = modo_analitico(xs, betaL[k])
    phi_L = phi[-1] / np.sqrt(m * integrar(phi**2, xs))
    wn = omega_an[k]
    zn = a_R / (2 * wn) + b_R * wn / 2
    Om = 2 * np.pi * f_barrido
    X_an += phi_L**2 * F0 / (wn**2 - Om**2 + 2j * zn * wn * Om)

print(f"\nCarga armónica F0 = {F0:.0f} N en la punta, ζ = {zeta_arm}")
print(f"Respuesta cuasiestática (Ω→0): FEM = {X_fem[0]*1e3:.4f} mm,"
      f" F0·L³/3EI = {F0*L**3/(3*E*I)*1e3:.4f} mm")
for k in range(4):
    j = np.argmin(abs(f_barrido - omega_fem[k] / (2 * np.pi)))
    print(f"  Resonancia {k+1}: f = {omega_fem[k]/(2*np.pi):8.2f} Hz,"
          f" amplitud en la punta ≈ {X_fem[j]*1e3:9.4f} mm")

fig3, ax = plt.subplots(figsize=(9, 4))
ax.semilogy(f_barrido, X_fem * 1e3, label="CALFEM (solución directa)")
ax.semilogy(f_barrido, np.abs(X_an) * 1e3, "k--", lw=1, label="Analítica (6 modos)")
for k in range(4):
    ax.axvline(omega_fem[k] / (2 * np.pi), color="gray", lw=0.5, ls=":")
ax.set_xlabel("Frecuencia de excitación f [Hz]")
ax.set_ylabel("Amplitud en la punta [mm]")
ax.set_title(f"Respuesta en frecuencia del voladizo (F0 = {F0:.0f} N, ζ = {zeta_arm})")
ax.grid(alpha=0.3, which="both")
ax.legend()
fig3.tight_layout()
fig3.savefig("respuesta_frecuencia_voladizo.png", dpi=150)


# 8b. Respuesta en el tiempo desde el reposo (Newmark con carga)
def newmark_forzado(Mf, Cf, Kf, fvec, Om, dt, nt):
    """Newmark (aceleración media) con F(t) = fvec·sin(Ω t), desde el reposo."""
    gam, bet = 0.5, 0.25
    d = np.zeros(len(fvec))
    v = np.zeros(len(fvec))
    acc = np.linalg.solve(Mf, fvec * np.sin(0.0))
    Keff_inv = np.linalg.inv(Kf + gam / (bet * dt) * Cf + 1.0 / (bet * dt**2) * Mf)
    hist = np.zeros(nt)
    for k in range(1, nt):
        tk = k * dt
        rhs = (fvec * np.sin(Om * tk)
               + Mf @ (d / (bet * dt**2) + v / (bet * dt) + (0.5 / bet - 1) * acc)
               + Cf @ (gam / (bet * dt) * d + (gam / bet - 1) * v
                       + dt * (gam / (2 * bet) - 1) * acc))
        d_new = Keff_inv @ rhs
        acc_new = (d_new - d) / (bet * dt**2) - v / (bet * dt) - (0.5 / bet - 1) * acc
        v = v + dt * ((1 - gam) * acc + gam * acc_new)
        d, acc = d_new, acc_new
        hist[k] = d[i_p]
    return hist


n_per = 40                       # periodos fundamentales a simular
dt_h = T1 / 100.0
nt_h = int(n_per * T1 / dt_h) + 1
t_h = np.arange(nt_h) * dt_h

fig4, axs4 = plt.subplots(len(razones), 1, figsize=(9, 3.2 * len(razones)), sharex=True)
axs4 = np.atleast_1d(axs4)
for ax, r in zip(axs4, razones):
    Om = r * omega_fem[0]
    w_h = newmark_forzado(Mf, Ch, Kf, fF, Om, dt_h, nt_h)
    X_ee = abs(np.linalg.solve(Kf - Om**2 * Mf + 1j * Om * Ch, fF)[i_p])
    ax.plot(t_h, w_h * 1e3, lw=0.8, label="CALFEM + Newmark")
    ax.axhline(X_ee * 1e3, color="k", ls="--", lw=0.8, label="± amplitud estacionaria")
    ax.axhline(-X_ee * 1e3, color="k", ls="--", lw=0.8)
    ax.set_ylabel("w punta [mm]")
    ax.set_title(f"Ω = {r:.2f}·ω1  (f = {Om/(2*np.pi):.2f} Hz)", fontsize=10, loc="left")
    ax.grid(alpha=0.3)
axs4[0].legend(loc="upper right", fontsize=8)
axs4[-1].set_xlabel("t [s]")
fig4.suptitle(f"Carga armónica en la punta (F0 = {F0:.0f} N, ζ = {zeta_arm})")
fig4.tight_layout()
fig4.savefig("respuesta_armonica_voladizo.png", dpi=150)

plt.show()
