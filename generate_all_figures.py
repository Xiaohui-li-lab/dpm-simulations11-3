
#!/usr/bin/env python3
"""
generate_all_figures.py
Generate all 5 figures for the paper:
"Quantum chirality and the relationship among the speed of light, energy and mass"
Author: Xiaohui Li
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['figure.dpi'] = 150
OUTDIR = Path('figures')
OUTDIR.mkdir(exist_ok=True)

# ------------------------------------------------------------
# Figure 1: Chiral decomposition of a Dirac spinor
# ------------------------------------------------------------
def fig1_chiral_decomposition():
    fig, ax = plt.subplots(figsize=(5, 5))
    theta = np.linspace(0, 2 * np.pi, 400)
    r = 1.0
    ax.fill_between(r * np.cos(theta), r * np.sin(theta),
                    where=(np.sin(theta) >= 0), color='steelblue', alpha=0.35)
    ax.fill_between(r * np.cos(theta), r * np.sin(theta),
                    where=(np.sin(theta) < 0), color='firebrick', alpha=0.35)
    ax.plot(r * np.cos(theta), r * np.sin(theta), 'k-', lw=2.2)
    ax.plot([-r, r], [0, 0], 'k-', lw=2.2)
    ax.text(-0.45, 0.45, r'$\psi_L$', fontsize=30, color='steelblue', fontweight='bold')
    ax.text(0.25, -0.45, r'$\psi_R$', fontsize=30, color='firebrick', fontweight='bold')
    ax.annotate('', xy=(2.3, 0.6), xytext=(1.3, 0.6),
                arrowprops=dict(arrowstyle='->', lw=2.2, color='black'))
    ax.text(1.85, 0.78, r'$v \to c$', fontsize=14)
    ax.text(1.35, 0.28, 'helicity → chirality', fontsize=11)
    ax.set_xlim(-1.6, 2.6)
    ax.set_ylim(-1.3, 1.3)
    ax.set_aspect('equal')
    ax.axis('off')
    fig.tight_layout()
    fig.savefig(OUTDIR / 'fig1_chiral_decomposition.png')
    plt.close(fig)
    print("Saved fig1_chiral_decomposition.png")

# ------------------------------------------------------------
# Figure 2: Helicity-chirality mixing vs v/c
# ------------------------------------------------------------
def fig2_helicity_mixing():
    v = np.linspace(0, 1, 300)
    p_flip = 0.5 * (1 - v)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(v, p_flip, lw=2.8, color='crimson',
            label=r'$\mathcal{P}_{\rm flip}=\frac{1}{2}(1-v/c)$')
    ax.axhline(0, color='gray', ls='--', lw=1)
    ax.set_xlabel(r'$v/c$', fontsize=13)
    ax.set_ylabel(r'helicity–chirality mixing probability', fontsize=12)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.05, 0.55)
    ax.legend(loc='upper right', fontsize=12)
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    fig.savefig(OUTDIR / 'fig2_helicity_mixing.png')
    plt.close(fig)
    print("Saved fig2_helicity_mixing.png")

# ------------------------------------------------------------
# Figure 3: Mass-energy-chirality conversion triangle
# ------------------------------------------------------------
def fig3_emc2_triangle():
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.set_xlim(-0.8, 6.0)
    ax.set_ylim(-0.8, 5.5)
    ax.set_aspect('equal')
    ax.axis('off')

    # nodes
    ax.add_patch(plt.Circle((0, 0), 0.7, color='gold', ec='black', lw=2.5))
    ax.text(0, 0, r'$m$', ha='center', va='center', fontsize=20, fontweight='bold')
    ax.add_patch(plt.Circle((5, 0), 0.7, color='lightgreen', ec='black', lw=2.5))
    ax.text(5, 0, r'$E$', ha='center', va='center', fontsize=20, fontweight='bold')
    ax.add_patch(plt.Circle((2.5, 4.33), 0.9, color='orange', ec='black', lw=2.5))
    ax.text(2.5, 4.33, r'$\gamma_{h=\pm1}$', ha='center', va='center', fontsize=15)

    # arrows
    ax.annotate('', xy=(4.25, 0), xytext=(0.7, 0),
                arrowprops=dict(arrowstyle='->', lw=2.8, color='black'))
    ax.text(2.5, -0.65, r'$E=mc^2$', ha='center', fontsize=14)

    ax.annotate('', xy=(2.05, 3.75), xytext=(0.45, 0.55),
                arrowprops=dict(arrowstyle='->', lw=2.5, color='black'))
    ax.text(0.55, 2.15, 'annihilation', fontsize=12, rotation=60)

    ax.annotate('', xy=(4.55, 0.55), xytext=(2.95, 3.8),
                arrowprops=dict(arrowstyle='->', lw=2.5, color='black'))
    ax.text(4.25, 2.15, 'conversion', fontsize=12, rotation=-60)

    fig.tight_layout()
    fig.savefig(OUTDIR / 'fig3_emc2_triangle.png')
    plt.close(fig)
    print("Saved fig3_emc2_triangle.png")

# ------------------------------------------------------------
# Toy 2D chiral-gas simulation + Figures 4 and 5
# ------------------------------------------------------------
class ChiralGas:
    def __init__(self, N=800, L0=1.0, seed=42,
                 k_s=50.0, k_o=0.5, r0=0.05, rc=0.15, ra=0.02,
                 dt=1e-4, vthermal=0.85, delta0=1e-9):
        self.N = N
        self.L0 = L0
        self.k_s = k_s
        self.k_o = k_o
        self.r0 = r0
        self.rc = rc
        self.ra = ra
        self.dt = dt
        rng = np.random.default_rng(seed)

        nL = N // 2 + int(delta0 * N)
        nR = N - nL
        self.chi = np.concatenate([np.ones(nL), -np.ones(nR)])
        rng.shuffle(self.chi)

        self.pos = rng.random((N, 2)) * L0
        angle = rng.uniform(0, 2 * np.pi, N)
        speed = rng.normal(vthermal, 0.1, N)
        self.vel = np.column_stack([speed * np.cos(angle), speed * np.sin(angle)])
        self.t = 0.0
        self.history = []

    def forces(self):
        F = np.zeros_like(self.pos)
        for i in range(self.N):
            rij = self.pos - self.pos[i]
            r = np.linalg.norm(rij, axis=1)
            mask = (r > 0) & (r < self.rc)
            for j in np.where(mask)[0]:
                rhat = rij[j] / r[j]
                if self.chi[i] * self.chi[j] > 0:
                    F[i] += -self.k_s * (r[j] - self.r0) * rhat
                else:
                    F[i] += self.k_o / (r[j]**2 + 1e-8) * rhat
        return F

    def annihilate(self):
        alive = np.ones(self.N, dtype=bool)
        for i in range(self.N):
            if not alive[i]:
                continue
            rij = self.pos - self.pos[i]
            r = np.linalg.norm(rij, axis=1)
            mask = (r > 0) & (r < self.ra) & (self.chi * self.chi[i] < 0) & alive
            if mask.any():
                j = np.argmin(r + (1 - alive) * 1e6)
                alive[i] = False
                alive[j] = False
        if not alive.all():
            self.pos = self.pos[alive]
            self.vel = self.vel[alive]
            self.chi = self.chi[alive]
            self.N = len(self.chi)

    def step(self):
        F = self.forces()
        self.vel += 0.5 * F * self.dt
        self.pos += self.vel * self.dt
        self.vel += 0.5 * F * self.dt
        self.t += self.dt
        for dim in range(2):
            hit_low = self.pos[:, dim] < 0
            hit_high = self.pos[:, dim] > self.L0
            self.vel[hit_low | hit_high, dim] *= -1
            self.pos[hit_low, dim] = 0.0
            self.pos[hit_high, dim] = self.L0

    def record(self):
        I = np.mean((self.pos - self.pos.mean(axis=0))**2)
        delta = self.chi.mean() if len(self.chi) else 0.0
        E = 0.5 * np.sum(self.vel**2)
        self.history.append({
            't': self.t,
            'N': self.N,
            'a': np.sqrt(I) / (self.L0 / np.sqrt(12)),
            'delta': delta,
            'E': E
        })

    def phase_separation_metric(self):
        """Simple nearest-neighbor phase separation metric xi."""
        xi_vals = []
        for i in range(self.N):
            r = np.linalg.norm(self.pos - self.pos[i], axis=1)
            r[i] = np.inf
            nn = np.argmin(r)
            same = int(self.chi[i] * self.chi[nn] > 0)
            xi_vals.append(same)
        return np.mean(xi_vals)

    def run(self, nsteps=200_000, record_every=20_000):
        self.record()
        for step in range(1, nsteps + 1):
            self.step()
            if step % 5_000 == 0:
                self.annihilate()
            if step % record_every == 0:
                xi = self.phase_separation_metric()
                self.record()
                self.history[-1]['xi'] = xi
                print(f"step {step:>6}: t={self.t:.3e}, N={self.N:>4}, "
                      f"a={self.history[-1]['a']:.3f}, "
                      f"delta={self.history[-1]['delta']:.3e}, xi={xi:.3f}")
        return self.history

    def fig4_snapshot(self, fname='fig4_simulation_snapshot.png'):
        fig, ax = plt.subplots(figsize=(6.5, 6.5))
        colors = np.where(self.chi > 0, 'steelblue', 'firebrick')
        ax.scatter(self.pos[:, 0], self.pos[:, 1], c=colors, s=25, alpha=0.8, edgecolors='k', linewidths=0.2)
        ax.set_xlim(0, self.L0)
        ax.set_ylim(0, self.L0)
        ax.set_aspect('equal')
        ax.set_xlabel(r'$x/L_0$', fontsize=13)
        ax.set_ylabel(r'$y/L_0$', fontsize=13)
        ax.set_title('Chiral-gas snapshot: blue = left-chiral, red = right-chiral', fontsize=12)
        fig.tight_layout()
        fig.savefig(OUTDIR / fname)
        plt.close(fig)
        print(f"Saved {fname}")

def fig5_expansion_history(history, fname='fig5_expansion_history.png'):
    t = np.array([h['t'] for h in history])
    a = np.array([h['a'] for h in history])
    delta = np.array([h['delta'] for h in history])
    E = np.array([h['E'] for h in history])
    H = np.gradient(np.log(a), t)

    fig, axes = plt.subplots(3, 1, figsize=(7.5, 9), sharex=True)

    axes[0].plot(t, a, 'o-', color='navy', lw=1.5, markersize=5)
    axes[0].set_ylabel(r'expansion factor $a(t)$', fontsize=12)
    axes[0].grid(True, alpha=0.35)

    axes[1].plot(t[1:], H[1:] * 1e3, 's-', color='darkgreen', lw=1.5, markersize=5)
    axes[1].set_ylabel(r'expansion rate $H(t)\,(10^{-3})$', fontsize=12)
    axes[1].grid(True, alpha=0.35)

    axes[2].plot(t, delta, '^-', color='purple', lw=1.5, markersize=5)
    axes[2].set_ylabel(r'chiral asymmetry $\delta(t)$', fontsize=12)
    axes[2].set_xlabel(r'time $t$', fontsize=12)
    axes[2].grid(True, alpha=0.35)

    for ax, label in zip(axes, ['(a)', '(b)', '(c)']):
        ax.text(0.02, 0.85, label, transform=ax.transAxes, fontsize=14, fontweight='bold')

    fig.tight_layout()
    fig.savefig(OUTDIR / fname)
    plt.close(fig)
    print(f"Saved {fname}")

    # Save CSV
    np.savetxt(OUTDIR / 'expansion_history.csv',
               np.column_stack([t, a, H, delta, E / E[0]]),
               delimiter=',',
               header='t,a(t),H(t),delta(t),E/E0',
               comments='')
    print("Saved expansion_history.csv")

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
if __name__ == '__main__':
    print("=" * 60)
    print("Generating concept figures 1-3...")
    print("=" * 60)
    fig1_chiral_decomposition()
    fig2_helicity_mixing()
    fig3_emc2_triangle()

    print("\n" + "=" * 60)
    print("Running chiral-gas simulation (N=800, ~200k steps)...")
    print("=" * 60)
    gas = ChiralGas(N=800, seed=42)
    history = gas.run(nsteps=200_000, record_every=20_000)

    print("\n" + "=" * 60)
    print("Generating simulation figures 4-5...")
    print("=" * 60)
    gas.fig4_snapshot()
    fig5_expansion_history(history)

    print("\nAll figures and data saved to:", OUTDIR.resolve())
