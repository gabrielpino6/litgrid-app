"""
Generate the data files plotted in 06-sweeps.tex.

Two parametric sweeps for the 110 kV asymmetric single-circuit line of the
worked example:

  * sweep_S.dat       -- spread vs |S| at fixed length ell = 25 km
  * sweep_length.dat  -- spread vs ell at fixed S = 80 + j10 MVA

The apparent-power direction in the |S| sweep is held parallel to
80 + j10 MVA (operating power factor of the example), so the |S| axis
is the magnitude along that direction.
"""
import sys
from pathlib import Path

import numpy as np
from scipy import linalg

# Make the line_parameters module reachable.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from line_parameters import compute_line_parameters  # noqa: E402

CONDUCTORS = [
    {"r_dc": 0.1897, "diameter": 17.1,  "x": -4.0, "y": 20.0},
    {"r_dc": 0.1897, "diameter": 17.1,  "x":  0.0, "y": 20.0},
    {"r_dc": 0.1897, "diameter": 17.1,  "x":  4.0, "y": 23.0},
    {"r_dc": 2.53,   "diameter": 11.11, "x":  0.0, "y": 25.0},
]
FREQ = 50.0
RHO  = 1000.0
V_LL = 110e3
V_S  = V_LL / np.sqrt(3)


def line_matrices():
    R, L, C, G = compute_line_parameters(
        CONDUCTORS, [0, 1, 2], [3], FREQ, RHO
    )
    omega = 2 * np.pi * FREQ
    Z = R + 1j * omega * L * 1e-3
    Y = (G * 1e-6) + 1j * omega * C * 1e-9
    F = np.block([[np.zeros((3, 3)), -Z], [-Y, np.zeros((3, 3))]])
    return F


def phase_currents(F, ell, S):
    M = linalg.expm(F * ell)
    A_m = M[:3, :3]
    B_m = M[:3, 3:]
    B_inv = np.linalg.inv(B_m)
    a = np.exp(1j * 2 * np.pi / 3)
    u = np.array([1.0, a ** 2, a])
    p_vec = B_inv @ u
    q_vec = B_inv @ A_m @ u
    alpha = u @ np.conj(p_vec)
    beta  = u @ np.conj(q_vec)
    W = (S + V_S ** 2 * beta) / (V_S * alpha)
    V_r = np.abs(W)
    delta = -np.angle(W)
    I0 = B_inv @ (V_r * np.exp(1j * delta) * u - V_S * A_m @ u)
    return np.abs(I0)


def write_dat(path, columns, header):
    arr = np.column_stack(columns)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(" ".join(header) + "\n")
        np.savetxt(fh, arr, fmt="%.6g")


def main():
    F = line_matrices()
    data_dir = Path(__file__).resolve().parent / "data"
    data_dir.mkdir(exist_ok=True)

    # === Sweep 1: |S| at ell = 25 km, arg(S) = arg(80 + j10) ===
    ell_fixed = 25.0
    S_unit = (80.0 + 1j * 10.0) / np.abs(80.0 + 1j * 10.0)
    S_mag = np.linspace(0.0, 200.0, 81)  # MVA
    spread = np.empty_like(S_mag)
    I_phase = np.empty((3, S_mag.size))
    for i, s in enumerate(S_mag):
        S = s * 1e6 * S_unit
        mag = phase_currents(F, ell_fixed, S)
        spread[i] = mag.max() - mag.min()
        I_phase[:, i] = mag
    write_dat(
        data_dir / "sweep_S.dat",
        [S_mag, spread, I_phase[0], I_phase[1], I_phase[2]],
        ["S_MVA", "spread_A", "IA_A", "IB_A", "IC_A"],
    )

    # === Sweep 2: ell at S = 80 + j10 MVA ===
    S_fixed = (80.0 + 1j * 10.0) * 1e6
    ell_sweep = np.linspace(1.0, 100.0, 100)  # km
    spread_l = np.empty_like(ell_sweep)
    I_phase_l = np.empty((3, ell_sweep.size))
    for i, ell in enumerate(ell_sweep):
        mag = phase_currents(F, ell, S_fixed)
        spread_l[i] = mag.max() - mag.min()
        I_phase_l[:, i] = mag
    write_dat(
        data_dir / "sweep_length.dat",
        [ell_sweep, spread_l, I_phase_l[0], I_phase_l[1], I_phase_l[2]],
        ["length_km", "spread_A", "IA_A", "IB_A", "IC_A"],
    )

    # Anchor at S = 80 + j10 MVA, ell = 25 km
    anchor_S = phase_currents(F, ell_fixed, S_fixed)
    print(f"anchor |S|=80.62 MVA @ ell=25 km: "
          f"spread = {anchor_S.max() - anchor_S.min():.4f} A")
    anchor_l = phase_currents(F, 25.0, S_fixed)
    print(f"sweep_length at ell=25 km: spread = "
          f"{anchor_l.max() - anchor_l.min():.4f} A")


if __name__ == "__main__":
    main()
