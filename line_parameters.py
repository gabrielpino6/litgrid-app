"""
Compute distributed RLCG matrices for overhead transmission lines
from conductor data and tower geometry.

Uses Carson's equations (series impedance) and the potential coefficient
method with image charges (shunt capacitance). Ground wires are eliminated
via Kron reduction.

Similar to MATLAB Simscape Electrical / Power Line Parameters block.
"""

import numpy as np

MU_0 = 4 * np.pi * 1e-7   # H/m
EPS_0 = 8.854187817e-12    # F/m


def compute_line_parameters(conductors, phase_idx, ground_idx, freq, rho_earth):
    """
    Compute 3x3 phase-domain R, L, C, G matrices for a transmission line.

    Parameters
    ----------
    conductors : list of dict
        Each conductor has:
            'r_dc'     : DC resistance at 20 °C (ohm/km)
            'diameter'  : outer diameter (mm)
            'x'        : horizontal position (m)
            'y'        : height above ground (m)
            'gmr'      : geometric mean radius (mm), optional.
                         If not given, computed as 0.7788 * radius (solid approx.)
    phase_idx : list of int
        Indices into `conductors` for the three phase conductors (A, B, C).
    ground_idx : list of int
        Indices into `conductors` for ground/shield wires. Can be empty.
    freq : float
        System frequency (Hz).
    rho_earth : float
        Earth resistivity (ohm·m).

    Returns
    -------
    R : ndarray (3, 3) — resistance matrix (ohm/km)
    L : ndarray (3, 3) — inductance matrix (mH/km)
    C : ndarray (3, 3) — capacitance matrix (nF/km)
    G : ndarray (3, 3) — conductance matrix (uS/km), diagonal, small
    """
    omega = 2 * np.pi * freq
    n = len(conductors)
    all_idx = phase_idx + ground_idx

    # --- Extract geometry and conductor properties ---
    x_pos = np.array([conductors[i]['x'] for i in all_idx])
    y_pos = np.array([conductors[i]['y'] for i in all_idx])
    r_dc = np.array([conductors[i]['r_dc'] for i in all_idx])  # ohm/km
    radius = np.array([conductors[i]['diameter'] / 2 for i in all_idx])  # mm

    gmr = np.zeros(n)
    for k, i in enumerate(all_idx):
        if 'gmr' in conductors[i] and conductors[i]['gmr'] is not None:
            gmr[k] = conductors[i]['gmr'] * 1e-3  # mm -> m
        else:
            gmr[k] = 0.7788 * radius[k] * 1e-3  # mm -> m

    radius_m = radius * 1e-3  # mm -> m
    n_all = len(all_idx)
    n_phase = len(phase_idx)

    # --- Carson's earth return depth ---
    D_e = 658.5 * np.sqrt(rho_earth / freq)  # m

    # --- Carson's earth return resistance (ohm/m) ---
    R_earth = omega * MU_0 / (8 * np.pi)  # ohm/m
    R_earth_km = R_earth * 1e3  # ohm/km

    # --- Series impedance matrix Z (ohm/km) for all conductors ---
    Z_full = np.zeros((n_all, n_all), dtype=complex)
    for i in range(n_all):
        for j in range(n_all):
            if i == j:
                Z_full[i, i] = (r_dc[i] + R_earth_km) + \
                    1j * omega * MU_0 / (2 * np.pi) * np.log(D_e / gmr[i]) * 1e3
            else:
                d_ij = np.sqrt((x_pos[i] - x_pos[j])**2 +
                               (y_pos[i] - y_pos[j])**2)
                Z_full[i, j] = R_earth_km + \
                    1j * omega * MU_0 / (2 * np.pi) * np.log(D_e / d_ij) * 1e3

    # --- Potential coefficient matrix P (m/F per km) for all conductors ---
    P_full = np.zeros((n_all, n_all))
    coeff = 1 / (2 * np.pi * EPS_0)
    for i in range(n_all):
        for j in range(n_all):
            if i == j:
                P_full[i, i] = coeff * np.log(2 * y_pos[i] / radius_m[i])
            else:
                d_ij = np.sqrt((x_pos[i] - x_pos[j])**2 +
                               (y_pos[i] - y_pos[j])**2)
                D_ij_image = np.sqrt((x_pos[i] - x_pos[j])**2 +
                                     (y_pos[i] + y_pos[j])**2)
                P_full[i, j] = coeff * np.log(D_ij_image / d_ij)

    # --- Kron reduction to eliminate ground wires ---
    if len(ground_idx) > 0:
        pp = slice(0, n_phase)
        gg = slice(n_phase, n_all)

        Z_pp = Z_full[pp, pp]
        Z_pg = Z_full[pp, gg]
        Z_gp = Z_full[gg, pp]
        Z_gg = Z_full[gg, gg]
        Z_phase = Z_pp - Z_pg @ np.linalg.inv(Z_gg) @ Z_gp

        P_pp = P_full[pp, pp]
        P_pg = P_full[pp, gg]
        P_gp = P_full[gg, pp]
        P_gg = P_full[gg, gg]
        P_phase = P_pp - P_pg @ np.linalg.inv(P_gg) @ P_gp
    else:
        Z_phase = Z_full
        P_phase = P_full

    # --- Extract R and L from Z ---
    R = Z_phase.real                       # ohm/km
    L = Z_phase.imag / omega * 1e3         # mH/km (H/km -> mH/km)

    # --- Capacitance from potential coefficients ---
    C_matrix = np.linalg.inv(P_phase)      # F/m
    C_matrix = C_matrix * 1e3 * 1e9        # F/m -> nF/km

    # --- Conductance (small, diagonal — corona/leakage) ---
    G = np.eye(n_phase) * 0.002            # uS/km

    return R, L, C_matrix, G
