"""Pre-compile all Numba JIT functions and populate the bytecode cache.

Run once after install (or after upgrading ochre/numba) to eliminate the
one-time ~1s compilation penalty on first simulation run:

    ochre-warmup        # via entry point
    python -m ochre.utils.warmup_jit   # direct invocation

Subsequent runs load pre-compiled bytecode from __pycache__/*.nbi/*.nbc.
"""

import numpy as np


def warmup():
    """Trigger Numba compilation for every @nb.njit function in OCHRE."""
    # --- psychrolib_jit.py (20 functions) ---
    from ochre.utils.psychrolib_jit import (
        get_t_kelvin_from_t_celsius,
        get_sat_vap_pressure,
        get_vap_pres_from_hum_ratio,
        get_hum_ratio_from_vap_pres,
        get_hum_ratio_from_rel_hum,
        get_rel_hum_from_hum_ratio,
        get_moist_air_enthalpy,
        get_moist_air_volume,
        get_moist_air_density,
        _d_ln_pws,
        get_dew_point_from_vap_pressure,
        get_dew_point_from_hum_ratio,
        get_hum_ratio_from_t_wet_bulb,
        get_t_wet_bulb_from_hum_ratio,
        _update_humidity,
        _iterate,
        _calculate_shr_jit,
    )

    t_db, w, p, rh = 20.0, 0.01, 101325.0, 0.5

    get_t_kelvin_from_t_celsius(t_db)
    get_sat_vap_pressure(t_db)
    vp = get_vap_pres_from_hum_ratio(w, p)
    get_hum_ratio_from_vap_pres(vp, p)
    get_hum_ratio_from_rel_hum(t_db, rh, p)
    get_rel_hum_from_hum_ratio(t_db, w, p)
    get_moist_air_enthalpy(t_db, w)
    get_moist_air_volume(t_db, w, p)
    get_moist_air_density(t_db, w, p)
    _d_ln_pws(t_db)
    get_dew_point_from_vap_pressure(t_db, vp)
    get_dew_point_from_hum_ratio(t_db, w, p)
    get_hum_ratio_from_t_wet_bulb(t_db, 15.0, p)
    get_t_wet_bulb_from_hum_ratio(t_db, w, p)
    _update_humidity(w, 0.001, 15.0, t_db, p)
    _iterate(1.0, 0.1, 0.9, -0.05, 0.8, 0.02, 3)
    _calculate_shr_jit(t_db, w, 101.325, 5.0, 0.5, 0.1)

    # --- Envelope.py (5 functions) ---
    from ochre.Models.Envelope import (
        _infiltration_ashrae,
        _natural_ventilation,
        _ventilation_flows_and_gain,
        _solve_interior_radiation,
        _solve_exterior_radiation,
    )

    _infiltration_ashrae(5.0, 3.0, 0.1, 0.02, 0.01, 0.5, 0.67)
    _natural_ventilation(22.0, 30.0, 22.78, 0.01, 3.0, 1.0, 0.1, 0.01, 200.0, 10000.0, 0.000278, 0.012)
    _ventilation_flows_and_gain(0.01, 0.0, 0.05, True, 0.7, 0.5, 1.2, 5.0, 500.0, True)

    n_surf = 6
    _solve_interior_radiation(
        np.ones(n_surf) * 1e-8,
        np.full(n_surf, 0.5),
        np.full(n_surf, 0.1),
        np.full(n_surf, 1.0 / n_surf),
        np.full(n_surf, 20.0),
        np.full(n_surf, 20.0),
        np.full(n_surf, 20.0),
        20.0,
        10,
        273.15,
    )

    n_bd = 3
    _solve_exterior_radiation(
        np.ones(n_bd) * 1e-8,
        np.full(n_bd, 0.5),
        np.full(n_bd, 0.5),
        np.full(n_bd, 0.1),
        np.full(n_bd, 15.0),
        np.full(n_bd, 15.0),
        np.full(n_bd, 15.0),
        np.zeros(n_bd),
        10.0,
        -10.0,
        True,
        5,
        273.15,
    )

    # --- HVAC.py (1 function) ---
    from ochre.Equipment.HVAC import _biquadratic

    coeffs_t = np.ones(6)
    coeffs_ff = np.ones(3)
    coeffs_plr = np.ones(3)
    _biquadratic(15.0, 35.0, coeffs_t, 1.0, coeffs_ff, 0.8, coeffs_plr, 5000.0, 10.0, 25.0, 20.0, 50.0, 0.5, 1.5, 0.1, 1.0)

    # --- Water.py (3 functions) ---
    from ochre.Models.Water import (
        _water_draw_general,
        _water_draw_2node,
        _inversion_mixing,
    )

    n_nodes = 6
    states = np.linspace(55.0, 45.0, n_nodes)
    vf = np.full(n_nodes, 1.0 / n_nodes)
    cap = np.full(n_nodes, 1000.0)
    q_out = np.zeros(n_nodes)
    _water_draw_general(states.copy(), vf, cap, 0.3, 15.0, 4186.0, 5.0, q_out)
    states_2 = np.array([55.0, 45.0])
    vf_2 = np.array([0.5, 0.5])
    cap_2 = np.array([1000.0, 1000.0])
    q_out_2 = np.zeros(2)
    _water_draw_2node(states_2, vf_2, cap_2, 0.3, 15.0, 4186.0, 5.0, 0.95, q_out_2)
    inv_states = np.array([45.0, 55.0, 50.0, 48.0, 46.0, 44.0])
    vol_cumsum = vf.cumsum()
    _inversion_mixing(inv_states, vf, vol_cumsum, cap)

    print("OCHRE JIT warmup complete: 29 functions compiled and cached.")


if __name__ == "__main__":
    warmup()
