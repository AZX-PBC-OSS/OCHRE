import math

import numba as nb


MIN_HUM_RATIO = 1e-7
TRIPLE_POINT_WATER_SI = 0.01
PSYCHROLIB_TOLERANCE = 0.001
MAX_ITER_COUNT = 100
ZERO_CELSIUS_AS_KELVIN = 273.15
R_DA_SI = 287.042
FREEZING_POINT_WATER_SI = 0.0


@nb.njit(cache=True)
def get_t_kelvin_from_t_celsius(t_celsius):
    return t_celsius + ZERO_CELSIUS_AS_KELVIN


@nb.njit(cache=True)
def get_sat_vap_pressure(t_dry_bulb):
    if t_dry_bulb < -100.0 or t_dry_bulb > 200.0:
        raise ValueError("Dry bulb temperature must be in range [-100, 200]°C")

    t = get_t_kelvin_from_t_celsius(t_dry_bulb)

    if t_dry_bulb <= TRIPLE_POINT_WATER_SI:
        ln_pws = (
            -5.6745359e03 / t
            + 6.3925247
            - 9.677843e-03 * t
            + 6.2215701e-07 * t * t
            + 2.0747825e-09 * t * t * t
            - 9.484024e-13 * t * t * t * t
            + 4.1635019 * math.log(t)
        )
    else:
        ln_pws = (
            -5.8002206e03 / t
            + 1.3914993
            - 4.8640239e-02 * t
            + 4.1764768e-05 * t * t
            - 1.4452093e-08 * t * t * t
            + 6.5459673 * math.log(t)
        )

    return math.exp(ln_pws)


@nb.njit(cache=True)
def get_vap_pres_from_hum_ratio(hum_ratio, pressure):
    if hum_ratio < 0.0:
        raise ValueError("Humidity ratio is negative")

    bounded_hum_ratio = max(hum_ratio, MIN_HUM_RATIO)
    return pressure * bounded_hum_ratio / (0.621945 + bounded_hum_ratio)


@nb.njit(cache=True)
def get_hum_ratio_from_vap_pres(vap_pres, pressure):
    if vap_pres < 0.0:
        raise ValueError("Partial pressure of water vapor in moist air cannot be negative")

    hum_ratio = 0.621945 * vap_pres / (pressure - vap_pres)
    return max(hum_ratio, MIN_HUM_RATIO)


@nb.njit(cache=True)
def get_hum_ratio_from_rel_hum(t_dry_bulb, rel_hum, pressure):
    if rel_hum < 0.0 or rel_hum > 1.0:
        raise ValueError("Relative humidity is outside range [0, 1]")

    vap_pres = rel_hum * get_sat_vap_pressure(t_dry_bulb)
    return get_hum_ratio_from_vap_pres(vap_pres, pressure)


@nb.njit(cache=True)
def get_rel_hum_from_hum_ratio(t_dry_bulb, hum_ratio, pressure):
    if hum_ratio < 0.0:
        raise ValueError("Humidity ratio cannot be negative")

    vap_pres = get_vap_pres_from_hum_ratio(hum_ratio, pressure)
    sat_vap_pres = get_sat_vap_pressure(t_dry_bulb)
    return vap_pres / sat_vap_pres


@nb.njit(cache=True)
def get_moist_air_enthalpy(t_dry_bulb, hum_ratio):
    if hum_ratio < 0.0:
        raise ValueError("Humidity ratio is negative")
    bounded_hum_ratio = max(hum_ratio, MIN_HUM_RATIO)
    return (1.006 * t_dry_bulb + bounded_hum_ratio * (2501.0 + 1.86 * t_dry_bulb)) * 1000.0


@nb.njit(cache=True)
def get_moist_air_volume(t_dry_bulb, hum_ratio, pressure):
    if hum_ratio < 0.0:
        raise ValueError("Humidity ratio is negative")
    bounded_hum_ratio = max(hum_ratio, MIN_HUM_RATIO)
    return R_DA_SI * get_t_kelvin_from_t_celsius(t_dry_bulb) * (1.0 + 1.607858 * bounded_hum_ratio) / pressure


@nb.njit(cache=True)
def get_moist_air_density(t_dry_bulb, hum_ratio, pressure):
    if hum_ratio < 0.0:
        raise ValueError("Humidity ratio is negative")
    bounded_hum_ratio = max(hum_ratio, MIN_HUM_RATIO)
    moist_air_volume = get_moist_air_volume(t_dry_bulb, bounded_hum_ratio, pressure)
    return (1.0 + bounded_hum_ratio) / moist_air_volume


@nb.njit(cache=True)
def _d_ln_pws(t_dry_bulb):
    t = get_t_kelvin_from_t_celsius(t_dry_bulb)
    if t_dry_bulb <= TRIPLE_POINT_WATER_SI:
        return (
            5.6745359e03 / (t * t)
            - 9.677843e-03
            + 2.0 * 6.2215701e-07 * t
            + 3.0 * 2.0747825e-09 * t * t
            - 4.0 * 9.484024e-13 * t * t * t
            + 4.1635019 / t
        )
    return (
        5.8002206e03 / (t * t)
        - 4.8640239e-02
        + 2.0 * 4.1764768e-05 * t
        - 3.0 * 1.4452093e-08 * t * t
        + 6.5459673 / t
    )


@nb.njit(cache=True)
def get_dew_point_from_vap_pressure(t_dry_bulb, vap_pres):
    min_bound = -100.0
    max_bound = 200.0

    if vap_pres < get_sat_vap_pressure(min_bound) or vap_pres > get_sat_vap_pressure(max_bound):
        raise ValueError("Partial pressure of water vapor is outside range of validity of equations")

    t_dew_point = t_dry_bulb
    ln_vp = math.log(vap_pres)
    index = 1

    while True:
        t_dew_point_iter = t_dew_point
        ln_vp_iter = math.log(get_sat_vap_pressure(t_dew_point_iter))
        d_ln_vp = _d_ln_pws(t_dew_point_iter)
        t_dew_point = t_dew_point_iter - (ln_vp_iter - ln_vp) / d_ln_vp
        t_dew_point = max(t_dew_point, min_bound)
        t_dew_point = min(t_dew_point, max_bound)

        if math.fabs(t_dew_point - t_dew_point_iter) <= PSYCHROLIB_TOLERANCE:
            break

        if index > MAX_ITER_COUNT:
            raise ValueError("Convergence not reached in GetTDewPointFromVapPres. Stopping.")

        index += 1

    return min(t_dew_point, t_dry_bulb)


@nb.njit(cache=True)
def get_dew_point_from_hum_ratio(t_dry_bulb, hum_ratio, pressure):
    if hum_ratio < 0.0:
        raise ValueError("Humidity ratio cannot be negative")

    vap_pres = get_vap_pres_from_hum_ratio(hum_ratio, pressure)
    return get_dew_point_from_vap_pressure(t_dry_bulb, vap_pres)


@nb.njit(cache=True)
def get_hum_ratio_from_t_wet_bulb(t_dry_bulb, t_wet_bulb, pressure):
    if t_wet_bulb > t_dry_bulb:
        raise ValueError("Wet bulb temperature is above dry bulb temperature")

    w_sat = get_hum_ratio_from_rel_hum(t_wet_bulb, 1.0, pressure)
    if t_wet_bulb >= FREEZING_POINT_WATER_SI:
        hum_ratio = ((2501.0 - 2.326 * t_wet_bulb) * w_sat - 1.006 * (t_dry_bulb - t_wet_bulb)) / (
            2501.0 + 1.86 * t_dry_bulb - 4.186 * t_wet_bulb
        )
    else:
        hum_ratio = ((2830.0 - 0.24 * t_wet_bulb) * w_sat - 1.006 * (t_dry_bulb - t_wet_bulb)) / (
            2830.0 + 1.86 * t_dry_bulb - 2.1 * t_wet_bulb
        )
    return max(hum_ratio, MIN_HUM_RATIO)


@nb.njit(cache=True)
def get_t_wet_bulb_from_hum_ratio(t_dry_bulb, hum_ratio, pressure):
    if hum_ratio < 0.0:
        raise ValueError("Humidity ratio cannot be negative")
    bounded_hum_ratio = max(hum_ratio, MIN_HUM_RATIO)

    t_dew_point = get_dew_point_from_hum_ratio(t_dry_bulb, bounded_hum_ratio, pressure)

    t_wet_bulb_sup = t_dry_bulb
    t_wet_bulb_inf = t_dew_point
    t_wet_bulb = (t_wet_bulb_inf + t_wet_bulb_sup) / 2.0

    index = 1
    while (t_wet_bulb_sup - t_wet_bulb_inf) > PSYCHROLIB_TOLERANCE:
        w_star = get_hum_ratio_from_t_wet_bulb(t_dry_bulb, t_wet_bulb, pressure)
        if w_star > bounded_hum_ratio:
            t_wet_bulb_sup = t_wet_bulb
        else:
            t_wet_bulb_inf = t_wet_bulb

        t_wet_bulb = (t_wet_bulb_sup + t_wet_bulb_inf) / 2.0

        if index >= MAX_ITER_COUNT:
            raise ValueError("Convergence not reached in GetTWetBulbFromHumRatio. Stopping.")
        index += 1
    return t_wet_bulb


@nb.njit(cache=True)
def _update_humidity(w, latent_gains_w, humidity_cap_mult, t_indoor, pressure):
    w_new = w + latent_gains_w / humidity_cap_mult
    if w_new < 0.0:
        w_new = 0.0

    rh = get_rel_hum_from_hum_ratio(t_indoor, w_new, pressure)
    if rh > 1.0:
        rh = 1.0
        w_new = get_hum_ratio_from_rel_hum(t_indoor, rh, pressure)

    density = get_moist_air_density(t_indoor, w_new, pressure)
    wet_bulb = get_t_wet_bulb_from_hum_ratio(t_indoor, w_new, pressure)
    return w_new, rh, density, wet_bulb


@nb.njit(cache=True)
def _iterate(x0, f0, x1, f1, x2, f2, icount, tol_rel=1e-5, small=1e-9):
    dx = 0.1

    if (abs(x0 - x1) < tol_rel * max(abs(x0), small) and icount != 1) or f0 == 0.0:
        return x0, True, x1, f1, x2, f2

    if icount == 1:
        mode = 1
    elif icount == 2:
        mode = 2
    else:
        mode = 3

    x_new = 0.0

    if mode == 3:
        if x0 == x1:
            x1 = x2
            f1 = f2
            mode = 2
        elif x0 == x2:
            mode = 2
        else:
            c = ((f2 - f0) / (x2 - x0) - (f1 - f0) / (x1 - x0)) / (x2 - x1)
            b = (f1 - f0) / (x1 - x0) - (x1 + x0) * c
            a = f0 - (b + c * x0) * x0

            if abs(c) < small:
                mode = 2
            elif abs((a + (b + c * x1) * x1 - f1) / f1) > small:
                mode = 2
            else:
                d = b * b - 4.0 * a * c
                if d < 0.0:
                    mode = 2
                else:
                    if d > 0.0:
                        x_new = (-b + math.sqrt(d)) / (2.0 * c)
                        x_other = -x_new - b / c
                        if abs(x_new - x0) > abs(x_other - x0):
                            x_new = x_other
                    else:
                        x_new = -b / (2.0 * c)

                    if f1 * f0 > 0.0 and f2 * f0 > 0.0:
                        if abs(f2) > abs(f1):
                            x2 = x1
                            f2 = f1
                    else:
                        if f2 * f0 > 0.0:
                            x2 = x1
                            f2 = f1
                    x1 = x0
                    f1 = f0

    if mode == 2:
        m = (f1 - f0) / (x1 - x0)
        if m == 0.0:
            mode = 1
        else:
            x_new = x0 - f0 / m
            x2 = x1
            f2 = f1
            x1 = x0
            f1 = f0

    if mode == 1:
        if abs(x0) > small:
            x_new = x0 * (1.0 + dx)
        else:
            x_new = dx
        x2 = x1
        f2 = f1
        x1 = x0
        f1 = f0

    return x_new, False, x1, f1, x2, f2


@nb.njit(cache=True)
def _calculate_shr_jit(db_in, w_in, p_kpa, q_kw, flow, ao):
    p_pa = p_kpa * 1000.0
    density = get_moist_air_density(db_in, w_in, p_pa)
    mfr = density * flow
    bf = math.exp(-ao / mfr) if mfr > 0.0 else 0.0

    h_in = get_moist_air_enthalpy(db_in, w_in)
    d_h = q_kw * 1000.0 / mfr if mfr > 0.0 else 0.0
    h_adp = h_in - d_h / (1.0 - bf)

    t_adp = get_dew_point_from_hum_ratio(db_in, w_in, p_pa)
    t_adp_1 = t_adp
    t_adp_2 = t_adp
    w_adp = get_hum_ratio_from_rel_hum(t_adp, 1.0, p_pa)
    error = h_adp - get_moist_air_enthalpy(t_adp, w_adp)
    error1 = error
    error2 = error

    cvg = False
    for i in range(1, 51):
        w_adp = get_hum_ratio_from_rel_hum(t_adp, 1.0, p_pa)
        error = h_adp - get_moist_air_enthalpy(t_adp, w_adp)
        t_adp, cvg, t_adp_1, error1, t_adp_2, error2 = _iterate(
            t_adp, error, t_adp_1, error1, t_adp_2, error2, i
        )
        if cvg:
            break

    h_tin_wadp = get_moist_air_enthalpy(db_in, w_adp)
    denom = h_in - h_adp
    if denom != 0.0:
        shr = min((h_tin_wadp - h_adp) / denom, 1.0)
    else:
        shr = 1.0
    return shr, cvg
