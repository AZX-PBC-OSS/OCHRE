import math
import unittest

import numpy as np
import psychrolib

from ochre.utils.equipment import _calculate_shr_legacy, iterate
from ochre.utils.psychrolib_jit import (
    _calculate_shr_jit,
    _iterate,
    _update_humidity,
    get_dew_point_from_hum_ratio,
    get_hum_ratio_from_rel_hum,
    get_moist_air_density,
    get_moist_air_enthalpy,
    get_rel_hum_from_hum_ratio,
    get_t_wet_bulb_from_hum_ratio,
)


psychrolib.SetUnitSystem(psychrolib.SI)


def _assert_rel_close(test_case, actual, expected, rtol=1e-6):
    denom = max(abs(expected), 1e-10)
    test_case.assertLess(abs(actual - expected) / denom, rtol)


class PsychrolibJitParityTestCase(unittest.TestCase):
    def test_formula_parity_dense_grid(self):
        t_grid = np.arange(-20.0, 51.0, 1.0)
        w_grid = np.arange(0.001, 0.0251, 0.002)
        p_grid = np.arange(80000.0, 105001.0, 5000.0)
        rh_grid = np.arange(0.0, 1.0001, 1.0 / 12.0)

        for t_db in t_grid:
            for p in p_grid:
                for w in w_grid:
                    jit_h = get_moist_air_enthalpy(t_db, w)
                    py_h = psychrolib.GetMoistAirEnthalpy(t_db, w)
                    _assert_rel_close(self, jit_h, py_h)

                    jit_density = get_moist_air_density(t_db, w, p)
                    py_density = psychrolib.GetMoistAirDensity(t_db, w, p)
                    _assert_rel_close(self, jit_density, py_density)

                    jit_rh = get_rel_hum_from_hum_ratio(t_db, w, p)
                    py_rh = psychrolib.GetRelHumFromHumRatio(t_db, w, p)
                    _assert_rel_close(self, jit_rh, py_rh)

                    jit_dp = get_dew_point_from_hum_ratio(t_db, w, p)
                    py_dp = psychrolib.GetTDewPointFromHumRatio(t_db, w, p)
                    self.assertLess(abs(jit_dp - py_dp), 0.001)

                for rh in rh_grid:
                    jit_w = get_hum_ratio_from_rel_hum(t_db, rh, p)
                    py_w = psychrolib.GetHumRatioFromRelHum(t_db, rh, p)
                    _assert_rel_close(self, jit_w, py_w)

    def test_wet_bulb_parity_operating_range(self):
        t_grid = np.linspace(-10.0, 45.0, 12)
        w_grid = np.linspace(0.001, 0.025, 13)
        p_grid = np.linspace(80000.0, 105000.0, 6)

        first = (
            float(t_grid[0]),
            float(w_grid[0]),
            float(p_grid[0]),
        )
        get_t_wet_bulb_from_hum_ratio(*first)
        self.assertTrue(len(get_t_wet_bulb_from_hum_ratio.nopython_signatures) > 0)

        for t_db in t_grid:
            for p in p_grid:
                for w in w_grid:
                    py_wb = psychrolib.GetTWetBulbFromHumRatio(float(t_db), float(w), float(p))
                    jit_wb = get_t_wet_bulb_from_hum_ratio(float(t_db), float(w), float(p))
                    self.assertLess(abs(jit_wb - py_wb), 0.01)

    def test_iterate_exact_parity_from_solver_trace(self):
        rng = np.random.default_rng(42)
        trace = []

        while len(trace) < 1000:
            db_in = float(rng.uniform(15.0, 35.0))
            w_in = float(rng.uniform(0.002, 0.02))
            p = float(rng.uniform(90.0, 103.0))
            q = float(rng.uniform(0.5, 15.0))
            flow = float(rng.uniform(0.05, 1.0))
            ao = float(rng.uniform(0.05, 3.0))

            mfr = flow * psychrolib.GetMoistAirDensity(db_in, w_in, p * 1000.0)
            bf = math.exp(-ao / mfr) if mfr > 0.0 else 0.0
            h_in = psychrolib.GetMoistAirEnthalpy(db_in, w_in)
            d_h = q * 1000.0 / mfr if mfr > 0.0 else 0.0
            h_adp = h_in - d_h / (1.0 - bf)

            t_adp = psychrolib.GetTDewPointFromHumRatio(db_in, w_in, p * 1000.0)
            t_adp_1 = t_adp
            t_adp_2 = t_adp
            w_adp = psychrolib.GetHumRatioFromRelHum(t_adp, 1.0, p * 1000.0)
            error = h_adp - psychrolib.GetMoistAirEnthalpy(t_adp, w_adp)
            error1 = error
            error2 = error

            for i in range(1, 51):
                w_adp = psychrolib.GetHumRatioFromRelHum(t_adp, 1.0, p * 1000.0)
                error = h_adp - psychrolib.GetMoistAirEnthalpy(t_adp, w_adp)
                trace.append((t_adp, error, t_adp_1, error1, t_adp_2, error2, i))
                t_adp, cvg, t_adp_1, error1, t_adp_2, error2 = iterate(
                    t_adp, error, t_adp_1, error1, t_adp_2, error2, i
                )
                if cvg or len(trace) >= 1000:
                    break

        for call in trace[:1000]:
            py_out = iterate(*call)
            jit_out = _iterate(*call)
            self.assertEqual(py_out, jit_out)

    def test_calculate_shr_jit_parity_and_nopython(self):
        rng = np.random.default_rng(42)
        test_inputs = []
        max_attempts = 20000
        attempts = 0
        while len(test_inputs) < 1000 and attempts < max_attempts:
            attempts += 1
            args = (
                float(rng.uniform(18.0, 33.0)),
                float(rng.uniform(0.003, 0.018)),
                float(rng.uniform(95.0, 102.0)),
                float(rng.uniform(1.0, 12.0)),
                float(rng.uniform(0.1, 0.8)),
                float(rng.uniform(0.1, 2.5)),
            )
            try:
                _calculate_shr_legacy(*args)
            except Exception:
                continue
            test_inputs.append(args)

        self.assertEqual(len(test_inputs), 1000)

        first = test_inputs[0]
        _calculate_shr_jit(*first)
        self.assertTrue(len(_calculate_shr_jit.nopython_signatures) > 0)

        for args in test_inputs:
            shr_jit, cvg_jit = _calculate_shr_jit(*args)
            shr_py, cvg_py = _calculate_shr_legacy(*args)
            denom = max(abs(shr_py), 1e-10)
            self.assertLess(abs(shr_jit - shr_py) / denom, 1e-5)
            self.assertEqual(cvg_jit, cvg_py)

    def test_update_humidity_compiles_nopython(self):
        _update_humidity(0.008, 0.0002, 15.0, 22.0, 101325.0)
        self.assertTrue(len(_update_humidity.nopython_signatures) > 0)


if __name__ == "__main__":
    unittest.main()
