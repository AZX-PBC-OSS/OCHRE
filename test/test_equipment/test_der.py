"""
DER (Distributed Energy Resource) sanity / integration tests.

Scenarios asserted:
  1. Clear sunny day, large PV + minimal load            -> reverse flow
  2. Large PV + battery (low SOC)                         -> surplus PV charges
     battery, net grid draw ~0
  3. Infinite-capacity battery in self-consumption        -> zero net load
  4. L2 11.5 kW BEV, charge-at-home, near-empty SOC       -> sensible peak load
"""

import unittest
import datetime as dt
import os
import numpy as np
import pandas as pd

from ochre.Equipment import PV, Battery, ElectricVehicle
from ochre.Equipment.PV import run_sam
from ochre.Simulator import KIND_EQUIPMENT, KIND_GENERATOR, KIND_BATTERY
from test.test_equipment import equip_init_args, start_time, duration, time_res


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sunny_pv_schedule(capacity_kw=10):
    """Synthetic PV schedule for a clear sunny day (sine-shaped 6am-6pm)."""
    times = pd.date_range(
        start_time, start_time + duration,
        freq=time_res, inclusive="left",
    )
    hours = (times - start_time).total_seconds() / 3600.0
    power = np.where(
        (hours >= 6) & (hours <= 18),
        -capacity_kw * np.sin(np.pi * (hours - 6) / 12),
        0.0,
    )
    return pd.DataFrame({"PV (kW)": power}, index=times)


def _make_ev_schedule(ambient_c=15):
    """Synthetic schedule with ambient temperature (required by EV)."""
    times = pd.date_range(
        start_time, start_time + duration,
        freq=time_res, inclusive="left",
    )
    return pd.DataFrame({"Ambient Dry Bulb (C)": [ambient_c] * len(times)}, index=times)


# ---------------------------------------------------------------------------
# Test 1 – PV reverse flow
# ---------------------------------------------------------------------------

class PVReverseFlowTestCase(unittest.TestCase):
    """Large PV system + minimal load -> net export (reverse flow)."""

    def setUp(self):
        args = equip_init_args.copy()
        args.update(
            {
                "capacity": 10,
                "tilt": 20,
                "azimuth": 180,
                "schedule": _make_sunny_pv_schedule(10),
            }
        )
        self.pv = PV(**args)

    def test_pv_generates_during_day(self):
        """PV power is negative (generating) during sun hours."""
        noon = start_time + dt.timedelta(hours=12)
        for _ in range(int(12 * 60)):
            self.pv.update()
        self.assertEqual(self.pv.current_time, noon)
        self.assertLess(self.pv.electric_kw, -5, "PV should generate > 5 kW at noon")
        self.assertLess(self.pv.electric_kw, 0, "PV power must be negative (export)")

    def test_reverse_flow_with_minimal_load(self):
        """When PV >> load, net power is negative (exporting to grid)."""
        for _ in range(int(12 * 60)):
            self.pv.update()
        pv_power = self.pv.electric_kw  # negative
        net = pv_power + 1.0  # +1 kW minimal load
        self.assertLess(net, 0, "Net power should be negative (reverse flow)")
        self.assertGreater(abs(pv_power) / 1.0, 3, "PV generation should dominate load")

    def test_pv_zero_at_night(self):
        """PV output is zero outside sun hours."""
        for _ in range(60):
            self.pv.update()
        self.assertAlmostEqual(self.pv.electric_kw, 0, msg="PV should be 0 at night")


# ---------------------------------------------------------------------------
# Test 2 – Battery self-consumption  (infinite capacity -> zero net load)
# ---------------------------------------------------------------------------

class BatteryZeroNetLoadTestCase(unittest.TestCase):
    """A very-large-capacity battery in self-consumption mode zeroes net load."""

    def setUp(self):
        args = equip_init_args.copy()
        args.update(
            {
                "capacity": 1000,  # kW — "infinite" power
                "capacity_kwh": 10000,  # kWh — "infinite" energy
                "soc_init": 0.5,
                "self_consumption_mode": True,
            }
        )
        self.battery = Battery(**args)

    def test_battery_kind_flags(self):
        self.assertTrue(self.battery._kind & KIND_EQUIPMENT)
        self.assertTrue(self.battery._kind & KIND_GENERATOR)
        self.assertTrue(self.battery._kind & KIND_BATTERY)

    def test_counter_import(self):
        """Battery discharges (-) to cancel import (+)."""
        self.battery.current_schedule["net_power"] = 5.0
        mode = self.battery.update_internal_control()
        self.assertEqual(mode, "On")
        self.assertLess(self.battery.power_setpoint, 0)
        self.assertAlmostEqual(self.battery.power_setpoint, -5.0, places=1)

    def test_counter_export(self):
        """Battery charges (+) to cancel export (-)."""
        self.battery.current_schedule["net_power"] = -5.0
        mode = self.battery.update_internal_control()
        self.assertEqual(mode, "On")
        self.assertGreater(self.battery.power_setpoint, 0)
        self.assertAlmostEqual(self.battery.power_setpoint, 5.0, places=1)

    def test_zeroes_net_load(self):
        """end-to-end: setpoint -> power matches -net_power."""
        self.battery.current_schedule["net_power"] = 3.0
        mode = self.battery.update_internal_control()
        self.battery.mode = mode
        self.battery.calculate_power_and_heat()
        self.assertAlmostEqual(self.battery.electric_kw, -3.0, places=1)

    def test_respects_import_limit(self):
        self.battery.import_limit = 2.0
        self.battery.current_schedule["net_power"] = 5.0
        self.battery.update_internal_control()
        # desired = max(min(5, 2), -0) = 2; setpoint = 2 - 5 = -3
        self.assertAlmostEqual(self.battery.power_setpoint, -3.0, places=1)

    def test_respects_export_limit(self):
        self.battery.export_limit = 2.0
        self.battery.current_schedule["net_power"] = -5.0
        self.battery.update_internal_control()
        # desired = max(min(-5, 0), -2) = -2; setpoint = -2 - (-5) = 3
        self.assertAlmostEqual(self.battery.power_setpoint, 3.0, places=1)

    def test_no_net_power_off(self):
        self.battery.current_schedule.pop("net_power", None)
        mode = self.battery.update_internal_control()
        self.assertEqual(mode, "Off")
        self.assertEqual(self.battery.power_setpoint, 0.0)

    def test_soc_max_blocks_charge(self):
        self.battery.soc = self.battery.soc_max
        self.battery.current_schedule["net_power"] = -10.0
        mode = self.battery.update_internal_control()
        self.assertEqual(mode, "Off")
        self.assertEqual(self.battery.power_setpoint, 0.0)

    def test_soc_min_blocks_discharge(self):
        self.battery.soc = self.battery.soc_min
        self.battery.current_schedule["net_power"] = 10.0
        mode = self.battery.update_internal_control()
        self.assertEqual(mode, "Off")
        self.assertEqual(self.battery.power_setpoint, 0.0)


# ---------------------------------------------------------------------------
# Test 3 – EV Level 2 charging (near-empty SOC)
# ---------------------------------------------------------------------------

class EVLevel2ChargingTestCase(unittest.TestCase):
    """L2 11.5 kW BEV with charge-at-home archetype yields sensible peak load."""

    def setUp(self):
        np.random.seed(42)
        args = equip_init_args.copy()
        args.update(
            {
                "vehicle_type": "BEV",
                "charging_level": "Level2",
                "range": 300,  # mi -> vehicle_num=4 -> 11.5 kW max
                "schedule": _make_ev_schedule(20),
            }
        )
        self.ev = ElectricVehicle(**args)

    def test_max_power_is_level2_rating(self):
        self.assertAlmostEqual(self.ev.max_power, 11.5, places=1)

    def test_capacity_from_range(self):
        expected = 300 / (1000.0 / 325.0)
        self.assertAlmostEqual(self.ev.capacity, expected, places=0)

    def test_charges_at_max_power_when_near_empty(self):
        self.ev.event_start = self.ev.current_time
        self.ev.event_end = self.ev.current_time + dt.timedelta(hours=4)
        self.ev.soc = 0.1
        self.ev.in_event = True
        mode = self.ev.update_internal_control()
        self.assertEqual(mode, "On")
        self.assertAlmostEqual(self.ev.p_setpoint, 11.5, places=1)

    def test_calculate_power_produces_sensible_load(self):
        self.ev.event_start = self.ev.current_time
        self.ev.event_end = self.ev.current_time + dt.timedelta(hours=4)
        self.ev.soc = 0.1
        self.ev.in_event = True
        self.ev.mode = "On"
        self.ev.p_setpoint = self.ev.max_power
        self.ev.calculate_power_and_heat()
        self.assertAlmostEqual(self.ev.electric_kw, 11.5, places=1)
        self.assertGreater(self.ev.electric_kw, 10)

    def test_soc_increases_during_charging(self):
        self.ev.event_start = self.ev.current_time
        self.ev.event_end = self.ev.current_time + dt.timedelta(hours=4)
        self.ev.soc = 0.1
        self.ev.in_event = True
        self.ev.mode = "On"
        self.ev.p_setpoint = self.ev.max_power
        soc_before = self.ev.soc
        self.ev.calculate_power_and_heat()
        self.assertGreater(self.ev.next_soc, soc_before)

    def test_power_limited_when_soc_full(self):
        self.ev.event_start = self.ev.current_time
        self.ev.event_end = self.ev.current_time + dt.timedelta(hours=4)
        self.ev.soc = 0.999
        self.ev.in_event = True
        self.ev.mode = "On"
        self.ev.p_setpoint = self.ev.max_power
        self.ev.calculate_power_and_heat()
        self.assertLess(self.ev.electric_kw, 11.5)


# ---------------------------------------------------------------------------
# Combined DER integration scenarios
# ---------------------------------------------------------------------------

class DERIntegrationTestCase(unittest.TestCase):
    """End-to-end: PV + Battery self-consumption in sequential update."""

    def setUp(self):
        pv_args = equip_init_args.copy()
        pv_args.update(
            {
                "capacity": 10,
                "tilt": 20,
                "azimuth": 180,
                "schedule": _make_sunny_pv_schedule(10),
            }
        )
        self.pv = PV(**pv_args)

        bat_args = equip_init_args.copy()
        bat_args.update(
            {
                "capacity": 1000,
                "capacity_kwh": 10000,
                "soc_init": 0.5,
                "self_consumption_mode": True,
            }
        )
        self.battery = Battery(**bat_args)

    def _battery_step(self, net_before):
        """Run battery for one timestep given net_power from rest of house."""
        self.battery.current_schedule["net_power"] = net_before
        mode = self.battery.update_internal_control()
        self.battery.mode = mode
        self.battery.calculate_power_and_heat()
        return net_before + self.battery.electric_kw

    # ------------------------------------------------------------------
    # Scenario A: surplus PV + low-SOC battery -> near-zero grid import
    # ------------------------------------------------------------------

    def test_pv_surplus_charges_low_soc_battery(self):
        """Large PV + battery with low SOC: surplus PV charges battery,
        resulting in near-zero grid draw."""
        # run PV to noon -> generating ~10 kW
        for _ in range(int(12 * 60)):
            self.pv.update()
        pv_kw = self.pv.electric_kw
        self.assertLess(pv_kw, -5, "PV should be generating at noon")

        # battery starts low
        self.battery.soc = 0.2
        self.assertLess(self.battery.soc, 0.5, "Battery should start low")

        # load is modest (2 kW)
        load_kw = 2.0
        net_before = load_kw + pv_kw  # ~ -8 kW (exporting)
        self.assertLess(net_before, 0, "Surplus PV = net export before battery")

        net_after = self._battery_step(net_before)
        self.assertAlmostEqual(net_after, 0, delta=0.05,
                               msg="Battery should absorb surplus PV, net ~0")

        # SOC should have increased (charging)
        self.assertGreater(self.battery.next_soc, self.battery.soc,
                           msg="SOC should increase when charging from PV")

    # ------------------------------------------------------------------
    # Scenario B: infinite battery zeroes import at night
    # ------------------------------------------------------------------

    def test_battery_zeroes_night_import(self):
        """At night (no PV), battery discharges to cancel house load."""
        net_after = self._battery_step(5.0)  # 5 kW import, no PV
        self.assertAlmostEqual(net_after, 0, delta=0.05)

    # ------------------------------------------------------------------
    # Scenario C: SOC moves correctly with charge/discharge
    # ------------------------------------------------------------------

    def test_soc_moves_correctly(self):
        self.battery.soc = 0.5

        # Export -> charge
        self._battery_step(-5.0)
        self.assertGreater(self.battery.next_soc, 0.5)

        # Import -> discharge
        self.battery.soc = 0.5
        self._battery_step(5.0)
        self.assertLess(self.battery.next_soc, 0.5)


# ---------------------------------------------------------------------------
# Scenario D: high-PV, low-load, low-SOC battery -> zero grid flow
# ---------------------------------------------------------------------------

class PVBatteryGridZeroTestCase(unittest.TestCase):
    """On a sunny midday with large PV, a low-SOC battery soaks up all
    surplus generation so that net grid flow is ~0."""

    def setUp(self):
        pv_args = equip_init_args.copy()
        pv_args.update(
            {
                "capacity": 10,
                "tilt": 20,
                "azimuth": 180,
                "schedule": _make_sunny_pv_schedule(10),
            }
        )
        self.pv = PV(**pv_args)

        bat_args = equip_init_args.copy()
        bat_args.update(
            {
                "capacity": 10,  # realistic 10 kW inverter
                "capacity_kwh": 13.5,  # typical residential (e.g. Powerwall)
                "soc_init": 0.2,
                "self_consumption_mode": True,
            }
        )
        self.battery = Battery(**bat_args)

    def _battery_step(self, net_before):
        self.battery.current_schedule["net_power"] = net_before
        mode = self.battery.update_internal_control()
        self.battery.mode = mode
        self.battery.calculate_power_and_heat()
        return net_before + self.battery.electric_kw

    def test_high_pv_low_soc_near_zero_grid(self):
        """Midday high PV, typical residential battery at low SOC,
        small house load -> battery charges on surplus, grid ~0 kW."""
        # Run PV to noon
        for _ in range(int(12 * 60)):
            self.pv.update()
        pv_kw = self.pv.electric_kw
        self.assertLess(pv_kw, -5)

        # Low SOC, small load
        self.battery.soc = 0.2
        load_kw = 1.5

        net_before = load_kw + pv_kw
        net_after = self._battery_step(net_before)

        # With 10 kW inverter and ~8ish kW surplus, battery should absorb
        # enough to bring grid draw near 0 (capped by inverter if surplus > capacity)
        self.assertLess(net_after, 0.5,
                        msg="Grid import should be small after battery soaks up PV")

        # Battery should have charged
        self.assertGreater(self.battery.next_soc, 0.2)

    def test_pv_surplus_limited_by_inverter(self):
        """When PV surplus exceeds battery inverter rating, grid exports
        the excess (battery saturates at capacity)."""
        for _ in range(int(12 * 60)):
            self.pv.update()
        pv_kw = self.pv.electric_kw  # ~ -10 kW

        # Undersized inverter — only 5 kW, but PV generating ~10 kW
        self.battery.capacity = 5
        self.battery.soc = 0.2
        net_before = pv_kw  # ~ -10 kW, no house load

        net_after = self._battery_step(net_before)

        # Battery inverter can only charge at 5 kW; remaining ~5 kW is exported
        self.assertLess(net_after, 0,
                        msg="Should still export if surplus exceeds inverter rating")
        self.assertGreater(net_after, net_before,
                           msg="Battery charging reduces net export")
        self.assertAlmostEqual(self.battery.electric_kw, 5.0, places=1,
                               msg="Battery charges at inverter max (5 kW)")


# ---------------------------------------------------------------------------
# Dwelling-level integration tests (real building + weather + schedule)
# ---------------------------------------------------------------------------


class DwellingDERIntegrationTestCase(unittest.TestCase):
    """Full Dwelling simulation with PV + Battery DER equipment,
    real weather, envelope, HVAC, and loads."""

    dwelling = None

    @classmethod
    def setUpClass(cls):
        from ochre import Dwelling
        from test import test_output_path

        args = {
            "name": "test_der_integration",
            "start_time": dt.datetime(2019, 5, 5, 12, 0, 0),
            "time_res": dt.timedelta(minutes=15),
            "duration": dt.timedelta(days=1),
            "ext_time_res": dt.timedelta(hours=1),
            "initialization_time": dt.timedelta(hours=1),
            "output_path": test_output_path,
            "hpxml_file": "BEopt_example.xml",
            "hpxml_schedule_file": "BEopt_example_schedule.csv",
            "weather_file": "USA_CO_Denver.Intl.AP.725650_TMY3.epw",
            "verbosity": 3,
            "metrics_verbosity": 1,
            "save_results": False,
            "Equipment": {
                "PV": {
                    "capacity": 10,
                    "tilt": 20,
                    "azimuth": 180,
                },
                "Battery": {
                    "capacity": 10,
                    "capacity_kwh": 13.5,
                    "soc_init": 0.5,
                    "self_consumption_mode": True,
                },
            },
        }
        cls.dwelling = Dwelling(**args)
        cls._cleanup_files = []
        for suffix in ["", "_schedule", "_metrics", "_hourly"]:
            f = os.path.join(test_output_path, f"test_der_integration{suffix}.csv")
            if os.path.exists(f):
                cls._cleanup_files.append(f)

    @classmethod
    def tearDownClass(cls):
        for f in cls._cleanup_files:
            try:
                os.remove(f)
            except OSError:
                pass

    def setUp(self):
        self.dwelling.reset_time()
        self.battery = self.dwelling.equipment["Battery"]
        self.pv = self.dwelling.equipment["PV"]
        # Ensure battery is in self-consumption mode
        self.battery.self_consumption_mode = True

    def _step(self, control=None):
        if control is None:
            control = {}
        return self.dwelling.update(control_signal=control)

    def _remaining_steps(self):
        """Number of steps remaining in the schedule."""
        end = self.dwelling.start_time + self.dwelling.duration
        return int((end - self.dwelling.current_time) / self.dwelling.time_res)

    # ---- Scenario: PV + low-SOC battery -> near-zero grid flow ----

    def test_pv_and_low_soc_battery_zeroes_grid(self):
        """Midday PV surplus with low-SOC battery: net grid ~0.
        Start time is May 5 noon in Denver on a sunny day."""
        # Battery at low SOC
        self.battery.soc = 0.2
        self.battery.self_consumption_mode = True

        # Take several steps at noon
        net_powers = []
        for _ in range(min(4, self._remaining_steps())):
            self._step({"Battery": {"Self Consumption Mode": True}})
            net_powers.append(self.dwelling.total_p_kw)

        # PV should be generating (negative)
        self.assertLess(self.pv.electric_kw, 0,
                        "PV should be generating at noon")

        # Battery should be charging (positive)
        self.assertGreater(self.battery.electric_kw, 0,
                           "Battery should charge from PV surplus")

        # Net grid draw should be near zero
        for net in net_powers:
            self.assertLess(abs(net), 2.0,
                            f"Net grid power {net:.2f} kW should be near 0")

        # SOC should have increased
        self.assertGreater(self.battery.next_soc, 0.2,
                           "Battery SOC should rise from PV charging")

    # ---- Scenario: battery fills up, then switches to export ----

    def test_battery_fills_then_exports(self):
        """Battery charges from PV until full, then surplus exports to grid."""
        # Battery near full - small headroom
        self.battery.soc = self.battery.soc_max - 0.03
        self.battery.self_consumption_mode = True

        net_flows = []
        bat_powers = []

        max_steps = min(6, self._remaining_steps())
        for _ in range(max_steps):
            self._step({"Battery": {"Self Consumption Mode": True}})
            net_flows.append(self.dwelling.total_p_kw)
            bat_powers.append(self.battery.electric_kw)

        self.assertGreater(len(net_flows), 1, "Need at least 2 steps")

        # Initially battery charges, net ~0
        self.assertGreater(bat_powers[0], 0,
                           "Battery should charge initially")

        # Eventually battery stops charging (power ~0 when full)
        final_bat_power = bat_powers[-1]
        self.assertAlmostEqual(final_bat_power, 0, delta=0.1,
                               msg="Battery should stop charging when SOC max reached")

        # SOC should be at or near max
        self.assertAlmostEqual(self.battery.soc, self.battery.soc_max, delta=0.03,
                               msg="SOC should reach max")

    # ---- Scenario: night-time battery discharge ----

    def test_battery_discharges_at_night(self):
        """At night (no PV), battery discharges to support house load."""
        steps_to_midnight = min(48, self._remaining_steps())
        for _ in range(steps_to_midnight):
            self._step({"Battery": {"Self Consumption Mode": True}})

        self.battery.soc = 0.7
        self.battery.self_consumption_mode = True

        if self._remaining_steps() > 0:
            self._step({"Battery": {"Self Consumption Mode": True}})

        # At midnight, PV should be ~0
        self.assertAlmostEqual(self.pv.electric_kw, 0, delta=0.1,
                               msg="PV should be zero at night")

        # Battery should discharge to support load
        if not self.dwelling.total_p_kw == 0:
            self.assertLess(self.battery.electric_kw, 0,
                            "Battery should discharge to support house load")


# ---------------------------------------------------------------------------
# Cloud cover test — SAM PVWatts on clear vs cloudy day
# ---------------------------------------------------------------------------

DENVER_LOCATION = {
    "timezone": -7,
    "altitude": 1656,  # m
    "latitude": 39.83,
    "longitude": -104.65,
}


def _make_weather_day(ghi_peak, dni_peak, dhi_peak, date=None):
    """Create a single-day hourly weather DataFrame with a sine-shaped
    irradiance profile centred on noon."""
    if date is None:
        date = start_time.date()
    times = pd.date_range(
        dt.datetime.combine(date, dt.time(0, 0)),
        dt.datetime.combine(date + dt.timedelta(days=1), dt.time(0, 0)),
        freq=dt.timedelta(hours=1),
        inclusive="left",
        tz="UTC",
    )
    hours = (times.hour + times.minute / 60.0).values
    # sine envelope: 0 at sunrise(6) / sunset(18), 1 at noon
    envelope = np.maximum(0, np.sin(np.pi * (hours - 6) / 12))
    return pd.DataFrame(
        {
            "GHI (W/m^2)": ghi_peak * envelope,
            "DNI (W/m^2)": dni_peak * envelope,
            "DHI (W/m^2)": dhi_peak * envelope,
            "Wind Speed (m/s)": 3.0,
            "Ambient Dry Bulb (C)": 20.0,
        },
        index=times,
    )


class PvCloudCoverTestCase(unittest.TestCase):
    """Verify that cloud cover (lower irradiance) reduces PV output."""

    PV_KW = 10

    def test_cloudy_reduces_pv_generation(self):
        """Same location & panel, clear vs cloudy weather -> cloudy peak
        power is substantially lower."""
        weather_clear = _make_weather_day(
            ghi_peak=1000, dni_peak=900, dhi_peak=100,
        )
        weather_cloudy = _make_weather_day(
            ghi_peak=250, dni_peak=40, dhi_peak=210,
        )

        ac_clear = run_sam(
            self.PV_KW, tilt=20, azimuth=180,
            weather=weather_clear, location=DENVER_LOCATION,
        )
        ac_cloudy = run_sam(
            self.PV_KW, tilt=20, azimuth=180,
            weather=weather_cloudy, location=DENVER_LOCATION,
        )

        peak_clear = abs(ac_clear.min())
        peak_cloudy = abs(ac_cloudy.min())

        self.assertGreater(peak_clear, 3,
                           "Clear-sky PV should generate > 3 kW")
        self.assertLess(peak_cloudy, peak_clear * 0.5,
                        f"Cloudy peak ({peak_cloudy:.1f} kW) should be "
                        f"< 50% of clear peak ({peak_clear:.1f} kW)")

    def test_zero_irradiance_gives_zero_power(self):
        """With GHI=DNI=DHI=0, PV output is zero."""
        weather_night = _make_weather_day(
            ghi_peak=0, dni_peak=0, dhi_peak=0,
        )
        ac = run_sam(
            self.PV_KW, tilt=20, azimuth=180,
            weather=weather_night, location=DENVER_LOCATION,
        )
        self.assertTrue((ac == 0).all(),
                        "PV output should be zero with no irradiance")

    def test_overcast_but_not_black_sky(self):
        """A heavily overcast (but not fully dark) day still produces
        some power — just much less than clear sky."""
        weather_overcast = _make_weather_day(
            ghi_peak=300, dni_peak=50, dhi_peak=250,
        )
        ac = run_sam(
            self.PV_KW, tilt=20, azimuth=180,
            weather=weather_overcast, location=DENVER_LOCATION,
        )
        peak = abs(ac.min())
        self.assertGreater(peak, 0.5,
                           "Overcast sky should still produce some power")
        self.assertLess(peak, 5,
                        f"Overcast peak ({peak:.1f} kW) should be well "
                        "below rated capacity")


if __name__ == "__main__":
    unittest.main()
