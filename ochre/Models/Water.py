import numpy as np
import numba as nb

from ochre.Models import RCModel, ModelException
from ochre.utils import convert

# Water Constants
water_density = 1000  # kg/m^3
water_density_liters = 1  # kg/L
water_cp = 4.183  # kJ/kg-K
water_conductivity = 0.6406  # W/m-K
water_c = water_cp * water_density_liters * 1000  # heat capacity with useful units: J/K-L


@nb.njit(cache=True)
def _water_draw_general(states, vol_fractions, capacitances, draw_fraction, mains_temp, water_c_val, draw_liters, q_nodes_out):
    """Returns (outlet_temp, q_delivered). Writes q_nodes into q_nodes_out."""
    n = len(states)

    min_vf = vol_fractions[0]
    for i in range(1, n):
        if vol_fractions[i] < min_vf:
            min_vf = vol_fractions[i]

    if draw_fraction < min_vf:
        outlet_temp = states[0]
        q_delivered = draw_liters * water_c_val * (outlet_temp - mains_temp)
        for i in range(n - 1):
            q_nodes_out[i] = draw_liters * water_c_val * (states[i + 1] - states[i])
        q_nodes_out[n - 1] = draw_liters * water_c_val * (mains_temp - states[n - 1])
        return outlet_temp, q_delivered

    n1 = n + 1
    vols_pre = np.empty(n1)
    vols_post = np.empty(n1)
    temps = np.empty(n1)

    running = 0.0
    for i in range(n):
        running += vol_fractions[i]
        vols_pre[i] = running
    vols_pre[n] = running + draw_fraction

    vols_post[0] = draw_fraction
    running = draw_fraction
    for i in range(n):
        running += vol_fractions[i]
        vols_post[i + 1] = running

    for i in range(n):
        temps[i] = states[i]
    temps[n] = mains_temp

    outlet_temp = 0.0
    prev = 0.0
    for i in range(n1):
        clipped = min(vols_pre[i], draw_fraction)
        vol_del = clipped - prev
        outlet_temp += temps[i] * vol_del
        prev = clipped
    outlet_temp /= draw_fraction
    q_delivered = draw_liters * water_c_val * (outlet_temp - mains_temp)

    for i in range(n):
        t_end = 0.0
        prev_v = vols_post[i]
        for j in range(n1):
            clipped = min(max(vols_pre[j], vols_post[i]), vols_post[i + 1])
            vol_del = clipped - prev_v
            t_end += temps[j] * vol_del
            prev_v = clipped
        t_end /= vol_fractions[i]
        q_nodes_out[i] = (t_end - states[i]) * capacitances[i]

    return outlet_temp, q_delivered


@nb.njit(cache=True)
def _water_draw_2node(
    states,
    vol_fractions,
    capacitances,
    draw_fraction,
    mains_temp,
    water_c_val,
    draw_liters,
    flow_fraction,
    q_nodes_out,
):
    """Returns (outlet_temp, q_delivered). 2-node fast path."""
    outlet_temp = states[0]
    if draw_fraction > vol_fractions[0]:
        outlet_temp = (states[0] * vol_fractions[0] + states[1] * (draw_fraction - vol_fractions[0])) / draw_fraction
    q_delivered = draw_liters * water_c_val * (outlet_temp - mains_temp)

    q_to_mains_lower = capacitances[1] * (states[1] - mains_temp)
    if q_delivered * flow_fraction > q_to_mains_lower:
        q_nodes_out[0] = q_to_mains_lower - q_delivered
        q_nodes_out[1] = -q_to_mains_lower
    else:
        q_nodes_out[0] = -q_delivered * (1.0 - flow_fraction)
        q_nodes_out[1] = -q_delivered * flow_fraction

    return outlet_temp, q_delivered


@nb.njit(cache=True)
def _inversion_mixing(next_states, vol_fractions, vol_cumsum, capacitances):
    """Modifies next_states in-place. Returns True if algorithm and energy checks pass."""
    n = len(next_states)
    init_heat = 0.0
    for i in range(n):
        init_heat += next_states[i] * capacitances[i]

    for node_idx in range(n - 1):
        current_temp = next_states[node_idx]

        new_temp = current_temp
        heat_sum = 0.0
        base_vol = 0.0
        if node_idx > 0:
            base_vol = vol_cumsum[node_idx - 1]

        for j in range(node_idx, n):
            heat_sum += next_states[j] * vol_fractions[j]
            vol_sum = vol_cumsum[j] - base_vol
            mixed = heat_sum / vol_sum
            if mixed > new_temp:
                new_temp = mixed

        if new_temp > current_temp + 0.001:
            q = (new_temp - current_temp) * vol_fractions[node_idx]
            next_states[node_idx] = new_temp
            next_states[node_idx + 1] -= q / vol_fractions[node_idx + 1]

            has_inversion = False
            for j in range(n - 1):
                if next_states[j + 1] - next_states[j] > 0.1:
                    has_inversion = True
                    break
            if not has_inversion:
                break
        elif new_temp < current_temp - 0.001:
            return False

    final_heat = 0.0
    for i in range(n):
        final_heat += next_states[i] * capacitances[i]
    return abs(final_heat - init_heat) < 1.0


class StratifiedWaterModel(RCModel):
    """
    Stratified Water Tank RC Thermal Model

    - Partitions a water tank into n nodes (12 by default).
    - Nodes can have different volumes, but are equal volume by default.
    - Node 1 is at the top of the tank (at outlet).
    - State names are [T_WH1, T_WH2, ...], length n
    - Input names are [T_AMB, H_WH1, H_WH2, ...], length n+1
    - The model can accept 2 additional inputs for water draw:
      - draw: volume of water to deliver
      - draw_tempered: volume to deliver at setpoint temperature.
      If tank temperature is higher than setpoint, the model assumes mixing with water mains.
    - The model considers the following effects on temperature at each time step:
      - Internal (node-to-node) conduction
      - External (node-to-ambient) conduction
      - Heat injections due to water heater
      - Heat injections due to water draw (calculated before the state-space update)
      - Heat transfer due to inversion mixing (assumes fast mixing, calculated after the state-space update)
    - At each time step, the model calculates:
      - The internal states (temperatures)
      - The heat delivered to the load (relative to mains temperature)
      - The heat lost to ambient air
    """

    name = "Water Tank"
    optional_inputs = [
        "Water Heating (L/min)",
        "Clothes Washer (L/min)",
        "Dishwasher (L/min)",
        "Mains Temperature (C)",
        "Zone Temperature (C)",
    ]

    def __init__(self, water_nodes=12, water_vol_fractions=None, **kwargs):
        if water_vol_fractions is None:
            self.n_nodes = water_nodes
            self.vol_fractions = np.ones(self.n_nodes) / self.n_nodes
        else:
            self.n_nodes = len(water_vol_fractions)
            self.vol_fractions = np.array(water_vol_fractions) / sum(water_vol_fractions)

        self.vol_fractions.flags.writeable = False
        self._vol_cumsum = self.vol_fractions.cumsum()

        self.volume = kwargs["Tank Volume (L)"]  # in L

        capacitances, resistances = self.load_rc_data(**kwargs)

        super().__init__(capacitances, resistances, external_nodes=["AMB"], **kwargs)
        self.next_states = self.states  # for holding state info for next time step

        self._control_buf = np.zeros(self.nx + 1)
        self._heats_buf = np.zeros(self.nx)
        self._inputs_init_buf = np.zeros(self.nx + 1)
        self._q_nodes_jit_buf = np.zeros(self.n_nodes)
        self._state_diff_buf = np.empty(self.n_nodes, dtype=float)

        self.t_amb_idx = self.input_names.index("T_AMB")
        assert self.t_amb_idx == 0  # should always be first
        self.t_1_idx = self.state_names.index("T_WH1")
        self.h_1_idx = self.input_names.index("H_WH1")

        # key variables for results
        self.draw_total = 0  # in L
        self.h_delivered = 0  # heat delivered in outlet water, in W
        self.h_injections = 0  # heat from water heater, in W
        self.h_loss = 0  # conduction heat loss from tank, in W
        self.h_unmet_load = 0  # unmet load from outlet temperature, fixtures only, in W
        self.mains_temp = 0  # water mains temperature, in C
        self.outlet_temp = 0  # temperature of outlet water, in C

        # mixed temperature (i.e. target temperature) setpoint for fixtures - Sink/Shower/Bath (SSB)
        self.tempered_draw_temp = kwargs.get("Mixed Delivery Temperature (C)", convert(105, "degF", "degC"))
        self.hot_draw_temp = kwargs.get("Tempering Valve Setpoint (C)", convert(125, "degF", "degC"))
        self.setpoint_temp = kwargs.get("Setpoint Temperature (C)", convert(125, "degF", "degC"))
        # Removing target temperature for clothes washers
        # self.washer_draw_temp = kwargs.get('Clothes Washer Delivery Temperature (C)', convert(92.5, 'degF', 'degC'))

    def load_rc_data(self, **kwargs):
        # Get properties from input file
        h = kwargs["Tank Height (m)"]  # in m
        top_area = self.volume / h / 1000  # in m^2
        r = (top_area / np.pi) ** 0.5

        if "Heat Transfer Coefficient (W/m^2/K)" in kwargs:
            u = kwargs["Heat Transfer Coefficient (W/m^2/K)"]
        elif "UA (W/K)" in kwargs:
            ua = kwargs["UA (W/K)"]
            total_area = 2 * top_area + 2 * np.pi * r * h
            u = ua / total_area
        else:
            raise ModelException("Missing heat transfer coefficient (UA) for {}".format(self.name))

        # calculate general RC parameters for whole tank
        c_water_tot = self.volume * water_c  # Heat capacity of water (J/K)
        r_int = (h / self.n_nodes) / water_conductivity / top_area  # R between nodes (K/W)
        r_side_tot = 1 / u / (2 * np.pi * r * h)  # R from side of tank (K/W)
        r_top = 1 / u / top_area  # R from top/bottom of tank (K/W)

        # Capacitance per node
        capacitances = {f"WH{i + 1}": c_water_tot * frac for i, frac in enumerate(self.vol_fractions)}

        # Resistance to exterior from side, top, and bottom
        resistances = {(f"WH{i + 1}", "AMB"): r_side_tot / frac for i, frac in enumerate(self.vol_fractions)}
        resistances[("WH1", "AMB")] = self.par(resistances[("WH1", "AMB")], r_top)
        resistances[(f"WH{self.n_nodes}", "AMB")] = self.par(resistances[(f"WH{self.n_nodes}", "AMB")], r_top)

        # Resistance between nodes
        if self.n_nodes > 1:
            resistances.update({(f"WH{i + 1}", f"WH{i + 2}"): r_int for i in range(self.n_nodes - 1)})

        return capacitances, resistances

    @staticmethod
    def initialize_state(state_names, input_names, A_c, B_c, **kwargs):
        t_init = kwargs.get("Initial Temperature (C)")
        if t_init is None:
            t_max = kwargs.get("Setpoint Temperature (C)", convert(125, "degF", "degC"))
            t_db = kwargs.get("Deadband Temperature (C)", convert(10, "degR", "K"))
            # temp = t_max - np.random.rand(1) * t_db

            # set initial temperature close to top of deadband
            t_init = t_max - t_db / 10

        # Return states as a dictionary
        return {name: t_init for name in state_names}

    def update_water_draw(self):
        self._heats_buf.fill(0)
        heats_to_model = self._heats_buf
        self.mains_temp = self.current_schedule.get("Mains Temperature (C)")
        self.outlet_temp = self.states[self.t_1_idx]  # initial outlet temp, for estimating draw volume

        # Note: removing target draw temperature for clothes washers, not implemented in ResStock
        draw_tempered = self.current_schedule.get("Water Heating (L/min)", 0)
        draw_hot = self.current_schedule.get("Clothes Washer (L/min)", 0) + self.current_schedule.get(
            "Dishwasher (L/min)", 0
        )
        # draw_cw = self.current_schedule.get('Clothes Washer (L/min)', 0)
        # draw_hot = self.current_schedule.get('Dishwasher (L/min)', 0)
        if not (draw_tempered + draw_hot):
            # No water draw
            self.draw_total = 0
            self.h_delivered = 0
            self.h_unmet_load = 0
            return heats_to_model

        if self.mains_temp is None:
            raise ModelException("Mains temperature required when water draw exists")

        # calculate total draw volume from tempered draw volume(s)
        # for tempered draw, assume outlet temperature == T1, slightly off if the water draw is very large
        if self.tempered_draw_temp < self.setpoint_temp:
            if self.outlet_temp <= self.hot_draw_temp:
                self.draw_total = draw_hot
            else:
                vol_ratio_hot = (self.hot_draw_temp - self.mains_temp) / (self.outlet_temp - self.mains_temp)
                self.draw_total = draw_hot * vol_ratio_hot
        else:
            self.draw_total = draw_hot

        if draw_tempered:
            if self.outlet_temp <= self.tempered_draw_temp:
                self.draw_total += draw_tempered
            else:
                vol_ratio = (self.tempered_draw_temp - self.mains_temp) / (self.outlet_temp - self.mains_temp)
                self.draw_total += draw_tempered * vol_ratio
        # if draw_cw:
        #     if self.outlet_temp <= self.washer_draw_temp:
        #         self.draw_total += draw_cw
        #     else:
        #         vol_ratio = (self.washer_draw_temp - self.mains_temp) / (self.outlet_temp - self.mains_temp)
        #         self.draw_total += draw_cw * vol_ratio

        t_s = self._dt_seconds
        draw_liters = self.draw_total * t_s / 60  # in liters
        draw_fraction = draw_liters / self.volume  # unitless
        q_nodes = self._q_nodes_jit_buf

        if self.n_nodes == 2 and draw_fraction < self.vol_fractions[1]:
            # Use empirical factor for determining water flow by node
            flow_fraction = 0.95  # Totally empirical factor based on detailed lab validation
            self.outlet_temp, q_delivered = _water_draw_2node(
                self.states,
                self.vol_fractions,
                self.capacitances,
                draw_fraction,
                self.mains_temp,
                water_c,
                draw_liters,
                flow_fraction,
                q_nodes,
            )
        else:
            self.outlet_temp, q_delivered = _water_draw_general(
                self.states,
                self.vol_fractions,
                self.capacitances,
                draw_fraction,
                self.mains_temp,
                water_c,
                draw_liters,
                q_nodes,
            )

        # convert heat transfer from J to W
        self.h_delivered = q_delivered / t_s
        heats_to_model += q_nodes / t_s

        # calculate unmet loads, fixtures only, in W
        self.h_unmet_load = max(draw_tempered / 60 * water_c * (self.tempered_draw_temp - self.outlet_temp), 0)  # in W

        return heats_to_model

    def update_inputs(self, schedule_inputs=None):
        # Note: self.inputs_init are not updated here, only self.current_schedule
        super().update_inputs(schedule_inputs)

        # get zone temperature from schedule
        t_zone = self.current_schedule["Zone Temperature (C)"]

        # update heat injections from water draw
        # FUTURE: revise CW and DW when event based schedules are added
        heats_to_model = self.update_water_draw()

        # update water tank model
        self._inputs_init_buf[0] = t_zone
        self._inputs_init_buf[1:] = heats_to_model
        self.inputs_init = self._inputs_init_buf

    def run_inversion_mixing_rule(self):
        # Inversion Mixing Rule
        # See https://energyplus.net/sites/all/modules/custom/nrel_custom/pdfs/pdfs_v9.1.0/EngineeringReference.pdf
        #     p. 1528
        # Starting from the top, check for mixing at each node

        ok = _inversion_mixing(self.next_states, self.vol_fractions, self._vol_cumsum, self.capacitances)
        if not ok:
            raise ModelException(
                "Error in water heater inversion mixing algorithm. Final state temperatures are: {}".format(self.next_states)
            )

    def update_model(self, control_signal=None):
        if control_signal is not None:
            # control signal must be heat injections from water heater, by node
            assert isinstance(control_signal, np.ndarray) and len(control_signal) == self.nx
            self.h_injections = control_signal.sum()
            self._control_buf[0] = 0.0
            self._control_buf[1:] = control_signal
            control_signal = self.inputs_init + self._control_buf
        else:
            self.h_injections = 0

        super().update_model(control_signal)

        np.subtract(self.next_states, self.states, out=self._state_diff_buf)
        q_change = self._state_diff_buf.dot(self.capacitances)  # in J
        h_change = q_change / self._dt_seconds

        # calculate heat loss, in W
        self.h_loss = self.h_injections - h_change - self.h_delivered
        if abs(self.h_loss) > 1000:
            raise ModelException("Error in calculating heat loss for {} model".format(self.name))

        # If any temperatures are inverted, run inversion mixing algorithm
        delta_t = 0.1 if self.high_res else 0.01
        if (np.diff(self.next_states) > delta_t).any():
            self.run_inversion_mixing_rule()

    def update_results(self):
        current_results = super().update_results()

        # check that states are within reasonable range
        # Note: default max temp on water heater model is 60C (140F). Temps may exceed that slightly
        if self.states.max() > 62 or self.states.min() < self.mains_temp - 10:
            if self.states.max() > 65 or self.states.min() < self.mains_temp - 15:
                raise ModelException(f"Water temperatures are outside acceptable range: {self.states}")
            else:
                self.warn(f"Water temperatures are outside acceptable range: {self.states}")

        return current_results

    def generate_results(self):
        # Note: most results are included in Dwelling/WH. Only inputs and states are saved to self.results
        results = super().generate_results()

        if self.verbosity >= 3:
            results["Hot Water Unmet Demand (kW)"] = self.h_unmet_load / 1000
            results["Hot Water Outlet Temperature (C)"] = self.outlet_temp
        if self.verbosity >= 4:
            results["Hot Water Delivered (L/min)"] = self.draw_total
            results["Hot Water Delivered (W)"] = self.h_delivered
        if self.verbosity >= 7:
            results["Hot Water Heat Injected (W)"] = self.h_injections
            results["Hot Water Heat Loss (W)"] = self.h_loss
            results["Hot Water Average Temperature (C)"] = self.states.dot(self.vol_fractions)
            results["Hot Water Maximum Temperature (C)"] = self.states.max()
            results["Hot Water Minimum Temperature (C)"] = self.states.min()
            results["Hot Water Mains Temperature (C)"] = self.mains_temp
        return results


class OneNodeWaterModel(StratifiedWaterModel):
    """
    1-node Water Tank Model
    """

    def __init__(self, **kwargs):
        kwargs.pop("water_nodes", None)
        super().__init__(water_nodes=1, **kwargs)


class TwoNodeWaterModel(StratifiedWaterModel):
    """
    2-node Water Tank Model

    - Partitions tank into 2 nodes
    - Top node is 1/3 of volume, Bottom node is 2/3
    """

    def __init__(self, **kwargs):
        kwargs.pop("water_nodes", None)
        super().__init__(water_nodes=2, water_vol_fractions=[1 / 3, 2 / 3], **kwargs)


class IdealWaterModel(OneNodeWaterModel):
    """
    Ideal water tank with near-perfect insulation. Used for TanklessWaterHeater. Modeled as 1-node tank.
    """

    def load_rc_data(self, **kwargs):
        # ignore RC parameters from the properties file
        self.volume = 1000
        return {"WH1": self.volume * water_c}, {("WH1", "AMB"): 1e6}

    @staticmethod
    def initialize_state(state_names, input_names, A_c, B_c, **kwargs):
        # set temperature to upper threshold
        t_max = kwargs.get("Setpoint Temperature (C)", convert(125, "degF", "degC"))

        # Return states as a dictionary
        return {name: t_max for name in state_names}
