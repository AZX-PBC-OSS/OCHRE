import argparse
import cProfile
import datetime as dt
import os
import pstats
from time import perf_counter

import numpy as np
import pandas as pd

from ochre import Dwelling
from ochre.utils import default_input_path
from test import test_output_path


GOLDEN_PATH = os.path.join(test_output_path, "benchmark_golden.parquet")
BENCHMARK_DIR = os.path.join(test_output_path, "benchmarks")


def get_benchmark_args(duration_days: int = 30):
    return {
        "name": "benchmark_dwelling",
        "start_time": dt.datetime(2018, 6, 1, 0, 0, 0),
        "time_res": dt.timedelta(minutes=1),
        "duration": dt.timedelta(days=duration_days),
        "initialization_time": dt.timedelta(hours=6),
        "ext_time_res": dt.timedelta(hours=1),
        "seed": 42,
        "output_path": test_output_path,
        "save_results": False,
        "verbosity": 3,
        "metrics_verbosity": 3,
        "hpxml_file": "BEopt_example.xml",
        "hpxml_schedule_file": "BEopt_example_schedule.csv",
        "weather_file": os.path.join(default_input_path, "Weather", "USA_CO_Denver.Intl.AP.725650_TMY3.epw"),
        "Equipment": {
            "PV": {
                "capacity": 5,
                "tilt": 20,
                "azimuth": 180,
            },
            "Battery": {
                "capacity_kwh": 6,
                "capacity": 3,
                "soc_init": 0.5,
                "self_consumption_mode": True,
            },
            "EV": {
                "vehicle_type": "BEV",
                "charging_level": "Level 2",
                "range": 150,
            },
        },
    }


def run_simulation(duration_days: int = 30):
    dwelling = Dwelling(**get_benchmark_args(duration_days=duration_days))
    df, _, _ = dwelling.simulate()
    return df


def print_top_hotspots(stats, count=10):
    entries = [
        (ct, tt, nc, filename, line, func)
        for (filename, line, func), (_, nc, tt, ct, _) in stats.stats.items()
    ]
    entries.sort(key=lambda item: item[0], reverse=True)
    print(f"\nTop {count} cumulative hotspots:")
    for rank, (ct, tt, nc, filename, line, func) in enumerate(entries[:count], start=1):
        print(
            f"{rank:>2}. {ct:>10.4f}s cum | {tt:>10.4f}s self | {nc:>8} calls | "
            f"{os.path.basename(filename)}:{line}::{func}"
        )


def run_profile(profile_path, duration_days: int = 30):
    profiler = cProfile.Profile()
    start = perf_counter()
    profiler.enable()
    df = run_simulation(duration_days=duration_days)
    profiler.disable()
    runtime = perf_counter() - start

    os.makedirs(os.path.dirname(profile_path), exist_ok=True)
    profiler.dump_stats(profile_path)

    stats = pstats.Stats(profiler).sort_stats("cumulative")
    stats.print_stats(50)
    print(f"\nTotal simulation runtime: {runtime:.4f}s")
    print_top_hotspots(stats, count=10)

    return df


def is_temperature_column(column):
    col = column.lower()
    return "temperature" in col and "(c)" in col


def is_power_column(column):
    return "power" in column.lower()


def compare_with_golden(df):
    if not os.path.exists(GOLDEN_PATH):
        raise FileNotFoundError(f"Golden parquet not found: {GOLDEN_PATH}")

    golden = pd.read_parquet(GOLDEN_PATH)

    pd.testing.assert_index_equal(df.index, golden.index)
    if set(df.columns) != set(golden.columns):
        missing = set(golden.columns) - set(df.columns)
        extra = set(df.columns) - set(golden.columns)
        raise AssertionError(f"Column mismatch vs golden. Missing: {missing}, Extra: {extra}")

    # Align column order to golden
    df = df[golden.columns]

    for column in df.columns:
        actual = df[column]
        expected = golden[column]

        if actual.dtype != expected.dtype:
            raise AssertionError(
                f"Dtype mismatch for column '{column}': {actual.dtype} != {expected.dtype}"
            )

        if pd.api.types.is_numeric_dtype(actual):
            if is_temperature_column(column):
                atol, rtol = 0.01, 1e-4
            elif is_power_column(column):
                atol, rtol = 0.001, 1e-3
            else:
                atol, rtol = 1e-10, 1e-7
            np.testing.assert_allclose(actual.to_numpy(), expected.to_numpy(), atol=atol, rtol=rtol, equal_nan=True)
        elif not actual.equals(expected):
            raise AssertionError(f"Non-numeric column '{column}' does not match golden reference.")

    print("Golden comparison passed.")


def verify_determinism(duration_days: int = 7):
    print(f"Determinism check: running {duration_days}-day simulation twice...")
    df_1 = run_simulation(duration_days=duration_days)
    df_2 = run_simulation(duration_days=duration_days)

    pd.testing.assert_index_equal(df_1.index, df_2.index)
    if list(df_1.columns) != list(df_2.columns):
        raise AssertionError("Determinism check failed: columns differ between runs.")
    if not df_1.dtypes.equals(df_2.dtypes):
        raise AssertionError("Determinism check failed: dtypes differ between runs.")
    if not df_1.equals(df_2):
        raise AssertionError("Determinism check failed: DataFrames are not identical.")

    print("Determinism check passed (index/columns/dtypes identical and df1.equals(df2)).")


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark/profiling harness for OCHRE dwelling simulation.")
    parser.add_argument("--generate-golden", action="store_true", help="Run simulation, save golden parquet and pstats.")
    parser.add_argument("--compare", action="store_true", help="Run simulation and compare with golden parquet.")
    parser.add_argument("--profile", action="store_true", help="Run simulation with cProfile and print cumulative stats.")
    parser.add_argument("--verify-determinism", action="store_true", help="Run simulation twice and verify outputs are identical.")
    parser.add_argument("--duration-days", type=int, default=30, help="Simulation duration in days (default: 30). Applies to --profile/--compare/--generate-golden.")
    parser.add_argument("--label", type=str, default="baseline", help="Label for output files (e.g. 'T0-001' or 'baseline'). Profile and results saved to test/outputs/benchmarks/<label>.pstats and <label>.parquet.")
    return parser.parse_args()


def main():
    args = parse_args()
    default_mode = not (args.generate_golden or args.compare or args.profile or args.verify_determinism)

    should_profile = args.profile or args.generate_golden or default_mode
    should_compare = args.compare or default_mode

    if args.verify_determinism:
        verify_determinism(duration_days=args.duration_days)
        return

    days = args.duration_days
    label = args.label
    os.makedirs(BENCHMARK_DIR, exist_ok=True)
    profile_path = os.path.join(BENCHMARK_DIR, f"{label}.pstats")
    results_path = os.path.join(BENCHMARK_DIR, f"{label}.parquet")

    if should_profile:
        df = run_profile(profile_path, duration_days=days)
    else:
        df = run_simulation(duration_days=days)

    df.to_parquet(results_path)
    print(f"Saved results: {results_path}")

    if args.generate_golden:
        os.makedirs(os.path.dirname(GOLDEN_PATH), exist_ok=True)
        df.to_parquet(GOLDEN_PATH)
        print(f"Saved golden parquet: {GOLDEN_PATH}")

    if should_compare:
        compare_with_golden(df)


if __name__ == "__main__":
    main()
