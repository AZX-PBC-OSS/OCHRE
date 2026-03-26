#!/usr/bin/env python3
"""Extract structural envelope data from OCHRE's BEopt example HPXML file.

Parses BEopt_example.xml and prints reference constants for use in Rust tests.
Areas in HPXML are ft²; volumes are ft³. Convert to SI using exact factors.
"""
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import NamedTuple

NS = "http://hpxmlonline.com/2019/10"
FT2_TO_M2 = 0.09290304
FT3_TO_M3 = 0.028316846592
# Pitch 6/12 roof tilt: atan(6/12) in radians
PITCH_6_TILT_RAD = math.atan(6.0 / 12.0)


def tag(name: str) -> str:
    return f"{{{NS}}}{name}"


def find_text(el: ET.Element, path: str) -> str | None:
    parts = path.split("/")
    cur = el
    for p in parts:
        found = cur.find(tag(p))
        if found is None:
            return None
        cur = found
    return cur.text


def find_float(el: ET.Element, path: str) -> float | None:
    t = find_text(el, path)
    return float(t) if t is not None else None


class Boundary(NamedTuple):
    btype: str
    bid: str
    area_ft2: float
    area_m2: float
    interior: str
    exterior: str
    solar_absorptance: float | None
    emittance: float | None
    r_value: float | None


class Window(NamedTuple):
    wid: str
    area_ft2: float
    area_m2: float
    azimuth: int
    orientation: str
    u_factor: float
    shgc: float


def parse_id(el: ET.Element) -> str:
    si = el.find(tag("SystemIdentifier"))
    if si is not None:
        return si.get("id", "")
    return ""


def main() -> None:
    xml_path = (
        Path(__file__).parent
        / "ochre"
        / "defaults"
        / "Input Files"
        / "BEopt_example.xml"
    )
    tree = ET.parse(xml_path)
    root = tree.getroot()

    building = root.find(tag("Building"))
    assert building is not None
    details = building.find(tag("BuildingDetails"))
    assert details is not None

    # ── Zone structure ────────────────────────────────────────────────────────
    construction = details.find(f"{tag('BuildingSummary')}/{tag('BuildingConstruction')}")
    assert construction is not None

    conditioned_area_ft2 = find_float(construction, "ConditionedFloorArea") or 0.0
    conditioned_vol_ft3 = find_float(construction, "ConditionedBuildingVolume") or 0.0
    conditioned_area_m2 = conditioned_area_ft2 * FT2_TO_M2
    conditioned_vol_m3 = conditioned_vol_ft3 * FT3_TO_M3
    print("=== Zone Structure ===")
    print(
        f"Zone: Conditioned, floor_area={conditioned_area_m2:.4f} m², "
        f"volume={conditioned_vol_m3:.4f} m³"
    )

    # Attic volume: OCHRE computes it from gable wall areas and roof pitch.
    # Formula: attic_volume = 0.5 * floor_area * attic_height
    # where attic_height = sqrt(gable_area * tan(roof_tilt))
    # Gable walls: Wall5 + Wall6 (each 144.5 ft²), pitch = 6/12.
    attic_floor_area_ft2 = conditioned_area_ft2  # 1-storey, attic over full footprint
    attic_floor_area_m2 = attic_floor_area_ft2 * FT2_TO_M2

    # Gable wall area (one gable, e.g. Wall5 = 144.5 ft²)
    gable_area_ft2 = 144.5
    gable_area_m2 = gable_area_ft2 * FT2_TO_M2

    attic_height_m = math.sqrt(gable_area_m2 * math.tan(PITCH_6_TILT_RAD))
    attic_volume_m3 = 0.5 * attic_floor_area_m2 * attic_height_m

    print(
        f"Zone: Attic, floor_area={attic_floor_area_m2:.4f} m², "
        f"volume={attic_volume_m3:.4f} m³"
        f"  (height={attic_height_m:.4f} m, pitch=6/12)"
    )

    # ── Boundaries ────────────────────────────────────────────────────────────
    enclosure = details.find(tag("Enclosure"))
    assert enclosure is not None

    boundaries: list[Boundary] = []

    # Roofs
    for roof in enclosure.findall(f"{tag('Roofs')}/{tag('Roof')}"):
        bid = parse_id(roof)
        area_ft2 = find_float(roof, "Area") or 0.0
        interior = find_text(roof, "InteriorAdjacentTo") or ""
        absorptance = find_float(roof, "SolarAbsorptance")
        emittance = find_float(roof, "Emittance")
        boundaries.append(
            Boundary("Roof", bid, area_ft2, area_ft2 * FT2_TO_M2, interior, "outside", absorptance, emittance, None)
        )

    # Walls
    for wall in enclosure.findall(f"{tag('Walls')}/{tag('Wall')}"):
        bid = parse_id(wall)
        area_ft2 = find_float(wall, "Area") or 0.0
        interior = find_text(wall, "InteriorAdjacentTo") or ""
        exterior = find_text(wall, "ExteriorAdjacentTo") or "outside"
        absorptance = find_float(wall, "SolarAbsorptance")
        emittance = find_float(wall, "Emittance")
        boundaries.append(
            Boundary("Wall", bid, area_ft2, area_ft2 * FT2_TO_M2, interior, exterior, absorptance, emittance, None)
        )

    # Floors (ceiling/floor assemblies)
    for floor in enclosure.findall(f"{tag('Floors')}/{tag('Floor')}"):
        bid = parse_id(floor)
        area_ft2 = find_float(floor, "Area") or 0.0
        interior = find_text(floor, "InteriorAdjacentTo") or ""
        exterior = find_text(floor, "ExteriorAdjacentTo") or ""
        floor_or_ceiling = find_text(floor, "FloorOrCeiling") or "floor"
        btype = "Ceiling" if floor_or_ceiling == "ceiling" else "Floor"
        boundaries.append(
            Boundary(btype, bid, area_ft2, area_ft2 * FT2_TO_M2, interior, exterior, None, None, None)
        )

    # Slabs
    for slab in enclosure.findall(f"{tag('Slabs')}/{tag('Slab')}"):
        bid = parse_id(slab)
        area_ft2 = find_float(slab, "Area") or 0.0
        interior = find_text(slab, "InteriorAdjacentTo") or ""
        boundaries.append(
            Boundary("Slab", bid, area_ft2, area_ft2 * FT2_TO_M2, interior, "ground", None, None, None)
        )

    # Doors
    for door in enclosure.findall(f"{tag('Doors')}/{tag('Door')}"):
        bid = parse_id(door)
        area_ft2 = find_float(door, "Area") or 0.0
        exterior = "outside"
        r_value = find_float(door, "RValue")
        boundaries.append(
            Boundary("Door", bid, area_ft2, area_ft2 * FT2_TO_M2, "living space", exterior, None, None, r_value)
        )

    print("\n=== Boundaries ===")
    for b in boundaries:
        absorptance_str = f", absorptance={b.solar_absorptance}" if b.solar_absorptance is not None else ""
        emittance_str = f", emittance={b.emittance}" if b.emittance is not None else ""
        rval_str = f", R={b.r_value}" if b.r_value is not None else ""
        print(
            f"Type: {b.btype}, id={b.bid}, area={b.area_m2:.4f} m²"
            f" ({b.area_ft2:.4f} ft²), interior={b.interior!r},"
            f" exterior={b.exterior!r}{absorptance_str}{emittance_str}{rval_str}"
        )

    # ── Summary by type ───────────────────────────────────────────────────────
    print("\n=== Summary by Type ===")
    type_groups: dict[str, list[Boundary]] = {}
    for b in boundaries:
        type_groups.setdefault(b.btype, []).append(b)
    for btype, group in sorted(type_groups.items()):
        total_m2 = sum(b.area_m2 for b in group)
        total_ft2 = sum(b.area_ft2 for b in group)
        print(
            f"{btype}: {len(group)} surfaces,"
            f" total {total_m2:.4f} m² ({total_ft2:.4f} ft²)"
        )

    # ── Windows ───────────────────────────────────────────────────────────────
    windows: list[Window] = []
    for win in enclosure.findall(f"{tag('Windows')}/{tag('Window')}"):
        wid = parse_id(win)
        area_ft2 = find_float(win, "Area") or 0.0
        azimuth = int(find_text(win, "Azimuth") or "0")
        orientation = find_text(win, "Orientation") or ""
        u_factor = find_float(win, "UFactor") or 0.0
        shgc = find_float(win, "SHGC") or 0.0
        windows.append(Window(wid, area_ft2, area_ft2 * FT2_TO_M2, azimuth, orientation, u_factor, shgc))

    print("\n=== Windows ===")
    for w in windows:
        print(
            f"Window: id={w.wid}, area={w.area_m2:.4f} m² ({w.area_ft2:.1f} ft²),"
            f" azimuth={w.azimuth}°, orientation={w.orientation},"
            f" U={w.u_factor:.4f} W/(m²·K) [HPXML units: Btu/(h·ft²·°F)],"
            f" SHGC={w.shgc}"
        )
    total_window_area_m2 = sum(w.area_m2 for w in windows)
    total_window_area_ft2 = sum(w.area_ft2 for w in windows)
    print(
        f"Total window area: {total_window_area_m2:.4f} m²"
        f" ({total_window_area_ft2:.1f} ft²)"
    )

    # ── Capacitance calculation ───────────────────────────────────────────────
    # OCHRE uses: C = rho_air * cp_air * V * TCM
    # rho_air = 1.2041 kg/m³, cp_air = 1006 J/(kg·K), TCM = 7 (thermal capacitance multiplier)
    rho_air = 1.2041
    cp_air = 1006.0
    tcm = 7.0
    indoor_c = rho_air * cp_air * conditioned_vol_m3 * tcm
    attic_c = rho_air * cp_air * attic_volume_m3 * tcm

    print("\n=== Capacitance Calculation ===")
    print(
        f"Indoor: C = {rho_air} * {cp_air} * {conditioned_vol_m3:.4f} * {tcm}"
        f" = {indoor_c:.2f} J/K"
    )
    print(
        f"Attic:  C = {rho_air} * {cp_air} * {attic_volume_m3:.4f} * {tcm}"
        f" = {attic_c:.2f} J/K"
    )

    print("\n=== Rust Test Constants ===")
    print(f"// Conditioned zone")
    print(f"const CONDITIONED_FLOOR_AREA_M2: f64 = {conditioned_area_m2:.6f};")
    print(f"const CONDITIONED_VOLUME_M3: f64 = {conditioned_vol_m3:.6f};")
    print(f"// Attic zone (OCHRE-derived: gable_area={gable_area_m2:.6f} m², pitch=6/12)")
    print(f"const ATTIC_FLOOR_AREA_M2: f64 = {attic_floor_area_m2:.6f};")
    print(f"const ATTIC_VOLUME_M3: f64 = {attic_volume_m3:.6f};")
    print(f"// Thermal capacitance (rho*cp*V*TCM, TCM=7)")
    print(f"const INDOOR_CAPACITANCE_J_PER_K: f64 = {indoor_c:.4f};")
    print(f"const ATTIC_CAPACITANCE_J_PER_K: f64 = {attic_c:.4f};")
    print(f"// Window totals")
    print(f"const TOTAL_WINDOW_AREA_M2: f64 = {total_window_area_m2:.6f};")
    print(f"const WINDOW_U_FACTOR_SI: f64 = {windows[0].u_factor * 5.678263:.6f};  // Btu/(h·ft²·°F) → W/(m²·K)")
    print(f"const WINDOW_SHGC: f64 = {windows[0].shgc};")
    # Wall totals (conditioned-facing exterior walls only)
    exterior_walls = [b for b in boundaries if b.btype == "Wall" and b.interior == "living space" and b.exterior == "outside"]
    total_ext_wall_m2 = sum(b.area_m2 for b in exterior_walls)
    print(f"// Exterior walls (living space ↔ outside)")
    print(f"const EXTERIOR_WALL_AREA_M2: f64 = {total_ext_wall_m2:.6f};")


if __name__ == "__main__":
    main()
