"""
flowvec.py -- Heat-pulse flow-vector analysis for the 3-ring / 8-sensor RRIV probe.

Geometry
--------
Three horizontal rings stacked vertically (Ring1 = top, Ring2 = middle, Ring3 = bottom).
Each ring carries 8 thermistors A..H at 45 degree spacing around a central heater.
A heat pulse is released at the centre; advection skews the thermal plume toward the
downstream sensors and (for a vertical flow component) toward the upper or lower ring.

Column map (RRIV CSV)
---------------------
value_1 .. value_8   -> Ring1 Sensor A..H
value_9 .. value_16  -> Ring2 Sensor A..H
value_17 .. value_24 -> Ring3 Sensor A..H
value_25             -> Heater state (0 / 1)

Method
------
Per heater cycle:
  1. Baseline = mean of each sensor over BASELINE_S seconds immediately before onset.
  2. dT_i(t) = T_i(t) - baseline_i
  3. Response S_i = time-integral of dT_i over the pulse + tail window (K.s).
     Integrating is more robust than peak-picking when dT is near the sensor's
     quantisation step.
  4. Per ring, fit the first circular harmonic to S around the 8 azimuths:
         S(theta) = a0 + a1 * cos(theta - phi)
     a0 = isotropic (conduction) response, a1 = advective anisotropy, phi = flow azimuth.
     Computed exactly via the discrete Fourier first harmonic.
  5. Horizontal vector = amplitude-weighted mean of the three ring harmonics.
  6. Vertical component from the linear gradient of a0 across ring elevations
     (top warmer than bottom => upward flow).
  7. Normalised so components are dimensionless ratios of the isotropic response,
     which cancels pulse-energy differences between cycles.

Nothing here converts to velocity -- that needs a calibration against known
discharge. The output is a direction plus a relative magnitude index.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Configuration -- edit to match the physical probe
# --------------------------------------------------------------------------

SENSORS = list("ABCDEFGH")
RINGS = [1, 2, 3]

#: Azimuth of each sensor, degrees clockwise from the probe's reference mark
#: (nominally magnetic/grid north). A at 0, then every 45 degrees.
SENSOR_AZIMUTH_DEG = {s: i * 45.0 for i, s in enumerate(SENSORS)}

#: Set True if sensor lettering runs counter-clockwise when viewed from above.
AZIMUTH_COUNTERCLOCKWISE = False

#: Elevation of each ring in metres, positive up. Ring1 is the top ring.
RING_ELEVATION_M = {1: +0.05, 2: 0.0, 3: -0.05}

#: Seconds of pre-onset record averaged for the baseline.
BASELINE_S = 60.0

#: Seconds after heater switch-off still counted as part of the response.
TAIL_S = 120.0

#: Cycles with fewer than this many samples while the heater is on are dropped.
MIN_ON_SAMPLES = 6

#: Sensor quantisation step (deg C). Used only for the SNR quality flag.
SENSOR_RESOLUTION_C = 0.0625


# --------------------------------------------------------------------------
# Loading and cleaning
# --------------------------------------------------------------------------

def column_map() -> dict[str, str]:
    """value_N -> Ring{r}_Sensor{s}."""
    out = {}
    n = 1
    for r in RINGS:
        for s in SENSORS:
            out[f"value_{n}"] = f"Ring{r}_Sensor{s}"
            n += 1
    out[f"value_{n}"] = "Heater"
    return out


TEMP_COLS = [f"Ring{r}_Sensor{s}" for r in RINGS for s in SENSORS]


def load(path: str) -> pd.DataFrame:
    """Read an RRIV export, rename columns, sort ascending, and flag bad records.

    Two failure modes exist in the raw feed and both are flagged rather than
    silently repaired:

    ``misaligned``  the record held fewer than 25 values and the API left-padded
                    with zeros, so every reading is shifted right and the Heater
                    column actually contains a temperature.
    ``dropout``     one or more sensors reported exactly 0.0 -- a failed read,
                    not a real 0 degree C measurement.
    """
    df = pd.read_csv(path)
    df = df.rename(columns=column_map())
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)

    # A shifted record is unrecoverable without knowing how many values were
    # dropped, so mark the whole row bad.
    df["misaligned"] = ~df["Heater"].isin([0, 1])

    temps = df[TEMP_COLS]
    df["n_dropout"] = (temps == 0).sum(axis=1)
    df["dropout"] = df["n_dropout"] > 0
    df["valid"] = ~df["misaligned"] & ~df["dropout"]

    # 0.0 is a failed read; keep it out of every average.
    df[TEMP_COLS] = temps.replace(0.0, np.nan)
    df.loc[df["misaligned"], TEMP_COLS] = np.nan
    df.loc[df["misaligned"], "Heater"] = np.nan

    df["elapsed_s"] = (df["timestamp"] - df["timestamp"].iloc[0]).dt.total_seconds()
    return df


def data_quality(df: pd.DataFrame) -> dict:
    """Summary counters for the dashboard header."""
    dt = df["timestamp"].diff().dt.total_seconds().dropna()
    return {
        "n_rows": len(df),
        "start": df["timestamp"].min(),
        "end": df["timestamp"].max(),
        "n_valid": int(df["valid"].sum()),
        "n_misaligned": int(df["misaligned"].sum()),
        "n_dropout": int(df["dropout"].sum()),
        "pct_valid": 100.0 * df["valid"].mean(),
        "median_dt_s": float(dt.median()),
        "max_gap_s": float(dt.max()),
    }


# --------------------------------------------------------------------------
# Cycle detection
# --------------------------------------------------------------------------

def find_cycles(df: pd.DataFrame) -> pd.DataFrame:
    """Locate every heater on-period and the analysis window around it."""
    h = df["Heater"].to_numpy()
    known = ~np.isnan(h)
    idx = np.flatnonzero(known)
    hk = h[idx].astype(int)

    if len(hk) == 0:
        return pd.DataFrame()

    edges = np.flatnonzero(np.diff(hk) != 0) + 1
    starts = np.r_[0, edges]
    stops = np.r_[edges, len(hk)]

    t = df["timestamp"].to_numpy()
    rows = []
    for a, b in zip(starts, stops):
        if hk[a] != 1:
            continue
        i0, i1 = idx[a], idx[b - 1]
        on_s = (t[i1] - t[i0]) / np.timedelta64(1, "s")
        rows.append(
            {
                "on_start": df["timestamp"].iloc[i0],
                "on_end": df["timestamp"].iloc[i1],
                "i_on_start": i0,
                "i_on_end": i1,
                "on_duration_s": on_s,
                "n_on_samples": b - a,
            }
        )

    cyc = pd.DataFrame(rows)
    if cyc.empty:
        return cyc
    cyc.insert(0, "cycle", np.arange(1, len(cyc) + 1))
    cyc["usable"] = cyc["n_on_samples"] >= MIN_ON_SAMPLES
    return cyc


# --------------------------------------------------------------------------
# Vector decomposition
# --------------------------------------------------------------------------

def _azimuths_rad() -> np.ndarray:
    a = np.array([SENSOR_AZIMUTH_DEG[s] for s in SENSORS], float)
    if AZIMUTH_COUNTERCLOCKWISE:
        a = -a
    return np.deg2rad(a)


def _first_harmonic(values: np.ndarray, theta: np.ndarray):
    """Fit S(theta) = a0 + a1*cos(theta - phi) to 8 points around a ring.

    Returns (a0, a1, phi_rad, r_squared, n_used). NaN sensors are dropped and the
    harmonic is solved by least squares so partial rings still resolve.
    """
    ok = np.isfinite(values)
    n = int(ok.sum())
    if n < 4:
        return np.nan, np.nan, np.nan, np.nan, n

    v, th = values[ok], theta[ok]
    design = np.column_stack([np.ones(n), np.cos(th), np.sin(th)])
    coef, *_ = np.linalg.lstsq(design, v, rcond=None)
    a0, c, s = coef

    a1 = float(np.hypot(c, s))
    phi = float(np.arctan2(s, c))

    fit = design @ coef
    ss_res = float(np.sum((v - fit) ** 2))
    ss_tot = float(np.sum((v - v.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return float(a0), a1, phi, r2, n


def analyse_cycle(df: pd.DataFrame, cycle_row: pd.Series) -> dict:
    """Baseline-correct, integrate, and decompose one heater cycle."""
    t0 = cycle_row["on_start"]
    t1 = cycle_row["on_end"]

    base = df[(df["timestamp"] < t0) & (df["timestamp"] >= t0 - pd.Timedelta(seconds=BASELINE_S))]
    win = df[(df["timestamp"] >= t0) & (df["timestamp"] <= t1 + pd.Timedelta(seconds=TAIL_S))]

    out = {
        "cycle": int(cycle_row["cycle"]),
        "on_start": t0,
        "on_duration_s": cycle_row["on_duration_s"],
        "n_on_samples": int(cycle_row["n_on_samples"]),
        "n_baseline_samples": len(base),
        "n_window_samples": len(win),
    }

    if len(base) < 2 or len(win) < 3:
        out["status"] = "insufficient samples"
        return out

    baseline = base[TEMP_COLS].mean()
    dT = win[TEMP_COLS] - baseline
    secs = (win["timestamp"] - t0).dt.total_seconds().to_numpy()

    # Integrate dT over the window; K.s, so a longer or hotter pulse scales up.
    integral = {}
    peak = {}
    for c in TEMP_COLS:
        y = dT[c].to_numpy()
        m = np.isfinite(y)
        integral[c] = np.trapezoid(y[m], secs[m]) if m.sum() >= 2 else np.nan
        peak[c] = np.nanmax(y) if m.any() else np.nan
    integral = pd.Series(integral)
    out["peak_dT_max_C"] = float(np.nanmax(list(peak.values())))
    out["peak_dT_mean_C"] = float(np.nanmean(list(peak.values())))

    # Residual scatter of the baseline is the noise this cycle has to beat.
    noise = float(np.nanmean(base[TEMP_COLS].std().to_numpy()))
    out["baseline_noise_C"] = noise
    out["snr"] = out["peak_dT_mean_C"] / noise if noise > 0 else np.nan
    out["above_resolution"] = bool(out["peak_dT_mean_C"] > SENSOR_RESOLUTION_C)

    theta = _azimuths_rad()
    per_ring = {}
    for r in RINGS:
        vals = np.array([integral[f"Ring{r}_Sensor{s}"] for s in SENSORS], float)
        a0, a1, phi, r2, n_used = _first_harmonic(vals, theta)
        per_ring[r] = dict(a0=a0, a1=a1, phi=phi, r2=r2, n=n_used)
        out[f"ring{r}_isotropic"] = a0
        out[f"ring{r}_anisotropy"] = a1
        out[f"ring{r}_azimuth_deg"] = np.degrees(phi) % 360 if np.isfinite(phi) else np.nan
        out[f"ring{r}_fit_r2"] = r2
        out[f"ring{r}_n_sensors"] = n_used

    # Horizontal: complex mean of the ring harmonics, weighted by isotropic
    # response so a ring that barely warmed does not dominate the direction.
    z = 0j
    wsum = 0.0
    for r in RINGS:
        p = per_ring[r]
        if not (np.isfinite(p["a1"]) and np.isfinite(p["a0"]) and p["a0"] > 0):
            continue
        w = p["a0"]
        z += w * p["a1"] * np.exp(1j * p["phi"])
        wsum += w
    if wsum > 0:
        z /= wsum
        vx, vy = float(z.real), float(z.imag)
    else:
        vx = vy = np.nan

    # Vertical: slope of isotropic response against ring elevation.
    zs, a0s = [], []
    for r in RINGS:
        if np.isfinite(per_ring[r]["a0"]):
            zs.append(RING_ELEVATION_M[r])
            a0s.append(per_ring[r]["a0"])
    if len(zs) >= 2:
        slope = np.polyfit(zs, a0s, 1)[0]          # K.s per metre
        span = max(RING_ELEVATION_M.values()) - min(RING_ELEVATION_M.values())
        vz = float(slope * span / 2.0)             # same units as vx, vy
        iso_mean = float(np.mean(a0s))
    else:
        vz = np.nan
        iso_mean = np.nan

    out["iso_mean"] = iso_mean
    out["vx"] = vx
    out["vy"] = vy
    out["vz"] = vz

    # Noise on an integrated response: per-sample sigma carried across the window
    # and beaten down by the sample count. Dividing by an isotropic term smaller
    # than this produces a magnitude that is pure rounding error, so gate on it.
    span_s = float(secs.max() - secs.min()) if len(secs) else 0.0
    iso_noise = noise * span_s / np.sqrt(max(len(win), 1)) if np.isfinite(noise) else np.nan
    out["iso_noise"] = iso_noise
    out["reliable"] = bool(
        np.isfinite(iso_mean) and np.isfinite(iso_noise) and iso_mean > 3.0 * iso_noise
    )

    # Dimensionless: fraction of the isotropic response that is directional.
    if out["reliable"]:
        out["vx_norm"] = vx / iso_mean
        out["vy_norm"] = vy / iso_mean
        out["vz_norm"] = vz / iso_mean
    else:
        out["vx_norm"] = out["vy_norm"] = out["vz_norm"] = np.nan

    v = np.array([out["vx_norm"], out["vy_norm"], out["vz_norm"]], float)
    out["magnitude"] = float(np.linalg.norm(v)) if np.all(np.isfinite(v)) else np.nan
    out["horizontal_magnitude"] = float(np.hypot(v[0], v[1])) if np.all(np.isfinite(v[:2])) else np.nan
    out["azimuth_deg"] = float(np.degrees(np.arctan2(v[1], v[0])) % 360) if np.all(np.isfinite(v[:2])) else np.nan
    out["inclination_deg"] = (
        float(np.degrees(np.arctan2(v[2], np.hypot(v[0], v[1])))) if np.all(np.isfinite(v)) else np.nan
    )

    r2s = [per_ring[r]["r2"] for r in RINGS if np.isfinite(per_ring[r]["r2"])]
    out["mean_fit_r2"] = float(np.mean(r2s)) if r2s else np.nan
    out["status"] = "ok"
    return out


def analyse_all(df: pd.DataFrame, cycles: pd.DataFrame | None = None) -> pd.DataFrame:
    """Run analyse_cycle over every usable cycle."""
    if cycles is None:
        cycles = find_cycles(df)
    if cycles.empty:
        return pd.DataFrame()
    rows = [analyse_cycle(df, c) for _, c in cycles[cycles["usable"]].iterrows()]
    return pd.DataFrame(rows)


def stack_cycles(df: pd.DataFrame, cycles: pd.DataFrame, cycle_ids=None) -> dict:
    """Average many cycles into one before decomposing.

    Per-cycle response here sits close to the sensor quantisation step, so a
    single pulse gives a direction dominated by rounding. Stacking N cycles cuts
    the noise by sqrt(N) while the advective signal, which has a fixed azimuth,
    adds coherently.
    """
    use = cycles[cycles["usable"]]
    if cycle_ids is not None:
        use = use[use["cycle"].isin(cycle_ids)]
    if use.empty:
        return {"status": "no cycles"}

    theta = _azimuths_rad()
    stack = []
    for _, c in use.iterrows():
        t0, t1 = c["on_start"], c["on_end"]
        base = df[(df["timestamp"] < t0) & (df["timestamp"] >= t0 - pd.Timedelta(seconds=BASELINE_S))]
        win = df[(df["timestamp"] >= t0) & (df["timestamp"] <= t1 + pd.Timedelta(seconds=TAIL_S))]
        if len(base) < 2 or len(win) < 3:
            continue
        dT = win[TEMP_COLS] - base[TEMP_COLS].mean()
        secs = (win["timestamp"] - t0).dt.total_seconds().to_numpy()
        vals = []
        for col in TEMP_COLS:
            y = dT[col].to_numpy()
            m = np.isfinite(y)
            # Normalise by duration so short and long pulses stack comparably.
            if m.sum() >= 2:
                span_s = float(secs[m].max() - secs[m].min())
                vals.append(np.trapezoid(y[m], secs[m]) / max(span_s, 1e-9))
            else:
                vals.append(np.nan)
        stack.append(vals)

    if not stack:
        return {"status": "no cycles"}

    S = np.nanmean(np.array(stack, float), axis=0)
    sd = np.nanstd(np.array(stack, float), axis=0)
    n = len(stack)

    out = {"status": "ok", "n_cycles": n, "cycles": list(use["cycle"])}
    per_ring = {}
    for i, r in enumerate(RINGS):
        vals = S[i * 8:(i + 1) * 8]
        a0, a1, phi, r2, n_used = _first_harmonic(vals, theta)
        per_ring[r] = dict(a0=a0, a1=a1, phi=phi, r2=r2)
        out[f"ring{r}_azimuth_deg"] = np.degrees(phi) % 360 if np.isfinite(phi) else np.nan
        out[f"ring{r}_anisotropy"] = a1
        out[f"ring{r}_isotropic"] = a0
        out[f"ring{r}_fit_r2"] = r2

    out["stacked_response"] = dict(zip(TEMP_COLS, S))
    out["stacked_sem"] = dict(zip(TEMP_COLS, sd / np.sqrt(n)))

    z, wsum = 0j, 0.0
    for r in RINGS:
        p = per_ring[r]
        if np.isfinite(p["a1"]) and np.isfinite(p["a0"]) and p["a0"] > 0:
            z += p["a0"] * p["a1"] * np.exp(1j * p["phi"])
            wsum += p["a0"]
    zs = [RING_ELEVATION_M[r] for r in RINGS if np.isfinite(per_ring[r]["a0"])]
    a0s = [per_ring[r]["a0"] for r in RINGS if np.isfinite(per_ring[r]["a0"])]

    if wsum > 0 and len(zs) >= 2:
        z /= wsum
        iso = float(np.mean(a0s))
        slope = np.polyfit(zs, a0s, 1)[0]
        span = max(RING_ELEVATION_M.values()) - min(RING_ELEVATION_M.values())
        vz = slope * span / 2.0
        out["vx_norm"] = z.real / iso
        out["vy_norm"] = z.imag / iso
        out["vz_norm"] = vz / iso
        v = np.array([out["vx_norm"], out["vy_norm"], out["vz_norm"]])
        out["magnitude"] = float(np.linalg.norm(v))
        out["azimuth_deg"] = float(np.degrees(np.arctan2(v[1], v[0])) % 360)
        out["inclination_deg"] = float(np.degrees(np.arctan2(v[2], np.hypot(v[0], v[1]))))
    return out
