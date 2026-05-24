import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch


# ============================================================
# IO helpers
# ============================================================


def load_csv_numeric(path: Path, delimiter: str = ",") -> np.ndarray:
    """Robust numeric CSV loader that tolerates header rows."""
    try:
        arr = np.loadtxt(path, delimiter=delimiter)
    except Exception:
        arr = np.genfromtxt(path, delimiter=delimiter, dtype=float)
        if arr.ndim == 1:
            arr = arr[None, :]
        arr = arr[~np.all(np.isnan(arr), axis=1)]
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D numeric CSV at {path}, got shape={arr.shape}")
    return arr.astype(np.float64)


def infer_time_scale_divisor(t: np.ndarray) -> float:
    dt = np.diff(np.asarray(t, dtype=np.float64))
    dt = dt[np.isfinite(dt)]
    if dt.size == 0:
        return 1.0
    med = float(np.median(np.abs(dt)))
    if med > 1e3:
        return 1e6   # microseconds-ish
    if med > 10:
        return 1e3   # milliseconds-ish
    return 1.0      # already seconds-ish


def timestamps_to_seconds_absolute(t_raw: np.ndarray, time_unit: str) -> np.ndarray:
    t_raw = np.asarray(t_raw, dtype=np.float64)
    if time_unit == "auto":
        div = infer_time_scale_divisor(t_raw)
    elif time_unit == "us":
        div = 1e6
    elif time_unit == "ms":
        div = 1e3
    elif time_unit == "s":
        div = 1.0
    else:
        raise ValueError(f"Unknown time_unit={time_unit}")
    return t_raw / div


# ============================================================
# Basic cleaning / interpolation / smoothing
# ============================================================


def sort_and_unique_time(t: np.ndarray, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    t = np.asarray(t, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    good = np.isfinite(t) & np.isfinite(x).all(axis=1)
    t = t[good]
    x = x[good]
    order = np.argsort(t)
    t = t[order]
    x = x[order]
    if t.size == 0:
        raise ValueError("No valid samples remain after cleaning.")
    uniq_t, uniq_idx = np.unique(t, return_index=True)
    return uniq_t, x[uniq_idx]


def interp_columns(t_src: np.ndarray, x_src: np.ndarray, t_tgt: np.ndarray) -> np.ndarray:
    x_src = np.asarray(x_src, dtype=np.float64)
    if x_src.ndim == 1:
        x_src = x_src[:, None]
    out = np.empty((t_tgt.shape[0], x_src.shape[1]), dtype=np.float64)
    for j in range(x_src.shape[1]):
        out[:, j] = np.interp(t_tgt, t_src, x_src[:, j])
    return out


def moving_average_same(x: np.ndarray, window: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if window <= 1:
        return x.copy()
    if x.ndim == 1:
        x = x[:, None]
        squeeze = True
    else:
        squeeze = False
    k = int(window)
    pad_l = k // 2
    pad_r = k - 1 - pad_l
    out = np.empty_like(x)
    kernel = np.ones(k, dtype=np.float64) / float(k)
    for d in range(x.shape[1]):
        xp = np.pad(x[:, d], (pad_l, pad_r), mode="edge")
        out[:, d] = np.convolve(xp, kernel, mode="valid")
    return out[:, 0] if squeeze else out


def thin_by_min_dt_keep_indices(t_abs: np.ndarray, min_dt: float) -> np.ndarray:
    if t_abs.size == 0:
        return np.zeros((0,), dtype=np.int64)
    kept = [0]
    last_t = float(t_abs[0])
    for i in range(1, t_abs.size):
        if float(t_abs[i]) - last_t >= min_dt:
            kept.append(i)
            last_t = float(t_abs[i])
    return np.asarray(kept, dtype=np.int64)


def choose_best_signed_axis(meas_3d: np.ndarray, ref_1d: np.ndarray) -> Tuple[int, float, float]:
    """
    在 3 个轴里自动选择与 ref_1d 最相关的轴及符号。
    返回: (best_axis, sign, abs_corr)
    """
    meas_3d = np.asarray(meas_3d, dtype=np.float64)
    ref_1d = np.asarray(ref_1d, dtype=np.float64).reshape(-1)

    ref = ref_1d - np.nanmean(ref_1d)
    ref_std = float(np.nanstd(ref))
    if ref_std < 1e-12:
        return 0, 1.0, 0.0

    best_axis = 0
    best_sign = 1.0
    best_score = -1.0

    for j in range(meas_3d.shape[1]):
        x = meas_3d[:, j]
        x = x - np.nanmean(x)
        x_std = float(np.nanstd(x))
        if x_std < 1e-12:
            continue

        corr = float(np.corrcoef(x, ref)[0, 1])
        if not np.isfinite(corr):
            continue

        score = abs(corr)
        sign = 1.0 if corr >= 0 else -1.0

        if score > best_score:
            best_score = score
            best_axis = j
            best_sign = sign

    if best_score < 0:
        return 0, 1.0, 0.0

    return best_axis, best_sign, best_score


def robust_scale_1d(x: np.ndarray, floor: float = 1e-3) -> float:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return floor
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    sigma = 1.4826 * mad
    sigma = max(sigma, float(np.std(x)), floor)
    return sigma


def quality_from_real_imu_residual(y_meas: np.ndarray, y_ref: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """Residual-based quality for real IMU after axis/sign calibration."""
    y_meas = np.asarray(y_meas, dtype=np.float64)
    y_ref = np.asarray(y_ref, dtype=np.float64)
    valid_mask = np.asarray(valid_mask, dtype=np.float64).reshape(-1)

    res = y_meas - y_ref
    s0 = robust_scale_1d(res[:, 0], floor=0.05)
    s1 = robust_scale_1d(res[:, 1], floor=0.01)

    z0 = np.abs(res[:, 0]) / s0
    z1 = np.abs(res[:, 1]) / s1
    q = np.exp(-0.35 * z0 - 0.35 * z1)
    q *= valid_mask
    q = np.clip(q, 0.0, 1.0)
    return q[:, None].astype(np.float32)


# ============================================================
# Stable derivative estimation: local polynomial fit
# ============================================================


def _poly_derivative_at_center(y_window: np.ndarray, tau: np.ndarray, order: int, deriv: int) -> float:
    A = np.vander(tau, N=order + 1, increasing=True)
    coef, *_ = np.linalg.lstsq(A, y_window, rcond=None)
    if deriv > order:
        return 0.0
    return float(math.factorial(deriv) * coef[deriv])


def local_poly_derivative_uniform(
    x: np.ndarray,
    dt: float,
    window: int = 9,
    order: int = 3,
    deriv: int = 1,
) -> np.ndarray:
    if window % 2 == 0:
        raise ValueError("window must be odd")
    if order >= window:
        raise ValueError("order must be < window")
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
        squeeze = True
    else:
        squeeze = False

    T, D = x.shape
    if T < 3:
        out = np.zeros_like(x)
        return out[:, 0] if squeeze else out

    half = window // 2
    out = np.zeros_like(x, dtype=np.float64)
    for i in range(T):
        left = max(0, i - half)
        right = min(T, i + half + 1)
        idx = np.arange(left, right)
        if idx.size < order + 1:
            if left == 0:
                idx = np.arange(0, min(T, order + 1))
            else:
                idx = np.arange(max(0, T - (order + 1)), T)
        tau = (idx - i).astype(np.float64) * dt
        yw = x[idx]
        for d in range(D):
            out[i, d] = _poly_derivative_at_center(yw[:, d], tau, order, deriv)

    return out[:, 0] if squeeze else out


# ============================================================
# Geometry / angle helpers
# ============================================================


def wrap_to_pi(x: np.ndarray) -> np.ndarray:
    return (x + np.pi) % (2 * np.pi) - np.pi


def latlon_to_local_xy_m(lat_rad: np.ndarray, lon_rad: np.ndarray, lat0_rad: float, lon0_rad: float) -> np.ndarray:
    """Simple local tangent-plane approximation, enough for dataset construction."""
    R = 6378137.0
    x_east = (lon_rad - lon0_rad) * math.cos(lat0_rad) * R
    y_north = (lat_rad - lat0_rad) * R
    # NCLT ground plane is often discussed as x=north, y=east. Return [north, east].
    return np.stack([y_north, x_east], axis=1)


# ============================================================
# Latent truth construction
# ============================================================


def build_uniform_time_grid(start_s: float, end_s: float, target_hz: float) -> np.ndarray:
    dt = 1.0 / float(target_hz)
    if end_s - start_s < 2 * dt:
        raise ValueError(f"Time range too short: start={start_s:.6f}, end={end_s:.6f}, dt={dt:.6f}")
    n = int(math.floor((end_s - start_s) / dt)) + 1
    return start_s + np.arange(n, dtype=np.float64) * dt


def infer_yaw_from_xy(gt_xy: np.ndarray, dt: float, poly_window: int, poly_order: int) -> np.ndarray:
    vel = local_poly_derivative_uniform(gt_xy, dt, window=poly_window, order=poly_order, deriv=1)
    yaw = np.arctan2(vel[:, 1], vel[:, 0])
    yaw = np.unwrap(yaw)
    return yaw


def build_latent_state_ctra6(
    gt_t_abs: np.ndarray,
    gt_xy: np.ndarray,
    *,
    t_latent_abs: np.ndarray,
    latent_dt: float,
    poly_window: int,
    poly_order: int,
    yaw_src_abs: Optional[np.ndarray] = None,
    yaw_src_val: Optional[np.ndarray] = None,
    smooth_window: int = 5,
    yaw_smooth_window: Optional[int] = None,
    prefer_gt_yaw: bool = True,
) -> np.ndarray:
    """
    Build 6D latent state:
        [px, py, v, yaw, a, yaw_rate]
    Shape: (T, 6)

    修正版语义：
    1) 若提供 GT yaw，则优先使用 GT yaw 构造 yaw / yaw_rate；
    2) 仅在 GT yaw 缺失或非法时，才回退到 xy 切向 yaw；
    3) yaw 的平滑窗口允许单独配置，避免由 xy 切向求导导致的 yaw_rate 尖峰污染。
    """
    gt_xy_ref = interp_columns(gt_t_abs, gt_xy, t_latent_abs)
    vel_xy = local_poly_derivative_uniform(gt_xy_ref, latent_dt, window=poly_window, order=poly_order, deriv=1)
    v = np.linalg.norm(vel_xy, axis=1)

    yaw_tangent = infer_yaw_from_xy(gt_xy_ref, latent_dt, poly_window, poly_order)

    if yaw_src_abs is not None and yaw_src_val is not None and prefer_gt_yaw:
        yaw_src_val = np.asarray(yaw_src_val, dtype=np.float64).reshape(-1)
        good = np.isfinite(yaw_src_abs) & np.isfinite(yaw_src_val)
        if good.sum() >= 2:
            yaw_src_abs_good = np.asarray(yaw_src_abs, dtype=np.float64)[good]
            yaw_src_val_good = np.unwrap(yaw_src_val[good])
            yaw_gt = np.interp(t_latent_abs, yaw_src_abs_good, yaw_src_val_good)
            yaw = yaw_gt.copy()
            bad = ~np.isfinite(yaw)
            if np.any(bad):
                yaw[bad] = yaw_tangent[bad]
        else:
            yaw = yaw_tangent
    else:
        yaw = yaw_tangent

    yaw = np.unwrap(yaw)

    v_s = moving_average_same(v, smooth_window)
    yaw_s_window = int(smooth_window if yaw_smooth_window is None else yaw_smooth_window)
    yaw_s = moving_average_same(yaw, yaw_s_window)

    a = local_poly_derivative_uniform(v_s, latent_dt, window=poly_window, order=poly_order, deriv=1)
    yaw_rate = local_poly_derivative_uniform(yaw_s, latent_dt, window=poly_window, order=poly_order, deriv=1)

    state = np.stack(
        [
            gt_xy_ref[:, 0],
            gt_xy_ref[:, 1],
            v_s,
            wrap_to_pi(yaw_s),
            a,
            yaw_rate,
        ],
        axis=1,
    )
    return state


def build_maneuver_labels(x_gt: np.ndarray, a_thr: float, yaw_rate_thr: float, strong_a_thr: float, strong_yaw_thr: float) -> np.ndarray:
    """
    Labels:
        0 = cruise / straight
        1 = accel / decel
        2 = steady turn
        3 = aggressive maneuver
    """
    a = np.abs(x_gt[:, 4])
    w = np.abs(x_gt[:, 5])
    lab = np.zeros(x_gt.shape[0], dtype=np.int64)
    lab[(a >= a_thr) & (w < yaw_rate_thr)] = 1
    lab[(w >= yaw_rate_thr) & (a < a_thr)] = 2
    lab[(a >= strong_a_thr) | (w >= strong_yaw_thr)] = 3
    return lab


# ============================================================
# Sensor schedule helpers
# ============================================================


def thin_by_min_dt(t_abs: np.ndarray, min_dt: float) -> np.ndarray:
    if t_abs.size == 0:
        return t_abs
    kept = [0]
    last_t = float(t_abs[0])
    for i in range(1, t_abs.size):
        if float(t_abs[i]) - last_t >= min_dt:
            kept.append(i)
            last_t = float(t_abs[i])
    return t_abs[np.asarray(kept, dtype=np.int64)]


def add_timestamp_jitter(rng: np.random.Generator, t_abs: np.ndarray, jitter_std: float, clamp: float) -> np.ndarray:
    if jitter_std <= 0.0:
        return t_abs.copy()
    noise = rng.normal(0.0, jitter_std, size=t_abs.shape[0])
    if clamp > 0.0:
        noise = np.clip(noise, -clamp, clamp)
    t_j = t_abs + noise
    t_j = np.maximum.accumulate(t_j)  # keep monotonic; weird but practical
    return t_j


def crop_schedule_to_range(t_abs: np.ndarray, start_s: float, end_s: float) -> np.ndarray:
    keep = (t_abs >= start_s) & (t_abs <= end_s)
    return t_abs[keep]


def keep_gps_valid_rows(gps_raw: np.ndarray, fix_mode_col: int, min_fix_mode: int) -> np.ndarray:
    return gps_raw[gps_raw[:, fix_mode_col] >= float(min_fix_mode)]


# ============================================================
# Degradation models
# ============================================================


@dataclass
class SensorProtocol:
    bernoulli_keep: float
    gap_lengths_sec: Sequence[float]
    n_gaps: int
    noise_std: Sequence[float]
    outlier_prob: float
    outlier_std: Sequence[float]
    bias_rw_std_per_s: Sequence[float]
    bias_burst_prob: float
    bias_burst_std: Sequence[float]
    scale_rw_std_per_s: float = 0.0
    scale_burst_prob: float = 0.0
    scale_burst_std: float = 0.0


@dataclass
class DayData:
    t_event: np.ndarray
    x_gt: np.ndarray
    y_imu: np.ndarray
    m_imu: np.ndarray
    q_imu: np.ndarray
    y_odom: np.ndarray
    m_odom: np.ndarray
    q_odom: np.ndarray
    y_gps: np.ndarray
    m_gps: np.ndarray
    q_gps: np.ndarray
    dt_global: np.ndarray
    dt_imu_last: np.ndarray
    dt_odom_last: np.ndarray
    dt_gps_last: np.ndarray
    maneuver_label: np.ndarray
    event_multihot: np.ndarray        # valid-measurement multihot: [imu, odom, gps]
    arrival_multihot: np.ndarray      # timestamp-arrival multihot: [imu, odom, gps]


# ============================================================
# Sensor observation generation from latent truth
# ============================================================


def sample_truth_at_times(t_latent: np.ndarray, x_latent: np.ndarray, t_sensor: np.ndarray) -> np.ndarray:
    return interp_columns(t_latent, x_latent, t_sensor)


def pick_gap_starts(rng: np.random.Generator, t_abs: np.ndarray, n_gaps: int, gap_lengths_sec: Sequence[float]) -> List[Tuple[float, float]]:
    if n_gaps <= 0 or t_abs.size == 0 or len(gap_lengths_sec) == 0:
        return []
    starts = []
    t0 = float(t_abs[0])
    t1 = float(t_abs[-1])
    if t1 - t0 < 1e-6:
        return []
    for _ in range(n_gaps):
        L = float(rng.choice(np.asarray(gap_lengths_sec, dtype=np.float64)))
        if L <= 0.0 or t1 - t0 <= L:
            continue
        s = float(rng.uniform(t0, t1 - L))
        starts.append((s, s + L))
    return starts


def apply_dropouts(rng: np.random.Generator, t_abs: np.ndarray, keep_prob: float, gap_ranges: List[Tuple[float, float]]) -> np.ndarray:
    keep = rng.random(t_abs.shape[0]) < float(keep_prob)
    if gap_ranges:
        for s, e in gap_ranges:
            keep &= ~((t_abs >= s) & (t_abs <= e))
    return keep.astype(np.float32)


def random_walk_bias(rng: np.random.Generator, t_abs: np.ndarray, std_per_s: Sequence[float]) -> np.ndarray:
    d = len(std_per_s)
    if t_abs.size == 0:
        return np.zeros((0, d), dtype=np.float64)
    out = np.zeros((t_abs.size, d), dtype=np.float64)
    dt = np.diff(t_abs, prepend=t_abs[0])
    std_per_s = np.asarray(std_per_s, dtype=np.float64)
    for i in range(1, t_abs.size):
        step_std = std_per_s * math.sqrt(max(dt[i], 0.0))
        out[i] = out[i - 1] + rng.normal(0.0, step_std)
    return out


def random_piecewise_bursts(rng: np.random.Generator, t_abs: np.ndarray, prob: float, burst_std: Sequence[float]) -> np.ndarray:
    d = len(burst_std)
    out = np.zeros((t_abs.size, d), dtype=np.float64)
    if t_abs.size == 0 or prob <= 0.0:
        return out
    active = False
    current = np.zeros(d, dtype=np.float64)
    for i in range(t_abs.size):
        if (not active) and (rng.random() < prob):
            active = True
            current = rng.normal(0.0, np.asarray(burst_std, dtype=np.float64))
            remain = int(rng.integers(5, 50))
        if active:
            out[i] = current
            remain -= 1
            if remain <= 0:
                active = False
                current = np.zeros(d, dtype=np.float64)
    return out


def scale_series_rw(rng: np.random.Generator, t_abs: np.ndarray, scale_rw_std_per_s: float, burst_prob: float, burst_std: float) -> np.ndarray:
    if t_abs.size == 0:
        return np.zeros((0,), dtype=np.float64)
    out = np.ones((t_abs.size,), dtype=np.float64)
    dt = np.diff(t_abs, prepend=t_abs[0])
    for i in range(1, t_abs.size):
        out[i] = out[i - 1] + rng.normal(0.0, scale_rw_std_per_s * math.sqrt(max(dt[i], 0.0)))
        if burst_prob > 0.0 and rng.random() < burst_prob:
            out[i] += rng.normal(0.0, burst_std)
    return out


def quality_from_components(mask: np.ndarray, noise_mag: np.ndarray, bias_mag: np.ndarray, burst_mag: np.ndarray, scale_dev: Optional[np.ndarray] = None) -> np.ndarray:
    q = np.ones((mask.shape[0],), dtype=np.float64)
    q *= np.exp(-0.15 * noise_mag)
    q *= np.exp(-0.25 * bias_mag)
    q *= np.exp(-0.20 * burst_mag)
    if scale_dev is not None:
        q *= np.exp(-1.0 * np.abs(scale_dev))
    q *= mask.astype(np.float64)
    q = np.clip(q, 0.0, 1.0)
    return q[:, None]


def synthesize_sensor_values(
    rng: np.random.Generator,
    t_sensor: np.ndarray,
    ideal: np.ndarray,
    protocol: SensorProtocol,
    *,
    apply_scale: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    gaps = pick_gap_starts(rng, t_sensor, protocol.n_gaps, protocol.gap_lengths_sec)
    mask = apply_dropouts(rng, t_sensor, protocol.bernoulli_keep, gaps)

    noise_std = np.asarray(protocol.noise_std, dtype=np.float64)
    noise = rng.normal(0.0, noise_std, size=ideal.shape)
    bias_rw = random_walk_bias(rng, t_sensor, protocol.bias_rw_std_per_s)
    bias_burst = random_piecewise_bursts(rng, t_sensor, protocol.bias_burst_prob, protocol.bias_burst_std)

    outliers = np.zeros_like(ideal)
    if protocol.outlier_prob > 0.0:
        hit = rng.random(t_sensor.shape[0]) < protocol.outlier_prob
        outliers[hit] = rng.normal(0.0, np.asarray(protocol.outlier_std, dtype=np.float64), size=(hit.sum(), ideal.shape[1]))

    scale_dev = None
    values = ideal + noise + bias_rw + bias_burst + outliers
    if apply_scale:
        scale = scale_series_rw(rng, t_sensor, protocol.scale_rw_std_per_s, protocol.scale_burst_prob, protocol.scale_burst_std)
        values = values * scale[:, None]
        scale_dev = scale - 1.0

    values = values * mask[:, None]

    noise_mag = np.linalg.norm(noise, axis=1)
    bias_mag = np.linalg.norm(bias_rw + bias_burst, axis=1)
    burst_mag = np.linalg.norm(outliers, axis=1)
    quality = quality_from_components(mask, noise_mag, bias_mag, burst_mag, scale_dev=scale_dev)
    return values, mask[:, None], quality


# ============================================================
# Day preprocessing
# ============================================================


def preprocess_one_day_multisource(
    *,
    gt_csv: Path,
    odom_csv: Path,
    gps_csv: Path,
    imu_csv: Path,
    time_unit: str,
    latent_hz: float,
    gt_xy_cols: Sequence[int],
    gt_yaw_col: Optional[int],
    odom_time_keep_hz: float,
    gps_time_keep_hz: float,
    imu_time_keep_hz: float,
    gps_fix_mode_col: int,
    min_gps_fix_mode: int,
    poly_window: int,
    poly_order: int,
    smooth_window: int,
    yaw_smooth_window: int,
    prefer_gt_yaw: bool,
    jitter_std: Dict[str, float],
    jitter_clip: Dict[str, float],
    seed: int,
    protocols: Dict[str, Dict],
    maneuver_thresholds: Dict[str, float],
    imu_acc_cols: Sequence[int],
    imu_gyro_cols: Sequence[int],
    imu_use_real_values: bool,
    imu_auto_axis_calib: bool,
) -> DayData:
    rng = np.random.default_rng(seed)

    gt_raw = load_csv_numeric(gt_csv)
    odom_raw = load_csv_numeric(odom_csv)
    gps_raw = load_csv_numeric(gps_csv)
    imu_raw = load_csv_numeric(imu_csv)

    gt_t_abs_raw = timestamps_to_seconds_absolute(gt_raw[:, 0], time_unit)
    odom_t_abs = timestamps_to_seconds_absolute(odom_raw[:, 0], time_unit)
    gps_t_abs = timestamps_to_seconds_absolute(gps_raw[:, 0], time_unit)
    imu_t_abs_raw = timestamps_to_seconds_absolute(imu_raw[:, 0], time_unit)

    gt_xy = gt_raw[:, [int(gt_xy_cols[0]), int(gt_xy_cols[1])]].astype(np.float64)
    gt_t_abs, gt_xy = sort_and_unique_time(gt_t_abs_raw, gt_xy)

    yaw_src_abs = None
    yaw_src_val = None
    if gt_yaw_col is not None and int(gt_yaw_col) < gt_raw.shape[1]:
        yaw_src_abs_raw = gt_t_abs_raw
        yaw_src_val_raw = gt_raw[:, int(gt_yaw_col)].astype(np.float64)
        yaw_src_abs, yaw_src_val2 = sort_and_unique_time(yaw_src_abs_raw, yaw_src_val_raw[:, None])
        yaw_src_val = yaw_src_val2[:, 0]

    gps_raw_valid = keep_gps_valid_rows(gps_raw, fix_mode_col=int(gps_fix_mode_col), min_fix_mode=int(min_gps_fix_mode))
    gps_t_abs_valid = timestamps_to_seconds_absolute(gps_raw_valid[:, 0], time_unit)

    odom_t_abs = np.unique(odom_t_abs)
    gps_t_abs_valid = np.unique(gps_t_abs_valid)

    if imu_use_real_values:
        max_col = max(max(int(c) for c in imu_acc_cols), max(int(c) for c in imu_gyro_cols))
        if imu_raw.shape[1] <= max_col:
            raise ValueError(
                f"IMU 文件列数不足: shape={imu_raw.shape}, 但需要访问到列 {max_col}. "
                f"请检查 imu_acc_cols / imu_gyro_cols 配置。"
            )
        imu_acc_raw = imu_raw[:, [int(c) for c in imu_acc_cols]].astype(np.float64)
        imu_gyro_raw = imu_raw[:, [int(c) for c in imu_gyro_cols]].astype(np.float64)
        imu_all_raw = np.concatenate([imu_acc_raw, imu_gyro_raw], axis=1)
        imu_t_abs, imu_all_raw = sort_and_unique_time(imu_t_abs_raw, imu_all_raw)
    else:
        imu_t_abs = np.unique(imu_t_abs_raw)
        imu_all_raw = None

    start_s = max(float(gt_t_abs[0]), float(odom_t_abs[0]), float(imu_t_abs[0]), float(gps_t_abs_valid[0]))
    end_s = min(float(gt_t_abs[-1]), float(odom_t_abs[-1]), float(imu_t_abs[-1]), float(gps_t_abs_valid[-1]))
    if end_s - start_s < 30.0:
        raise ValueError(f"Common overlap too short: {end_s - start_s:.3f}s")

    latent_dt = 1.0 / float(latent_hz)
    t_latent_abs = build_uniform_time_grid(start_s, end_s, target_hz=latent_hz)
    x_latent = build_latent_state_ctra6(
        gt_t_abs,
        gt_xy,
        t_latent_abs=t_latent_abs,
        latent_dt=latent_dt,
        poly_window=int(poly_window),
        poly_order=int(poly_order),
        yaw_src_abs=yaw_src_abs,
        yaw_src_val=yaw_src_val,
        smooth_window=int(smooth_window),
        yaw_smooth_window=int(yaw_smooth_window),
        prefer_gt_yaw=bool(prefer_gt_yaw),
    )
    maneuver_label_latent = build_maneuver_labels(
        x_latent,
        a_thr=float(maneuver_thresholds["a_thr"]),
        yaw_rate_thr=float(maneuver_thresholds["yaw_rate_thr"]),
        strong_a_thr=float(maneuver_thresholds["strong_a_thr"]),
        strong_yaw_thr=float(maneuver_thresholds["strong_yaw_thr"]),
    )

    # Build sensor schedules from real file timestamps, then optionally thin and jitter.
    odom_t = crop_schedule_to_range(odom_t_abs, start_s, end_s)
    gps_t = crop_schedule_to_range(gps_t_abs_valid, start_s, end_s)

    if imu_use_real_values:
        imu_keep = (imu_t_abs >= start_s) & (imu_t_abs <= end_s)
        imu_t = imu_t_abs[imu_keep]
        imu_all = imu_all_raw[imu_keep]
    else:
        imu_t = crop_schedule_to_range(imu_t_abs, start_s, end_s)
        imu_all = None

    odom_t = thin_by_min_dt(odom_t, 1.0 / float(odom_time_keep_hz)) if odom_time_keep_hz > 0 else odom_t
    gps_t = thin_by_min_dt(gps_t, 1.0 / float(gps_time_keep_hz)) if gps_time_keep_hz > 0 else gps_t

    if imu_time_keep_hz > 0:
        if imu_use_real_values:
            keep_idx = thin_by_min_dt_keep_indices(imu_t, 1.0 / float(imu_time_keep_hz))
            imu_t = imu_t[keep_idx]
            imu_all = imu_all[keep_idx]
        else:
            imu_t = thin_by_min_dt(imu_t, 1.0 / float(imu_time_keep_hz))

    odom_t = add_timestamp_jitter(rng, odom_t, float(jitter_std["odom"]), float(jitter_clip["odom"]))
    gps_t = add_timestamp_jitter(rng, gps_t, float(jitter_std["gps"]), float(jitter_clip["gps"]))
    imu_t = add_timestamp_jitter(rng, imu_t, float(jitter_std["imu"]), float(jitter_clip["imu"]))

    odom_t = np.unique(crop_schedule_to_range(odom_t, start_s, end_s))
    gps_t = np.unique(crop_schedule_to_range(gps_t, start_s, end_s))

    if imu_use_real_values:
        keep = (imu_t >= start_s) & (imu_t <= end_s)
        imu_t = imu_t[keep]
        imu_all = imu_all[keep]
        imu_t, imu_all = sort_and_unique_time(imu_t, imu_all)
    else:
        imu_t = np.unique(crop_schedule_to_range(imu_t, start_s, end_s))

    # Ideal sensor values sampled from latent truth.
    x_odom = sample_truth_at_times(t_latent_abs, x_latent, odom_t)
    x_gps = sample_truth_at_times(t_latent_abs, x_latent, gps_t)

    ideal_odom = np.stack([x_odom[:, 2], x_odom[:, 5]], axis=1)      # [v, yaw_rate]
    ideal_gps = np.stack([x_gps[:, 0], x_gps[:, 1]], axis=1)         # [px, py]

    odom_protocol = SensorProtocol(**protocols["odom"])
    gps_protocol = SensorProtocol(**protocols["gps"])
    imu_protocol = SensorProtocol(**protocols["imu"])

    y_odom_raw, m_odom_raw, q_odom_raw = synthesize_sensor_values(rng, odom_t, ideal_odom, odom_protocol, apply_scale=True)
    y_gps_raw, m_gps_raw, q_gps_raw = synthesize_sensor_values(rng, gps_t, ideal_gps, gps_protocol, apply_scale=False)

    if imu_use_real_values:
        imu_acc = imu_all[:, :3]
        imu_gyro = imu_all[:, 3:]

        x_imu_ref = sample_truth_at_times(t_latent_abs, x_latent, imu_t)
        a_ref = x_imu_ref[:, 4]
        yaw_rate_ref = x_imu_ref[:, 5]

        if imu_auto_axis_calib:
            acc_axis, acc_sign, acc_corr = choose_best_signed_axis(imu_acc, a_ref)
            gyro_axis, gyro_sign, gyro_corr = choose_best_signed_axis(imu_gyro, yaw_rate_ref)
        else:
            acc_axis, acc_sign, acc_corr = 0, 1.0, 0.0
            gyro_axis, gyro_sign, gyro_corr = 2, 1.0, 0.0

        a_m_real = acc_sign * imu_acc[:, acc_axis]
        yaw_rate_m_real = gyro_sign * imu_gyro[:, gyro_axis]
        y_imu_raw = np.stack([a_m_real, yaw_rate_m_real], axis=1).astype(np.float64)

        finite = np.isfinite(y_imu_raw).all(axis=1)
        m_imu_raw = finite.astype(np.float32)[:, None]
        y_imu_raw = np.where(finite[:, None], y_imu_raw, 0.0)
        q_imu_raw = quality_from_real_imu_residual(
            y_imu_raw,
            np.stack([a_ref, yaw_rate_ref], axis=1),
            m_imu_raw[:, 0],
        )

        print(
            f"[IMU-REAL] {imu_csv.name} | "
            f"acc_axis={acc_axis} sign={acc_sign:+.0f} corr={acc_corr:.4f} | "
            f"gyro_axis={gyro_axis} sign={gyro_sign:+.0f} corr={gyro_corr:.4f}"
        )
    else:
        x_imu = sample_truth_at_times(t_latent_abs, x_latent, imu_t)
        ideal_imu = np.stack([x_imu[:, 4], x_imu[:, 5]], axis=1)     # [a_m, yaw_rate_m]
        y_imu_raw, m_imu_raw, q_imu_raw = synthesize_sensor_values(rng, imu_t, ideal_imu, imu_protocol, apply_scale=False)

    # Global event timeline.
    t_event = np.unique(np.concatenate([odom_t, gps_t, imu_t]))
    t_event.sort()
    T = t_event.shape[0]

    x_gt_event = sample_truth_at_times(t_latent_abs, x_latent, t_event)
    maneuver_event = np.interp(t_event, t_latent_abs, maneuver_label_latent.astype(np.float64))
    maneuver_event = np.rint(maneuver_event).astype(np.int64)

    y_odom = np.zeros((T, 2), dtype=np.float64)
    m_odom = np.zeros((T, 1), dtype=np.float32)
    q_odom = np.zeros((T, 1), dtype=np.float32)

    y_gps = np.zeros((T, 2), dtype=np.float64)
    m_gps = np.zeros((T, 1), dtype=np.float32)
    q_gps = np.zeros((T, 1), dtype=np.float32)

    y_imu = np.zeros((T, 2), dtype=np.float64)
    m_imu = np.zeros((T, 1), dtype=np.float32)
    q_imu = np.zeros((T, 1), dtype=np.float32)

    # 两套 multihot 语义分开存：
    # 1) arrival_multihot: 时间戳到达，不关心数值是否有限；
    # 2) event_multihot  : 有效观测 multihot，与 m_* 保持一致，便于统一 reader / 审查。
    event_multihot = np.zeros((T, 3), dtype=np.float32)    # valid-measurement multihot
    arrival_multihot = np.zeros((T, 3), dtype=np.float32)  # timestamp-arrival multihot

    def _scatter_by_exact_times(
        t_src: np.ndarray,
        y_src: np.ndarray,
        m_src: np.ndarray,
        q_src: np.ndarray,
        y_tgt: np.ndarray,
        m_tgt: np.ndarray,
        q_tgt: np.ndarray,
        sensor_idx: int,
    ):
        idx = np.searchsorted(t_event, t_src)
        ok = (idx >= 0) & (idx < T) & (np.abs(t_event[idx] - t_src) < 1e-8)
        ii = idx[ok]
        y_tgt[ii] = y_src[ok]
        m_tgt[ii, 0] = m_src[ok, 0].astype(np.float32)
        q_tgt[ii, 0] = q_src[ok, 0].astype(np.float32)
        arrival_multihot[ii, sensor_idx] = 1.0
        event_multihot[ii, sensor_idx] = m_src[ok, 0].astype(np.float32)

    _scatter_by_exact_times(imu_t, y_imu_raw, m_imu_raw, q_imu_raw, y_imu, m_imu, q_imu, sensor_idx=0)
    _scatter_by_exact_times(odom_t, y_odom_raw, m_odom_raw, q_odom_raw, y_odom, m_odom, q_odom, sensor_idx=1)
    _scatter_by_exact_times(gps_t, y_gps_raw, m_gps_raw, q_gps_raw, y_gps, m_gps, q_gps, sensor_idx=2)

    # Delta times.
    dt_global = np.zeros((T,), dtype=np.float64)
    dt_global[1:] = t_event[1:] - t_event[:-1]

    def _compute_dt_last(m_src: np.ndarray) -> np.ndarray:
        out = np.zeros((T,), dtype=np.float64)
        last_t = None
        for i in range(T):
            if last_t is None:
                out[i] = 0.0
            else:
                out[i] = t_event[i] - last_t
            if m_src[i, 0] > 0.5:
                last_t = t_event[i]
        return out

    dt_imu_last = _compute_dt_last(m_imu)
    dt_odom_last = _compute_dt_last(m_odom)
    dt_gps_last = _compute_dt_last(m_gps)

    # Shift time origin to zero for storage.
    t0 = float(t_event[0])
    t_event = t_event - t0

    return DayData(
        t_event=t_event.astype(np.float32),
        x_gt=x_gt_event.astype(np.float32),
        y_imu=y_imu.astype(np.float32),
        m_imu=m_imu.astype(np.float32),
        q_imu=q_imu.astype(np.float32),
        y_odom=y_odom.astype(np.float32),
        m_odom=m_odom.astype(np.float32),
        q_odom=q_odom.astype(np.float32),
        y_gps=y_gps.astype(np.float32),
        m_gps=m_gps.astype(np.float32),
        q_gps=q_gps.astype(np.float32),
        dt_global=dt_global.astype(np.float32),
        dt_imu_last=dt_imu_last.astype(np.float32),
        dt_odom_last=dt_odom_last.astype(np.float32),
        dt_gps_last=dt_gps_last.astype(np.float32),
        maneuver_label=maneuver_event.astype(np.int64),
        event_multihot=event_multihot.astype(np.float32),
        arrival_multihot=arrival_multihot.astype(np.float32),
    )


# ============================================================
# Windowing with variable event counts
# ============================================================


def build_time_windows(
    t_event: np.ndarray,
    duration_sec: float,
    stride_sec: float,
    *,
    random_offset_sec: float,
    rng: np.random.Generator,
) -> List[Tuple[int, int]]:
    if t_event.size == 0:
        return []
    offset = 0.0
    if random_offset_sec > 0.0:
        offset = float(rng.uniform(0.0, random_offset_sec))
    start_times = []
    s = float(t_event[0] + offset)
    end_limit = float(t_event[-1])
    while s + duration_sec <= end_limit + 1e-8:
        start_times.append(s)
        s += stride_sec
    windows: List[Tuple[int, int]] = []
    for s in start_times:
        e = s + duration_sec
        i0 = int(np.searchsorted(t_event, s, side="left"))
        i1 = int(np.searchsorted(t_event, e, side="left"))
        if i1 - i0 >= 2:
            windows.append((i0, i1))
    return windows


def pad_variable_windows(arrays: List[np.ndarray], max_len: int, *, as_channels: bool) -> np.ndarray:
    if len(arrays) == 0:
        raise ValueError("No arrays to pad")
    if arrays[0].ndim == 1:
        out = np.zeros((len(arrays), max_len), dtype=arrays[0].dtype)
        for i, a in enumerate(arrays):
            out[i, : a.shape[0]] = a
        return out
    else:
        D = arrays[0].shape[1]
        if as_channels:
            out = np.zeros((len(arrays), D, max_len), dtype=arrays[0].dtype)
            for i, a in enumerate(arrays):
                out[i, :, : a.shape[0]] = a.T
        else:
            out = np.zeros((len(arrays), max_len, D), dtype=arrays[0].dtype)
            for i, a in enumerate(arrays):
                out[i, : a.shape[0], :] = a
        return out


def save_day_windows(out_dir: Path, day: DayData, duration_sec: float, stride_sec: float, *, random_offset_sec: float, seed: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    windows = build_time_windows(day.t_event.astype(np.float64), duration_sec, stride_sec, random_offset_sec=random_offset_sec, rng=rng)
    if len(windows) == 0:
        raise ValueError(f"No windows created for {out_dir.name}")

    seqs = {
        "t_global": [],
        "event_valid": [],
        "x_gt": [],
        "y_imu": [],
        "m_imu": [],
        "q_imu": [],
        "y_odom": [],
        "m_odom": [],
        "q_odom": [],
        "y_gps": [],
        "m_gps": [],
        "q_gps": [],
        "dt_global": [],
        "dt_imu_last": [],
        "dt_odom_last": [],
        "dt_gps_last": [],
        "maneuver_label": [],
        "event_multihot": [],
        "arrival_multihot": [],
    }

    lens = []
    for i0, i1 in windows:
        lens.append(i1 - i0)
        t_rel = day.t_event[i0:i1].astype(np.float64).copy()
        t_rel = t_rel - t_rel[0]

        m_imu = day.m_imu[i0:i1]
        m_odom = day.m_odom[i0:i1]
        m_gps = day.m_gps[i0:i1]

        dt_global_local = _compute_dt_global_from_t_rel(t_rel)
        dt_imu_last_local = _compute_dt_last_from_mask_and_t_rel(m_imu[:, 0], t_rel)
        dt_odom_last_local = _compute_dt_last_from_mask_and_t_rel(m_odom[:, 0], t_rel)
        dt_gps_last_local = _compute_dt_last_from_mask_and_t_rel(m_gps[:, 0], t_rel)
        event_multihot_local = np.concatenate([m_imu, m_odom, m_gps], axis=1).astype(np.float32)

        seqs["t_global"].append(t_rel.astype(np.float32))
        seqs["event_valid"].append(np.ones((i1 - i0,), dtype=np.float32))
        seqs["x_gt"].append(day.x_gt[i0:i1])
        seqs["y_imu"].append(day.y_imu[i0:i1])
        seqs["m_imu"].append(m_imu)
        seqs["q_imu"].append(day.q_imu[i0:i1])
        seqs["y_odom"].append(day.y_odom[i0:i1])
        seqs["m_odom"].append(m_odom)
        seqs["q_odom"].append(day.q_odom[i0:i1])
        seqs["y_gps"].append(day.y_gps[i0:i1])
        seqs["m_gps"].append(m_gps)
        seqs["q_gps"].append(day.q_gps[i0:i1])
        seqs["dt_global"].append(dt_global_local.astype(np.float32))
        seqs["dt_imu_last"].append(dt_imu_last_local.astype(np.float32))
        seqs["dt_odom_last"].append(dt_odom_last_local.astype(np.float32))
        seqs["dt_gps_last"].append(dt_gps_last_local.astype(np.float32))
        seqs["maneuver_label"].append(day.maneuver_label[i0:i1])
        seqs["event_multihot"].append(event_multihot_local)
        seqs["arrival_multihot"].append(day.arrival_multihot[i0:i1])

    max_len = int(max(lens))

    save_map = {
        "t_global": pad_variable_windows(seqs["t_global"], max_len, as_channels=False),
        "event_valid": pad_variable_windows(seqs["event_valid"], max_len, as_channels=False),
        "x_gt": pad_variable_windows(seqs["x_gt"], max_len, as_channels=True),
        "y_imu": pad_variable_windows(seqs["y_imu"], max_len, as_channels=True),
        "m_imu": pad_variable_windows(seqs["m_imu"], max_len, as_channels=True),
        "q_imu": pad_variable_windows(seqs["q_imu"], max_len, as_channels=True),
        "y_odom": pad_variable_windows(seqs["y_odom"], max_len, as_channels=True),
        "m_odom": pad_variable_windows(seqs["m_odom"], max_len, as_channels=True),
        "q_odom": pad_variable_windows(seqs["q_odom"], max_len, as_channels=True),
        "y_gps": pad_variable_windows(seqs["y_gps"], max_len, as_channels=True),
        "m_gps": pad_variable_windows(seqs["m_gps"], max_len, as_channels=True),
        "q_gps": pad_variable_windows(seqs["q_gps"], max_len, as_channels=True),
        "dt_global": pad_variable_windows(seqs["dt_global"], max_len, as_channels=False),
        "dt_imu_last": pad_variable_windows(seqs["dt_imu_last"], max_len, as_channels=False),
        "dt_odom_last": pad_variable_windows(seqs["dt_odom_last"], max_len, as_channels=False),
        "dt_gps_last": pad_variable_windows(seqs["dt_gps_last"], max_len, as_channels=False),
        "maneuver_label": pad_variable_windows(seqs["maneuver_label"], max_len, as_channels=False),
        "event_multihot": pad_variable_windows(seqs["event_multihot"], max_len, as_channels=True),
        "arrival_multihot": pad_variable_windows(seqs["arrival_multihot"], max_len, as_channels=True),
    }

    for name, arr in save_map.items():
        torch.save(torch.from_numpy(arr), out_dir / f"{name}.pt")

    meta = {
        "num_windows": int(len(windows)),
        "max_events": int(max_len),
        "event_count_min": int(min(lens)),
        "event_count_mean": float(np.mean(lens)),
        "event_count_max": int(max(lens)),
        "duration_sec": float(duration_sec),
        "stride_sec": float(stride_sec),
        "x_gt_shape": list(save_map["x_gt"].shape),
        "y_imu_shape": list(save_map["y_imu"].shape),
        "y_odom_shape": list(save_map["y_odom"].shape),
        "y_gps_shape": list(save_map["y_gps"].shape),
        "t_global_shape": list(save_map["t_global"].shape),
        "arrival_multihot_shape": list(save_map["arrival_multihot"].shape),
    }
    dt = np.diff(day.t_event.astype(np.float64))
    if dt.size > 0:
        meta["global_dt_min"] = float(dt.min())
        meta["global_dt_mean"] = float(dt.mean())
        meta["global_dt_max"] = float(dt.max())
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"[{out_dir.name}] windows={meta['num_windows']} | max_events={meta['max_events']} | mean_events={meta['event_count_mean']:.1f}")


# ============================================================
# Stats
# ============================================================


def masked_mean_std_channels(arr_nct: np.ndarray, mask_n1t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mask = mask_n1t.astype(np.float64)
    num = (arr_nct.astype(np.float64) * mask).sum(axis=(0, 2))
    den = mask.sum(axis=(0, 2))
    den = np.maximum(den, 1.0)
    mean = num / den
    var = (((arr_nct - mean[None, :, None]) ** 2) * mask).sum(axis=(0, 2)) / den
    std = np.sqrt(np.maximum(var, 1e-12))
    return mean, std


def masked_mean_std_time(arr_nt: np.ndarray, valid_nt: np.ndarray) -> Tuple[float, float]:
    x = arr_nt.astype(np.float64)
    m = valid_nt.astype(np.float64)
    den = max(float(m.sum()), 1.0)
    mean = float((x * m).sum() / den)
    var = float((((x - mean) ** 2) * m).sum() / den)
    return mean, math.sqrt(max(var, 1e-12))


def compute_train_stats(out_root: Path) -> Dict:
    x_gt = torch.load(out_root / "train" / "x_gt.pt").numpy()
    y_imu = torch.load(out_root / "train" / "y_imu.pt").numpy()
    m_imu = torch.load(out_root / "train" / "m_imu.pt").numpy()
    y_odom = torch.load(out_root / "train" / "y_odom.pt").numpy()
    m_odom = torch.load(out_root / "train" / "m_odom.pt").numpy()
    y_gps = torch.load(out_root / "train" / "y_gps.pt").numpy()
    m_gps = torch.load(out_root / "train" / "m_gps.pt").numpy()
    dt_global = torch.load(out_root / "train" / "dt_global.pt").numpy()
    event_valid = torch.load(out_root / "train" / "event_valid.pt").numpy()

    x_mean = x_gt.mean(axis=(0, 2))
    x_std = x_gt.std(axis=(0, 2))
    x_std = np.maximum(x_std, 1e-6)

    imu_mean, imu_std = masked_mean_std_channels(y_imu, m_imu)
    odom_mean, odom_std = masked_mean_std_channels(y_odom, m_odom)
    gps_mean, gps_std = masked_mean_std_channels(y_gps, m_gps)
    dt_mean, dt_std = masked_mean_std_time(dt_global, event_valid)

    stats = {
        "state_mode": "ctra6_latent_async_multisource_realimu",
        "state_order": ["px", "py", "v", "yaw", "a", "yaw_rate"],
        "imu_obs_order": ["a_m", "yaw_rate_m"],
        "odom_obs_order": ["v_odom", "yaw_rate_odom"],
        "gps_obs_order": ["px_gps", "py_gps"],
        "state_mean": x_mean.tolist(),
        "state_std": x_std.tolist(),
        "imu_mean": imu_mean.tolist(),
        "imu_std": imu_std.tolist(),
        "odom_mean": odom_mean.tolist(),
        "odom_std": odom_std.tolist(),
        "gps_mean": gps_mean.tolist(),
        "gps_std": gps_std.tolist(),
        "dt_global_mean": float(dt_mean),
        "dt_global_std": float(dt_std),
        "recommended_train_loss_weights": {
            "pos": 1.0,
            "vel": 0.5,
            "yaw": 0.2,
            "aux": 0.1,
        },
        "position_state_indices": [0, 1],
        "velocity_scalar_index": 2,
        "yaw_index": 3,
        "acc_index": 4,
        "yaw_rate_index": 5,
    }
    with open(out_root / "train_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    return stats


# ============================================================
# Main (PyCharm direct-run style)
# ============================================================


PRESET_EXP_A = {
    "jitter_std": {
        "imu": 0.001,
        "odom": 0.005,
        "gps": 0.020,
    },
    "jitter_clip": {
        "imu": 0.003,
        "odom": 0.010,
        "gps": 0.040,
    },
    "protocols": {
        "odom": {
            "bernoulli_keep": 0.995,
            "gap_lengths_sec": [0.0],
            "n_gaps": 0,
            "noise_std": [0.08, 0.02],
            "outlier_prob": 0.0,
            "outlier_std": [0.0, 0.0],
            "bias_rw_std_per_s": [0.0, 0.0],
            "bias_burst_prob": 0.0,
            "bias_burst_std": [0.0, 0.0],
            "scale_rw_std_per_s": 0.0,
            "scale_burst_prob": 0.0,
            "scale_burst_std": 0.0,
        },
        "gps": {
            "bernoulli_keep": 0.999,
            "gap_lengths_sec": [0.0],
            "n_gaps": 0,
            "noise_std": [2.0, 2.0],
            "outlier_prob": 0.0,
            "outlier_std": [0.0, 0.0],
            "bias_rw_std_per_s": [0.0, 0.0],
            "bias_burst_prob": 0.0,
            "bias_burst_std": [0.0, 0.0],
            "scale_rw_std_per_s": 0.0,
            "scale_burst_prob": 0.0,
            "scale_burst_std": 0.0,
        },
        "imu": {
            "bernoulli_keep": 0.998,
            "gap_lengths_sec": [0.0],
            "n_gaps": 0,
            "noise_std": [0.05, 0.010],
            "outlier_prob": 0.0,
            "outlier_std": [0.0, 0.0],
            "bias_rw_std_per_s": [0.0, 0.0],
            "bias_burst_prob": 0.0,
            "bias_burst_std": [0.0, 0.0],
            "scale_rw_std_per_s": 0.0,
            "scale_burst_prob": 0.0,
            "scale_burst_std": 0.0,
        },
    },
}

PRESET_EXP_B = {
    "jitter_std": {
        "imu": 0.002,
        "odom": 0.015,
        "gps": 0.050,
    },
    "jitter_clip": {
        "imu": 0.005,
        "odom": 0.030,
        "gps": 0.100,
    },
    "protocols": {
        "odom": {
            "bernoulli_keep": 0.97,
            "gap_lengths_sec": [0.0, 0.5, 1.0],
            "n_gaps": 8,
            "noise_std": [0.10, 0.03],
            "outlier_prob": 0.003,
            "outlier_std": [1.0, 0.25],
            "bias_rw_std_per_s": [0.01, 0.002],
            "bias_burst_prob": 0.002,
            "bias_burst_std": [0.25, 0.05],
            "scale_rw_std_per_s": 0.002,
            "scale_burst_prob": 0.001,
            "scale_burst_std": 0.08,
        },
        "gps": {
            "bernoulli_keep": 0.95,
            "gap_lengths_sec": [1.0, 2.0, 5.0, 10.0],
            "n_gaps": 12,
            "noise_std": [2.5, 2.5],
            "outlier_prob": 0.015,
            "outlier_std": [12.0, 12.0],
            "bias_rw_std_per_s": [0.03, 0.03],
            "bias_burst_prob": 0.006,
            "bias_burst_std": [4.0, 4.0],
            "scale_rw_std_per_s": 0.0,
            "scale_burst_prob": 0.0,
            "scale_burst_std": 0.0,
        },
        "imu": {
            "bernoulli_keep": 0.995,
            "gap_lengths_sec": [0.0, 0.1, 0.2],
            "n_gaps": 6,
            "noise_std": [0.08, 0.015],
            "outlier_prob": 0.002,
            "outlier_std": [0.8, 0.20],
            "bias_rw_std_per_s": [0.02, 0.004],
            "bias_burst_prob": 0.004,
            "bias_burst_std": [0.4, 0.08],
            "scale_rw_std_per_s": 0.0,
            "scale_burst_prob": 0.0,
            "scale_burst_std": 0.0,
        },
    },
}

PRESET_LIBRARY = {
    "expA_easy": PRESET_EXP_A,
    "expB_longgap": PRESET_EXP_B,
}


CONFIG = {
    # 路径（None 表示默认脚本同目录下 real_data 和输出目录）
    "raw_root": None,
    "out_root": None,

    # Missing-Gap 实验使用哪个 matched 数据协议作为训练/验证底座：
    #   - expA_easy    : easy matched
    #   - expB_longgap : long-gap matched（默认，更贴近主战场）
    "base_protocol": "expA_easy",

    # 日期
    "train_date": "2012-01-22",
    "test_date": "2012-04-29",

    # 文件前缀
    "gt_prefix": "groundtruth_",
    "odom_prefix": "odometry_mu_100hz_",
    "gps_prefix": "gps_",
    "imu_prefix": "ms25_",

    # 时间单位：NCLT 基本是 us，这里保留 auto 更稳
    "time_unit": "auto",

    # latent backbone
    "latent_hz": 20.0,
    "poly_window": 9,
    "poly_order": 3,
    "smooth_window": 5,
    "yaw_smooth_window": 11,
    "prefer_gt_yaw": True,

    # GT 列号：默认第 0 列时间，第 1/2 列 x/y，第 6 列 yaw
    "gt_x_col": 1,
    "gt_y_col": 2,
    "gt_yaw_col": 6,

    # GPS 列号：第 1 列 fix_mode
    "gps_fix_mode_col": 1,
    "min_gps_fix_mode": 2,

    # 真实 IMU 列号（基于当前 ms25 文件推断）
    "imu_acc_cols": [4, 5, 6],
    "imu_gyro_cols": [7, 8, 9],
    "imu_use_real_values": True,
    "imu_auto_axis_calib": True,

    # 传感器目标保留频率（固定基准频率，变量步长由 jitter 后 union event 自动形成）
    "odom_time_keep_hz": 10.0,
    "gps_time_keep_hz": 1.0,
    "imu_time_keep_hz": 50.0,

    # 仅测试集施加更强 timestamp jitter。
    # train / valid 的 jitter 由 base_protocol 自动继承。
    # 这会自然带来：
    #   1) 更强的 timestamp 随机扰动；
    #   2) 更明显的异步到达；
    #   3) 更不规则的 event-level variable dt。
    "test_jitter_std": {
        "imu": 0.006,
        "odom": 0.040,
        "gps": 0.120,
    },
    "test_jitter_clip": {
        "imu": 0.015,
        "odom": 0.080,
        "gps": 0.250,
    },

    # 如需在 Missing-Gap 测试中同时切换观测退化协议，可在此覆盖；None 表示沿用 base_protocol。
    "test_protocols_override": None,

    # maneuver 标签阈值
    "maneuver_thresholds": {
        "a_thr": 0.30,
        "yaw_rate_thr": 0.08,
        "strong_a_thr": 1.20,
        "strong_yaw_thr": 0.35,
    },

    # 切窗：保持与现有 matched / long-gap 主线一致，便于对比
    "train_split": 0.875,
    "train_duration_sec": 20.0,
    "train_stride_sec": 4.0,
    "valid_duration_sec": 40.0,
    "valid_stride_sec": 10.0,
    "test_duration_sec": 80.0,
    "test_stride_sec": 20.0,
    "train_random_offset_sec": 4.0,
    "valid_random_offset_sec": 0.0,
    "test_random_offset_sec": 0.0,

    # 随机种子
    "seed": 42,
}



def _compute_dt_global_from_t_rel(t_rel: np.ndarray) -> np.ndarray:
    t_rel = np.asarray(t_rel, dtype=np.float64)
    out = np.zeros((t_rel.shape[0],), dtype=np.float64)
    if t_rel.shape[0] >= 2:
        out[1:] = t_rel[1:] - t_rel[:-1]
    return out


def _compute_dt_last_from_mask_and_t_rel(m_src: np.ndarray, t_rel: np.ndarray) -> np.ndarray:
    m_src = np.asarray(m_src, dtype=np.float64).reshape(-1)
    t_rel = np.asarray(t_rel, dtype=np.float64)
    out = np.zeros((t_rel.shape[0],), dtype=np.float64)
    last_t = None
    for i in range(t_rel.shape[0]):
        if last_t is None:
            out[i] = 0.0
        else:
            out[i] = float(t_rel[i] - last_t)
        if m_src[i] > 0.5:
            last_t = float(t_rel[i])
    return out


def _resolve_config(cfg: Dict) -> Dict:
    script_dir = Path(__file__).resolve().parent
    out = dict(cfg)

    base_protocol = str(out["base_protocol"])
    if base_protocol not in PRESET_LIBRARY:
        raise ValueError(f"Unknown base_protocol={base_protocol}. Expected one of {list(PRESET_LIBRARY.keys())}")

    preset = PRESET_LIBRARY[base_protocol]
    out["train_jitter_std"] = json.loads(json.dumps(preset["jitter_std"]))
    out["train_jitter_clip"] = json.loads(json.dumps(preset["jitter_clip"]))
    out["protocols"] = json.loads(json.dumps(preset["protocols"]))

    if out.get("test_protocols_override") is not None:
        out["test_protocols"] = json.loads(json.dumps(out["test_protocols_override"]))
    else:
        out["test_protocols"] = json.loads(json.dumps(out["protocols"]))

    if out["raw_root"] is None:
        out["raw_root"] = str(script_dir / "real_data")
    if out["out_root"] is None:
        suffix = "FromExpA" if base_protocol == "expA_easy" else "FromExpB"
        out["out_root"] = str(script_dir / f"Data_NCLT_MultiSourceAsync_CTRA6_RealIMU_ExpD_MissingGap_{suffix}")
    return out


def _time_slice_day(day: DayData, start_idx: int, end_idx: int) -> DayData:
    t = day.t_event[start_idx:end_idx].astype(np.float64).copy()
    t = t - t[0]

    m_imu = day.m_imu[start_idx:end_idx]
    m_odom = day.m_odom[start_idx:end_idx]
    m_gps = day.m_gps[start_idx:end_idx]

    event_multihot = np.concatenate([m_imu, m_odom, m_gps], axis=1).astype(np.float32)
    arrival_multihot = day.arrival_multihot[start_idx:end_idx]

    return DayData(
        t_event=t.astype(np.float32),
        x_gt=day.x_gt[start_idx:end_idx],
        y_imu=day.y_imu[start_idx:end_idx],
        m_imu=m_imu,
        q_imu=day.q_imu[start_idx:end_idx],
        y_odom=day.y_odom[start_idx:end_idx],
        m_odom=m_odom,
        q_odom=day.q_odom[start_idx:end_idx],
        y_gps=day.y_gps[start_idx:end_idx],
        m_gps=m_gps,
        q_gps=day.q_gps[start_idx:end_idx],
        dt_global=_compute_dt_global_from_t_rel(t).astype(np.float32),
        dt_imu_last=_compute_dt_last_from_mask_and_t_rel(m_imu[:, 0], t).astype(np.float32),
        dt_odom_last=_compute_dt_last_from_mask_and_t_rel(m_odom[:, 0], t).astype(np.float32),
        dt_gps_last=_compute_dt_last_from_mask_and_t_rel(m_gps[:, 0], t).astype(np.float32),
        maneuver_label=day.maneuver_label[start_idx:end_idx],
        event_multihot=event_multihot,
        arrival_multihot=arrival_multihot,
    )


def main():
    cfg = _resolve_config(CONFIG)
    raw_root = Path(cfg["raw_root"])
    out_root = Path(cfg["out_root"])
    out_root.mkdir(parents=True, exist_ok=True)

    train_date = cfg["train_date"]
    test_date = cfg["test_date"]

    train_files = {
        "gt": raw_root / f"{cfg['gt_prefix']}{train_date}.csv",
        "odom": raw_root / f"{cfg['odom_prefix']}{train_date}.csv",
        "gps": raw_root / f"{cfg['gps_prefix']}{train_date}.csv",
        "imu": raw_root / f"{cfg['imu_prefix']}{train_date}.csv",
    }
    test_files = {
        "gt": raw_root / f"{cfg['gt_prefix']}{test_date}.csv",
        "odom": raw_root / f"{cfg['odom_prefix']}{test_date}.csv",
        "gps": raw_root / f"{cfg['gps_prefix']}{test_date}.csv",
        "imu": raw_root / f"{cfg['imu_prefix']}{test_date}.csv",
    }

    for p in list(train_files.values()) + list(test_files.values()):
        if not p.exists():
            raise FileNotFoundError(f"缺少文件: {p}")

    print("=" * 100)
    print("Building NCLT multi-source async dataset (Exp-D Missing Gap, REAL IMU)")
    print(f"raw_root  : {raw_root}")
    print(f"out_root  : {out_root}")
    print(f"state     : [px, py, v, yaw, a, yaw_rate]")
    print(f"odom obs  : [v_odom, yaw_rate_odom]   (synthetic from latent truth)")
    print(f"gps  obs  : [px_gps, py_gps]          (synthetic from latent truth)")
    print(f"imu  obs  : [a_m, yaw_rate_m]         (REAL values from ms25)")
    print(f"latent_hz : {cfg['latent_hz']}")
    print(f"base_protocol: {cfg['base_protocol']}")
    print(f"train_jitter : {cfg['train_jitter_std']}")
    print(f"test_jitter  : {cfg['test_jitter_std']}")
    print("=" * 100)

    train_day_full = preprocess_one_day_multisource(
        gt_csv=train_files["gt"],
        odom_csv=train_files["odom"],
        gps_csv=train_files["gps"],
        imu_csv=train_files["imu"],
        time_unit=cfg["time_unit"],
        latent_hz=float(cfg["latent_hz"]),
        gt_xy_cols=(int(cfg["gt_x_col"]), int(cfg["gt_y_col"])),
        gt_yaw_col=None if cfg["gt_yaw_col"] is None else int(cfg["gt_yaw_col"]),
        odom_time_keep_hz=float(cfg["odom_time_keep_hz"]),
        gps_time_keep_hz=float(cfg["gps_time_keep_hz"]),
        imu_time_keep_hz=float(cfg["imu_time_keep_hz"]),
        gps_fix_mode_col=int(cfg["gps_fix_mode_col"]),
        min_gps_fix_mode=int(cfg["min_gps_fix_mode"]),
        poly_window=int(cfg["poly_window"]),
        poly_order=int(cfg["poly_order"]),
        smooth_window=int(cfg["smooth_window"]),
        yaw_smooth_window=int(cfg["yaw_smooth_window"]),
        prefer_gt_yaw=bool(cfg["prefer_gt_yaw"]),
        jitter_std=cfg["train_jitter_std"],
        jitter_clip=cfg["train_jitter_clip"],
        seed=int(cfg["seed"]),
        protocols=cfg["protocols"],
        maneuver_thresholds=cfg["maneuver_thresholds"],
        imu_acc_cols=cfg["imu_acc_cols"],
        imu_gyro_cols=cfg["imu_gyro_cols"],
        imu_use_real_values=bool(cfg["imu_use_real_values"]),
        imu_auto_axis_calib=bool(cfg["imu_auto_axis_calib"]),
    )

    test_day = preprocess_one_day_multisource(
        gt_csv=test_files["gt"],
        odom_csv=test_files["odom"],
        gps_csv=test_files["gps"],
        imu_csv=test_files["imu"],
        time_unit=cfg["time_unit"],
        latent_hz=float(cfg["latent_hz"]),
        gt_xy_cols=(int(cfg["gt_x_col"]), int(cfg["gt_y_col"])),
        gt_yaw_col=None if cfg["gt_yaw_col"] is None else int(cfg["gt_yaw_col"]),
        odom_time_keep_hz=float(cfg["odom_time_keep_hz"]),
        gps_time_keep_hz=float(cfg["gps_time_keep_hz"]),
        imu_time_keep_hz=float(cfg["imu_time_keep_hz"]),
        gps_fix_mode_col=int(cfg["gps_fix_mode_col"]),
        min_gps_fix_mode=int(cfg["min_gps_fix_mode"]),
        poly_window=int(cfg["poly_window"]),
        poly_order=int(cfg["poly_order"]),
        smooth_window=int(cfg["smooth_window"]),
        yaw_smooth_window=int(cfg["yaw_smooth_window"]),
        prefer_gt_yaw=bool(cfg["prefer_gt_yaw"]),
        jitter_std=cfg["test_jitter_std"],
        jitter_clip=cfg["test_jitter_clip"],
        seed=int(cfg["seed"]) + 1,
        protocols=cfg["test_protocols"],
        maneuver_thresholds=cfg["maneuver_thresholds"],
        imu_acc_cols=cfg["imu_acc_cols"],
        imu_gyro_cols=cfg["imu_gyro_cols"],
        imu_use_real_values=bool(cfg["imu_use_real_values"]),
        imu_auto_axis_calib=bool(cfg["imu_auto_axis_calib"]),
    )

    T_train = train_day_full.t_event.shape[0]
    split_idx = int(math.floor(T_train * float(cfg["train_split"])))
    if split_idx <= 10 or split_idx >= T_train - 10:
        raise ValueError("train_split 导致 train/valid 过短")

    train_day = _time_slice_day(train_day_full, 0, split_idx)
    valid_day = _time_slice_day(train_day_full, split_idx, T_train)

    save_day_windows(
        out_root / "train",
        train_day,
        duration_sec=float(cfg["train_duration_sec"]),
        stride_sec=float(cfg["train_stride_sec"]),
        random_offset_sec=float(cfg["train_random_offset_sec"]),
        seed=int(cfg["seed"]),
    )
    save_day_windows(
        out_root / "valid",
        valid_day,
        duration_sec=float(cfg["valid_duration_sec"]),
        stride_sec=float(cfg["valid_stride_sec"]),
        random_offset_sec=float(cfg["valid_random_offset_sec"]),
        seed=int(cfg["seed"]) + 10,
    )
    save_day_windows(
        out_root / "test",
        test_day,
        duration_sec=float(cfg["test_duration_sec"]),
        stride_sec=float(cfg["test_stride_sec"]),
        random_offset_sec=float(cfg["test_random_offset_sec"]),
        seed=int(cfg["seed"]) + 20,
    )

    save_cfg = dict(cfg)
    save_cfg["experiment_name"] = "Exp-D Missing Gap"
    save_cfg["train_jitter_semantics"] = "base_protocol_matched_jitter_for_train_and_valid"
    save_cfg["test_jitter_semantics"] = "test_only_timestamp_random_perturbation_inducing_async_arrival_and_variable_event_dt"
    save_cfg["protocol_source_train_valid"] = cfg["base_protocol"]
    save_cfg["protocol_source_test"] = "override" if cfg.get("test_protocols_override") is not None else cfg["base_protocol"]
    save_cfg["state_mode"] = "ctra6_latent_async_multisource_realimu_missinggap"
    save_cfg["state_order"] = ["px", "py", "v", "yaw", "a", "yaw_rate"]
    save_cfg["odom_obs_order"] = ["v_odom", "yaw_rate_odom"]
    save_cfg["gps_obs_order"] = ["px_gps", "py_gps"]
    save_cfg["imu_obs_order"] = ["a_m", "yaw_rate_m"]
    save_cfg["event_multihot_semantics"] = "valid_measurement_multihot_[imu,odom,gps]"
    save_cfg["arrival_multihot_semantics"] = "timestamp_arrival_multihot_[imu,odom,gps]"
    save_cfg["dt_semantics"] = "window_local; dt_global[0]=0; dt_sensor_last resets within each split/window"
    save_cfg["yaw_rate_source"] = "preferred_gt_yaw_derivative_with_tangent_fallback"
    with open(out_root / "build_config.json", "w", encoding="utf-8") as f:
        json.dump(save_cfg, f, indent=2, ensure_ascii=False)

    stats = compute_train_stats(out_root)
    print(f"[stats] saved -> {out_root / 'train_stats.json'}")
    print("state_mean:", np.round(np.asarray(stats["state_mean"]), 4).tolist())
    print("state_std :", np.round(np.asarray(stats["state_std"]), 4).tolist())
    print()
    print("Done.")
    print(f"Output root: {out_root}")


if __name__ == "__main__":
    main()
