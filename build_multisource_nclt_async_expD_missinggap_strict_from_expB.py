from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch


CONFIG: Dict[str, Any] = {
    # Existing Exp-B dataset root (source of fixed observation realization)
    "src_root": r"C:\Users\22401\Desktop\simulations\KalmanNet_TSP-main\Data_NCLT_MultiSourceAsync_CTRA6_RealIMU_V2",
    # New strict-control-variable Exp-D root
    "out_root": r"C:\Users\22401\Desktop\simulations\KalmanNet_TSP-main\Data_NCLT_MultiSourceAsync_CTRA6_RealIMU_ExpD_MissingGap_Strict_FromExpB",
    # Apply timestamp perturbation ONLY to test split. train/valid are copied unchanged.
    "apply_to_split": "test",
    "test_jitter_std": {"imu": 0.006, "odom": 0.040, "gps": 0.120},
    "test_jitter_clip": {"imu": 0.015, "odom": 0.080, "gps": 0.250},
    "seed": 42,
    # Small epsilon to preserve within-sensor ordering / sample count after perturbation.
    "strict_eps_sec": 1e-6,
}

PT_FILENAMES = {
    "t_global": "t_global.pt",
    "event_valid": "event_valid.pt",
    "x_gt": "x_gt.pt",
    "y_imu": "y_imu.pt",
    "m_imu": "m_imu.pt",
    "q_imu": "q_imu.pt",
    "y_odom": "y_odom.pt",
    "m_odom": "m_odom.pt",
    "q_odom": "q_odom.pt",
    "y_gps": "y_gps.pt",
    "m_gps": "m_gps.pt",
    "q_gps": "q_gps.pt",
    "dt_global": "dt_global.pt",
    "dt_imu_last": "dt_imu_last.pt",
    "dt_odom_last": "dt_odom_last.pt",
    "dt_gps_last": "dt_gps_last.pt",
    "maneuver_label": "maneuver_label.pt",
    "event_multihot": "event_multihot.pt",
    "arrival_multihot": "arrival_multihot.pt",
}

SENSORS = ("imu", "odom", "gps")
SENSOR_INDEX = {name: i for i, name in enumerate(SENSORS)}
OBS_DIM = {"imu": 2, "odom": 2, "gps": 2}
STATE_DIM = 6
YAW_INDEX = 3


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def load_split_tensors(split_dir: Path) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for key, filename in PT_FILENAMES.items():
        path = split_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing tensor file: {path}")
        out[key] = torch.load(path, map_location="cpu")
    return out


def copy_split_dir(src_dir: Path, dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for item in src_dir.iterdir():
        if item.is_file():
            shutil.copy2(item, dst_dir / item.name)


def wrap_to_pi(x: np.ndarray) -> np.ndarray:
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def interp_state(t_old: np.ndarray, x_old: np.ndarray, t_new: np.ndarray) -> np.ndarray:
    """x_old: [T,6], t_old/t_new in seconds."""
    out = np.empty((t_new.shape[0], x_old.shape[1]), dtype=np.float64)
    for d in range(x_old.shape[1]):
        vals = x_old[:, d]
        if d == YAW_INDEX:
            vals = np.unwrap(vals)
            out[:, d] = wrap_to_pi(np.interp(t_new, t_old, vals))
        else:
            out[:, d] = np.interp(t_new, t_old, vals)
    return out


def interp_labels_nearest(t_old: np.ndarray, lab_old: np.ndarray, t_new: np.ndarray) -> np.ndarray:
    xf = np.interp(t_new, t_old, lab_old.astype(np.float64))
    return np.rint(xf).astype(np.int64)


def perturb_sensor_times(
    t_src: np.ndarray,
    *,
    jitter_std: float,
    jitter_clip: float,
    duration: float,
    rng: np.random.Generator,
    eps: float,
) -> np.ndarray:
    if t_src.size == 0:
        return t_src.copy()
    noise = rng.normal(0.0, float(jitter_std), size=t_src.shape[0])
    if jitter_clip > 0:
        noise = np.clip(noise, -float(jitter_clip), float(jitter_clip))
    t = t_src.astype(np.float64) + noise
    t = np.clip(t, 0.0, max(float(duration), 0.0))
    # Preserve count and order: strict monotone nondecreasing with tiny epsilon tie-break.
    out = np.empty_like(t)
    out[0] = max(0.0, min(float(duration), float(t[0])))
    for i in range(1, t.shape[0]):
        out[i] = max(float(t[i]), float(out[i - 1]) + eps)
        if out[i] > duration:
            out[i] = min(duration, float(out[i]))
            if out[i] <= out[i - 1]:
                out[i] = out[i - 1] + eps
    return out


def compute_dt_last(t_event: np.ndarray, m: np.ndarray) -> np.ndarray:
    """PRE-update recency semantics, same as current builder/reader."""
    T = t_event.shape[0]
    out = np.zeros((T,), dtype=np.float64)
    last_t = None
    for i in range(T):
        out[i] = 0.0 if last_t is None else float(t_event[i] - last_t)
        if m[i] > 0.5:
            last_t = float(t_event[i])
    return out


def transform_test_window(
    tensors: Dict[str, torch.Tensor],
    bidx: int,
    cfg: Dict[str, Any],
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    padding = tensors["event_valid"][bidx].numpy().astype(bool)
    L = int(padding.sum())
    if L <= 0:
        raise ValueError(f"Empty test window at index {bidx}")

    t_old = tensors["t_global"][bidx, :L].numpy().astype(np.float64)
    x_old = tensors["x_gt"][bidx, :, :L].numpy().transpose(1, 0).astype(np.float64)  # [L,6]
    lab_old = tensors["maneuver_label"][bidx, :L].numpy().astype(np.int64)
    arrival_old = tensors["arrival_multihot"][bidx, :, :L].numpy().astype(np.float64)  # [3,L]

    duration = float(t_old[-1]) if L > 0 else 0.0

    sensor_packets: Dict[str, Dict[str, np.ndarray]] = {}
    for sensor in SENSORS:
        sidx = SENSOR_INDEX[sensor]
        arr_mask = arrival_old[sidx] > 0.5
        y = tensors[f"y_{sensor}"][bidx, :, :L].numpy().transpose(1, 0).astype(np.float64)
        m = tensors[f"m_{sensor}"][bidx, 0, :L].numpy().astype(np.float64)
        q = tensors[f"q_{sensor}"][bidx, 0, :L].numpy().astype(np.float64)

        t_sensor_old = t_old[arr_mask]
        y_sensor = y[arr_mask]
        m_sensor = m[arr_mask]
        q_sensor = q[arr_mask]
        t_sensor_new = perturb_sensor_times(
            t_sensor_old,
            jitter_std=float(cfg["test_jitter_std"][sensor]),
            jitter_clip=float(cfg["test_jitter_clip"][sensor]),
            duration=duration,
            rng=rng,
            eps=float(cfg["strict_eps_sec"]),
        )
        sensor_packets[sensor] = {
            "t_old": t_sensor_old,
            "t_new": t_sensor_new,
            "y": y_sensor,
            "m": m_sensor,
            "q": q_sensor,
        }

    # Rebuild global event timeline from perturbed sensor arrival times.
    all_times = [pkt["t_new"] for pkt in sensor_packets.values() if pkt["t_new"].size > 0]
    if not all_times:
        raise ValueError(f"No arrivals remain after transformation for window {bidx}")
    t_event = np.unique(np.concatenate(all_times))
    t_event.sort()
    Tnew = int(t_event.shape[0])

    x_new = interp_state(t_old, x_old, t_event)
    lab_new = interp_labels_nearest(t_old, lab_old, t_event)

    y_out = {sensor: np.zeros((Tnew, OBS_DIM[sensor]), dtype=np.float64) for sensor in SENSORS}
    m_out = {sensor: np.zeros((Tnew,), dtype=np.float64) for sensor in SENSORS}
    q_out = {sensor: np.zeros((Tnew,), dtype=np.float64) for sensor in SENSORS}
    event_multihot = np.zeros((Tnew, 3), dtype=np.float64)
    arrival_multihot = np.zeros((Tnew, 3), dtype=np.float64)

    for sensor in SENSORS:
        sidx = SENSOR_INDEX[sensor]
        pkt = sensor_packets[sensor]
        if pkt["t_new"].size == 0:
            continue
        idx = np.searchsorted(t_event, pkt["t_new"])
        ok = (idx >= 0) & (idx < Tnew) & (np.abs(t_event[idx] - pkt["t_new"]) < 1e-10)
        ii = idx[ok]
        y_out[sensor][ii] = pkt["y"][ok]
        m_out[sensor][ii] = pkt["m"][ok]
        q_out[sensor][ii] = pkt["q"][ok]
        arrival_multihot[ii, sidx] = 1.0
        event_multihot[ii, sidx] = pkt["m"][ok]

    dt_global = np.zeros((Tnew,), dtype=np.float64)
    if Tnew > 1:
        dt_global[1:] = t_event[1:] - t_event[:-1]

    dt_last = {sensor: compute_dt_last(t_event, m_out[sensor]) for sensor in SENSORS}

    return {
        "t_global": t_event.astype(np.float32),
        "x_gt": x_new.T.astype(np.float32),  # [6,T]
        "y_imu": y_out["imu"].T.astype(np.float32),
        "m_imu": m_out["imu"][None, :].astype(np.float32),
        "q_imu": q_out["imu"][None, :].astype(np.float32),
        "y_odom": y_out["odom"].T.astype(np.float32),
        "m_odom": m_out["odom"][None, :].astype(np.float32),
        "q_odom": q_out["odom"][None, :].astype(np.float32),
        "y_gps": y_out["gps"].T.astype(np.float32),
        "m_gps": m_out["gps"][None, :].astype(np.float32),
        "q_gps": q_out["gps"][None, :].astype(np.float32),
        "dt_global": dt_global.astype(np.float32),
        "dt_imu_last": dt_last["imu"].astype(np.float32),
        "dt_odom_last": dt_last["odom"].astype(np.float32),
        "dt_gps_last": dt_last["gps"].astype(np.float32),
        "maneuver_label": lab_new.astype(np.int64),
        "event_multihot": event_multihot.T.astype(np.float32),  # [3,T]
        "arrival_multihot": arrival_multihot.T.astype(np.float32),
        "event_valid": np.ones((Tnew,), dtype=np.bool_),
    }


def pad_stack(entries: list[Dict[str, np.ndarray]]) -> Dict[str, torch.Tensor]:
    N = len(entries)
    Tmax = max(int(e["t_global"].shape[0]) for e in entries)
    out: Dict[str, torch.Tensor] = {}

    def alloc(shape, dtype):
        return torch.zeros(shape, dtype=dtype)

    out["t_global"] = alloc((N, Tmax), torch.float32)
    out["event_valid"] = torch.zeros((N, Tmax), dtype=torch.bool)
    out["x_gt"] = alloc((N, STATE_DIM, Tmax), torch.float32)
    for sensor in SENSORS:
        out[f"y_{sensor}"] = alloc((N, OBS_DIM[sensor], Tmax), torch.float32)
        out[f"m_{sensor}"] = alloc((N, 1, Tmax), torch.float32)
        out[f"q_{sensor}"] = alloc((N, 1, Tmax), torch.float32)
    out["dt_global"] = alloc((N, Tmax), torch.float32)
    for sensor in SENSORS:
        out[f"dt_{sensor}_last"] = alloc((N, Tmax), torch.float32)
    out["maneuver_label"] = torch.zeros((N, Tmax), dtype=torch.long)
    out["event_multihot"] = alloc((N, 3, Tmax), torch.float32)
    out["arrival_multihot"] = alloc((N, 3, Tmax), torch.float32)

    seq_lens = []
    for i, e in enumerate(entries):
        L = int(e["t_global"].shape[0])
        seq_lens.append(L)
        out["t_global"][i, :L] = torch.from_numpy(e["t_global"])
        out["event_valid"][i, :L] = torch.from_numpy(e["event_valid"])
        out["x_gt"][i, :, :L] = torch.from_numpy(e["x_gt"])
        for sensor in SENSORS:
            out[f"y_{sensor}"][i, :, :L] = torch.from_numpy(e[f"y_{sensor}"])
            out[f"m_{sensor}"][i, :, :L] = torch.from_numpy(e[f"m_{sensor}"])
            out[f"q_{sensor}"][i, :, :L] = torch.from_numpy(e[f"q_{sensor}"])
            out[f"dt_{sensor}_last"][i, :L] = torch.from_numpy(e[f"dt_{sensor}_last"])
        out["dt_global"][i, :L] = torch.from_numpy(e["dt_global"])
        out["maneuver_label"][i, :L] = torch.from_numpy(e["maneuver_label"])
        out["event_multihot"][i, :, :L] = torch.from_numpy(e["event_multihot"])
        out["arrival_multihot"][i, :, :L] = torch.from_numpy(e["arrival_multihot"])

    meta = {
        "num_windows": N,
        "max_events": Tmax,
        "event_count_min": int(min(seq_lens)),
        "event_count_mean": float(sum(seq_lens) / max(len(seq_lens), 1)),
        "event_count_max": int(max(seq_lens)),
        "x_gt_shape": list(out["x_gt"].shape),
        "y_imu_shape": list(out["y_imu"].shape),
        "y_odom_shape": list(out["y_odom"].shape),
        "y_gps_shape": list(out["y_gps"].shape),
        "t_global_shape": list(out["t_global"].shape),
        "arrival_multihot_shape": list(out["arrival_multihot"].shape),
        "global_dt_min": float(out["dt_global"][out["event_valid"]].min().item()) if int(out["event_valid"].sum()) > 0 else 0.0,
        "global_dt_mean": float(out["dt_global"][out["event_valid"]].float().mean().item()) if int(out["event_valid"].sum()) > 0 else 0.0,
        "global_dt_max": float(out["dt_global"][out["event_valid"]].max().item()) if int(out["event_valid"].sum()) > 0 else 0.0,
    }
    return out, meta


def save_split(split_dir: Path, tensors: Dict[str, torch.Tensor], meta: Dict[str, Any]) -> None:
    split_dir.mkdir(parents=True, exist_ok=True)
    for key, filename in PT_FILENAMES.items():
        torch.save(tensors[key], split_dir / filename)
    save_json(split_dir / "meta.json", meta)


def main() -> None:
    cfg = dict(CONFIG)
    src_root = Path(cfg["src_root"]).resolve()
    out_root = Path(cfg["out_root"]).resolve()
    rng = np.random.default_rng(int(cfg["seed"]))

    if not src_root.exists():
        raise FileNotFoundError(f"src_root not found: {src_root}")
    out_root.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("Building STRICT Exp-D Missing-Gap from existing Exp-B dataset")
    print(f"src_root        : {src_root}")
    print(f"out_root        : {out_root}")
    print(f"apply_to_split  : {cfg['apply_to_split']}")
    print(f"test_jitter_std : {cfg['test_jitter_std']}")
    print(f"test_jitter_clip: {cfg['test_jitter_clip']}")
    print("Semantics       : FIX observation realization (y/m/q), perturb timestamps only")
    print("=" * 100)

    # Copy train/valid unchanged.
    for split in ("train", "valid"):
        copy_split_dir(src_root / split, out_root / split)

    # Transform test split only.
    src_test_dir = src_root / cfg["apply_to_split"]
    test_tensors = load_split_tensors(src_test_dir)
    N = int(test_tensors["t_global"].shape[0])
    entries = []
    old_lens = []
    new_lens = []
    for bidx in range(N):
        old_lens.append(int(test_tensors["event_valid"][bidx].sum().item()))
        e = transform_test_window(test_tensors, bidx, cfg, rng)
        entries.append(e)
        new_lens.append(int(e["t_global"].shape[0]))
        if (bidx + 1) % 25 == 0 or bidx == N - 1:
            print(f"[test-transform] {bidx + 1:4d}/{N} | old_mean={np.mean(old_lens):.1f} | new_mean={np.mean(new_lens):.1f}")

    test_out, test_meta = pad_stack(entries)
    src_test_meta = load_json(src_test_dir / "meta.json")
    # Preserve split slicing settings from source meta when present.
    for k in ("duration_sec", "stride_sec"):
        if k in src_test_meta:
            test_meta[k] = src_test_meta[k]
    save_split(out_root / "test", test_out, test_meta)

    # Root jsons
    build_cfg = load_json(src_root / "build_config.json")
    train_stats = load_json(src_root / "train_stats.json")
    build_cfg["out_root"] = str(out_root)
    build_cfg["experiment_name"] = "Exp-D Missing Gap STRICT from Exp-B"
    build_cfg["protocol_source_train_valid"] = str(src_root)
    build_cfg["protocol_source_test"] = str(src_root)
    build_cfg["strict_control_variable"] = True
    build_cfg["strict_semantics"] = (
        "train/valid copied unchanged from source Exp-B; "
        "test keeps source observation realization (y/m/q per sensor arrival) fixed and perturbs timestamps only"
    )
    build_cfg["strict_source_root"] = str(src_root)
    build_cfg["test_jitter_std"] = cfg["test_jitter_std"]
    build_cfg["test_jitter_clip"] = cfg["test_jitter_clip"]
    build_cfg["dt_semantics"] = "window_local; dt_global[0]=0; dt_sensor_last uses PRE-update recency semantics"
    save_json(out_root / "build_config.json", build_cfg)
    save_json(out_root / "train_stats.json", train_stats)
    save_json(
        out_root / "strict_transform_report.json",
        {
            "src_root": str(src_root),
            "out_root": str(out_root),
            "copied_splits": ["train", "valid"],
            "transformed_split": cfg["apply_to_split"],
            "seed": int(cfg["seed"]),
            "test_jitter_std": cfg["test_jitter_std"],
            "test_jitter_clip": cfg["test_jitter_clip"],
            "strict_eps_sec": float(cfg["strict_eps_sec"]),
            "old_test_seq_len_mean": float(np.mean(old_lens)),
            "new_test_seq_len_mean": float(np.mean(new_lens)),
            "old_test_seq_len_max": int(max(old_lens)),
            "new_test_seq_len_max": int(max(new_lens)),
        },
    )

    print("Done.")
    print(f"Output root: {out_root}")
    print(f"Old test mean/max events: {float(np.mean(old_lens)):.1f} / {int(max(old_lens))}")
    print(f"New test mean/max events: {float(np.mean(new_lens)):.1f} / {int(max(new_lens))}")


if __name__ == "__main__":
    main()
