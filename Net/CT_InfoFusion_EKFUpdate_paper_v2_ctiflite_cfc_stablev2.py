from __future__ import annotations

import math
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from KNet.CFCNet import SparseCfcCell


Tensor = torch.Tensor


def _symmetrize(A: Tensor) -> Tensor:
    return 0.5 * (A + A.transpose(-1, -2))


def _clone_obj(x: Any) -> Any:
    if torch.is_tensor(x):
        return x.detach().clone()
    if isinstance(x, dict):
        return {k: _clone_obj(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_clone_obj(v) for v in x]
    if isinstance(x, tuple):
        return tuple(_clone_obj(v) for v in x)
    return x


def _detach_obj(x: Any) -> Any:
    if torch.is_tensor(x):
        return x.detach()
    if isinstance(x, dict):
        return {k: _detach_obj(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_detach_obj(v) for v in x]
    if isinstance(x, tuple):
        return tuple(_detach_obj(v) for v in x)
    return x


def _safe_log_ratio(x: Tensor, denom: float, clip: float = 100.0) -> Tensor:
    denom = max(float(denom), 1e-6)
    r = torch.clamp(x / denom, min=1e-6, max=max(clip, 1.0))
    return torch.log(r)


def _normalize_dt(dt: Optional[Tensor], B: int, device: torch.device, dtype: torch.dtype, nominal_dt: float, max_timespan: float) -> Tensor:
    if dt is None:
        out = torch.full((B, 1), float(nominal_dt), device=device, dtype=dtype)
    else:
        if not torch.is_tensor(dt):
            out = torch.tensor(dt, device=device, dtype=dtype)
        else:
            out = dt.to(device=device, dtype=dtype)
        while out.dim() > 2:
            out = out.squeeze(-1)
        if out.dim() == 0:
            out = out.view(1, 1)
        elif out.dim() == 1:
            out = out.unsqueeze(1)
        if out.shape[0] == 1 and B > 1:
            out = out.expand(B, 1)
        out = out.reshape(B, 1)
    return torch.clamp(out, min=1e-4, max=float(max_timespan))


class TimeAwareCFCCell(nn.Module):
    """CFC-based continuous-time cell.

    This is the CFC replacement of the previous TimeAwareGRUCell.
    Time enters through SparseCfcCell(timespan=dt_feat), so the local
    split-gain controller preserves the CFC temporal semantics instead
    of using a GRU + manual decay gate.
    """

    def __init__(self, input_dim: int, hidden_dim: int, *, sparsity: float = 0.6, use_mask_input: bool = False):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.use_mask_input = bool(use_mask_input)
        self.cfc = SparseCfcCell(
            input_size=input_dim,
            hidden_size=hidden_dim,
            sparsity=float(sparsity),
            use_mask_input=bool(use_mask_input),
        )

    def forward(self, x: Tensor, h: Optional[Tensor], dt_feat: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        B = x.shape[0]
        if h is None or h.shape[0] != B or h.device != x.device:
            h = torch.zeros(B, self.hidden_dim, device=x.device, dtype=x.dtype)
        if mask is not None:
            if not torch.is_tensor(mask):
                mask = torch.tensor(mask, device=x.device, dtype=x.dtype)
            else:
                mask = mask.to(device=x.device, dtype=x.dtype)
            while mask.dim() > 2:
                mask = mask.squeeze(-1)
            if mask.dim() == 1:
                mask = mask.unsqueeze(1)
        out, new_h = self.cfc(x, h, timespan=dt_feat.to(device=x.device, dtype=x.dtype), mask=mask)
        return new_h


class SplitGainCore(nn.Module):
    """Split-KalmanNet style local controller.

    State-side branch learns a bounded surrogate of prior uncertainty.
    Measurement-side branch learns a bounded surrogate of innovation
    uncertainty. The final learned split gain is
        K_split = G1 * H^T * G2
    in diagonalized form, and it is softly mixed with the EKF gain.
    """

    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        state_feat_dim: int,
        meas_feat_dim: int,
        ctx_dim: int,
        hidden_dim: int = 96,
        alpha_cap: float = 0.25,
        cfc_sparsity: float = 0.6,
        cfc_use_mask_input: bool = False,
    ):
        super().__init__()
        self.x_dim = int(x_dim)
        self.y_dim = int(y_dim)
        self.hidden_dim = int(hidden_dim)
        self.alpha_cap = float(alpha_cap)

        self.state_cell = TimeAwareCFCCell(
            state_feat_dim + ctx_dim,
            hidden_dim,
            sparsity=cfc_sparsity,
            use_mask_input=cfc_use_mask_input,
        )
        self.meas_cell = TimeAwareCFCCell(
            meas_feat_dim + ctx_dim,
            hidden_dim,
            sparsity=cfc_sparsity,
            use_mask_input=cfc_use_mask_input,
        )

        self.state_to_q = nn.Linear(hidden_dim, x_dim)
        self.state_to_g1 = nn.Linear(hidden_dim, x_dim)
        self.meas_to_r = nn.Linear(hidden_dim, y_dim)
        self.meas_to_g2 = nn.Linear(hidden_dim, y_dim)
        self.mix_head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + ctx_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.h_state: Optional[Tensor] = None
        self.h_meas: Optional[Tensor] = None
        self._stable_init()

    def _stable_init(self) -> None:
        for layer in [self.state_to_q, self.state_to_g1, self.meas_to_r, self.meas_to_g2]:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        last = self.mix_head[-1]
        if isinstance(last, nn.Linear):
            nn.init.zeros_(last.weight)
            nn.init.constant_(last.bias, -2.2)  # bias towards EKF at the beginning

    def initialize_hidden(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> None:
        self.h_state = torch.zeros(batch_size, self.hidden_dim, device=device, dtype=dtype)
        self.h_meas = torch.zeros(batch_size, self.hidden_dim, device=device, dtype=dtype)

    def reset_hidden(self) -> None:
        self.h_state = None
        self.h_meas = None

    def snapshot_runtime(self) -> Dict[str, Any]:
        return {"h_state": _clone_obj(self.h_state), "h_meas": _clone_obj(self.h_meas)}

    def restore_runtime(self, state: Dict[str, Any]) -> None:
        self.h_state = _clone_obj(state.get("h_state", None))
        self.h_meas = _clone_obj(state.get("h_meas", None))

    def detach_runtime(self) -> None:
        self.h_state = _detach_obj(self.h_state)
        self.h_meas = _detach_obj(self.h_meas)

    def forward(
        self,
        state_feat: Tensor,
        meas_feat: Tensor,
        ctx_feat: Tensor,
        dt_feat: Tensor,
        mask: Optional[Tensor] = None,
    ) -> SimpleNamespace:
        x_in = torch.cat([state_feat, ctx_feat], dim=1)
        y_in = torch.cat([meas_feat, ctx_feat], dim=1)
        self.h_state = self.state_cell(x_in, self.h_state, dt_feat, mask=mask)
        self.h_meas = self.meas_cell(y_in, self.h_meas, dt_feat, mask=mask)

        q_diag_scale = 0.50 + F.softplus(self.state_to_q(self.h_state))
        r_diag_scale = 0.50 + F.softplus(self.meas_to_r(self.h_meas))
        g1_diag = 0.05 + F.softplus(self.state_to_g1(self.h_state))
        g2_inv_diag = 0.05 + F.softplus(self.meas_to_g2(self.h_meas))
        alpha = self.alpha_cap * torch.sigmoid(self.mix_head(torch.cat([self.h_state, self.h_meas, ctx_feat], dim=1)))

        return SimpleNamespace(
            q_diag_scale=q_diag_scale,
            r_diag_scale=r_diag_scale,
            g1_diag=g1_diag,
            g2_inv_diag=g2_inv_diag,
            split_alpha=alpha,
        )


class CTInfoFusionCTIFLite(nn.Module):
    """Stable CTIF-lite replacement for the failed free-correction EKF-backbone branch.

    Design principles retained from the papers:
      - KalmanNet: explicit predict-update flow, learned gain-related statistics,
        F1-F4 style temporal features, short-window TBPTT.
      - Split-KalmanNet: split state-side / measurement-side branches with
        K_split = G1 H^T G2, softly mixed with EKF gain.
      - Vehicle heterogeneous multi-source fusion: heterogeneous sensors stay
        sensor-specific; the absolute stream only touches position.
      - CFC / continuous-time: hidden-state evolution depends explicitly on dt via SparseCfcCell.
    """

    def __init__(self):
        super().__init__()

    def NNBuild(self, SysModel, args, gru_scale_s: float = 1.0, normalize_inputs: bool = False):
        self.device = torch.device("cuda") if getattr(args, "use_cuda", False) and torch.cuda.is_available() else torch.device("cpu")
        self.SysModel = SysModel
        self.m = int(SysModel.m)
        self.normalize_inputs = bool(normalize_inputs)
        self.nominal_dt = float(getattr(args, "nominal_dt", 0.01))
        self.max_timespan = float(getattr(args, "max_timespan", 0.1))
        self.sensor_order = tuple(str(s).lower() for s in getattr(args, "sensor_order", getattr(SysModel, "sensor_order", ["imu", "odom", "gps"])))
        self.sensor_to_idx = {s: i for i, s in enumerate(self.sensor_order)}
        self.sensor_code_dim = len(self.sensor_order)
        self.sensor_nominal_dt = {
            s: float(getattr(args, "sensor_nominal_dt", {}).get(s, self.nominal_dt)) if isinstance(getattr(args, "sensor_nominal_dt", None), dict) else float(self.nominal_dt)
            for s in self.sensor_order
        }

        self.scale_x = None
        sx = getattr(args, "feature_scale_x", None)
        if sx is not None:
            if isinstance(sx, (int, float)):
                sx = [float(sx)] * self.m
            self.scale_x = torch.tensor(sx, dtype=torch.float32, device=self.device).view(1, self.m)

        self.state_clip_pos = float(getattr(args, "state_clip_pos", 1e4))
        self.state_clip_vel = float(getattr(args, "state_clip_vel", 20.0))
        self.state_clip_acc = float(getattr(args, "state_clip_acc", 10.0))
        self.state_clip_yaw_rate = float(getattr(args, "state_clip_yaw_rate", 5.0))
        self.corr_clip_pos = float(getattr(args, "corr_clip_pos", 5.0))
        self.corr_clip_vel = float(getattr(args, "corr_clip_vel", 1.5))
        self.corr_clip_yaw = float(getattr(args, "corr_clip_yaw", 0.25))
        self.corr_clip_acc = float(getattr(args, "corr_clip_acc", 0.75))
        self.corr_clip_yaw_rate = float(getattr(args, "corr_clip_yaw_rate", 0.15))

        self.long_gap_sec = float(getattr(args, "long_gap_sec", 2.0))
        self.return_gap_sec = float(getattr(args, "return_gap_sec", 1.0))
        self.split_alpha_cap = float(getattr(args, "split_alpha_cap", 0.25))
        self.cfc_sparsity = float(getattr(args, "cfc_sparsity", 0.60))
        self.cfc_use_mask_input = bool(getattr(args, "cfc_use_mask_input", False))

        hidden_dim = max(64, int(round(float(gru_scale_s) * 96)))
        state_feat_dim = (self.m * 2) + self.m
        meas_feat_dim = 2 + 2 + 2
        ctx_dim = 8 + self.sensor_code_dim + len(self.sensor_order)
        self.local_core = SplitGainCore(
            x_dim=self.m,
            y_dim=2,
            state_feat_dim=state_feat_dim,
            meas_feat_dim=meas_feat_dim,
            ctx_dim=ctx_dim,
            hidden_dim=hidden_dim,
            alpha_cap=self.split_alpha_cap,
            cfc_sparsity=self.cfc_sparsity,
            cfc_use_mask_input=self.cfc_use_mask_input,
        ).to(self.device)

        self.pos_dims = torch.tensor([0, 1], dtype=torch.long, device=self.device)
        self.anchor_conf_decay_tau = float(getattr(args, "anchor_conf_decay_tau", 4.0))
        self.anchor_conf_floor = float(getattr(args, "anchor_conf_floor", 0.03))
        self.anchor_return_horizon = int(getattr(args, "anchor_return_horizon", 16))
        self.anchor_return_gap = float(getattr(args, "anchor_return_gap", 1.0))
        self.abs_gate_floor = float(getattr(args, "abs_gate_floor", 0.02))
        self.abs_gate_cap = float(getattr(args, "abs_gate_cap", 0.85))
        self.anchor_pos_clip = float(getattr(args, "anchor_pos_clip", 30.0))

        anchor_ctx = 2 + 2 + 2 + 1 + 1 + 1 + 1 + self.sensor_code_dim
        self.anchor_update_head = nn.Sequential(
            nn.Linear(anchor_ctx, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
        ).to(self.device)
        self.abs_gate_head = nn.Sequential(
            nn.Linear(anchor_ctx + self.m, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        ).to(self.device)
        self._init_abs_modules()

        self.m2x_0 = getattr(SysModel, "m2x_0", torch.eye(self.m, dtype=torch.float32))
        self.m2x_0 = self.m2x_0.to(self.device)

        self._active_sensor = self.sensor_order[0]
        self._sensor_gap: Dict[str, Tensor] = {}
        self._sensor_seen: Dict[str, Tensor] = {}
        self._sensor_prev_y: Dict[str, Optional[Tensor]] = {}
        self._last_step_aux: Dict[str, Tensor] = {}
        self._last_step_raw: Dict[str, Tensor] = {}
        self._anchor_pos: Optional[Tensor] = None
        self._anchor_conf: Optional[Tensor] = None
        self._anchor_age: Optional[Tensor] = None
        self._anchor_valid: Optional[Tensor] = None
        self._anchor_return_count: Optional[Tensor] = None
        return self

    def _init_abs_modules(self) -> None:
        for mod in [self.anchor_update_head, self.abs_gate_head]:
            for layer in mod.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight, gain=0.7)
                    nn.init.zeros_(layer.bias)
        last = self.abs_gate_head[-1]
        if isinstance(last, nn.Linear):
            nn.init.zeros_(last.weight)
            nn.init.constant_(last.bias, -1.5)
        last_u = self.anchor_update_head[-1]
        if isinstance(last_u, nn.Linear):
            nn.init.zeros_(last_u.weight)
            nn.init.zeros_(last_u.bias)

    # ------------------------------------------------------------------
    # system helpers
    # ------------------------------------------------------------------
    def _get_sensor_name(self, sensor: Optional[str] = None) -> str:
        s = self._active_sensor if sensor is None else str(sensor).lower()
        if s not in self.sensor_to_idx:
            s = self.sensor_order[0]
        return s

    def set_active_sensor(self, sensor: str) -> None:
        sensor = self._get_sensor_name(sensor)
        self._active_sensor = sensor
        if hasattr(self.SysModel, "set_active_sensor"):
            self.SysModel.set_active_sensor(sensor)

    def _get_sensor_code(self, B: int, dtype: torch.dtype, sensor: str) -> Tensor:
        code = torch.zeros(B, self.sensor_code_dim, device=self.device, dtype=dtype)
        code[:, self.sensor_to_idx[self._get_sensor_name(sensor)]] = 1.0
        return code

    def _call_f(self, x: Tensor, dt: Optional[Tensor] = None, jacobian: bool = False):
        try:
            return self.SysModel.f(x, dt=dt, jacobian=jacobian)
        except TypeError:
            return self.SysModel.f(x, dt=dt)

    def _call_h(self, x: Tensor, sensor: str):
        sensor = self._get_sensor_name(sensor)
        try:
            return self.SysModel.h(x, sensor=sensor)
        except TypeError:
            return self.SysModel.h(x)

    def _get_sensor_H(self, sensor: str, B: int, dtype: torch.dtype, x: Optional[Tensor] = None) -> Tensor:
        sensor = self._get_sensor_name(sensor)
        H = None
        if hasattr(self.SysModel, "Jacobian_h"):
            try:
                if x is None:
                    H = self.SysModel.Jacobian_h(sensor=sensor, B=B, dtype=dtype, device=self.device)
                else:
                    H = self.SysModel.Jacobian_h(x=x, sensor=sensor)
            except TypeError:
                H = None
        if H is None:
            try:
                H = getattr(self.SysModel, "H_dict")[sensor]
            except Exception:
                H = getattr(self.SysModel, "H")
        if not torch.is_tensor(H):
            H = torch.tensor(H, device=self.device, dtype=dtype)
        else:
            H = H.to(device=self.device, dtype=dtype)
        if H.dim() == 2:
            H = H.unsqueeze(0)
        if H.shape[0] == 1 and B > 1:
            H = H.expand(B, -1, -1)
        return H

    def _get_sensor_R(self, sensor: str, B: int, dtype: torch.dtype) -> Tensor:
        sensor = self._get_sensor_name(sensor)
        R = None
        if hasattr(self.SysModel, "Rk"):
            try:
                R = self.SysModel.Rk(sensor=sensor, B=B, dtype=dtype, device=self.device)
            except TypeError:
                R = None
        if R is None:
            try:
                R = getattr(self.SysModel, "R_dict")[sensor]
            except Exception:
                R = getattr(self.SysModel, "R")
        if not torch.is_tensor(R):
            R = torch.tensor(R, device=self.device, dtype=dtype)
        else:
            R = R.to(device=self.device, dtype=dtype)
        if R.dim() == 2:
            R = R.unsqueeze(0)
        if R.shape[0] == 1 and B > 1:
            R = R.expand(B, -1, -1)
        return R

    def _get_base_Q(self, dt: Tensor, B: int, dtype: torch.dtype) -> Tensor:
        if hasattr(self.SysModel, "Qk"):
            Q = self.SysModel.Qk(dt=dt, B=B, dtype=dtype, device=self.device)
        else:
            Q = getattr(self.SysModel, "Q", torch.eye(self.m, dtype=dtype))
            if not torch.is_tensor(Q):
                Q = torch.tensor(Q, device=self.device, dtype=dtype)
            else:
                Q = Q.to(device=self.device, dtype=dtype)
            if Q.dim() == 2:
                Q = Q.unsqueeze(0).expand(B, -1, -1)
        if Q.dim() == 2:
            Q = Q.unsqueeze(0)
        return Q

    def _clip_state(self, x: Tensor) -> Tensor:
        if x.dim() != 3 or x.shape[1] < 6:
            return x
        px_py = torch.clamp(x[:, 0:2, :], min=-self.state_clip_pos, max=self.state_clip_pos)
        v = torch.clamp(x[:, 2:3, :], min=-self.state_clip_vel, max=self.state_clip_vel)
        yaw = ((x[:, 3:4, :] + torch.pi) % (2.0 * torch.pi)) - torch.pi
        a = torch.clamp(x[:, 4:5, :], min=-self.state_clip_acc, max=self.state_clip_acc)
        yaw_rate = torch.clamp(x[:, 5:6, :], min=-self.state_clip_yaw_rate, max=self.state_clip_yaw_rate)
        return torch.cat([px_py, v, yaw, a, yaw_rate], dim=1)

    def _clip_delta(self, dx: Tensor) -> Tensor:
        if dx.dim() != 3 or dx.shape[1] < 6:
            return dx
        parts = [
            dx[:, 0:2, :].clamp(-self.corr_clip_pos, self.corr_clip_pos),
            dx[:, 2:3, :].clamp(-self.corr_clip_vel, self.corr_clip_vel),
            dx[:, 3:4, :].clamp(-self.corr_clip_yaw, self.corr_clip_yaw),
            dx[:, 4:5, :].clamp(-self.corr_clip_acc, self.corr_clip_acc),
            dx[:, 5:6, :].clamp(-self.corr_clip_yaw_rate, self.corr_clip_yaw_rate),
        ]
        return torch.cat(parts, dim=1)

    def _event_mask_bool(self, event_mask: Optional[Tensor], B: int) -> Tensor:
        if event_mask is None:
            return torch.ones(B, device=self.device, dtype=torch.bool)
        if not torch.is_tensor(event_mask):
            mask = torch.tensor(event_mask, device=self.device)
        else:
            mask = event_mask.to(device=self.device)
        while mask.dim() > 1:
            mask = mask.squeeze(-1)
        return mask.to(dtype=torch.bool).reshape(B)

    def _blend_rows(self, new: Tensor, old: Tensor, mask: Tensor) -> Tensor:
        if new.dim() == 3:
            m = mask.view(-1, 1, 1).to(device=new.device, dtype=new.dtype)
        elif new.dim() == 2:
            m = mask.view(-1, 1).to(device=new.device, dtype=new.dtype)
        else:
            raise ValueError(f"Unsupported rank {new.dim()} for blending")
        return m * new + (1.0 - m) * old

    def _sanitize_tensor(self, x: Tensor, *, nan: float = 0.0, posinf: float = 1e4, neginf: float = -1e4) -> Tensor:
        return torch.nan_to_num(x, nan=nan, posinf=posinf, neginf=neginf)

    def _safe_solve_gain(self, P: Tensor, H: Tensor, R: Tensor) -> Tensor:
        PHt = torch.bmm(P, H.transpose(1, 2))
        S = torch.bmm(torch.bmm(H, P), H.transpose(1, 2)) + R
        S = _symmetrize(S) + 1e-6 * torch.eye(S.shape[-1], device=S.device, dtype=S.dtype).unsqueeze(0)
        solve_dtype = torch.float32 if S.dtype in (torch.float16, torch.bfloat16) else S.dtype
        try:
            K = torch.linalg.solve(S.to(solve_dtype), PHt.transpose(1, 2).to(solve_dtype)).transpose(1, 2)
        except RuntimeError:
            K = torch.bmm(PHt.to(solve_dtype), torch.linalg.pinv(S.to(solve_dtype)))
        return K.to(dtype=P.dtype), S

    # ------------------------------------------------------------------
    # runtime state helpers
    # ------------------------------------------------------------------
    def _init_sensor_memory(self, B: int, dtype: torch.dtype) -> None:
        self._sensor_gap = {
            s: torch.full((B, 1), self.sensor_nominal_dt.get(s, self.nominal_dt), device=self.device, dtype=dtype)
            for s in self.sensor_order
        }
        self._sensor_seen = {s: torch.zeros((B, 1), device=self.device, dtype=dtype) for s in self.sensor_order}
        self._sensor_prev_y = {s: None for s in self.sensor_order}

    def _advance_sensor_gaps(self, dt: Tensor, event_mask: Tensor) -> None:
        for s in self.sensor_order:
            updated = self._sensor_gap[s] + dt
            self._sensor_gap[s] = self._blend_rows(updated, self._sensor_gap[s], event_mask)

    def _commit_sensor_observation(self, sensor: str, y: Tensor, event_mask: Tensor) -> None:
        sensor = self._get_sensor_name(sensor)
        zero_gap = torch.zeros_like(self._sensor_gap[sensor])
        one_seen = torch.ones_like(self._sensor_seen[sensor])
        self._sensor_gap[sensor] = self._blend_rows(zero_gap, self._sensor_gap[sensor], event_mask)
        self._sensor_seen[sensor] = self._blend_rows(one_seen, self._sensor_seen[sensor], event_mask)

        y_old = self._sensor_prev_y[sensor]
        y_new = y.detach().clone()
        if y_new.dim() == 2:
            y_new = y_new.unsqueeze(-1)
        if y_old is not None and y_old.dim() == 2:
            y_old = y_old.unsqueeze(-1)

        self._sensor_prev_y[sensor] = y_new if y_old is None else self._blend_rows(y_new, y_old, event_mask)

    def _init_anchor_memory(self, B: int, dtype: torch.dtype, init_pos: Optional[Tensor] = None) -> None:
        if init_pos is None:
            init_pos = torch.zeros(B, 2, device=self.device, dtype=dtype)
        self._anchor_pos = init_pos.clone()
        self._anchor_conf = torch.zeros(B, 1, device=self.device, dtype=dtype)
        self._anchor_age = torch.full((B, 1), 1e3, device=self.device, dtype=dtype)
        self._anchor_valid = torch.zeros(B, 1, device=self.device, dtype=dtype)
        self._anchor_return_count = torch.zeros(B, 1, device=self.device, dtype=dtype)

    def _ensure_anchor_memory(self, B: int, dtype: torch.dtype, init_pos: Optional[Tensor] = None) -> None:
        if self._anchor_pos is None or self._anchor_pos.shape[0] != B or self._anchor_pos.device != self.device:
            self._init_anchor_memory(B, dtype, init_pos)

    def _decay_anchor(self, dt: Tensor, event_mask: Tensor) -> None:
        self._anchor_age = self._blend_rows(self._anchor_age + dt, self._anchor_age, event_mask)
        decay = torch.exp(-dt / max(self.anchor_conf_decay_tau, 1e-3))
        self._anchor_conf = self._blend_rows(self._anchor_conf * decay, self._anchor_conf, event_mask)
        self._anchor_return_count = self._blend_rows(torch.clamp(self._anchor_return_count - 1.0, min=0.0), self._anchor_return_count, event_mask)
        self._anchor_valid = (self._anchor_conf > self.anchor_conf_floor).to(self._anchor_conf.dtype)

    def _build_anchor_context(self, local_state: Tensor, sensor_code: Tensor, gps_gap: Tensor, q_gps: Tensor) -> Tensor:
        local_pos = local_state[:, self.pos_dims, 0]
        return torch.cat(
            [
                local_pos,
                self._anchor_pos,
                local_pos - self._anchor_pos,
                self._anchor_conf,
                self._anchor_age,
                _safe_log_ratio(gps_gap, self.sensor_nominal_dt.get("gps", self.nominal_dt), clip=100.0),
                q_gps,
                sensor_code,
            ],
            dim=1,
        )

    def _update_anchor_from_gps(self, local_state: Tensor, y_gps: Tensor, q_gps: Tensor, gps_dt_last: Tensor, event_mask: Tensor, sensor_code: Tensor) -> None:
        ctx = self._build_anchor_context(local_state, sensor_code, gps_dt_last, q_gps)
        raw = self.anchor_update_head(ctx)
        delta = raw[:, :2].clamp(-self.anchor_pos_clip, self.anchor_pos_clip)
        conf_gain = torch.sigmoid(raw[:, 2:3])
        candidate = y_gps[:, :2] + delta
        upd = torch.clamp(0.10 + 0.30 * q_gps, min=0.05, max=0.45)
        new_anchor = (1.0 - upd) * self._anchor_pos + upd * candidate
        new_conf = torch.clamp(torch.maximum(self._anchor_conf, conf_gain * q_gps), min=0.0, max=1.0)
        new_age = torch.zeros_like(self._anchor_age)
        is_return = (gps_dt_last >= self.anchor_return_gap).to(dtype=q_gps.dtype)
        new_return = torch.maximum(self._anchor_return_count, is_return * float(self.anchor_return_horizon))
        self._anchor_pos = self._blend_rows(new_anchor, self._anchor_pos, event_mask)
        self._anchor_conf = self._blend_rows(new_conf, self._anchor_conf, event_mask)
        self._anchor_age = self._blend_rows(new_age, self._anchor_age, event_mask)
        self._anchor_return_count = self._blend_rows(new_return, self._anchor_return_count, event_mask)
        self._anchor_valid = (self._anchor_conf > self.anchor_conf_floor).to(self._anchor_conf.dtype)

    def _apply_absolute_stream(self, local_state: Tensor, sensor_code: Tensor, gps_gap: Tensor, q_gps: Tensor, event_mask: Tensor) -> tuple[Tensor, Tensor]:
        ctx = self._build_anchor_context(local_state, sensor_code, gps_gap, q_gps)
        gate_in = torch.cat([ctx, local_state[:, :, 0]], dim=1)
        base_gate = torch.sigmoid(self.abs_gate_head(gate_in))
        freshness = torch.exp(-gps_gap / max(self.anchor_return_gap, 1e-3))
        return_strength = torch.clamp(self._anchor_return_count / max(float(self.anchor_return_horizon), 1.0), 0.0, 1.0)
        gate = base_gate * (0.20 + 0.50 * self._anchor_conf + 0.20 * freshness + 0.25 * return_strength)
        gate = torch.clamp(gate, min=self.abs_gate_floor, max=self.abs_gate_cap)
        gate = gate * self._anchor_valid
        fused = local_state.clone()
        local_pos = local_state[:, self.pos_dims, 0]
        fused_pos = (1.0 - gate) * local_pos + gate * self._anchor_pos
        fused[:, self.pos_dims, 0] = fused_pos
        fused = self._blend_rows(fused, local_state, event_mask)
        gate = self._blend_rows(gate, torch.zeros_like(gate), event_mask)
        return fused, gate

    # ------------------------------------------------------------------
    # sequence control
    # ------------------------------------------------------------------
    def InitSequence(self, M1_0: Tensor, T: int):
        self.T = int(T)
        self.state_post = M1_0.to(self.device)
        if self.state_post.dim() == 2:
            self.state_post = self.state_post.unsqueeze(-1)
        B = self.state_post.shape[0]
        dtype = self.state_post.dtype
        if self.m2x_0.dim() == 2:
            self.P_post = self.m2x_0.to(device=self.device, dtype=dtype).unsqueeze(0).expand(B, -1, -1).clone()
        else:
            self.P_post = self.m2x_0.to(device=self.device, dtype=dtype).clone()
        self.P_post = _symmetrize(self.P_post)
        self.m1x_prior = self.state_post.detach().clone()
        self.m1y = None
        self.x_post_prev = self.state_post.detach().clone()
        self.x_post_prev2 = self.state_post.detach().clone()
        self.x_prior_prev = self.state_post.detach().clone()
        self.step_idx = 0
        self.local_core.initialize_hidden(B, self.device, dtype)
        self._init_sensor_memory(B, dtype)
        self._ensure_anchor_memory(B, dtype, init_pos=self.state_post[:, self.pos_dims, 0])
        self._last_step_aux = {}
        self._last_step_raw = {}

    def reset(self, clean_history: bool = False):
        self.local_core.reset_hidden()
        self._sensor_gap = {}
        self._sensor_seen = {}
        self._sensor_prev_y = {}
        self._last_step_aux = {}
        self._last_step_raw = {}
        self._anchor_pos = None
        self._anchor_conf = None
        self._anchor_age = None
        self._anchor_valid = None
        self._anchor_return_count = None
        if clean_history:
            self.state_post = None
            self.P_post = None
            self.m1x_prior = None
            self.m1y = None
            self.x_post_prev = None
            self.x_post_prev2 = None
            self.x_prior_prev = None
            self.step_idx = 0

    def snapshot_runtime_state(self) -> Dict[str, Any]:
        return {
            "state_post": _clone_obj(getattr(self, "state_post", None)),
            "P_post": _clone_obj(getattr(self, "P_post", None)),
            "m1x_prior": _clone_obj(getattr(self, "m1x_prior", None)),
            "m1y": _clone_obj(getattr(self, "m1y", None)),
            "x_post_prev": _clone_obj(getattr(self, "x_post_prev", None)),
            "x_post_prev2": _clone_obj(getattr(self, "x_post_prev2", None)),
            "x_prior_prev": _clone_obj(getattr(self, "x_prior_prev", None)),
            "step_idx": int(getattr(self, "step_idx", 0)),
            "sensor_gap": _clone_obj(self._sensor_gap),
            "sensor_seen": _clone_obj(self._sensor_seen),
            "sensor_prev_y": _clone_obj(self._sensor_prev_y),
            "anchor_pos": _clone_obj(self._anchor_pos),
            "anchor_conf": _clone_obj(self._anchor_conf),
            "anchor_age": _clone_obj(self._anchor_age),
            "anchor_valid": _clone_obj(self._anchor_valid),
            "anchor_return_count": _clone_obj(self._anchor_return_count),
            "core": self.local_core.snapshot_runtime(),
            "last_step_aux": _clone_obj(self._last_step_aux),
            "last_step_raw": _clone_obj(self._last_step_raw),
        }

    def restore_runtime_state(self, state: Dict[str, Any]) -> None:
        self.state_post = _clone_obj(state.get("state_post", None))
        self.P_post = _clone_obj(state.get("P_post", None))
        self.m1x_prior = _clone_obj(state.get("m1x_prior", None))
        self.m1y = _clone_obj(state.get("m1y", None))
        self.x_post_prev = _clone_obj(state.get("x_post_prev", None))
        self.x_post_prev2 = _clone_obj(state.get("x_post_prev2", None))
        self.x_prior_prev = _clone_obj(state.get("x_prior_prev", None))
        self.step_idx = int(state.get("step_idx", 0))
        self._sensor_gap = _clone_obj(state.get("sensor_gap", {}))
        self._sensor_seen = _clone_obj(state.get("sensor_seen", {}))
        self._sensor_prev_y = _clone_obj(state.get("sensor_prev_y", {}))
        self._anchor_pos = _clone_obj(state.get("anchor_pos", None))
        self._anchor_conf = _clone_obj(state.get("anchor_conf", None))
        self._anchor_age = _clone_obj(state.get("anchor_age", None))
        self._anchor_valid = _clone_obj(state.get("anchor_valid", None))
        self._anchor_return_count = _clone_obj(state.get("anchor_return_count", None))
        self.local_core.restore_runtime(state.get("core", {}))
        self._last_step_aux = _clone_obj(state.get("last_step_aux", {}))
        self._last_step_raw = _clone_obj(state.get("last_step_raw", {}))

    def detach_runtime_state(self) -> None:
        for name in [
            "state_post", "P_post", "m1x_prior", "m1y", "x_post_prev", "x_post_prev2", "x_prior_prev",
            "_anchor_pos", "_anchor_conf", "_anchor_age", "_anchor_valid", "_anchor_return_count",
        ]:
            x = getattr(self, name, None)
            if torch.is_tensor(x):
                setattr(self, name, x.detach())
        self._sensor_gap = _detach_obj(self._sensor_gap)
        self._sensor_seen = _detach_obj(self._sensor_seen)
        self._sensor_prev_y = _detach_obj(self._sensor_prev_y)
        self._last_step_aux = _detach_obj(self._last_step_aux)
        self._last_step_raw = _detach_obj(self._last_step_raw)
        self.local_core.detach_runtime()

    def get_step_aux(self) -> Dict[str, Tensor]:
        out = {}
        for src in [self._last_step_aux, self._last_step_raw]:
            for k, v in src.items():
                if torch.is_tensor(v):
                    out[k] = v.detach().clone()
        return out

    # ------------------------------------------------------------------
    # prediction / update
    # ------------------------------------------------------------------
    def predict_only(self, dt=None, event_mask=None):
        if getattr(self, "state_post", None) is None:
            raise RuntimeError("predict_only requires initialized sequence state")
        B = self.state_post.shape[0]
        dtype = self.state_post.dtype
        dt_t = _normalize_dt(dt, B, self.device, dtype, self.nominal_dt, self.max_timespan)
        mask = self._event_mask_bool(event_mask, B)
        self._ensure_anchor_memory(B, dtype, init_pos=self.state_post[:, self.pos_dims, 0])
        self._decay_anchor(dt_t, mask)
        self._advance_sensor_gaps(dt_t, mask)

        pred = self._call_f(self.state_post, dt=dt_t, jacobian=True)
        if isinstance(pred, tuple):
            x_prior, Fk = pred
        else:
            x_prior = pred
            if not hasattr(self.SysModel, "Jacobian_f"):
                raise RuntimeError("SystemModel must provide Jacobian_f")
            Fk = self.SysModel.Jacobian_f(self.state_post, dt=dt_t)
        Q = self._get_base_Q(dt_t, B=B, dtype=dtype)
        P_prior = torch.bmm(torch.bmm(Fk, self.P_post), Fk.transpose(1, 2)) + Q
        P_prior = _symmetrize(self._sanitize_tensor(P_prior)) + 1e-6 * torch.eye(self.m, device=self.device, dtype=dtype).unsqueeze(0)

        self.m1x_prior = self._clip_state(self._sanitize_tensor(x_prior))
        self.state_post = self._blend_rows(self.m1x_prior, self.state_post, mask)
        self.P_post = self._blend_rows(P_prior, self.P_post, mask)
        self.x_prior_prev = self.m1x_prior.detach().clone()
        self._last_step_aux = {
            "timespan": dt_t.detach(),
            "anchor_pos": self._anchor_pos.detach().clone(),
            "anchor_conf": self._anchor_conf.detach().clone(),
            "anchor_age": self._anchor_age.detach().clone(),
            "split_alpha": torch.zeros(B, 1, device=self.device, dtype=dtype),
            "anchor_gate": torch.zeros(B, 1, device=self.device, dtype=dtype),
        }
        self._last_step_raw = {}
        return self.state_post

    def update(self, sensor: str, y: Tensor, dt_last: Tensor, q: Tensor, event_mask=None):
        sensor = self._get_sensor_name(sensor)
        self.set_active_sensor(sensor)
        if getattr(self, "state_post", None) is None:
            raise RuntimeError("update requires initialized sequence state")
        y_raw = y.to(self.device)
        if y_raw.dim() == 2:
            y_raw = y_raw.unsqueeze(-1)
        B = y_raw.shape[0]
        dtype = y_raw.dtype
        mask = self._event_mask_bool(event_mask, B)
        q = _normalize_dt(q, B, self.device, dtype, 1.0, 1.0)
        dt_last = _normalize_dt(dt_last, B, self.device, dtype, self.sensor_nominal_dt.get(sensor, self.nominal_dt), 100.0)
        self._ensure_anchor_memory(B, dtype, init_pos=self.state_post[:, self.pos_dims, 0])

        H = self._get_sensor_H(sensor, B=B, dtype=dtype, x=self.state_post)
        y_pred = self._call_h(self.state_post, sensor)
        residual = y_raw - y_pred
        prev_y = self._sensor_prev_y.get(sensor, None)
        if prev_y is not None and prev_y.dim() == 2:
            prev_y = prev_y.unsqueeze(-1)
        obs_diff = torch.zeros_like(y_raw) if prev_y is None else y_raw - prev_y
        evol_diff = self.x_post_prev - self.x_post_prev2 if torch.is_tensor(getattr(self, "x_post_prev", None)) else torch.zeros_like(self.state_post)
        upd_diff = self.x_post_prev - self.x_prior_prev if torch.is_tensor(getattr(self, "x_post_prev", None)) else torch.zeros_like(self.state_post)
        diagP = torch.diagonal(self.P_post, dim1=1, dim2=2)

        if self.scale_x is not None:
            sx = torch.clamp(self.scale_x.to(device=self.device, dtype=dtype), min=1e-6)
            evol_feat = evol_diff.squeeze(-1) / sx
            upd_feat = upd_diff.squeeze(-1) / sx
        else:
            evol_feat = evol_diff.squeeze(-1)
            upd_feat = upd_diff.squeeze(-1)

        state_feat = torch.cat([upd_feat, evol_feat, torch.sqrt(torch.clamp(diagP, min=1e-8))], dim=1)
        residual_norm = torch.norm(residual.squeeze(-1), dim=1, keepdim=True)
        obs_diff_norm = torch.norm(obs_diff.squeeze(-1), dim=1, keepdim=True)
        S_diag_proxy = torch.diagonal(torch.bmm(torch.bmm(H, self.P_post), H.transpose(1, 2)), dim1=1, dim2=2)
        meas_feat = torch.cat([residual.squeeze(-1), obs_diff.squeeze(-1), torch.sqrt(torch.clamp(S_diag_proxy, min=1e-8))], dim=1)

        gps_gap = self._sensor_gap.get("gps", dt_last)
        is_gps = torch.full((B, 1), 1.0 if sensor == "gps" else 0.0, device=self.device, dtype=dtype)
        is_long_gap = (dt_last >= self.long_gap_sec).to(dtype)
        is_return = ((self._sensor_seen[sensor] > 0.5) & (dt_last >= self.return_gap_sec)).to(dtype)
        sensor_gap = self._sensor_gap[sensor]
        all_gap_logs = torch.cat([
            _safe_log_ratio(self._sensor_gap[s], self.sensor_nominal_dt.get(s, self.nominal_dt), clip=100.0)
            for s in self.sensor_order
        ], dim=1)
        sensor_code = self._get_sensor_code(B, dtype, sensor)
        ctx_feat = torch.cat(
            [
                _safe_log_ratio(dt_last, self.sensor_nominal_dt.get(sensor, self.nominal_dt), clip=100.0),
                _safe_log_ratio(gps_gap, self.sensor_nominal_dt.get("gps", self.nominal_dt), clip=100.0),
                q,
                is_gps,
                is_long_gap,
                is_return,
                self._anchor_conf,
                self._anchor_age.clamp(max=100.0),
                sensor_code,
                all_gap_logs,
            ],
            dim=1,
        )
        dt_feat = _safe_log_ratio(sensor_gap.clamp(min=1e-4), self.sensor_nominal_dt.get(sensor, self.nominal_dt), clip=100.0)

        core = self.local_core(state_feat, meas_feat, ctx_feat, dt_feat, mask=mask.to(dtype=dtype).view(B, 1))
        Dq = torch.diag_embed(torch.clamp(core.q_diag_scale, min=0.75, max=2.0))
        Dr = torch.diag_embed(torch.clamp(core.r_diag_scale, min=0.75, max=3.0))
        P_eff = torch.bmm(torch.bmm(Dq, self.P_post), Dq)
        R_base = self._get_sensor_R(sensor, B=B, dtype=dtype)
        R_eff = torch.bmm(torch.bmm(Dr, R_base), Dr)

        K_ekf, S_eff = self._safe_solve_gain(P_eff, H, R_eff)
        G1 = torch.diag_embed(torch.clamp(core.g1_diag, min=0.05, max=6.0))
        G2 = torch.diag_embed(1.0 / torch.clamp(core.g2_inv_diag, min=0.10, max=10.0))
        K_split = torch.bmm(torch.bmm(G1, H.transpose(1, 2)), G2)
        alpha = torch.clamp(core.split_alpha, min=0.0, max=self.split_alpha_cap)
        K = (1.0 - alpha.unsqueeze(-1)) * K_ekf + alpha.unsqueeze(-1) * K_split

        delta = torch.bmm(K, residual)
        delta = self._clip_delta(self._sanitize_tensor(delta, nan=0.0, posinf=self.corr_clip_pos, neginf=-self.corr_clip_pos))
        local_state = self._clip_state(self.state_post + delta)

        I = torch.eye(self.m, device=self.device, dtype=dtype).unsqueeze(0).expand(B, -1, -1)
        KH = torch.bmm(K, H)
        I_KH = I - KH
        P_post = torch.bmm(torch.bmm(I_KH, P_eff), I_KH.transpose(1, 2)) + torch.bmm(torch.bmm(K, R_eff), K.transpose(1, 2))
        P_post = _symmetrize(P_post) + 1e-6 * torch.eye(self.m, device=self.device, dtype=dtype).unsqueeze(0)

        q_gps = q if sensor == "gps" else torch.ones_like(q) * 0.5
        if sensor == "gps":
            self._update_anchor_from_gps(local_state, y_raw.squeeze(-1), q_gps, dt_last, mask, sensor_code)
        fused_state, anchor_gate = self._apply_absolute_stream(local_state, sensor_code, gps_gap, q_gps, mask)

        self.state_post = self._blend_rows(fused_state, self.state_post, mask)
        self.P_post = self._blend_rows(P_post, self.P_post, mask)
        self.state_post = self._clip_state(self._sanitize_tensor(self.state_post))
        self.P_post = _symmetrize(self._sanitize_tensor(self.P_post))

        self.x_post_prev2 = self.x_post_prev.detach().clone()
        self.x_post_prev = self.state_post.detach().clone()
        self.m1y = y_pred.detach().clone()
        self._commit_sensor_observation(sensor, y_raw, mask)
        self.step_idx += 1

        self._last_step_raw = {
            "q_diag_scale": core.q_diag_scale.detach(),
            "r_diag_scale": core.r_diag_scale.detach(),
            "split_alpha": alpha.detach(),
            "K_ekf": K_ekf.detach(),
            "K_split": K_split.detach(),
            "K_gain": K.detach(),
        }
        self._last_step_aux = {
            "anchor_pos": self._anchor_pos.detach().clone(),
            "anchor_conf": self._anchor_conf.detach().clone(),
            "anchor_age": self._anchor_age.detach().clone(),
            "anchor_gate": anchor_gate.detach(),
            "residual": residual.detach(),
            "delta": delta.detach(),
            "P_post": self.P_post.detach(),
            "S_eff": S_eff.detach(),
        }
        return self.state_post


# Backward-compatible aliases.
CTInfoFusionEKFBackbonePaperV2 = CTInfoFusionCTIFLite
CTInfoFusionEKFBackboneStable = CTInfoFusionCTIFLite


CTInfoFusionEKFBackbonePaperV2CFC = CTInfoFusionCTIFLite
