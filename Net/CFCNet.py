"""
CFCNet.py
Platinum Version (Sealed)
-------------------------
Final Patches Applied:
1. Scalar-based Missingness Check (.item() < 0.5)
2. Channel-wise Zeroing (Protects concatenated mask channel)
3. Strict Exception Handling (No swallowing of real bugs)
"""

import torch
import torch.nn as nn
import torch.nn.functional as func
import math

# ==========================================
# 1. Sparse Continuous-Time Flow Cell (CFC)
# ==========================================
class SparseCfcCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True, activation="tanh",
                 sparsity=0.0, use_mask_input=False):
        super(SparseCfcCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.sparsity = sparsity
        self.use_mask_input = use_mask_input # [Context] Logic switch

        # Backbone & Time Gates
        self.backbone = nn.Linear(input_size + hidden_size, hidden_size, bias=bias)
        self.time_a = nn.Linear(input_size + hidden_size, hidden_size, bias=bias)
        self.time_b = nn.Linear(input_size + hidden_size, hidden_size, bias=bias)

        self.activation = nn.Tanh() if activation == "tanh" else nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        if sparsity > 0:
            self.register_buffer("mask_backbone", self._create_mask(self.backbone.weight, sparsity))
            self.register_buffer("mask_time_a", self._create_mask(self.time_a.weight, sparsity))
            self.register_buffer("mask_time_b", self._create_mask(self.time_b.weight, sparsity))

    def _create_mask(self, weight, sparsity):
        k = int((1 - sparsity) * weight.numel())
        mask = torch.zeros_like(weight)
        if k > 0:
            indices = torch.randperm(weight.numel())[:k]
            mask.view(-1)[indices] = 1
        return mask

    def forward(self, input, hx, timespan, mask=None):
        """
        :param input: [Batch, Input_Dim] (Concatenated with mask if configured)
        :param hx: [Batch, Hidden_Dim]
        :param timespan: [Batch, 1] Clamped Dt
        :param mask: [Batch, 1] Binary Mask for Zeroing
        """
        # [Patch 2] Smart Zeroing (Protect Mask Channel)
        if mask is not None:
            if mask.dim() == 1: mask = mask.unsqueeze(-1)

            if self.use_mask_input:
                # input is [Batch, Feat + 1]
                # Split features and mask channel
                feat = input[:, :-1]
                mask_channel = input[:, -1:]

                # Only zero out the features
                masked_feat = feat * mask.expand_as(feat)

                # Re-assemble: features are zeroed, mask channel preserves its value
                # (Crucial for future soft-mask/confidence scenarios)
                masked_input = torch.cat([masked_feat, mask_channel], dim=1)
            else:
                # Standard full zeroing
                masked_input = input * mask.expand_as(input)
        else:
            masked_input = input

        x_cat = torch.cat([masked_input, hx], 1)

        # Apply Sparsity
        if self.sparsity > 0:
            w_backbone = self.backbone.weight * self.mask_backbone
            w_time_a = self.time_a.weight * self.mask_time_a
            w_time_b = self.time_b.weight * self.mask_time_b

            res_out = func.linear(x_cat, w_backbone, self.backbone.bias)
            time_a_out = func.linear(x_cat, w_time_a, self.time_a.bias)
            time_b_out = func.linear(x_cat, w_time_b, self.time_b.bias)
        else:
            res_out = self.backbone(x_cat)
            time_a_out = self.time_a(x_cat)
            time_b_out = self.time_b(x_cat)

        # Time-Continuous Gating
        time_param = time_a_out * timespan + time_b_out
        gate = self.sigmoid(time_param)

        new_h = gate * hx + (1 - gate) * self.activation(res_out)

        return new_h, new_h


# ==========================================
# 2. KalmanNet Wrapper (Sealed)
# ==========================================
class KalmanNetCFC(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def NNBuild(self, SysModel, args):
        self.device = torch.device('cuda' if args.use_cuda and torch.cuda.is_available() else 'cpu')

        # [Assertion] Semantic Safety
        self.m = SysModel.m
        self.n = SysModel.n

        self.nominal_dt = getattr(args, 'nominal_dt', 0.01)
        self.use_mask_input = getattr(args, 'use_mask_input', False)

        self.InitSystemDynamics(SysModel)
        self.InitKGainNet(SysModel.prior_Q, SysModel.prior_Sigma, SysModel.prior_S, args)

    def InitSystemDynamics(self, SysModel):
        self.f = getattr(SysModel, 'f', None)
        self.h = getattr(SysModel, 'h', None)

        self.f_mode = self._probe_dynamics()

        print(f"[CFCNet] Dynamics Mode: {self.f_mode}")
        print(f"[CFCNet] Semantics: State(m)={self.m}, Obs(n)={self.n}")

    def _probe_dynamics(self):
        """探测 f 的签名"""
        if self.f is None: return 'linear'

        dummy_x = torch.zeros(1, self.m, 1).to(self.device)
        dummy_dt = torch.ones(1, 1).to(self.device) * 0.01

        # [Patch 3] Strict Exception Handling
        # 1. Try 3D Discrete
        try:
            _ = self.f(dummy_x, dummy_dt)
            return 'discrete'
        except TypeError:
            return 'continuous' # Args mismatch -> Continuous
        except (RuntimeError, ValueError):
            pass # Shape mismatch -> Try 2D

        # 2. Try 2D Discrete
        x_2d = dummy_x.squeeze(-1)

        # 2a. Try with dt [B, 1]
        try:
            _ = self.f(x_2d, dummy_dt)
            return 'discrete'
        except (TypeError, RuntimeError, ValueError):
            pass

        # 2b. Try with dt [B]
        try:
            _ = self.f(x_2d, dummy_dt.squeeze(-1))
            return 'discrete'
        except (TypeError, RuntimeError, ValueError):
            pass

        # 3. Fallback
        return 'continuous'

    def _ensure_3d(self, tensor):
        if tensor.dim() == 2:
            return tensor.unsqueeze(-1)
        return tensor

    def _ensure_dt(self, dt, batch_size):
        if dt is None:
            return torch.ones(batch_size, 1).to(self.device) * self.nominal_dt

        if isinstance(dt, (float, int)):
            return torch.ones(batch_size, 1).to(self.device) * dt

        if dt.dim() == 0:
             return torch.ones(batch_size, 1).to(self.device) * dt
        if dt.dim() == 1:
            return dt.unsqueeze(-1)
        return dt

    def InitKGainNet(self, prior_Q, prior_Sigma, prior_S, args):
        self.batch_size = args.n_batch

        self.prior_Q = prior_Q.to(self.device)
        self.prior_Sigma = prior_Sigma.to(self.device)
        self.prior_S = prior_S.to(self.device)

        MY_SPARSITY = 0.6

        # Mask Dim Logic
        self.mask_dim = 1 if self.use_mask_input else 0

        # 1. Q-Block
        self.d_input_Q = self.m * args.in_mult_KNet + self.mask_dim
        self.d_hidden_Q = self.m ** 2
        self.CFC_Q = SparseCfcCell(self.d_input_Q, self.d_hidden_Q,
                                   sparsity=MY_SPARSITY, use_mask_input=self.use_mask_input).to(self.device)

        # 2. Sigma-Block
        self.d_input_Sigma = self.d_hidden_Q + self.m * args.in_mult_KNet + self.mask_dim
        self.d_hidden_Sigma = self.m ** 2
        self.CFC_Sigma = SparseCfcCell(self.d_input_Sigma, self.d_hidden_Sigma,
                                       sparsity=MY_SPARSITY, use_mask_input=self.use_mask_input).to(self.device)

        # 3. S-Block
        self.d_input_S = self.n ** 2 + 2 * self.n * args.in_mult_KNet + self.mask_dim
        self.d_hidden_S = self.n ** 2
        self.CFC_S = SparseCfcCell(self.d_input_S, self.d_hidden_S,
                                   sparsity=MY_SPARSITY, use_mask_input=self.use_mask_input).to(self.device)

        # FC Layers
        self.d_input_FC1 = self.d_hidden_Sigma
        self.d_output_FC1 = self.n ** 2
        self.FC1 = nn.Sequential(nn.Linear(self.d_input_FC1, self.d_output_FC1), nn.ReLU()).to(self.device)

        self.d_input_FC2 = self.d_hidden_S + self.d_hidden_Sigma
        self.d_output_FC2 = self.m * self.n
        reduced_out_mult = max(2, args.out_mult_KNet // 8)
        self.d_hidden_FC2 = self.d_input_FC2 * reduced_out_mult

        self.FC2 = nn.Sequential(
            nn.Linear(self.d_input_FC2, self.d_hidden_FC2), nn.ReLU(),
            nn.Linear(self.d_hidden_FC2, self.d_output_FC2)
        ).to(self.device)

        self.d_input_FC3 = self.d_hidden_S + self.d_output_FC2
        self.d_output_FC3 = self.m ** 2
        self.FC3 = nn.Sequential(nn.Linear(self.d_input_FC3, self.d_output_FC3), nn.ReLU()).to(self.device)

        self.d_input_FC4 = self.d_hidden_Sigma + self.d_output_FC3
        self.d_output_FC4 = self.d_hidden_Sigma
        self.FC4 = nn.Sequential(nn.Linear(self.d_input_FC4, self.d_output_FC4), nn.ReLU()).to(self.device)

        # Encoders
        self.d_input_FC5 = self.m
        self.d_output_FC5 = self.m * args.in_mult_KNet
        self.FC5 = nn.Sequential(nn.Linear(self.d_input_FC5, self.d_output_FC5), nn.ReLU()).to(self.device)

        self.d_input_FC6 = self.m
        self.d_output_FC6 = self.m * args.in_mult_KNet
        self.FC6 = nn.Sequential(nn.Linear(self.d_input_FC6, self.d_output_FC6), nn.ReLU()).to(self.device)

        self.d_input_FC7 = 2 * self.n
        self.d_output_FC7 = 2 * self.n * args.in_mult_KNet
        self.FC7 = nn.Sequential(nn.Linear(self.d_input_FC7, self.d_output_FC7), nn.ReLU()).to(self.device)

    def InitSequence(self, M1_0, T):
        self.T = T
        self.batch_size = M1_0.shape[0]

        assert M1_0.shape[1] == self.m, f"Input dim {M1_0.shape[1]} != State dim {self.m}"

        self.m1x_posterior = self._ensure_3d(M1_0.to(self.device))
        self.m1x_posterior_previous = self.m1x_posterior
        self.m1x_prior_previous = self.m1x_posterior

        if self.h is not None:
            try:
                self.y_previous = self._ensure_3d(self.h(self.m1x_posterior))
            except (RuntimeError, TypeError, ValueError):
                 self.y_previous = self._ensure_3d(self.h(self.m1x_posterior.squeeze(-1)))
        else:
            self.y_previous = torch.zeros(self.batch_size, self.n, 1).to(self.device)

        self.init_hidden_KNet()

    def step_prior(self, dt):
        # Numerical Safety
        if self.training:
            assert torch.isfinite(dt).all(), "DT contains NaN/Inf"
            assert torch.isfinite(self.m1x_posterior).all(), "Posterior contains NaN/Inf"

        if self.f_mode == 'discrete':
            try:
                self.m1x_prior = self.f(self.m1x_posterior, dt)
            except (RuntimeError, TypeError, ValueError):
                # Optimization: Try Squeezed 2D inputs directly
                x_sq = self.m1x_posterior.squeeze(-1)
                try:
                    self.m1x_prior = self.f(x_sq, dt.squeeze(-1)) # [Patch] Try most likely 2D case first
                except (RuntimeError, TypeError, ValueError):
                    self.m1x_prior = self.f(x_sq, dt) # Fallback

        elif self.f_mode == 'continuous':
            try:
                x_dot = self.f(self.m1x_posterior)
            except (RuntimeError, TypeError, ValueError):
                x_dot = self.f(self.m1x_posterior.squeeze(-1))

            x_dot = self._ensure_3d(x_dot)
            self.m1x_prior = self.m1x_posterior + x_dot * dt
        else:
            self.m1x_prior = self.m1x_posterior

        self.m1x_prior = self._ensure_3d(self.m1x_prior)

        if self.h is not None:
            try:
                pred_y = self.h(self.m1x_prior)
            except (RuntimeError, TypeError, ValueError):
                pred_y = self.h(self.m1x_prior.squeeze(-1))
            self.m1y = self._ensure_3d(pred_y)
        else:
             self.m1y = self.y_previous

    def step_KGain_est(self, y, dt, mask):
        obs_diff = y - self.y_previous
        obs_innov_diff = y - self.m1y
        fw_evol_diff = self.m1x_posterior - self.m1x_posterior_previous
        fw_update_diff = self.m1x_posterior - self.m1x_prior_previous

        # Pre-Normalize Masking
        if mask is not None:
            mask_bool = (mask > 0).float().view(-1, 1, 1)
            obs_diff = obs_diff * mask_bool
            obs_innov_diff = obs_innov_diff * mask_bool

        # Normalize
        obs_diff = func.normalize(obs_diff.squeeze(-1), p=2, dim=1, eps=1e-12)
        obs_innov_diff = func.normalize(obs_innov_diff.squeeze(-1), p=2, dim=1, eps=1e-12)
        fw_evol_diff = func.normalize(fw_evol_diff.squeeze(-1), p=2, dim=1, eps=1e-12)
        fw_update_diff = func.normalize(fw_update_diff.squeeze(-1), p=2, dim=1, eps=1e-12)

        KG = self.KGain_step(obs_diff, obs_innov_diff, fw_evol_diff, fw_update_diff, dt, mask)

        self.KGain = torch.reshape(KG, (self.batch_size, self.m, self.n))

    def KGain_step(self, obs_diff, obs_innov_diff, fw_evol_diff, fw_update_diff, dt, mask):
        # External Mask Concatenation
        mask2 = None
        if mask is not None:
            mask2 = mask.float()
            if mask2.dim() == 1:
                mask2 = mask2.unsqueeze(1)
            elif mask2.dim() > 2:
                mask2 = mask2.view(mask2.size(0), -1)[:, :1]

        def _cat_mask(tensor):
            if self.mask_dim > 0 and mask2 is not None:
                return torch.cat([tensor, mask2], dim=1)
            return tensor

        def _blend(old, new):
            if mask2 is None:
                return new
            return mask2 * new + (1.0 - mask2) * old

        hQ_old = self.h_Q
        out_FC5 = self.FC5(fw_update_diff)
        in_Q = _cat_mask(out_FC5)

        out_Q, hQ_new = self.CFC_Q(in_Q, hQ_old, timespan=dt, mask=mask2)
        self.h_Q = _blend(hQ_old, hQ_new)
        out_Q = _blend(hQ_old, out_Q)  # 让下游看到的 out_Q 也在 mask=0 时不变

        # =========================
        # 2) Sigma block
        # =========================
        hSig_old = self.h_Sigma
        out_FC6 = self.FC6(fw_evol_diff)
        in_Sigma = _cat_mask(torch.cat((out_Q, out_FC6), dim=1))

        out_Sigma, hSig_new = self.CFC_Sigma(in_Sigma, hSig_old, timespan=dt, mask=mask2)
        out_Sigma = _blend(hSig_old, out_Sigma)

        # =========================
        # 3) S block
        # =========================
        hS_old = self.h_S
        out_FC1 = self.FC1(out_Sigma)
        out_FC7 = self.FC7(torch.cat((obs_diff, obs_innov_diff), dim=1))
        in_S = _cat_mask(torch.cat((out_FC1, out_FC7), dim=1))

        out_S, hS_new = self.CFC_S(in_S, hS_old, timespan=dt, mask=mask2)
        self.h_S = _blend(hS_old, hS_new)
        out_S = _blend(hS_old, out_S)

        # =========================
        # 4) FC stack -> KGain
        # =========================
        out_FC2 = self.FC2(torch.cat((out_Sigma, out_S), dim=1))
        out_FC3 = self.FC3(torch.cat((out_S, out_FC2), dim=1))
        out_FC4 = self.FC4(torch.cat((out_Sigma, out_FC3), dim=1))

        # 关键：Sigma 的“跨步记忆”也要 mask=0 冻结
        # 保留你原结构：用 out_FC4 作为下一步的 Sigma hidden（但做冻结）
        self.h_Sigma = _blend(hSig_old, out_FC4)

        return out_FC2

    def KNet_step(self, y, dt, mask):
        # [Patch 1] Safe Scalar Missingness Check
        if mask is not None and mask.detach().max().item() < 0.5:
            # Skip NN, just run physics. Saves gradients and compute.
            self.step_prior(dt)
            self.m1x_posterior_previous = self.m1x_posterior
            self.m1x_posterior = self.m1x_prior
            self.m1x_prior_previous = self.m1x_prior
            self.y_previous = self.m1y.detach()
            return self.m1x_posterior

        self.step_prior(dt)

        y = self._ensure_3d(y)
        self.step_KGain_est(y, dt, mask)

        dy = y - self.m1y
        if mask is not None:
            mask_exp = mask.view(-1, 1, 1)
            dy = dy * mask_exp

        INOV = torch.bmm(self.KGain, dy)

        self.m1x_posterior_previous = self.m1x_posterior
        self.m1x_posterior = self.m1x_prior + INOV
        self.m1x_prior_previous = self.m1x_prior

        if mask is not None:
            mask_bool = (mask > 0).view(-1, 1, 1)
            self.y_previous = torch.where(mask_bool, y, self.m1y.detach())
        else:
            self.y_previous = y

        return self.m1x_posterior

    def forward(self, y, dt=None, mask=None):
        y = y.to(self.device)
        bs = y.shape[0]
        self.batch_size = bs

        # Robust Shape Handling
        if y.shape[1] != self.n:
             if y.dim() == 3 and y.shape[2] == self.n:
                 y = y.squeeze(1)
             elif y.dim() == 3 and y.shape[1] == self.n:
                 pass
             else:
                 raise ValueError(f"Obs Dim {y.shape} mismatch SysModel.n {self.n}")

        dt = self._ensure_dt(dt, bs)
        dt = torch.clamp(dt, min=1e-4, max=0.1)

        if mask is None:
            mask = torch.ones(bs, 1).to(self.device)
        else:
            mask = mask.to(self.device)
            mask = (mask > 0.5).float() # Binary Mask

        return self.KNet_step(y, dt, mask)

    def init_hidden_KNet(self):
        if not hasattr(self, 'prior_S'):
             raise RuntimeError("Call NNBuild before InitSequence!")

        self.h_S = self.prior_S.flatten().reshape(1, -1).repeat(self.batch_size, 1).detach()
        self.h_Sigma = self.prior_Sigma.flatten().reshape(1, -1).repeat(self.batch_size, 1).detach()
        self.h_Q = self.prior_Q.flatten().reshape(1, -1).repeat(self.batch_size, 1).detach()