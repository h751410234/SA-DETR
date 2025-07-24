# bisar_style_adapter.py
# ----------------------------------------------------------
# Bi-SAR: Bidirectional Style-Adaptive Retrieval
# 完整实现：StyleMemoryBank + BiSARAdapter
# Apache-2.0 License
# ----------------------------------------------------------
from __future__ import annotations
from typing import Tuple, Dict

import torch
from torch import nn
import torch.nn.functional as F

###############################################################################
# Helper: 计算每张特征图的均值 / 标准差
###############################################################################
def calculate_mu_sig(x: torch.Tensor, eps: float = 1e-6
                     ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Channel-wise μ / σ for each feature map. 形状 [B,C]."""
    mu  = x.mean(dim=[2, 3])                     # [B,C]
    var = x.var (dim=[2, 3], unbiased=False)
    sig = (var + eps).sqrt()
    return mu.detach(), sig.detach()

###############################################################################
# Style Memory Bank
###############################################################################
class StyleMemoryBank(nn.Module):
    """Ring-buffer style prototype bank (GPU-friendly, DDP-safe)."""

    def __init__(self,
                 max_size : int,
                 channel  : int,
                 device   = torch.device("cpu"),
                 tau_scale: float = 0.7,
                 gamma    : float = 0.995,
                 use_organize = True
                 ):
        """
        max_size : 容量 (条数)
        channel  : 特征通道数
        tau_scale: 距离阈值比例，用于替换策略
        gamma    : aging 衰减系数
        """
        super().__init__()
        self.register_buffer("mu_bank",  torch.zeros(max_size, channel, device=device))
        self.register_buffer("sig_bank", torch.ones (max_size, channel, device=device))
        self.register_buffer("age",      torch.zeros(max_size, device=device))          # aging 分数
        self.register_buffer("ptr",      torch.zeros(1, dtype=torch.long, device=device))  # 写入计数
        self.max_size      = max_size
        self.tau_scale     = tau_scale
        self.gamma         = gamma
        self.replace_count = 0  # 统计替换次数，用于debug
        self.use_organize = use_organize

    # ------------------------------------------------------------------ #
    # 写入 / 更新
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def add_or_update(self,
                      mu     : torch.Tensor,
                      sig    : torch.Tensor,
                      m_intra: float = 0.1):
        """
        根据新样本 (mu, sig) 自动执行 “填充 / 替换 / EMA 融合”。

        m_intra : EMA 动量（0..1），值越大 → 新样本占比越高
        """
        B = mu.size(0)
        for b in range(B):
            cur_mu, cur_sig = mu[b], sig[b]

            # -------- 阶段 A：环形填充 ------------------------------------
            if int(self.ptr) < self.max_size:
                idx = int(self.ptr)
                self.mu_bank[idx]  = cur_mu
                self.sig_bank[idx] = cur_sig
                self.age[idx]      = 1
                self.ptr += 1
                continue

            # -------- 阶段 B/C：最近距离检索 ------------------------------
            valid = self.cur_size()                       # == max_size
            dist  = self._distance(
                        cur_mu, cur_sig,
                        self.mu_bank[:valid],
                        self.sig_bank[:valid])            # [valid]
            d_min, j_min = dist.min(0)
            tau = self.tau_scale * dist.mean()            # 适应性阈值

            if self.use_organize: #开启替换
                if d_min > tau:                               # 替换最老槽
                    j_rep = torch.argmin(self.age[:valid])
                    self.mu_bank[j_rep], self.sig_bank[j_rep] = cur_mu, cur_sig
                    self.age[j_rep] = 1
                    self.replace_count += 1
                else:                                         # EMA 融合
                    self.mu_bank[j_min]  = (1 - m_intra) * self.mu_bank[j_min]  + m_intra * cur_mu
                    self.sig_bank[j_min] = (1 - m_intra) * self.sig_bank[j_min] + m_intra * cur_sig
                    self.age[j_min]      = 1
            else:  # 不启用替换
                self.mu_bank[j_min] = (1 - m_intra) * self.mu_bank[j_min] + m_intra * cur_mu
                self.sig_bank[j_min] = (1 - m_intra) * self.sig_bank[j_min] + m_intra * cur_sig
                self.age[j_min] = 1



        # Aging 所有槽
        self.age.mul_(self.gamma)

    # ------------------------------------------------------------------ #
    # KNN 检索
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def knn(self,
            mu_q : torch.Tensor,
            sig_q: torch.Tensor,
            k    : int   = 4,
            metric: str  = "was"):
        """
        mu_q / sig_q : [B,C]
        return (dist, idx) or (None, None) if bank 不足 k 条
        """
        if self.cur_size() < k:                         # 至少 k 条才能检索
            self.add_or_update(mu_q.detach(), sig_q.detach(), m_intra=1.0)
            return None, None

        N = self.cur_size()
        mu_q, sig_q = mu_q.unsqueeze(1), sig_q.unsqueeze(1)             # [B,1,C]
        mu_b, sig_b = self.mu_bank[:N].unsqueeze(0), self.sig_bank[:N].unsqueeze(0)

        if metric == "was":
            dist = (mu_q - mu_b).pow(2) + (sig_q.pow(2) + sig_b.pow(2) - 2 * sig_q * sig_b)
        elif metric == "abs":
            dist = torch.abs(mu_q / (sig_q + 1e-6) - mu_b / (sig_b + 1e-6))
        else:
            raise ValueError(metric)
        dist = dist.mean(dim=-1)                                        # [B,N]
        d, idx = torch.topk(dist, k=k, largest=False)                   # [B,k]
        return d, idx

    # ------------------------------------------------------------------ #
    # 工具函数
    # ------------------------------------------------------------------ #
    def _distance(self, mu_q, sig_q, mu_b, sig_b):
        return ((mu_q - mu_b) ** 2 +
                sig_q ** 2 + sig_b ** 2 -
                2 * sig_q * sig_b).mean(dim=-1)

    def cur_size(self) -> int:
        """返回已写入的原型数（int）。"""
        return min(int(self.ptr), self.max_size)

    @torch.no_grad()
    def sync_between_gpus(self):
        """DDP 下：把各卡 Bank 取平均（简单实用）。"""
        if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
            return
        world = torch.distributed.get_world_size()
        for buf in [self.mu_bank, self.sig_bank, self.age]:
            torch.distributed.all_reduce(buf)
            buf /= world
        torch.distributed.all_reduce(self.ptr)          # 写入计数
        self.ptr //= world

###############################################################################
#  Adapter
###############################################################################
class Adapter(nn.Module):
    """KNN-AdaIN + 可选在线更新。"""

    def __init__(self,
                 channel : int,
                 mem_bank: StyleMemoryBank,
                 k       : int   = 4,
                 metric  : str   = "was"):
        super().__init__()
        self.channel  = channel
        self.mem      = mem_bank
        self.k        = k
        self.metric   = metric

    def forward(self,
                fea       : torch.Tensor,
                *,
                update_mem: bool  = False,
                online_m  : float = 0.05) -> torch.Tensor:
        """
        fea        : [B,C,H,W] 特征图
        update_mem : True → 写新统计进 Bank
        online_m   : 推理期动量
        """
        mu, sig = calculate_mu_sig(fea)                       # [B,C]

        # ---------- KNN 检索 ----------
        with torch.no_grad():
            out = self.mem.knn(mu, sig, k=self.k, metric=self.metric)
            if out == (None, None):                           # Bank 不足 k 条
                return fea
            dist, idx = out                                   # [B,k]
            mu_p  = self.mem.mu_bank[idx]                     # [B,k,C]
            sig_p = self.mem.sig_bank[idx]
            alpha = F.softmax(-dist, dim=1).unsqueeze(-1)     # [B,k,1]
            mix_mu  = (alpha * mu_p ).sum(dim=1)              # [B,C]
            mix_sig = (alpha * sig_p).sum(dim=1)

        # ---------- AdaIN ----------
        fea = (fea - mu[:, :, None, None]) / sig[:, :, None, None]
        fea =  fea * mix_sig[:, :, None, None] + mix_mu[:, :, None, None]
        # ---------- 在线更新 ----------
        if self.training or update_mem:  #训练时或开启TTA时进行更新
            with torch.no_grad():
                self.mem.add_or_update(mu.detach(), sig.detach(),
                                       m_intra=online_m)
        return fea


