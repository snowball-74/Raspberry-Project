"""
model.py
========
CNN + GRU 手语识别模型（双手版本）。

输入维度变化（相对于单手版本）：
  feat_dim : 27  →  72   （双手 + 身体相对距离，去掉双手绝对坐标后）
  pose_dim : 12  →  24   （右手12 + 左手12，起止手型）

网络结构不变，只有入口维度跟着走。
"""

import torch
import torch.nn as nn


class Model_CNNGRU(nn.Module):
    def __init__(self,
                 num_classes,
                 feat_dim=78,        # 双手版默认78维（30+30+18）
                 pose_dim=24,        # 双手版默认24维
                 hidden_size=128,
                 bidirectional=False):
        super().__init__()
        self.bidirectional = bidirectional

        # ── 1. 局部特征提取（CNN） ─────────────────────────────
        self.local_cnn = nn.Sequential(
            nn.Conv1d(feat_dim, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(),
        )

        # ── 2. 时序特征提取（GRU） ─────────────────────────────
        self.gru = nn.GRU(
            128, hidden_size, num_layers=2,
            batch_first=True, dropout=0.3,
            bidirectional=bidirectional
        )

        # ── 3. 姿态（起止手型）分支 ───────────────────────────
        #    pose_dim * 2：start_pose 和 end_pose 拼接
        self.pose_fc = nn.Sequential(
            nn.Linear(pose_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # ── 4. 分类头 ──────────────────────────────────────────
        gru_out_dim = hidden_size * (2 if bidirectional else 1)
        self.head = nn.Sequential(
            nn.Linear(gru_out_dim + 128, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, seq, sp, ep):
        """
        seq : (B, T, feat_dim)
        sp  : (B, pose_dim)   起始手型
        ep  : (B, pose_dim)   终止手型
        """
        # CNN 需要 (B, D, T)
        x = self.local_cnn(seq.transpose(1, 2)).transpose(1, 2)   # (B, T, 128)

        # GRU
        _, h = self.gru(x)   # h: (num_layers * directions, B, hidden)

        if self.bidirectional:
            x = torch.cat([h[-2], h[-1]], dim=1)   # (B, hidden*2)
        else:
            x = h[-1]                               # (B, hidden)

        # 姿态分支
        p = self.pose_fc(torch.cat([sp, ep], dim=1))   # (B, 128)

        # 融合分类
        return self.head(torch.cat([x, p], dim=1))