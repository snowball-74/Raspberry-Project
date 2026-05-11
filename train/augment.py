import numpy as np
from typing import Optional, Dict, Any


# ─────────────────────────────────────────────────────────────────
#  特征索引常量（与 preprocess v2 完全对应）
# ─────────────────────────────────────────────────────────────────

FEAT_DIM = 172
POSE_DIM = 40

# ── 右手通道（offset=0）────────────────────────────────────────────
RH_OFF = 0

# 手型静态
RH_R1_VECS   = slice(RH_OFF+0,  RH_OFF+15)   # R1→食/中/无名/小/掌心（5×3）
RH_R2_HTIPS  = slice(RH_OFF+15, RH_OFF+27)   # R2→手部4指尖（4×3）
RH_R2_POSE   = slice(RH_OFF+27, RH_OFF+45)   # R2→pose 6点（6×3）
RH_R3_POSE   = slice(RH_OFF+45, RH_OFF+60)   # R3→pose 5点（5×3）
RH_BEND      = slice(RH_OFF+60, RH_OFF+65)   # 5指弯曲角度
# 轨迹动态
RH_TIP_VEL   = slice(RH_OFF+65, RH_OFF+80)   # 五指尖相对R2速度（5×3）
RH_PALM_VEL  = slice(RH_OFF+80, RH_OFF+83)   # R2相对肩中心速度
RH_WRIST_VEL = slice(RH_OFF+83, RH_OFF+86)   # R3相对肩中心速度

# ── 左手通道（offset=86）──────────────────────────────────────────
LH_OFF = 86

LH_L1_VECS   = slice(LH_OFF+0,  LH_OFF+15)
LH_L2_HTIPS  = slice(LH_OFF+15, LH_OFF+27)
LH_L2_POSE   = slice(LH_OFF+27, LH_OFF+45)
LH_L3_POSE   = slice(LH_OFF+45, LH_OFF+60)
LH_BEND      = slice(LH_OFF+60, LH_OFF+65)
LH_TIP_VEL   = slice(LH_OFF+65, LH_OFF+80)
LH_PALM_VEL  = slice(LH_OFF+80, LH_OFF+83)
LH_WRIST_VEL = slice(LH_OFF+83, LH_OFF+86)

# ── 左手所有通道（遮挡时清零）──────────────────────────────────────
LH_ALL_SLICES = [
    LH_L1_VECS, LH_L2_HTIPS, LH_L2_POSE,
    LH_L3_POSE, LH_BEND,
    LH_TIP_VEL, LH_PALM_VEL, LH_WRIST_VEL,
]

# ── 右手所有通道（遮挡时清零）──────────────────────────────────────
RH_ALL_SLICES = [
    RH_R1_VECS, RH_R2_HTIPS, RH_R2_POSE,
    RH_R3_POSE, RH_BEND,
    RH_TIP_VEL, RH_PALM_VEL, RH_WRIST_VEL,
]

# ── Pose 中左/右手分量 ─────────────────────────────────────────────
# 右手20维 = [0:20]；左手20维 = [20:40]
POSE_RH = slice(0,  20)
POSE_LH = slice(20, 40)


# ─────────────────────────────────────────────────────────────────
#  默认增广配置
# ─────────────────────────────────────────────────────────────────

DEFAULT_CFG: Dict[str, Any] = {
    "noise": {
        "prob": 0.8,
        "scale": 0.006,
        "vel_scale_ratio": 0.5,   # 速度通道噪声倍率
        "angle_scale_ratio": 0.3, # 弯曲角度通道噪声倍率（弧度单位，需更小）
    },
    "time_warp": {
        "prob": 0.5,
        "max_warp": 0.15,
        "n_anchors": 4,
    },
    "time_scale": {
        "prob": 0.5,
        "min_scale": 0.8,
        "max_scale": 1.2,
    },
    "spatial_shift": {
        "prob": 0.7,
        "xy_range": 0.12,
        "z_range":  0.05,
    },
    "spatial_zoom": {
        "prob": 0.9,
        "min_scale": 0.5,
        "max_scale": 2.2,
    },
    "distance_simulate": {
        "prob": 0.0,   # 由 spatial_zoom 转发，不单独注册触发
        "min_scale": 0.5,
        "max_scale": 2.2,
    },
    "frame_drop": {
        "prob": 0.4,
        "drop_ratio": 0.15,
    },
    "random_erase": {
        "prob": 0.3,
        "max_len": 5,
    },
    "occlude_left_hand": {
        "prob": 0.35,
    },
    "occlude_right_hand": {
        "prob": 0.10,
    },
    "mirror": {
        "prob": 0.0,   # 手语不对称时保持 0
    },
}


# ─────────────────────────────────────────────────────────────────
#  工具函数
# ─────────────────────────────────────────────────────────────────

def _align_sequence(arr: np.ndarray, n_frames: int) -> np.ndarray:
    T, D = arr.shape
    if T == n_frames:
        return arr.astype(np.float32)
    t_src = np.linspace(0, 1, T)
    t_dst = np.linspace(0, 1, n_frames)
    return np.stack(
        [np.interp(t_dst, t_src, arr[:, d]) for d in range(D)],
        axis=1,
    ).astype(np.float32)


def _hand_is_active(seq: np.ndarray, slc: slice, thresh: float = 0.01) -> bool:
    """判断指定手是否有实质信号（取指尖相对掌心向量均值判定）"""
    return float(np.abs(seq[:, slc]).mean()) > thresh


def _rh_is_active(seq: np.ndarray, thresh: float = 0.01) -> bool:
    return _hand_is_active(seq, RH_R2_HTIPS, thresh)

def _lh_is_active(seq: np.ndarray, thresh: float = 0.01) -> bool:
    return _hand_is_active(seq, LH_L2_HTIPS, thresh)


# ─────────────────────────────────────────────────────────────────
#  各增广函数
# ─────────────────────────────────────────────────────────────────

def aug_noise(seq: np.ndarray, sp: np.ndarray, ep: np.ndarray,
              rng: np.random.Generator, cfg: dict):
    """全通道高斯噪声，速度通道和角度通道单独控制倍率。"""
    scale = cfg["scale"]
    vr    = cfg["vel_scale_ratio"]
    ar    = cfg["angle_scale_ratio"]

    noise = rng.standard_normal(seq.shape).astype(np.float32) * scale
    # 速度通道降噪
    for sl in [RH_TIP_VEL, RH_PALM_VEL, RH_WRIST_VEL,
               LH_TIP_VEL, LH_PALM_VEL, LH_WRIST_VEL]:
        noise[:, sl] *= vr
    # 弯曲角度通道进一步降噪（弧度单位本身就小）
    noise[:, RH_BEND] *= ar
    noise[:, LH_BEND] *= ar

    seq = seq + noise
    pose_noise = rng.standard_normal(sp.shape).astype(np.float32) * scale
    sp = sp + pose_noise
    ep = ep + rng.standard_normal(ep.shape).astype(np.float32) * scale

    return seq, sp, ep


def aug_time_warp(seq: np.ndarray, sp: np.ndarray, ep: np.ndarray,
                  rng: np.random.Generator, cfg: dict):
    """时间轴非线性扭曲，改变动作节奏。"""
    T         = len(seq)
    n_anchors = cfg["n_anchors"]
    max_warp  = cfg["max_warp"]

    anchor_x = np.linspace(0, T - 1, n_anchors + 2)
    offsets  = np.concatenate([[0],
                                rng.uniform(-max_warp * T, max_warp * T, n_anchors),
                                [0]])
    warped_x = np.clip(anchor_x + offsets, 0, T - 1)
    t_dst    = np.linspace(0, T - 1, T)
    t_src    = np.interp(t_dst, anchor_x, warped_x)

    seq_new = np.stack(
        [np.interp(t_src, np.arange(T), seq[:, d]) for d in range(FEAT_DIM)],
        axis=1,
    ).astype(np.float32)
    return seq_new, sp, ep


def aug_time_scale(seq: np.ndarray, sp: np.ndarray, ep: np.ndarray,
                   rng: np.random.Generator, cfg: dict):
    """时间缩放，模拟快/慢手语。"""
    T     = len(seq)
    scale = rng.uniform(cfg["min_scale"], cfg["max_scale"])
    new_len = max(4, int(T * scale))
    idx     = np.round(np.linspace(0, T - 1, new_len)).astype(int)
    seq     = _align_sequence(seq[idx], T)
    return seq, sp, ep


def aug_spatial_shift(seq: np.ndarray, sp: np.ndarray, ep: np.ndarray,
                      rng: np.random.Generator, cfg: dict):
    """
    相对坐标整体平移。
    新布局中没有掌心绝对坐标，平移作用于 R2/L2 中心的 pose 方向向量
    （即"掌心→身体各点"的相对量），以及 R3/L3 的 pose 向量。
    速度通道是差分量，不受平移影响，无需处理。
    弯曲角度不受平移影响，无需处理。
    """
    dx = rng.uniform(-cfg["xy_range"], cfg["xy_range"])
    dy = rng.uniform(-cfg["xy_range"], cfg["xy_range"])
    dz = rng.uniform(-cfg["z_range"],  cfg["z_range"])
    delta = np.array([dx, dy, dz], dtype=np.float32)

    seq = seq.copy()

    # R2 pose 向量（6×3=18维）：每个向量 x/y/z 分量各加偏移
    # 这些向量是"掌心→鼻/腕/肩/肘"，整体平移场景下方向向量同方向偏移
    for k in range(6):
        seq[:, RH_OFF + 27 + k*3 : RH_OFF + 27 + k*3+3] += delta
    for k in range(6):
        seq[:, LH_OFF + 27 + k*3 : LH_OFF + 27 + k*3+3] += delta

    # R3/L3 pose 向量（5×3=15维）
    for k in range(5):
        seq[:, RH_OFF + 45 + k*3 : RH_OFF + 45 + k*3+3] += delta
    for k in range(5):
        seq[:, LH_OFF + 45 + k*3 : LH_OFF + 45 + k*3+3] += delta

    return seq, sp, ep


def aug_spatial_zoom(seq: np.ndarray, sp: np.ndarray, ep: np.ndarray,
                     rng: np.random.Generator, cfg: dict):
    """内部转发到 aug_distance_simulate。"""
    return aug_distance_simulate(seq, sp, ep, rng, cfg)


def aug_distance_simulate(seq: np.ndarray, sp: np.ndarray, ep: np.ndarray,
                          rng: np.random.Generator, cfg: dict):
    """
    距离模拟缩放（核心！）。
    scale < 1 → 人站得更远；scale > 1 → 人站得更近。
    默认范围 [0.5, 2.2] 覆盖 0.45m~2m 使用距离。

    新布局无绝对坐标，缩放逻辑：
      - R1/L1 中心向量（手形大小相关）：同比缩放
      - R2/L2 手部4指尖向量（手形）：同比缩放
      - R2/L2 pose向量（位置）：同比缩放
      - R3/L3 pose向量（位置）：同比缩放
      - 速度通道（差分量）：同比缩放
      - 弯曲角度（角度量，与距离无关）：不缩放
      - Pose（手型+角度）：手型部分缩放，角度部分不缩放
    """
    scale = rng.uniform(cfg["min_scale"], cfg["max_scale"])
    seq   = seq.copy()

    # 所有"坐标/向量"类通道同比缩放（跳过弯曲角度）
    for sl in [RH_R1_VECS, RH_R2_HTIPS, RH_R2_POSE, RH_R3_POSE,
               RH_TIP_VEL, RH_PALM_VEL, RH_WRIST_VEL,
               LH_L1_VECS, LH_L2_HTIPS, LH_L2_POSE, LH_L3_POSE,
               LH_TIP_VEL, LH_PALM_VEL, LH_WRIST_VEL]:
        seq[:, sl] *= scale

    # Pose 缩放：
    # 右手pose [0:20] = R2→4指尖(12) + R1→掌心(3) + 弯曲角度(5)
    # 坐标部分 [0:15] 缩放，角度部分 [15:20] 不动
    sp = sp.copy(); ep = ep.copy()
    sp[0:15]  *= scale;  sp[20:35] *= scale   # 右手前15 + 左手前15
    ep[0:15]  *= scale;  ep[20:35] *= scale

    return seq, sp, ep


def aug_frame_drop(seq: np.ndarray, sp: np.ndarray, ep: np.ndarray,
                   rng: np.random.Generator, cfg: dict):
    """随机丢帧后插值重建，模拟追踪抖动。"""
    T      = len(seq)
    drop_n = max(1, int(T * rng.uniform(0, cfg["drop_ratio"])))
    mask   = np.ones(T, dtype=bool)
    mask[rng.choice(T, drop_n, replace=False)] = False
    kept = seq[mask]
    if len(kept) < 2:
        return seq, sp, ep
    seq = _align_sequence(kept, T)
    return seq, sp, ep


def aug_random_erase(seq: np.ndarray, sp: np.ndarray, ep: np.ndarray,
                     rng: np.random.Generator, cfg: dict):
    """随机连续帧置零，模拟短暂遮挡。"""
    T       = len(seq)
    erase_n = int(rng.integers(1, cfg["max_len"] + 1))
    start   = int(rng.integers(0, max(1, T - erase_n)))
    seq     = seq.copy()
    seq[start: start + erase_n] = 0.0
    return seq, sp, ep


def aug_occlude_left_hand(seq: np.ndarray, sp: np.ndarray, ep: np.ndarray,
                          rng: np.random.Generator, cfg: dict):
    """
    【核心增广】左手全通道清零。
    让模型学会"只看右手也能识别"，增强对左手缺失的鲁棒性。
    """
    seq = seq.copy()
    for sl in LH_ALL_SLICES:
        seq[:, sl] = 0.0
    sp = sp.copy(); sp[POSE_LH] = 0.0
    ep = ep.copy(); ep[POSE_LH] = 0.0
    return seq, sp, ep


def aug_occlude_right_hand(seq: np.ndarray, sp: np.ndarray, ep: np.ndarray,
                           rng: np.random.Generator, cfg: dict):
    """右手全通道清零（低概率辅助），防止完全忽略左手。"""
    if not _lh_is_active(seq):
        return seq, sp, ep
    seq = seq.copy()
    for sl in RH_ALL_SLICES:
        seq[:, sl] = 0.0
    sp = sp.copy(); sp[POSE_RH] = 0.0
    ep = ep.copy(); ep[POSE_RH] = 0.0
    return seq, sp, ep


def aug_mirror(seq: np.ndarray, sp: np.ndarray, ep: np.ndarray,
               rng: np.random.Generator, cfg: dict):
    """
    左右手镜像互换。

    新布局下的镜像操作：
      1. 整体右手86维 ↔ 左手86维 互换
      2. x 坐标分量取反（所有 xyz 向量的 x 方向）
      3. Pose 右手20维 ↔ 左手20维 互换，x 方向取反

    注意：
      - R2/L2 的 pose 向量中含对侧腕/掌心/肘，互换后语义自动对应
      - 弯曲角度不含方向信息，x取反不影响，交换即可
      - 速度通道 x 方向同样取反

    仅当手语语义左右对称时开启（prob > 0）。
    """
    seq = seq.copy()
    sp  = sp.copy()
    ep  = ep.copy()

    # ── 1. 整块互换右手[0:86] / 左手[86:172] ─────────────────────
    rh_chunk = seq[:, 0:86].copy()
    lh_chunk = seq[:, 86:172].copy()
    seq[:, 0:86]   = lh_chunk
    seq[:, 86:172] = rh_chunk

    # ── 2. x 坐标取反（所有 xyz 组的第0个分量）──────────────────────
    # 每个 xyz 三元组起始 offset 在单手86维内的位置：
    # R1中心5个向量：offset 0,3,6,9,12 → 双手 +0/+86
    # R2手部4个向量：offset 15,18,21,24
    # R2 pose 6个向量：offset 27,30,33,36,39,42
    # R3 pose 5个向量：offset 45,48,51,54,57
    # 速度 5+1+1=7 个向量：offset 65,68,71,74,77,80,83
    # 弯曲角度 [60:65] 是标量，不取反
    _xyz_offsets_in_hand = (
        list(range(0,  15, 3)) +    # R1中心5向量
        list(range(15, 27, 3)) +    # R2手部4向量
        list(range(27, 45, 3)) +    # R2 pose 6向量
        list(range(45, 60, 3)) +    # R3 pose 5向量
        list(range(65, 80, 3)) +    # 五指尖速度5向量
        [80, 83]                    # 掌心速度、腕点速度
    )

    for base_off in [0, 86]:          # 右手和左手各自的起始偏移
        for rel_off in _xyz_offsets_in_hand:
            seq[:, base_off + rel_off] *= -1   # x 分量取反

    # ── 3. Pose 互换 + x 取反 ──────────────────────────────────────
    # Pose 结构：右手[0:20] = R2→4指尖(12) + R1→掌心(3) + 角度(5)
    #            左手[20:40] = 同上
    # 坐标部分 [0:15] 和 [20:35] 的 x 方向取反
    rh_p = sp[POSE_RH].copy(); lh_p = sp[POSE_LH].copy()
    rh_p[:15:3] *= -1;  lh_p[:15:3] *= -1   # x 分量取反（步长3取第0）
    sp[POSE_RH] = lh_p; sp[POSE_LH] = rh_p

    rh_p = ep[POSE_RH].copy(); lh_p = ep[POSE_LH].copy()
    rh_p[:15:3] *= -1;  lh_p[:15:3] *= -1
    ep[POSE_RH] = lh_p; ep[POSE_LH] = rh_p

    return seq, sp, ep


# ─────────────────────────────────────────────────────────────────
#  主类：SignAugmentor
# ─────────────────────────────────────────────────────────────────

_AUG_REGISTRY = {
    "noise":               aug_noise,
    "time_warp":           aug_time_warp,
    "time_scale":          aug_time_scale,
    "spatial_shift":       aug_spatial_shift,
    "spatial_zoom":        aug_spatial_zoom,
    "distance_simulate":   aug_distance_simulate,
    "frame_drop":          aug_frame_drop,
    "random_erase":        aug_random_erase,
    "occlude_left_hand":   aug_occlude_left_hand,
    "occlude_right_hand":  aug_occlude_right_hand,
    "mirror":              aug_mirror,
}

# 执行顺序（几何类先，噪声类后，遮挡类放最后）
_AUG_ORDER = [
    "time_scale",
    "time_warp",
    "frame_drop",
    "spatial_shift",
    "spatial_zoom",
    "random_erase",
    "mirror",
    "occlude_left_hand",
    "occlude_right_hand",
    "noise",
]


class SignAugmentor:
    """
    手语序列增强器。

    参数
    ----
    cfg  : dict, optional  增广配置（只写需修改项，其余用默认值）
    seed : int, optional   随机种子

    用法
    ----
    aug = SignAugmentor()
    seq_aug, sp_aug, ep_aug = aug(seq, sp, ep)
    # seq: (T, 172)  sp/ep: (40,)
    """

    def __init__(self,
                 cfg: Optional[Dict[str, Any]] = None,
                 seed: Optional[int] = None):
        self.cfg = {}
        for key, default in DEFAULT_CFG.items():
            user = (cfg or {}).get(key, {})
            self.cfg[key] = {**default, **user}
        self.rng = np.random.default_rng(seed)

    def __call__(self,
                 seq: np.ndarray,
                 sp:  np.ndarray,
                 ep:  np.ndarray) -> tuple:
        """
        seq : (T, 172)  float32
        sp  : (40,)     float32
        ep  : (40,)     float32
        """
        seq = seq.astype(np.float32)
        sp  = sp.astype(np.float32)
        ep  = ep.astype(np.float32)

        for name in _AUG_ORDER:
            prob = self.cfg[name]["prob"]
            if prob <= 0.0:
                continue
            if self.rng.random() < prob:
                seq, sp, ep = _AUG_REGISTRY[name](seq, sp, ep, self.rng, self.cfg[name])

        return seq, sp, ep

    def offline_augment(self,
                        seqs:   np.ndarray,
                        starts: np.ndarray,
                        ends:   np.ndarray,
                        labels: np.ndarray,
                        n_aug:  int = 3) -> tuple:
        """
        离线批量增广，每条样本生成 n_aug 条新样本。

        seqs   : (N, T, 172)
        starts : (N, 40)
        ends   : (N, 40)
        labels : (N,)
        """
        aug_seqs, aug_sps, aug_eps, aug_lbls = [], [], [], []
        for i in range(len(seqs)):
            for _ in range(n_aug):
                s, sp, ep = self(seqs[i], starts[i], ends[i])
                aug_seqs.append(s)
                aug_sps.append(sp)
                aug_eps.append(ep)
                aug_lbls.append(labels[i])
        return (
            np.array(aug_seqs,  dtype=np.float32),
            np.array(aug_sps,   dtype=np.float32),
            np.array(aug_eps,   dtype=np.float32),
            np.array(aug_lbls,  dtype=np.int64),
        )


# ─────────────────────────────────────────────────────────────────
#  快速自测
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  augment.py v2 自测（172维特征 / 40维Pose）")
    print("=" * 55)

    rng_test = np.random.default_rng(0)
    T = 20

    seq_ori = rng_test.standard_normal((T, FEAT_DIM)).astype(np.float32) * 0.5
    sp_ori  = rng_test.standard_normal(POSE_DIM).astype(np.float32) * 0.3
    ep_ori  = rng_test.standard_normal(POSE_DIM).astype(np.float32) * 0.3

    aug = SignAugmentor(seed=42)

    # 在线增广
    seq_a, sp_a, ep_a = aug(seq_ori.copy(), sp_ori.copy(), ep_ori.copy())
    print(f"\n[在线增广]")
    print(f"  seq : {seq_ori.shape} → {seq_a.shape}")
    print(f"  sp  : {sp_ori.shape} → {sp_a.shape}")
    print(f"  diff(seq max): {np.abs(seq_a - seq_ori).max():.4f}")
    assert seq_a.shape == (T, FEAT_DIM),  f"seq维度错误: {seq_a.shape}"
    assert sp_a.shape  == (POSE_DIM,),    f"sp维度错误:  {sp_a.shape}"

    # 左手遮挡验证
    cfg_occ = {k: dict(v) for k, v in DEFAULT_CFG.items()}
    for k in cfg_occ:
        cfg_occ[k] = dict(cfg_occ[k])
        cfg_occ[k]["prob"] = 0.0
    cfg_occ["occlude_left_hand"]["prob"] = 1.0

    aug_occ = SignAugmentor(cfg=cfg_occ, seed=0)
    seq_occ, sp_occ, ep_occ = aug_occ(seq_ori.copy(), sp_ori.copy(), ep_ori.copy())

    lh_zero    = all(np.all(seq_occ[:, sl] == 0) for sl in LH_ALL_SLICES)
    pose_lh_z  = np.all(sp_occ[POSE_LH] == 0)
    print(f"\n[左手遮挡验证]")
    print(f"  左手所有通道全零: {lh_zero}")
    print(f"  Pose左手全零:     {pose_lh_z}")
    assert lh_zero and pose_lh_z, "❌ 左手遮挡存在漏网通道！"
    print("  ✅ 所有左手通道正确清零")

    # 镜像验证：互换后形状不变
    cfg_mir = {k: dict(v) for k, v in DEFAULT_CFG.items()}
    for k in cfg_mir:
        cfg_mir[k] = dict(cfg_mir[k])
        cfg_mir[k]["prob"] = 0.0
    cfg_mir["mirror"]["prob"] = 1.0
    aug_mir = SignAugmentor(cfg=cfg_mir, seed=0)
    seq_m, sp_m, ep_m = aug_mir(seq_ori.copy(), sp_ori.copy(), ep_ori.copy())
    assert seq_m.shape == (T, FEAT_DIM), f"镜像后seq维度错误: {seq_m.shape}"
    assert sp_m.shape  == (POSE_DIM,),   f"镜像后sp维度错误:  {sp_m.shape}"
    print(f"\n[镜像验证]  seq形状: {seq_m.shape}  ✅")

    # 离线批量增广
    N = 20
    seqs_b  = rng_test.standard_normal((N, T, FEAT_DIM)).astype(np.float32)
    starts_b = rng_test.standard_normal((N, POSE_DIM)).astype(np.float32)
    ends_b   = rng_test.standard_normal((N, POSE_DIM)).astype(np.float32)
    labels_b = rng_test.integers(0, 5, N).astype(np.int64)
    aug_s, aug_sp, aug_ep, aug_l = aug.offline_augment(
        seqs_b, starts_b, ends_b, labels_b, n_aug=3
    )
    print(f"\n[离线增广]  原始: {seqs_b.shape}  →  增广: {aug_s.shape}")
    assert aug_s.shape == (N * 3, T, FEAT_DIM), f"离线增广维度错误: {aug_s.shape}"
    print(f"  ✅ 离线增广输出维度正确")

    print("\n✅ 所有测试通过！\n")