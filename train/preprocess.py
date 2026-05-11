import numpy as np
import argparse

# ── 常量 ──────────────────────────────────────────────────────────
N_FRAMES    = 20
POSE_FRAMES = 3          # 起/止手型取前/后几帧均值

# 手部关键点 id（MediaPipe 21点）
ID_WRIST   = 0           # R3/L3  腕点（hand wrist）
ID_THUMB   = 4           # R1/L1  拇指尖
ID_INDEX   = 8           # 食指尖
ID_MIDDLE  = 12          # 中指尖
ID_RING    = 16          # 无名指尖
ID_PINKY   = 20          # 小指尖
# 掌心用 wrist(0) 和 中指根(9) 的中点

# 五指尖
FIVE_TIPS = [ID_THUMB, ID_INDEX, ID_MIDDLE, ID_RING, ID_PINKY]  # R1/L1 也在里面
# 四指尖（不含拇指），用于 R2/L2 中心的手部向量
FOUR_TIPS = [ID_INDEX, ID_MIDDLE, ID_RING, ID_PINKY]

# 弯曲角度所需关节三元组（根→中→尖），每根手指取一个夹角
# 拇指：1-2-4；食指：5-6-8；中指：9-10-12；无名：13-14-16；小指：17-18-20
FINGER_JOINTS = [
    (1, 2, 4),    # 拇指
    (5, 6, 8),    # 食指
    (9, 10, 12),  # 中指
    (13, 14, 16), # 无名指
    (17, 18, 20), # 小指
]

# ── Pose 165维里的索引 ────────────────────────────────────────────
IDX_NOSE       = 0
IDX_L_SHOULDER = 1
IDX_R_SHOULDER = 2
IDX_L_ELBOW    = 3
IDX_R_ELBOW    = 4
IDX_L_WRIST    = 5
IDX_R_WRIST    = 6

POSE_END = 39
RH_START = 39;  RH_END = 102
LH_START = 102; LH_END = 165


# ═══════════════════════════════════════════════════════════════════
#  工具函数
# ═══════════════════════════════════════════════════════════════════

def palm_center(lm_21x3):
    """掌心 R2/L2 = (wrist + 中指根) / 2"""
    return (lm_21x3[0] + lm_21x3[9]) * 0.5


def finger_bend_angles(lm_21x3):
    """
    计算五根手指弯曲角度（弧度），返回 (5,)。
    用关节三元组 (A, B, C) 的 BA·BC 夹角。
    手不存在（全零）时返回全零。
    """
    if np.all(lm_21x3 == 0):
        return np.zeros(5, dtype=np.float32)

    angles = np.zeros(5, dtype=np.float32)
    for k, (a, b, c) in enumerate(FINGER_JOINTS):
        ba = lm_21x3[a] - lm_21x3[b]
        bc = lm_21x3[c] - lm_21x3[b]
        n_ba = np.linalg.norm(ba)
        n_bc = np.linalg.norm(bc)
        if n_ba < 1e-6 or n_bc < 1e-6:
            angles[k] = 0.0
        else:
            cos_a = np.clip(np.dot(ba, bc) / (n_ba * n_bc), -1.0, 1.0)
            angles[k] = np.arccos(cos_a)
    return angles


def safe_vec(src, dst):
    """向量 src→dst，若任一为全零则返回零向量 (3,)"""
    if np.all(src == 0) or np.all(dst == 0):
        return np.zeros(3, dtype=np.float32)
    return (dst - src).astype(np.float32)


def shoulder_center(l_sho, r_sho):
    """肩中心"""
    return (l_sho + r_sho) * 0.5


# ═══════════════════════════════════════════════════════════════════
#  Step 1：解析单帧 165 维
# ═══════════════════════════════════════════════════════════════════

def parse_frame(frame_165):
    """
    返回：
      nose         (3,)
      l_shoulder   (3,)
      r_shoulder   (3,)
      l_elbow      (3,)
      r_elbow      (3,)
      pose_l_wrist (3,)   pose 里的左腕（用于鲁棒左右手判定）
      pose_r_wrist (3,)
      rh_raw       (21,3) npz 里的"右手槽位"
      lh_raw       (21,3) npz 里的"左手槽位"
    """
    pose   = frame_165[0:POSE_END].reshape(13, 3)
    rh_raw = frame_165[RH_START:RH_END].reshape(21, 3)
    lh_raw = frame_165[LH_START:LH_END].reshape(21, 3)

    return (
        pose[IDX_NOSE],
        pose[IDX_L_SHOULDER],
        pose[IDX_R_SHOULDER],
        pose[IDX_L_ELBOW],
        pose[IDX_R_ELBOW],
        pose[IDX_L_WRIST],
        pose[IDX_R_WRIST],
        rh_raw,
        lh_raw,
    )


# ═══════════════════════════════════════════════════════════════════
#  Step 2：鲁棒左右手判定
# ═══════════════════════════════════════════════════════════════════

def robust_assign_hands(pose_l_wrist, pose_r_wrist, rh_raw, lh_raw):
    zero21 = np.zeros((21, 3), dtype=np.float32)

    rh_blank = np.all(rh_raw == 0)
    lh_blank = np.all(lh_raw == 0)

    if rh_blank and lh_blank:
        return zero21.copy(), zero21.copy()

    pw_r = pose_r_wrist[:2]
    pw_l = pose_l_wrist[:2]

    if not rh_blank and not lh_blank:
        d_rh_to_r = np.linalg.norm(rh_raw[0][:2] - pw_r)
        d_lh_to_r = np.linalg.norm(lh_raw[0][:2] - pw_r)
        if d_rh_to_r <= d_lh_to_r:
            return rh_raw.copy(), lh_raw.copy()
        else:
            return lh_raw.copy(), rh_raw.copy()

    hand = rh_raw if not rh_blank else lh_raw
    d_to_r = np.linalg.norm(hand[0][:2] - pw_r)
    d_to_l = np.linalg.norm(hand[0][:2] - pw_l)
    if d_to_r <= d_to_l:
        return hand.copy(), zero21.copy()
    else:
        return zero21.copy(), hand.copy()


# ═══════════════════════════════════════════════════════════════════
#  Step 3：去除空白帧
# ═══════════════════════════════════════════════════════════════════

def remove_blank_frames(rh_seq, lh_seq, nose_seq,
                        l_sho_seq, r_sho_seq,
                        l_elbow_seq, r_elbow_seq,
                        pose_lw_seq, pose_rw_seq):
    T = len(rh_seq)
    rh_flat = rh_seq.reshape(T, -1)
    lh_flat = lh_seq.reshape(T, -1)
    blank = np.all(rh_flat == 0, axis=1) & np.all(lh_flat == 0, axis=1)
    v = ~blank
    return (rh_seq[v], lh_seq[v], nose_seq[v],
            l_sho_seq[v], r_sho_seq[v],
            l_elbow_seq[v], r_elbow_seq[v],
            pose_lw_seq[v], pose_rw_seq[v])


# ═══════════════════════════════════════════════════════════════════
#  Step 4：对齐到固定帧数
# ═══════════════════════════════════════════════════════════════════

def align_to_n_frames(arr, n_frames=N_FRAMES):
    orig_shape = arr.shape
    T = orig_shape[0]
    if T == n_frames:
        return arr.astype(np.float32)
    flat = arr.reshape(T, -1).astype(np.float32)
    t_orig   = np.linspace(0, 1, T)
    t_target = np.linspace(0, 1, n_frames)
    result = np.stack(
        [np.interp(t_target, t_orig, flat[:, d]) for d in range(flat.shape[1])],
        axis=1
    )
    return result.reshape((n_frames,) + orig_shape[1:]).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════
#  Step 5：提取单手 86 维特征（单帧）
# ═══════════════════════════════════════════════════════════════════

def hand_features_86(
        lm,           # (21,3)  本帧当前手
        lm_prev,      # (21,3)  上一帧当前手（None 或全零则速度为0）
        nose,         # (3,)
        opp_pose_wrist,  # (3,)  对侧 pose 腕点
        opp_palm,        # (3,)  对侧掌心（用于 R2中心的 pose 向量）
        opp_shoulder,    # (3,)  对侧肩膀
        same_shoulder,   # (3,)  同侧肩膀
        opp_elbow,       # (3,)  对侧肘点
        l_sho,           # (3,)  左肩（用于算肩中心）
        r_sho,           # (3,)  右肩
):
    """
    返回 (86,) 特征向量。手不存在（全零）时返回全零。
    """
    out = np.zeros(86, dtype=np.float32)
    sho_center = shoulder_center(l_sho, r_sho)

    hand_blank = np.all(lm == 0)

    # ── R1/L1（拇指尖）中心，5条边向量 ───────────────────────────
    # → 食、中、无名、小指尖、掌心 (5×3=15)
    if not hand_blank:
        r1 = lm[ID_THUMB]
        palm = palm_center(lm)
        targets_r1 = [lm[ID_INDEX], lm[ID_MIDDLE], lm[ID_RING], lm[ID_PINKY], palm]
        for k, tgt in enumerate(targets_r1):
            out[k*3 : k*3+3] = safe_vec(r1, tgt)
    # [0:15]

    # ── R2/L2（掌心）中心 ─────────────────────────────────────────
    if not hand_blank:
        palm = palm_center(lm)
        # 4条手部向量 → 食/中/无名/小指尖 (4×3=12)  [15:27]
        for k, tip_id in enumerate(FOUR_TIPS):
            out[15 + k*3 : 15 + k*3+3] = safe_vec(palm, lm[tip_id])

        # 6条 pose 向量 [27:45]
        # 右手：鼻、左腕(opp_pose_wrist)、左掌心(opp_palm)、左肩(opp_shoulder)、右肩(same_shoulder)、左肘(opp_elbow)
        pose_targets_r2 = [nose, opp_pose_wrist, opp_palm,
                           opp_shoulder, same_shoulder, opp_elbow]
        for k, tgt in enumerate(pose_targets_r2):
            out[27 + k*3 : 27 + k*3+3] = safe_vec(palm, tgt)
    # [15:45]

    # ── R3/L3（腕点=hand wrist landmark[0]）中心 ─────────────────
    # 5条 pose 向量 → 鼻、对侧腕、对侧肘、同侧肩、对侧肩 [45:60]
    if not hand_blank:
        r3 = lm[ID_WRIST]
        pose_targets_r3 = [nose, opp_pose_wrist, opp_elbow,
                           same_shoulder, opp_shoulder]
        for k, tgt in enumerate(pose_targets_r3):
            out[45 + k*3 : 45 + k*3+3] = safe_vec(r3, tgt)

    # ── 五指弯曲角度 [60:65] ──────────────────────────────────────
    out[60:65] = finger_bend_angles(lm)

    # ── 轨迹：五指尖相对掌心速度 Δ(tip-palm) [65:80] ──────────────
    if not hand_blank:
        palm = palm_center(lm)
        prev_blank = (lm_prev is None) or np.all(lm_prev == 0)
        if not prev_blank:
            prev_palm = palm_center(lm_prev)
            for k, tip_id in enumerate(FIVE_TIPS):
                cur_rel  = lm[tip_id]      - palm
                prev_rel = lm_prev[tip_id] - prev_palm
                out[65 + k*3 : 65 + k*3+3] = (cur_rel - prev_rel).astype(np.float32)
        # else 保持0

    # ── 轨迹：掌心相对肩中心速度 [80:83] ──────────────────────────
    if not hand_blank:
        prev_blank = (lm_prev is None) or np.all(lm_prev == 0)
        if not prev_blank:
            palm      = palm_center(lm)
            prev_palm = palm_center(lm_prev)
            cur_rel  = palm      - sho_center
            prev_rel = prev_palm - sho_center   # 肩中心在对齐后近似不变，用当前帧
            out[80:83] = (cur_rel - prev_rel).astype(np.float32)

    # ── 轨迹：腕点相对肩中心速度 [83:86] ──────────────────────────
    if not hand_blank:
        prev_blank = (lm_prev is None) or np.all(lm_prev == 0)
        if not prev_blank:
            r3      = lm[ID_WRIST]
            prev_r3 = lm_prev[ID_WRIST]
            cur_rel  = r3      - sho_center
            prev_rel = prev_r3 - sho_center
            out[83:86] = (cur_rel - prev_rel).astype(np.float32)

    return out   # (86,)


# ═══════════════════════════════════════════════════════════════════
#  Step 6：提取 172 维特征序列
# ═══════════════════════════════════════════════════════════════════

def extract_features_172(rh_al, lh_al,
                         nose_al, l_sho_al, r_sho_al,
                         l_elbow_al, r_elbow_al,
                         pose_lw_al, pose_rw_al):
    """
    输入均为对齐后的 (N_FRAMES, ...) 数组。
    返回 (N_FRAMES, 172) 特征矩阵。

    右手 86 维 [0:86]：
      以 R2(掌心) 为中心的 pose 向量用 → 鼻、左腕、左掌心、左肩、右肩、左肘
    左手 86 维 [86:172]：
      以 L2(掌心) 为中心的 pose 向量用 → 鼻、右腕、右掌心、右肩、左肩、右肘
    """
    T = N_FRAMES
    feats = np.zeros((T, 172), dtype=np.float32)

    for t in range(T):
        rh = rh_al[t]        # (21,3)
        lh = lh_al[t]
        prev_rh = rh_al[t-1] if t > 0 else None
        prev_lh = lh_al[t-1] if t > 0 else None

        nose  = nose_al[t]
        l_sho = l_sho_al[t]
        r_sho = r_sho_al[t]
        l_elbow = l_elbow_al[t]
        r_elbow = r_elbow_al[t]
        pose_lw = pose_lw_al[t]   # pose 左腕
        pose_rw = pose_rw_al[t]   # pose 右腕

        l_palm = palm_center(lh)
        r_palm = palm_center(rh)

        # 右手：对侧 = 左侧
        feats[t, 0:86] = hand_features_86(
            lm=rh, lm_prev=prev_rh,
            nose=nose,
            opp_pose_wrist=pose_lw,    # 左腕
            opp_palm=l_palm,           # 左掌心
            opp_shoulder=l_sho,        # 左肩
            same_shoulder=r_sho,       # 右肩
            opp_elbow=l_elbow,         # 左肘
            l_sho=l_sho, r_sho=r_sho,
        )

        # 左手：对侧 = 右侧
        feats[t, 86:172] = hand_features_86(
            lm=lh, lm_prev=prev_lh,
            nose=nose,
            opp_pose_wrist=pose_rw,    # 右腕
            opp_palm=r_palm,           # 右掌心
            opp_shoulder=r_sho,        # 右肩
            same_shoulder=l_sho,       # 左肩
            opp_elbow=r_elbow,         # 右肘
            l_sho=l_sho, r_sho=r_sho,
        )

    return feats   # (30, 172)


# ═══════════════════════════════════════════════════════════════════
#  Step 7：起止 Pose（手型描述向量）
# ═══════════════════════════════════════════════════════════════════

def get_pose_v2(feat_172, n=POSE_FRAMES):
    """
    从 (30, 172) 特征里提取起止手型向量。

    右手手型片段（20维）：
      R2中心4指尖向量 [15:27] = 12维
      R1中心掌心向量  [12:15] =  3维（R1→palm，也就是 targets_r1[4]）
      弯曲角度        [60:65] =  5维
      共 20 维

    左手手型片段（20维，偏移86）：
      同上结构

    start_pose / end_pose 各 40 维
    """
    def hand_pose_slice(offset):
        # R2中心4指尖向量
        s1 = slice(offset + 15, offset + 27)   # 12维
        # R1中心掌心向量（targets_r1[4]，排在 4×3=12 之后，即 [12:15]）
        s2 = slice(offset + 12, offset + 15)   #  3维
        # 弯曲角度
        s3 = slice(offset + 60, offset + 65)   #  5维
        return s1, s2, s3

    rh_s1, rh_s2, rh_s3 = hand_pose_slice(0)
    lh_s1, lh_s2, lh_s3 = hand_pose_slice(86)

    def mean_concat(f, slices):
        return np.concatenate([f[:, s].mean(axis=0) for s in slices])

    sp = np.concatenate([
        mean_concat(feat_172[:n],  [rh_s1, rh_s2, rh_s3]),
        mean_concat(feat_172[:n],  [lh_s1, lh_s2, lh_s3]),
    ])   # (40,)

    ep = np.concatenate([
        mean_concat(feat_172[-n:], [rh_s1, rh_s2, rh_s3]),
        mean_concat(feat_172[-n:], [lh_s1, lh_s2, lh_s3]),
    ])   # (40,)

    return sp, ep


# ═══════════════════════════════════════════════════════════════════
#  主处理流程
# ═══════════════════════════════════════════════════════════════════

def batch_process(input_npz, output_npz, n_frames=N_FRAMES):
    print(f"[加载] {input_npz}")
    data     = np.load(input_npz, allow_pickle=True)
    raw_seqs = data['sequences']
    labels   = data['labels']
    lens     = data['seq_lengths'] if 'seq_lengths' in data.keys() \
               else np.array([len(s) for s in raw_seqs])

    new_seqs, new_sp, new_ep, new_labels = [], [], [], []
    cnt_ok, cnt_skip, cnt_interp, cnt_sample = 0, 0, 0, 0

    print(f"[处理] 共 {len(raw_seqs)} 条样本  目标帧数={n_frames}")

    for i in range(len(raw_seqs)):
        T        = int(lens[i])
        raw_clip = np.array(raw_seqs[i], dtype=np.float32)[:T]   # (T, 165)

        # ── 逐帧解析 ──────────────────────────────────────────────
        rh_list, lh_list        = [], []
        nose_list               = []
        l_sho_list, r_sho_list  = [], []
        l_elbow_list, r_elbow_list = [], []
        pose_lw_list, pose_rw_list = [], []

        for frame in raw_clip:
            (nose, l_sho, r_sho, l_elbow, r_elbow,
             pw_l, pw_r, rh_raw, lh_raw) = parse_frame(frame)

            true_rh, true_lh = robust_assign_hands(pw_l, pw_r, rh_raw, lh_raw)

            rh_list.append(true_rh)
            lh_list.append(true_lh)
            nose_list.append(nose)
            l_sho_list.append(l_sho)
            r_sho_list.append(r_sho)
            l_elbow_list.append(l_elbow)
            r_elbow_list.append(r_elbow)
            pose_lw_list.append(pw_l)
            pose_rw_list.append(pw_r)

        rh_seq      = np.array(rh_list,      dtype=np.float32)   # (T,21,3)
        lh_seq      = np.array(lh_list,      dtype=np.float32)
        nose_seq    = np.array(nose_list,    dtype=np.float32)   # (T,3)
        l_sho_seq   = np.array(l_sho_list,   dtype=np.float32)
        r_sho_seq   = np.array(r_sho_list,   dtype=np.float32)
        l_elbow_seq = np.array(l_elbow_list, dtype=np.float32)
        r_elbow_seq = np.array(r_elbow_list, dtype=np.float32)
        pose_lw_seq = np.array(pose_lw_list, dtype=np.float32)
        pose_rw_seq = np.array(pose_rw_list, dtype=np.float32)

        # ── 去除空白帧 ────────────────────────────────────────────
        (rh_seq, lh_seq, nose_seq,
         l_sho_seq, r_sho_seq,
         l_elbow_seq, r_elbow_seq,
         pose_lw_seq, pose_rw_seq) = remove_blank_frames(
            rh_seq, lh_seq, nose_seq,
            l_sho_seq, r_sho_seq,
            l_elbow_seq, r_elbow_seq,
            pose_lw_seq, pose_rw_seq,
        )
        T_clean = len(rh_seq)

        if T_clean < 3:
            cnt_skip += 1
            continue

        # ── 对齐到固定帧数 ────────────────────────────────────────
        if T_clean >= n_frames:
            cnt_sample += 1
        else:
            cnt_interp += 1

        rh_al      = align_to_n_frames(rh_seq,      n_frames)   # (30,21,3)
        lh_al      = align_to_n_frames(lh_seq,      n_frames)
        nose_al    = align_to_n_frames(nose_seq,    n_frames)   # (30,3)
        l_sho_al   = align_to_n_frames(l_sho_seq,   n_frames)
        r_sho_al   = align_to_n_frames(r_sho_seq,   n_frames)
        l_elbow_al = align_to_n_frames(l_elbow_seq, n_frames)
        r_elbow_al = align_to_n_frames(r_elbow_seq, n_frames)
        pose_lw_al = align_to_n_frames(pose_lw_seq, n_frames)
        pose_rw_al = align_to_n_frames(pose_rw_seq, n_frames)

        # ── 提取 172 维特征 ───────────────────────────────────────
        feat = extract_features_172(
            rh_al, lh_al, nose_al,
            l_sho_al, r_sho_al,
            l_elbow_al, r_elbow_al,
            pose_lw_al, pose_rw_al,
        )   # (30, 172)

        # ── 起止手型 Pose ─────────────────────────────────────────
        sp, ep = get_pose_v2(feat)   # (40,), (40,)

        new_seqs.append(feat)
        new_sp.append(sp)
        new_ep.append(ep)
        new_labels.append(labels[i])
        cnt_ok += 1

        if (cnt_ok + cnt_skip) % 500 == 0:
            print(f"  进度: {cnt_ok + cnt_skip}/{len(raw_seqs)}")

    print(f"\n✅ 处理完成！")
    print(f"  成功样本:     {cnt_ok}")
    print(f"  跳过(太短):   {cnt_skip}")
    print(f"  均匀采样:     {cnt_sample}")
    print(f"  插值补帧:     {cnt_interp}")
    print(f"  特征维度:     (30, 172)  pose维度: (40,)")

    np.savez_compressed(
        output_npz,
        sequences   = np.array(new_seqs,   dtype=object),
        start_poses = np.array(new_sp,     dtype=object),
        end_poses   = np.array(new_ep,     dtype=object),
        labels      = np.array(new_labels),
    )
    print(f"  已保存至: {output_npz}")


# ═══════════════════════════════════════════════════════════════════
#  入口
# ═══════════════════════════════════════════════════════════════════

def get_args():
    parser = argparse.ArgumentParser(
        description='双手关键点预处理 v2：去空白→对齐→提取172维特征'
    )
    parser.add_argument('--input_npz',  default='dataset_keypoints_101_valid.npz',
                        help='输入 npz（165维关键点，由 extract_keypoints.py 生成）')
    parser.add_argument('--output_npz', default='dataset_aligned.npz',
                        help='输出 npz（30帧×172维特征）')
    parser.add_argument('--n_frames',   type=int, default=20,
                        help='目标帧数（默认30）')
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    batch_process(args.input_npz, args.output_npz, args.n_frames)