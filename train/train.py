import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json, os, random
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from model import Model_CNNGRU
from augment import SignAugmentor

augmentor = SignAugmentor()

# ═══════════════════════════════════════════════════════════════
#  配置区  ← 只需改这里
# ═══════════════════════════════════════════════════════════════
DATASET_PATH = "dataset_aligned.npz"
SAVE_PATH    = "dynamic_model.pth"
CLASS_JSON   = "class_names.json"

SEQ_LEN  = 20

# ── 特征维度（与 preprocess v2 保持一致） ─────────────────────
FEAT_DIM = 172   # (30, 172) 全部送入模型
POSE_DIM = 40    # 右手20 + 左手20

EPOCHS     = 200
BATCH_SIZE = 16
LR         = 1e-3
VAL_RATIO  = 0.2
SEED       = 42

# ── 消融开关 ─────────────────────────────────────────────────
BIDIRECTIONAL = True
GRU_HIDDEN    = 128

# ── 数据增强 ─────────────────────────────────────────────────
AUGMENT = True

# ── vel_boost：放大各类通道权重，突出关键信息 ─────────────────
# 右手（[0:86]）和左手（[86:172]）同结构，对称设置
VEL_BOOST = np.ones(FEAT_DIM, dtype=np.float32)

# 手型核心：R1/L1中心向量、R2/L2手部指尖向量（手形最关键）
VEL_BOOST[0:15]    = 2.0   # 右手 R1中心向量
VEL_BOOST[15:27]   = 2.5   # 右手 R2手部指尖向量（手型核心，权重最高）
VEL_BOOST[86:101]  = 2.0   # 左手 L1中心向量
VEL_BOOST[101:113] = 2.5   # 左手 L2手部指尖向量

# 手位置（pose方向向量，手相对于身体的位置）
VEL_BOOST[27:45]   = 1.5   # 右手 R2→pose
VEL_BOOST[45:60]   = 1.5   # 右手 R3→pose
VEL_BOOST[113:131] = 1.5   # 左手 L2→pose
VEL_BOOST[131:146] = 1.5   # 左手 L3→pose

# 弯曲角度（弧度量，放大使其与坐标量级接近）
VEL_BOOST[60:65]   = 3.0   # 右手五指弯曲角度
VEL_BOOST[146:151] = 3.0   # 左手五指弯曲角度

# 轨迹速度通道（放大突出运动信息）
VEL_BOOST[65:80]   = 3.0   # 右手五指尖速度
VEL_BOOST[80:83]   = 4.0   # 右手掌心相对肩中心速度（轨迹最关键）
VEL_BOOST[83:86]   = 3.0   # 右手腕点相对肩中心速度
VEL_BOOST[151:166] = 3.0   # 左手五指尖速度
VEL_BOOST[166:169] = 4.0   # 左手掌心相对肩中心速度
VEL_BOOST[169:172] = 3.0   # 左手腕点相对肩中心速度


# ═══════════════════════════════════════════════════════════════
#  序列对齐工具
# ═══════════════════════════════════════════════════════════════

def align_sequence(arr, n_frames=SEQ_LEN):
    """(T, D) → (n_frames, D)，线性插值"""
    T, D = arr.shape
    t_orig   = np.linspace(0, 1, T)
    t_target = np.linspace(0, 1, n_frames)
    result   = np.stack([
        np.interp(t_target, t_orig, arr[:, d]) for d in range(D)
    ], axis=1)
    return result.astype(np.float32)


# ═══════════════════════════════════════════════════════════════
#  数据集
# ═══════════════════════════════════════════════════════════════

class SignDataset(Dataset):
    def __init__(self, seqs, starts, ends, labels, augment=False):
        """
        seqs   : (N, 30, FEAT_DIM=172)
        starts : (N, POSE_DIM=40)
        ends   : (N, POSE_DIM=40)
        labels : (N,)
        """
        self.seqs    = seqs
        self.starts  = starts
        self.ends    = ends
        self.labels  = labels
        self.augment = augment

    def __len__(self):
        return len(self.seqs)

    def _aug(self, seq, sp, ep):
        return augmentor(seq, sp, ep)

    def __getitem__(self, idx):
        seq = self.seqs[idx].astype(np.float32)
        sp  = self.starts[idx].astype(np.float32)
        ep  = self.ends[idx].astype(np.float32)
        if self.augment and np.random.rand() < 0.7:
            seq, sp, ep = self._aug(seq, sp, ep)
        return (
            torch.from_numpy(seq),
            torch.from_numpy(sp),
            torch.from_numpy(ep),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )


# ═══════════════════════════════════════════════════════════════
#  数据加载 + 归一化
# ═══════════════════════════════════════════════════════════════

def load_data(path):
    d = np.load(path, allow_pickle=True)

    # ── 序列 ──────────────────────────────────────────────────
    raw_seqs = [np.array(s, dtype=np.float32) for s in d['sequences']]
    seqs = np.array([
        align_sequence(s, SEQ_LEN) if s.shape[0] != SEQ_LEN else s
        for s in raw_seqs
    ])   # (N, 30, 172)

    # ── Pose（起止手型） ───────────────────────────────────────
    starts = np.array([np.array(s, dtype=np.float32) for s in d['start_poses']])
    ends   = np.array([np.array(s, dtype=np.float32) for s in d['end_poses']])

    # ── 标签 ──────────────────────────────────────────────────
    raw_labels     = d['labels']
    cleaned_labels = [str(lb).rsplit('_', 1)[0] for lb in raw_labels]
    video_ids      = [str(lb).rsplit('_', 1)[1] for lb in raw_labels]

    le      = LabelEncoder()
    labels  = le.fit_transform(cleaned_labels).astype(np.int64)
    classes = list(le.classes_)

    print(f"[数据] {len(seqs)} 条  {len(classes)} 类")
    print(f"[特征] 序列 {seqs.shape}  Pose {starts.shape}")
    print(f"[类别] {classes}")

    # ── vel_boost ─────────────────────────────────────────────
    seqs = seqs * VEL_BOOST[None, None, :]

    # ── z-score 归一化（序列） ────────────────────────────────
    seq_mean = seqs.mean(axis=(0, 1), keepdims=True)   # (1, 1, 172)
    seq_std  = seqs.std(axis=(0, 1),  keepdims=True) + 1e-8
    seqs     = (seqs - seq_mean) / seq_std

    # ── z-score 归一化（Pose） ────────────────────────────────
    # start_poses 和 end_poses 各 40 维，合并后各维度独立归一化
    p_all  = np.concatenate([starts, ends], axis=1)   # (N, 80)
    p_mean = p_all.mean(axis=0, keepdims=True)         # (1, 80)
    p_std  = p_all.std(axis=0,  keepdims=True) + 1e-8
    starts = (starts - p_mean[:, :POSE_DIM]) / p_std[:, :POSE_DIM]
    ends   = (ends   - p_mean[:, POSE_DIM:]) / p_std[:, POSE_DIM:]

    norm = {
        'seq_mean':  seq_mean.tolist(),
        'seq_std':   seq_std.tolist(),
        'vel_boost': VEL_BOOST.tolist(),
        'p_mean':    p_mean.tolist(),
        'p_std':     p_std.tolist(),
        'pose_dim':  POSE_DIM,
        'feat_dim':  FEAT_DIM,
    }

    return seqs, starts, ends, labels, classes, norm, video_ids


# ═══════════════════════════════════════════════════════════════
#  训练主流程
# ═══════════════════════════════════════════════════════════════

def train():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    seqs, starts, ends, labels, classes, norm, video_ids = load_data(DATASET_PATH)
    num_classes = len(classes)

    # ── 按人划分：随机选2个ID作为验证集，其余训练 ──────────────
    all_ids = sorted(set(video_ids))
    random.shuffle(all_ids)
    val_ids   = set(all_ids[:2])
    train_ids = set(all_ids[2:])

    print(f"[划分] 训练ID: {sorted(train_ids)}  验证ID: {sorted(val_ids)}")

    tr_i = np.array([i for i, v in enumerate(video_ids) if v in train_ids])
    va_i = np.array([i for i, v in enumerate(video_ids) if v in val_ids])
    np.random.shuffle(tr_i)
    np.random.shuffle(va_i)
    print(f"[样本数] 训练={len(tr_i)}  验证={len(va_i)}")

    # ── DataLoader ────────────────────────────────────────────
    tr_ds = SignDataset(seqs[tr_i], starts[tr_i], ends[tr_i], labels[tr_i], AUGMENT)
    va_ds = SignDataset(seqs[va_i], starts[va_i], ends[va_i], labels[va_i], False)
    tr_dl = DataLoader(tr_ds, BATCH_SIZE, shuffle=True,  drop_last=False)
    va_dl = DataLoader(va_ds, BATCH_SIZE, shuffle=False)

    # ── 模型 ──────────────────────────────────────────────────
    model = Model_CNNGRU(
        num_classes   = num_classes,
        feat_dim      = FEAT_DIM,    # 172
        pose_dim      = POSE_DIM,    # 40
        hidden_size   = GRU_HIDDEN,
        bidirectional = BIDIRECTIONAL,
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = model.to(device)

    direction = "双向" if BIDIRECTIONAL else "单向"
    print(f"[模型] CNN + {direction}GRU  "
          f"参数量: {sum(p.numel() for p in model.parameters()):,}")

    opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    crit  = nn.CrossEntropyLoss(label_smoothing=0.05)

    best_acc, best_ep = 0.0, 0
    history = {'train_loss': [], 'train_acc': [], 'val_acc': []}

    print(f"\n{'Ep':>5} {'TrLoss':>9} {'TrAcc':>7} {'VaAcc':>7}")
    print("-" * 35)

    for ep in range(1, EPOCHS + 1):
        # ── 训练 ──────────────────────────────────────────────
        model.train()
        tl, tc, tt = 0.0, 0, 0
        for seq, sp, ep_, lbl in tr_dl:
            seq, sp, ep_, lbl = (seq.to(device), sp.to(device),
                                  ep_.to(device), lbl.to(device))
            opt.zero_grad()
            out  = model(seq, sp, ep_)
            loss = crit(out, lbl)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl += loss.item() * len(lbl)
            tc += (out.argmax(1) == lbl).sum().item()
            tt += len(lbl)
        sched.step()

        # ── 验证 ──────────────────────────────────────────────
        model.eval()
        vc, vt = 0, 0
        with torch.no_grad():
            for seq, sp, ep_, lbl in va_dl:
                seq, sp, ep_, lbl = (seq.to(device), sp.to(device),
                                      ep_.to(device), lbl.to(device))
                vc += (model(seq, sp, ep_).argmax(1) == lbl).sum().item()
                vt += len(lbl)

        ta = tc / tt
        va = vc / vt
        history['train_loss'].append(tl / tt)
        history['train_acc'].append(ta)
        history['val_acc'].append(va)

        if ep % 20 == 0 or ep == 1:
            flag = " ◀ best" if va > best_acc else ""
            print(f"{ep:>5} {tl/tt:>9.4f} {ta:>7.3f} {va:>7.3f}{flag}")

        if va > best_acc:
            best_acc, best_ep = va, ep
            torch.save({
                'model_state':   model.state_dict(),
                'class_names':   classes,
                'norm_params':   norm,
                'model_type':    'Model_CNNGRU',
                'bidirectional': BIDIRECTIONAL,
                'num_classes':   num_classes,
                'feat_dim':      FEAT_DIM,
                'pose_dim':      POSE_DIM,
                'val_acc':       va,
            }, SAVE_PATH)

    print(f"\n[最佳] val_acc={best_acc:.3f}  epoch={best_ep}")
    print(f"[保存] {SAVE_PATH}")

    with open(CLASS_JSON, 'w', encoding='utf-8') as f:
        json.dump(classes, f, ensure_ascii=False)

    # ── 训练曲线 ─────────────────────────────────────────────
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], color='red', label='Train Loss')
    plt.title('Training Loss'); plt.xlabel('Epochs')
    plt.grid(True); plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], color='blue',  label='Train Acc')
    plt.plot(history['val_acc'],   color='green', label='Val Acc')
    plt.title(f'Accuracy  ({direction}GRU)')
    plt.xlabel('Epochs'); plt.grid(True); plt.legend()
    plt.tight_layout()

    fname = "curves_bigru_aug.png"
    plt.savefig(fname)
    print(f"[曲线] {fname}")
    plt.show()

    # ── 混淆矩阵 ─────────────────────────────────────────────
    ckpt = torch.load(SAVE_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    all_p, all_t = [], []
    with torch.no_grad():
        for seq, sp, ep_, lbl in va_dl:
            seq, sp, ep_ = seq.to(device), sp.to(device), ep_.to(device)
            all_p.extend(model(seq, sp, ep_).argmax(1).cpu().numpy())
            all_t.extend(lbl.numpy())

    cm = np.zeros((num_classes, num_classes), int)
    for t, p in zip(all_t, all_p):
        cm[t][p] += 1

    print(f"\n[混淆矩阵]  行=真实  列=预测")
    print(f"{'':12s}" + "".join(f"{c:>8s}" for c in classes))
    for i, row in enumerate(cm):
        print(f"{classes[i]:12s}" + "".join(f"{v:>8d}" for v in row))

    per = cm.diagonal() / (cm.sum(1) + 1e-8)
    print("\n[每类准确率]")
    for c, a in zip(classes, per):
        bar = '█' * int(a * 20) + '░' * (20 - int(a * 20))
        print(f"  {c:14s}: {a:.2f}  {bar}")


if __name__ == "__main__":
    if not os.path.exists(DATASET_PATH):
        print(f"❌ 找不到: {DATASET_PATH}")
        print("请先运行 preprocess.py 生成 dataset_aligned_v2.npz")
    else:
        train()