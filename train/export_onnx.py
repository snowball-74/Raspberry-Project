import argparse
import json
import os

import numpy as np
import torch
import onnx

# ── 导入你的模型定义（确保 model.py 在同一目录） ────────────────
from model import Model_CNNGRU


# ═══════════════════════════════════════════════════════════════
#  工具函数
# ═══════════════════════════════════════════════════════════════

def load_checkpoint(pth_path: str, device: torch.device):
    """加载 checkpoint，返回 (model, ckpt_dict)"""
    print(f"[加载] {pth_path}")
    ckpt = torch.load(pth_path, map_location=device, weights_only=False)

    # 从 checkpoint 读取超参数（与 train.py torch.save 对应）
    num_classes   = ckpt['num_classes']
    feat_dim      = ckpt.get('feat_dim', 172)
    pose_dim      = ckpt.get('pose_dim', 40)
    bidirectional = ckpt.get('bidirectional', True)

    print(f"  num_classes={num_classes}  feat_dim={feat_dim}  "
          f"pose_dim={pose_dim}  bidirectional={bidirectional}")

    model = Model_CNNGRU(
        num_classes   = num_classes,
        feat_dim      = feat_dim,
        pose_dim      = pose_dim,
        hidden_size   = 128,
        bidirectional = bidirectional,
    )
    model.load_state_dict(ckpt['model_state'])
    model.to(device).eval()
    return model, ckpt


def export_norm_params(ckpt: dict, out_dir: str):
    """将归一化参数单独保存为 JSON，推理时用来预处理输入数据"""
    norm = ckpt.get('norm_params', {})
    if not norm:
        print("[警告] checkpoint 中没有 norm_params，跳过导出")
        return

    out_path = os.path.join(out_dir, "norm_params.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(norm, f, ensure_ascii=False, indent=2)
    print(f"[归一化] 已保存 → {out_path}")


def export_class_names(ckpt: dict, out_dir: str):
    """保存类别名称"""
    classes = ckpt.get('class_names', [])
    if not classes:
        print("[警告] checkpoint 中没有 class_names，跳过导出")
        return

    out_path = os.path.join(out_dir, "class_names.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(classes, f, ensure_ascii=False, indent=2)
    print(f"[类别] {len(classes)} 类 → {out_path}")


def export_to_onnx(model: torch.nn.Module,
                   feat_dim: int,
                   pose_dim: int,
                   seq_len: int,
                   out_path: str,
                   opset: int = 12):
    """执行 ONNX 导出"""
    # 构造虚拟输入（batch=1）
    dummy_seq = torch.randn(1, seq_len, feat_dim)   # (B, T, feat_dim)
    dummy_sp  = torch.randn(1, pose_dim)            # (B, pose_dim)
    dummy_ep  = torch.randn(1, pose_dim)            # (B, pose_dim)

    print(f"\n[导出] opset={opset}  输入形状: "
          f"seq={tuple(dummy_seq.shape)}  "
          f"sp={tuple(dummy_sp.shape)}  ep={tuple(dummy_ep.shape)}")

    torch.onnx.export(
        model,
        (dummy_seq, dummy_sp, dummy_ep),
        out_path,
        opset_version        = opset,
        input_names          = ['seq', 'start_pose', 'end_pose'],
        output_names         = ['logits'],
        dynamic_axes         = {
            'seq':        {0: 'batch'},
            'start_pose': {0: 'batch'},
            'end_pose':   {0: 'batch'},
            'logits':     {0: 'batch'},
        },
        do_constant_folding  = True,
        export_params        = True,
    )
    print(f"[导出] 完成 → {out_path}")

    # 验证图结构
    onnx_model = onnx.load(out_path)
    onnx.checker.check_model(onnx_model)
    print("[验证] ONNX 图结构检查通过")


def verify_with_onnxruntime(model: torch.nn.Module,
                             feat_dim: int,
                             pose_dim: int,
                             seq_len: int,
                             onnx_path: str,
                             rtol: float = 1e-3,
                             atol: float = 1e-5):
    """用 onnxruntime 推理，与 torch 输出对比精度"""
    try:
        import onnxruntime as ort
    except ImportError:
        print("[跳过] 未安装 onnxruntime，无法做精度验证")
        print("       安装命令: pip install onnxruntime")
        return

    # 随机输入
    np.random.seed(0)
    seq_np = np.random.randn(1, seq_len, feat_dim).astype(np.float32)
    sp_np  = np.random.randn(1, pose_dim).astype(np.float32)
    ep_np  = np.random.randn(1, pose_dim).astype(np.float32)

    # Torch 推理
    with torch.no_grad():
        torch_out = model(
            torch.from_numpy(seq_np),
            torch.from_numpy(sp_np),
            torch.from_numpy(ep_np),
        ).numpy()

    # ONNX Runtime 推理
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(onnx_path,
                                sess_options=sess_options,
                                providers=['CPUExecutionProvider'])
    ort_out = sess.run(
        ['logits'],
        {'seq': seq_np, 'start_pose': sp_np, 'end_pose': ep_np}
    )[0]

    max_diff = np.abs(torch_out - ort_out).max()
    match    = np.allclose(torch_out, ort_out, rtol=rtol, atol=atol)

    print(f"\n[精度验证]  最大误差={max_diff:.2e}  "
          f"{'✓ 通过' if match else '✗ 超出容差，请检查'}")
    if not match:
        print(f"  torch  输出前5: {torch_out[0, :5]}")
        print(f"  onnxrt 输出前5: {ort_out[0, :5]}")


# ═══════════════════════════════════════════════════════════════
#  主入口
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="PTH → ONNX 转换工具（手语识别）")
    parser.add_argument('--pth',    default='dynamic_model.pth', help='输入 .pth 路径')
    parser.add_argument('--out',    default='dynamic_model.onnx', help='输出 .onnx 路径')
    parser.add_argument('--seq',    type=int, default=20,  help='序列长度（默认30）')
    parser.add_argument('--opset', type=int, default=12,  help='ONNX opset（默认12）')
    args = parser.parse_args()

    device   = torch.device('cpu')   # 导出时固定用 CPU
    out_dir  = os.path.dirname(os.path.abspath(args.out)) or '.'

    # 1. 加载模型
    model, ckpt = load_checkpoint(args.pth, device)

    feat_dim = ckpt.get('feat_dim', 172)
    pose_dim = ckpt.get('pose_dim', 40)

    # 2. 导出归一化参数 & 类别名
    export_norm_params(ckpt, out_dir)
    export_class_names(ckpt, out_dir)

    # 3. 导出 ONNX
    export_to_onnx(model, feat_dim, pose_dim, args.seq, args.out, args.opset)

    # 4. 精度验证
    verify_with_onnxruntime(model, feat_dim, pose_dim, args.seq, args.out)

    # 5. 打印文件大小
    size_mb = os.path.getsize(args.out) / 1024 / 1024
    print(f"\n[完成] {args.out}  ({size_mb:.2f} MB)")
    print("\n树莓派部署步骤：")
    print("  1. pip install onnxruntime   (ARM 版本已官方支持)")
    print("  2. 将 dynamic_model.onnx / norm_params.json / class_names.json 复制到树莓派")
    print("  3. 参考 infer_rpi.py 进行推理")


if __name__ == '__main__':
    main()