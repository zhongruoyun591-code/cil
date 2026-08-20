# -*- coding: utf-8 -*-
"""
run_diag.py / train_with_diag.py 共享的工具函数。
"""
import csv
import os
import sys

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(REPO_ROOT, 'src')
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import yaml

from acmap.backbone.vit_acmap import Adapter
from acmap.utils.config import Config

COLUMNS = [
    'task', 'n_seen', 'A_sep', 'A_route', 'R_route',
    'A_merge_acmap', 'A_merge_tid', 'A_merge_fresh',
    'LMC_0', 'LMC_25', 'LMC_50', 'LMC_75', 'LMC_100', 'LMC_barrier',
    'overlap_avg_cum', 'cos_flat_avg_cum', 'norm_task', 'norm_merged',
]


class NullLogger:
    """ACMapNet 只调用 logger.info,这里给个最小实现。"""

    def info(self, message):
        print(f'[diag] {message}')

    def log(self, data):
        pass

    def print_args(self):
        pass


def build_config(args):
    with open(args.config) as f:
        config = yaml.safe_load(f)
    config.update(vars(args))
    config.update({'seed': 0})  # dummy,官方 train.py 同样处理
    config = Config(**config)
    config = Config.model_validate(config)
    config.seed = args.seed
    return config


def build_empty_adapter(config):
    adapter = nn.ModuleList()
    for _ in range(config.transformer.depth):
        adapter.append(Adapter(config=config))
    return adapter


def safe_torch_load(path):
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(path, map_location='cpu')


def interp_adapter(adapter_a, adapter_b, lam, config, device):
    """返回 (1-lam)*a + lam*b 的新适配器(ModuleList)。"""
    out = build_empty_adapter(config).to(device)
    sa, sb = adapter_a.state_dict(), adapter_b.state_dict()
    with torch.no_grad():
        for key in sa.keys():
            out.state_dict()[key].copy_((1.0 - lam) * sa[key] + lam * sb[key])
    return out


def fro_norm(module_list):
    total = 0.0
    for p in module_list.parameters():
        total += (p.detach().float() ** 2).sum().item()
    return float(total ** 0.5)


def extract_feats(net, loader, adapter, device):
    feats = []
    with torch.no_grad():
        for _, (_, data, _) in enumerate(loader):
            data = data.to(device)
            feats.append(net.backbone.forward_proto(data, adapter).cpu())
    return torch.cat(feats, dim=0)


def extract_protos(net, dataset, loader, adapter, device):
    feats = extract_feats(net, loader, adapter, device)
    labels = torch.as_tensor(dataset.labels, dtype=torch.long)

    classes = np.unique(dataset.labels)
    proto_rows = []
    for c in classes:
        mask = labels == c
        proto_rows.append(feats[mask].mean(dim=0, keepdim=True))
    proto_mat = torch.cat(proto_rows, dim=0)
    return np.asarray(classes), proto_mat


def cosine_pred(feats, classes, proto_mat):
    sims = F.normalize(feats, dim=1) @ F.normalize(proto_mat, dim=1).T
    idx = sims.argmax(dim=1)
    return classes[idx.numpy()]


def task_of_class(c, ranges):
    for t, (lo, hi) in enumerate(ranges, start=1):
        if lo <= c < hi:
            return t
    return len(ranges)


def pairwise_overlaps(adapter_a, adapter_b, config):
    """两个任务适配器之间的 (逐层行空间重叠, 展平余弦相似度)。"""
    per_layer = []
    for layer in range(config.transformer.depth):
        wa = adapter_a[layer].down_proj.weight.detach().float().cpu()  # [down, d_model]
        wb = adapter_b[layer].down_proj.weight.detach().float().cpu()
        qa = torch.linalg.qr(wa.T).Q  # [d_model, down]
        qb = torch.linalg.qr(wb.T).Q
        per_layer.append(float((qa.T @ qb).pow(2).sum() / wa.shape[0]))
    ov = float(np.mean(per_layer))

    va = torch.cat([p.detach().float().cpu().flatten() for p in adapter_a.parameters()])
    vb = torch.cat([p.detach().float().cpu().flatten() for p in adapter_b.parameters()])
    cf = float(F.cosine_similarity(va, vb, dim=0))
    return ov, cf


def overlap_matrices(task_adapters, config):
    """task_adapters: {task: adapter}; 返回 (ov, cf) 两个 (T+1)x(T+1) 矩阵。"""
    T = len(task_adapters)
    ov = np.zeros((T + 1, T + 1))
    cf = np.zeros((T + 1, T + 1))
    for t in range(1, T + 1):
        for s in range(t + 1, T + 1):
            ov[t, s], cf[t, s] = pairwise_overlaps(task_adapters[t], task_adapters[s], config)
            ov[s, t], cf[s, t] = ov[t, s], cf[t, s]
    return ov, cf


def eval_acmap_net(net, loader, ranges, device):
    """
    用 ACMapNet 的 forward(test=True)(合并 adapter + fc 原型分类器)评估。
    返回 (A_merge_acmap, A_merge_tid)。
    """
    net.eval()
    preds, labels = [], []
    tid_correct, total = 0, 0
    with torch.no_grad():
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(device)
            logits = net.forward(inputs, test=True)['logits']  # [N, n_seen_classes]
            preds.append(logits.argmax(dim=1).cpu())
            labels.append(targets)

            # task-id 掩码评估(等价于官方代码里的 task_logits 逻辑)
            for i, tg in enumerate(targets):
                t_true = task_of_class(tg.item(), ranges)
                lo, hi = ranges[t_true - 1]
                masked = logits[i].clone()
                masked[:lo] = float('-inf')
                masked[hi:] = float('-inf')
                tid_correct += int(masked.argmax() == tg.to(device))
                total += 1

    preds = torch.cat(preds)
    labels = torch.cat(labels)
    a_merge = float((preds == labels).float().mean() * 100)
    a_tid = float(tid_correct / total * 100)
    return a_merge, a_tid


def write_matrix(path, mat):
    T = mat.shape[0] - 1
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['task'] + list(range(1, T + 1)))
        for t in range(1, T + 1):
            writer.writerow([t] + [f'{mat[t, s]:.6f}' for s in range(1, T + 1)])


def write_amatrices(out_dir, A_mat, Atid_mat, T):
    """写遗忘矩阵 A[t,s] 与 A_tid[t,s](s>t 留空)。

    A[t,s]      = 时间 t 的模型(合并适配器 + 当时分类器)在任务 s 测试集上的全口径精度;
    A_tid[t,s]  = 同上,但 argmax 限制在任务 s 的类范围内(task-id 掩码)。
    两者之差 = 跨任务混淆; 对角 A_tid 与列方向衰减之差 = 特征级遗忘。
    """
    for name, mat in [('A_matrix.csv', A_mat), ('A_tid_matrix.csv', Atid_mat)]:
        path = os.path.join(out_dir, name)
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['model_task'] + [f'task{s}' for s in range(1, T + 1)])
            for t in range(1, T + 1):
                row = [t]
                for s in range(1, T + 1):
                    v = mat[t].get(s, float('nan'))
                    row.append('' if (s > t or np.isnan(v)) else f'{v:.4f}')
                writer.writerow(row)
    print(f'[diag] 遗忘矩阵已写入 {os.path.join(out_dir, "A_matrix.csv")} / {os.path.join(out_dir, "A_tid_matrix.csv")}')


def finalize_results(rows, out_dir, ov=None, cf=None):
    """写 CSV、写重叠矩阵、打印 H1-H3 快速检查。"""
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, 'diag_results.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    if ov is not None:
        write_matrix(os.path.join(out_dir, 'overlap_matrix.csv'), ov)
    if cf is not None:
        write_matrix(os.path.join(out_dir, 'cos_matrix.csv'), cf)

    print('\n[diag] 快速检查(排除首任务):')
    sub = [r for r in rows if r['task'] > 1]
    if sub:
        ts = np.array([r['task'] for r in sub])
        d_sep = np.array([r['A_sep'] - r['A_merge_acmap'] for r in sub])
        ovc = np.array([r['overlap_avg_cum'] for r in sub])
        print(f'  H1: corr(Δ_sep, task) = {np.corrcoef(d_sep, ts)[0, 1]:+.3f}')
        print(f'  H2: corr(Δ_sep, overlap_cum) = {np.corrcoef(d_sep, ovc)[0, 1]:+.3f}')
        d_route = np.array([r['A_sep'] - r['A_route'] for r in sub])
        print(f'  H3: mean routing loss = {d_route.mean():.2f}, mean merge loss = {d_sep.mean():.2f}')

    print(f'\n[diag] 结果已写入 {csv_path}')
    print(f'[diag] 下一步: python diag/plot.py --csv {csv_path}')
    return csv_path
