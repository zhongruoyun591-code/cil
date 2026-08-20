# -*- coding: utf-8 -*-
"""
Phase 8 探针: post-hoc margin 校准能追回多少"跨任务混淆"损失?

读取 diag_out/<run>/merged/task{T}.pkl(最终合并适配器 θ̄_T),在冻结主干上:
  1) 用全部已见类的训练数据算原型 μ_c(合并适配器特征空间);
  2) 三个变体评估(全部在"末时刻、任务无关、全类 argmax"口径下):
     - baseline : cos(x, μ_c)                     (≈ diag 的 A_merge_fresh)
     - oracle   : s·cos(x, μ_c) + b_c, 用全部已见类训练特征做 CE 训练(校准天花板,非增量合法)
     - geometry : cos(x, μ_c) − α·max_{c'∈其他任务} cos(μ_c, μ_{c'}),α 仅用最后任务训练数据网格选(增量合法)
  3) 对标 diag 的 A_tid 末行(任务边界已知的上限),报告每个变体对"混淆缺口"的追回率。

用法(在 ACMap 仓库根目录):
  python diag/calib_probe.py --config exps/cifar.yaml --init_cls 0 --increment 5 --seed 1993
  python diag/calib_probe.py --config exps/imagenet-r.yaml --init_cls 0 --increment 20 --seed 1993

输出: <diag_out_dir>/calib_probe.csv + 控制台汇总表。
"""

import argparse
import csv
import os

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from diag_common import (
    NullLogger,
    build_config,
    build_empty_adapter,
    safe_torch_load,
    extract_feats,
)

from acmap.utils.context import Context
from acmap.utils.data_manager import DataManager
from acmap.utils.inc_net import ACMapNet
from acmap.utils.toolkit import set_random


# ----------------------------------------------------------------------------
# 数据准备
# ----------------------------------------------------------------------------
def make_loaders(data_manager, ranges, batch_size, num_workers):
    """全部已见类的训练 loader + 每任务测试 loader。"""
    final_total = ranges[-1][1]
    train_ds = data_manager.get_dataset(np.arange(0, final_total), source='train', mode='test')
    train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    test_loaders, test_labels = {}, {}
    for s, (lo, hi) in enumerate(ranges, start=1):
        ds = data_manager.get_dataset(np.arange(lo, hi), source='test', mode='test')
        test_loaders[s] = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        test_labels[s] = np.asarray(ds.labels)
    return train_ld, test_loaders, test_labels


def extract_feats_labels(net, loader, adapter, device):
    feats, labels = [], []
    with torch.no_grad():
        for _, (_, data, label) in enumerate(loader):
            feats.append(net.backbone.forward_proto(data.to(device), adapter).cpu())
            labels.append(label)
    return torch.cat(feats, dim=0), torch.cat(labels)


def class_prototypes(feats, labels, n_classes):
    P = torch.zeros(n_classes, feats.shape[1])
    for c in range(n_classes):
        P[c] = feats[labels == c].mean(dim=0)
    return P


# ----------------------------------------------------------------------------
# 变体实现
# ----------------------------------------------------------------------------
def eval_variant(scores_dict, test_labels):
    """scores_dict: {task: [N_s, C] 打分矩阵} → {task: top-1 精度(任务无关口径)}。"""
    accs = {}
    for s, scores in scores_dict.items():
        preds = scores.argmax(dim=1).numpy()
        accs[s] = float((preds == test_labels[s]).mean() * 100)
    return accs


def variant_oracle(tr_feats, tr_labels, P, n_classes, epochs, lr, device):
    """per-class bias b_c + 全局 scale s,在全量训练特征上做 CE(天花板)。"""
    z = F.normalize(tr_feats, dim=1).to(device)
    p = F.normalize(P, dim=1).to(device)
    y = tr_labels.to(device)

    b = torch.zeros(n_classes, device=device, requires_grad=True)
    log_s = torch.zeros(1, device=device, requires_grad=True)
    opt = torch.optim.Adam([b, log_s], lr=lr)
    base = z @ p.T  # [N, C] 余弦
    for _ in range(epochs):
        opt.zero_grad()
        logits = base * log_s.exp() + b
        loss = F.cross_entropy(logits, y)
        loss.backward()
        opt.step()
    return b.detach(), float(log_s.detach().exp())


def apply_oracle(P, feats_dict, b, s, device):
    out = {}
    p = F.normalize(P, dim=1).to(device)
    for k, f in feats_dict.items():
        out[k] = F.normalize(f, dim=1).to(device) @ p.T * s + b
    return out


def variant_geometry(P, ranges):
    """几何 margin: m_c = α·max_{跨任务 c'} cos(μ_c, μ_c')。返回 (归一化原型, 各类跨任务最近邻余弦)。"""
    Pn = F.normalize(P, dim=1)
    C = P.shape[0]
    sim = Pn @ Pn.T  # [C, C]
    task_of = np.zeros(C, dtype=np.int64)
    for t, (lo, hi) in enumerate(ranges, start=1):
        task_of[lo:hi] = t
    cross = np.full((C, C), -np.inf)
    for c in range(C):
        mask = task_of != task_of[c]
        cross[c, mask] = sim[c, mask].numpy()
    d_c = torch.as_tensor(cross.max(axis=1), dtype=torch.float32)  # 每类与"其他任务"最近原型的余弦
    return Pn, d_c


def main():
    parser = argparse.ArgumentParser(description='Phase8: post-hoc margin 校准探针')
    parser.add_argument('--config', type=str, default=os.path.join('exps', 'cifar.yaml'))
    parser.add_argument('--init_cls', type=int, default=0)
    parser.add_argument('--increment', type=int, default=5)
    parser.add_argument('--seed', type=int, default=1993)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--dataset_dir', type=str, default='dataset')
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=300, help='oracle 校准的 CE 训练轮数')
    parser.add_argument('--lr', type=float, default=0.1, help='oracle 校准学习率')
    parser.add_argument('--alpha_grid', type=str, default='0.0,0.1,0.2,0.3,0.5,0.8,1.0,1.5,2.0')
    parser.add_argument(
        '--merged_dir', type=str, default=None,
        help='合并适配器快照目录(默认 diag_out/<dataset>-b<init>i<inc>-s<seed>/merged)',
    )
    parser.add_argument('--out_dir', type=str, default=None, help='输出目录(默认 diag 目录)')
    args = parser.parse_args()

    config = build_config(args)
    set_random(config.seed)
    device = config.device
    batch_size = args.batch_size if args.batch_size else config.exp.batch_size
    num_workers = config.exp.num_workers

    run_tag = f'{config.exp.dataset}-b{config.init_cls}-i{config.increment}-s{config.seed}'
    merged_dir = args.merged_dir or os.path.join('diag_out', run_tag, 'merged')
    out_dir = args.out_dir or os.path.join('diag_out', run_tag)

    # ---- 数据与任务范围 ----
    data_manager = DataManager(
        dataset_name=config.exp.dataset, shuffle=config.exp.shuffle,
        seed=config.seed, dataset_dir=config.dataset_dir,
    )
    sim_ctx = Context(config=config, logger=NullLogger(), class_order=data_manager.class_order)
    ranges = []
    for _ in range(sim_ctx.num_tasks):
        ranges.append((sim_ctx.known_classes, sim_ctx.total_classes))
        sim_ctx.next_task()
    T = len(ranges)
    n_classes = ranges[-1][1]

    # ---- 模型 + 最终合并适配器 ----
    net = ACMapNet(context=Context(config=config, logger=NullLogger(), class_order=data_manager.class_order))
    net.to(device)
    net.eval()
    ckpt = safe_torch_load(os.path.join(merged_dir, f'task{T}.pkl'))
    merged_adapter = build_empty_adapter(config).to(device)
    merged_adapter.load_state_dict(ckpt['state_dict'])
    print(f'[calib] merged snapshot: {os.path.join(merged_dir, f"task{T}.pkl")} (T={T}, {n_classes} classes)')

    # ---- 特征 ----
    train_ld, test_loaders, test_labels = make_loaders(data_manager, ranges, batch_size, num_workers)
    print('[calib] extracting features with merged adapter ...')
    tr_feats, tr_labels = extract_feats_labels(net, train_ld, merged_adapter, device)
    te_feats = {s: extract_feats(net, ld, merged_adapter, device) for s, ld in test_loaders.items()}

    # ---- 原型 ----
    P = class_prototypes(tr_feats, tr_labels, n_classes)
    print(f'[calib] prototypes {P.shape}, train feats {tr_feats.shape}')

    # ---- baseline ----
    base_scores = {s: F.normalize(f, dim=1) @ F.normalize(P, dim=1).T for s, f in te_feats.items()}
    base_acc = eval_variant(base_scores, test_labels)

    # ---- oracle 校准(天花板) ----
    print('[calib] training oracle per-class bias + scale (all train data) ...')
    b, s = variant_oracle(tr_feats, tr_labels, P, n_classes, args.epochs, args.lr, device)
    oracle_scores = apply_oracle(P, te_feats, b, s, device)
    oracle_acc = eval_variant({k: v.cpu() for k, v in oracle_scores.items()}, test_labels)

    # ---- 几何 margin(增量合法,α 在最后任务训练数据上网格选) ----
    Pn, d_c = variant_geometry(P, ranges)
    last_tr = tr_feats[tr_labels >= ranges[-1][0]]
    last_tr_labels = tr_labels[tr_labels >= ranges[-1][0]]
    sim_last = F.normalize(last_tr, dim=1) @ Pn.T  # [N_T, C]
    alpha_grid = [float(x) for x in args.alpha_grid.split(',')]
    best_alpha, best_acc = None, -1
    for alpha in alpha_grid:
        scores = sim_last - alpha * d_c
        acc = float((scores.argmax(1) == last_tr_labels).float().mean() * 100)
        if acc > best_acc:
            best_acc, best_alpha = acc, alpha
    geom_scores = {
        s: F.normalize(f, dim=1) @ Pn.T - best_alpha * d_c for s, f in te_feats.items()
    }
    geom_acc = eval_variant(geom_scores, test_labels)
    print(f'[calib] geometry margin: best alpha={best_alpha} (last-task train acc {best_acc:.2f})')

    # ---- 对标 A_tid(任务边界上限) ----
    atid_path = os.path.join(out_dir, 'A_tid_matrix.csv')
    atid_final = None
    if os.path.exists(atid_path):
        with open(atid_path) as f:
            lines = [l.strip().split(',') for l in f]
        atid_final = np.array([float(v) for v in lines[T][1:]])
    else:
        print(f'[calib] 未找到 {atid_path},跳过 A_tid 对标')

    # ---- 输出 ----
    csv_path = os.path.join(out_dir, 'calib_probe.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['task', 'baseline', 'oracle', 'geometry', 'A_tid',
                    'oracle_recovery%', 'geometry_recovery%'])
        rows = []
        for s in range(1, T + 1):
            bv, ov, gv = base_acc[s], oracle_acc[s], geom_acc[s]
            tv = float(atid_final[s - 1]) if atid_final is not None else float('nan')
            gap = tv - bv
            rec_o = 100 * (ov - bv) / gap if gap > 0 else float('nan')
            rec_g = 100 * (gv - bv) / gap if gap > 0 else float('nan')
            w.writerow([s, f'{bv:.2f}', f'{ov:.2f}', f'{gv:.2f}', f'{tv:.2f}',
                        f'{rec_o:.1f}', f'{rec_g:.1f}'])
            rows.append((bv, ov, gv, tv, rec_o, rec_g))

    def mean(i):
        return np.nanmean([r[i] for r in rows])

    print('\n[calib] 汇总(末时刻,任务无关口径, mean over tasks):')
    print(f'  baseline        : {mean(0):.2f}')
    print(f'  oracle (天花板) : {mean(1):.2f}   (追回率 {mean(4):.1f}%)')
    print(f'  geometry (合法) : {mean(2):.2f}   (追回率 {mean(5):.1f}%)')
    if atid_final is not None:
        print(f'  A_tid (上限)    : {mean(3):.2f}')
    print(f'\n[calib] 明细已写入 {csv_path}')


if __name__ == '__main__':
    main()
