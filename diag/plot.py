# -*- coding: utf-8 -*-
"""
C1 诊断绘图: 读取 run_diag.py 输出的 CSV,生成 4 张诊断图 + 1 张汇总图。

用法:
  python diag/plot.py --csv diag_out/diag_results.csv                    # 单次运行
  python diag/plot.py --csv run1/diag_results.csv run2/diag_results.csv  # 多次运行(不同 seed/顺序)→ 均值±std
  python diag/plot.py --csv diag_out/diag_results.csv --overlap diag_out/overlap_matrix.csv

输出(默认 diag_out/):
  fig1_acc.png       三条精度曲线: A_sep / A_route / A_merge_acmap(+A_merge_fresh)
  fig2_loss.png      损失分解: 路由误差 vs 合并损失(Δ_sep = A_sep - A_merge_acmap)
  fig3_overlap.png   左: 任务对重叠热图; 右: Δ_sep vs 累计重叠散点(Pearson r)
  fig4_lmc.png       左: 最后一个任务的 LMC 曲线; 右: LMC 屏障随任务变化
  fig_all.png        2x2 汇总
"""

import argparse
import csv
import os

import numpy as np

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 1. 支持中文（用于标题、轴标签中的中文部分）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
# 2. 禁用 Unicode 减号（防止非数学模式下的减号缺失，但数学模式下不需要）
plt.rcParams['axes.unicode_minus'] = False

METRICS = ['A_sep', 'A_route', 'A_merge_acmap', 'A_merge_tid', 'A_merge_fresh']


def load_csv(path):
    with open(path, newline='') as f:
        rows = list(csv.DictReader(f))
    data = {key: [] for key in rows[0].keys()}
    for r in rows:
        for k, v in r.items():
            try:
                data[k].append(float(v))
            except (TypeError, ValueError):
                data[k].append(np.nan)
    return {k: np.asarray(v) for k, v in data.items()}


def mean_std(runs, key):
    """按 task 对齐后求均值与标准差。"""
    stacks = np.vstack([run[key] for run in runs])  # [n_runs, T]
    return stacks.mean(axis=0), stacks.std(axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', nargs='+', required=True, help='一个或多个 diag_results.csv')
    parser.add_argument('--overlap', type=str, default=None, help='overlap_matrix.csv(默认取第一个 csv 同目录)')
    parser.add_argument('--out_dir', type=str, default=None)
    args = parser.parse_args()

    runs = [load_csv(p) for p in args.csv]
    T = min(len(r['task']) for r in runs)
    runs = [{k: v[:T] for k, v in r.items()} for r in runs]
    task = runs[0]['task'].astype(int)

    out_dir = args.out_dir or os.path.dirname(args.csv[0])
    os.makedirs(out_dir, exist_ok=True)

    has_fresh = all(not np.isnan(r['A_merge_fresh']).all() for r in runs)

    # ---------------------------------------------------------------- Fig 1
    fig, ax = plt.subplots(figsize=(7, 5))
    for key, color, label in [
        ('A_sep', '#2ca02c', 'A_sep (oracle task-id)'),
        ('A_route', '#1f77b4', 'A_route (prototype routing)'),
        ('A_merge_acmap', '#d62728', 'A_merge (ACMap)'),
    ]:
        m, s = mean_std(runs, key)
        ax.plot(task, m, color=color, label=label, marker='o', ms=4)
        ax.fill_between(task, m - s, m + s, color=color, alpha=0.15)
    if has_fresh:
        m, s = mean_std(runs, 'A_merge_fresh')
        ax.plot(task, m, color='#9467bd', label='A_merge_fresh (no proto drift)', marker='s', ms=4)
        ax.fill_between(task, m - s, m + s, color='#9467bd', alpha=0.15)
    ax.set_xlabel('Task t')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Fig1: 合并损失主图 (merge-loss curves)')
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'fig1_acc.png'), dpi=200)
    plt.close(fig)

    # ---------------------------------------------------------------- Fig 2
    fig, ax = plt.subplots(figsize=(7, 5))
    m_merge, s_merge = mean_std(runs, 'A_sep')
    m_route, s_route = mean_std(runs, 'A_route')
    m_acmap, s_acmap = mean_std(runs, 'A_merge_acmap')
    loss_merge = m_merge - m_acmap
    loss_route = m_merge - m_route
    ax.bar(task - 0.2, loss_merge, width=0.35, 
       label=r'merge loss $\Delta_{\mathrm{sep}} = A_{\mathrm{sep}} - A_{\mathrm{merge}}$', 
       color='#d62728', alpha=0.8)
    ax.bar(task + 0.2, loss_route, width=0.35, 
       label=r'routing loss $= A_{\mathrm{sep}} - A_{\mathrm{route}}$', 
       color='#1f77b4', alpha=0.8)
    ax.set_xlabel('Task t')
    ax.set_ylabel('Accuracy loss (%)')
    ax.set_title('Fig2: 损失分解 — 合并干扰 vs 路由误差')
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'fig2_loss.png'), dpi=200)
    plt.close(fig)

    # ---------------------------------------------------------------- Fig 3
    overlap_path = args.overlap or os.path.join(os.path.dirname(args.csv[0]), 'overlap_matrix.csv')
    mat = None
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    if os.path.exists(overlap_path):
        with open(overlap_path) as f:
            f.readline()  # skip header
            mat = np.array([[float(x) for x in line.strip().split(',')[1:]] for line in f])
        im = axes[0].imshow(mat, cmap='viridis', aspect='auto')
        axes[0].set_xticks(np.arange(mat.shape[0]))
        axes[0].set_yticks(np.arange(mat.shape[0]))
        axes[0].set_xticklabels(np.arange(1, mat.shape[0] + 1), fontsize=7)
        axes[0].set_yticklabels(np.arange(1, mat.shape[0] + 1), fontsize=7)
        axes[0].set_title('任务适配器行空间重叠 overlap(t, s)')
        axes[0].set_xlabel('task s')
        axes[0].set_ylabel('task t')
        fig.colorbar(im, ax=axes[0], fraction=0.046)
    else:
        axes[0].text(0.5, 0.5, f'未找到 {overlap_path}', ha='center')
        axes[0].set_title('overlap')

    d_sep = mean_std(runs, 'A_sep')[0] - mean_std(runs, 'A_merge_acmap')[0]
    ov_cum = mean_std(runs, 'overlap_avg_cum')[0]
    mask = ~np.isnan(ov_cum)
    axes[1].scatter(ov_cum[mask], d_sep[mask], s=30, alpha=0.8)
    if mask.sum() > 2:
        r = np.corrcoef(ov_cum[mask], d_sep[mask])[0, 1]
        k, b = np.polyfit(ov_cum[mask], d_sep[mask], 1)
        xs = np.linspace(ov_cum[mask].min(), ov_cum[mask].max(), 50)
        axes[1].plot(xs, k * xs + b, '--', color='gray')
        axes[1].set_title(f'Δ_sep vs 累计重叠 (Pearson r = {r:+.3f})')
    else:
        axes[1].set_title('Δ_sep vs 累计重叠')
    axes[1].set_xlabel('mean overlap(θ_t, θ_{s<t})')
    axes[1].set_ylabel('merge loss Δ_sep (%)')
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'fig3_overlap.png'), dpi=200)
    plt.close(fig)

    # ---------------------------------------------------------------- Fig 4
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    lams = [0.0, 0.25, 0.5, 0.75, 1.0]
    keys = ['LMC_0', 'LMC_25', 'LMC_50', 'LMC_75', 'LMC_100']
    curve = np.vstack([mean_std(runs, k)[0] for k in keys])  # [5, T]
    axes[0].plot(lams, curve[:, -1], marker='o', color='#d62728', label='last task')
    axes[0].plot(lams, curve[:, max(0, len(task) - 3)], marker='s', color='#1f77b4', label=f'task {max(1, len(task)-2)}')
    axes[0].set_xlabel(r'$\lambda: \theta(\lambda) = (1-\lambda)\cdot\bar{\theta}_{t-1} + \lambda\cdot\theta_t$')
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_title('Fig4a: LMC 路径曲线(固定本任务原型分类器)')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    m_bar, s_bar = mean_std(runs, 'LMC_barrier')
    axes[1].plot(task, m_bar, marker='o', color='#2ca02c')
    axes[1].fill_between(task, m_bar - s_bar, m_bar + s_bar, color='#2ca02c', alpha=0.15)
    axes[1].set_xlabel('Task t')
    axes[1].set_ylabel('LMC barrier (err)')
    axes[1].set_title('Fig4b: LMC 损失屏障随任务变化')
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'fig4_lmc.png'), dpi=200)
    plt.close(fig)

    # ---------------------------------------------------------------- Fig 5/6: 遗忘矩阵
    amat_path = os.path.join(os.path.dirname(args.csv[0]), 'A_matrix.csv')
    atid_path = os.path.join(os.path.dirname(args.csv[0]), 'A_tid_matrix.csv')

    def _read_amat(p):
        with open(p) as f:
            lines = [l.strip().split(',') for l in f]
        T = len(lines) - 1
        M = np.full((T, T), np.nan)
        for i, l in enumerate(lines[1:]):
            for j in range(T):
                v = l[1 + j]
                if v != '':
                    M[i, j] = float(v)
        return M

    if os.path.exists(amat_path):
        A = _read_amat(amat_path)
        Atid = _read_amat(atid_path) if os.path.exists(atid_path) else None
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        im = axes[0].imshow(A, cmap='viridis', aspect='auto', vmin=np.nanmin(A), vmax=np.nanmax(A))
        axes[0].set_title('A[t,s]: time-t 模型在任务 s 上的精度')
        axes[0].set_xlabel('task s')
        axes[0].set_ylabel('model time t')
        fig.colorbar(im, ax=axes[0], fraction=0.046)
        if Atid is not None:
            im2 = axes[1].imshow(Atid, cmap='viridis', aspect='auto', vmin=np.nanmin(Atid), vmax=np.nanmax(Atid))
            axes[1].set_title('A_tid[t,s]: task-id 掩码精度')
            axes[1].set_xlabel('task s')
            axes[1].set_ylabel('model time t')
            fig.colorbar(im2, ax=axes[1], fraction=0.046)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, 'fig5_amat.png'), dpi=200)
        plt.close(fig)

        # Fig6: 末时刻损失分解(特征遗忘 vs 跨任务混淆)
        fig, ax = plt.subplots(figsize=(8, 5))
        T = A.shape[0]
        s_idx = np.arange(1, T + 1)
        diag = np.diag(A)
        last = A[T - 1]
        if Atid is not None:
            feat = np.diag(Atid) - Atid[T - 1]
            conf = Atid[T - 1] - last
            ax.bar(s_idx, feat, color='#2ca02c', label=r'特征遗忘 (diag $A_{\mathrm{tid}} - A_{\mathrm{tid}[T,s]}$)')
            ax.bar(s_idx, conf, bottom=feat, color='#d62728', label=r'跨任务混淆 ($A_{\mathrm{tid}[T,s]} - A_{[T,s]}$)')
            total = feat + conf
        else:
            total = diag - last
            ax.bar(s_idx, total, color='#d62728', label=r'total loss (diag $A - A_{[T,s]}$)')
        ax.set_xlabel('task s')
        ax.set_ylabel('Accuracy loss at final time (%)')
        ax.set_title('Fig6: 末时刻损失分解(特征遗忘 vs 跨任务混淆)')
        ax.legend()
        ax.grid(alpha=0.3, axis='y')
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, 'fig6_decomp.png'), dpi=200)
        plt.close(fig)

    # ---------------------------------------------------------------- 汇总
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for key, color, label in [
        ('A_sep', '#2ca02c', 'A_sep (oracle)'),
        ('A_route', '#1f77b4', 'A_route (routing)'),
        ('A_merge_acmap', '#d62728', 'A_merge (ACMap)'),
    ]:
        m, s = mean_std(runs, key)
        axes[0, 0].plot(task, m, color=color, label=label, marker='o', ms=3)
        axes[0, 0].fill_between(task, m - s, m + s, color=color, alpha=0.15)
    axes[0, 0].set_title('Acc curves'); axes[0, 0].legend(fontsize=8); axes[0, 0].grid(alpha=0.3)

    axes[0, 1].bar(task - 0.2, loss_merge, width=0.35, color='#d62728', alpha=0.8, label='merge loss')
    axes[0, 1].bar(task + 0.2, loss_route, width=0.35, color='#1f77b4', alpha=0.8, label='routing loss')
    axes[0, 1].set_title('Loss decomposition'); axes[0, 1].legend(fontsize=8); axes[0, 1].grid(alpha=0.3, axis='y')

    if mat is not None:
        im = axes[1, 0].imshow(mat, cmap='viridis', aspect='auto')
        axes[1, 0].set_title('Adapter overlap matrix')
        fig.colorbar(im, ax=axes[1, 0], fraction=0.046)
    axes[1, 0].set_xlabel('task s'); axes[1, 0].set_ylabel('task t')

    axes[1, 1].plot(task, m_bar, marker='o', color='#2ca02c')
    axes[1, 1].set_title('LMC barrier'); axes[1, 1].grid(alpha=0.3)
    fig.suptitle('C1 诊断汇总 (ACMap replay)')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'fig_all.png'), dpi=200)
    plt.close(fig)

    print(
        f'[plot] 图已输出到 {out_dir}: fig1_acc.png / fig2_loss.png / fig3_overlap.png / fig4_lmc.png / fig_all.png'
    )
    if os.path.exists(amat_path):
        print(f'[plot] 遗忘矩阵图: fig5_amat.png / fig6_decomp.png')


if __name__ == '__main__':
    main()
