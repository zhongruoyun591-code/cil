# -*- coding: utf-8 -*-
"""
C1 诊断(事后重放模式): 基于训练时保存的 task{t}.pkl,重放 ACMap 的合并/推理协议,
测量合并损失分解与干扰指标。训练代码零改动。

如果不想保存 checkpoint,请改用在线模式: python diag/train_with_diag.py ...

用法(在 ACMap 仓库根目录):
  python diag/run_diag.py --config exps/cifar.yaml --dataset_dir ./dataset --seed 1993

输出(默认 diag_out/):
  diag_results.csv / overlap_matrix.csv / cos_matrix.csv / merged/task{t}.pkl(重放出的合并快照)
"""

import argparse
import os
import re
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

from diag_common import (
    NullLogger,
    build_config,
    build_empty_adapter,
    extract_feats,
    extract_protos,
    cosine_pred,
    interp_adapter,
    fro_norm,
    overlap_matrices,
    eval_acmap_net,
    finalize_results,
    safe_torch_load,
    write_amatrices,
)

from acmap.utils.context import Context
from acmap.utils.data_manager import DataManager
from acmap.utils.inc_net import ACMapNet
from acmap.utils.toolkit import set_random


def find_run_dir(config, args):
    """定位某次训练运行保存 checkpoint 的目录(data/acmap/ckpts/<group>/<run_name>)。"""
    if args.run_dir:
        return args.run_dir

    group = os.path.join(
        config.ckpts_dir, config.exp.dataset, f'b{config.init_cls}-inc{config.increment}'
    )
    name = f'{config.exp.name}-{config.our.merge_method}'
    name += '-in21k' if '_in21k' in config.exp.backbone_type else ''
    name += f'-seed{config.seed}'

    cand = os.path.join(group, name)
    if os.path.isdir(cand):
        return cand

    if os.path.isdir(group):
        matches = [os.path.join(group, d) for d in os.listdir(group) if os.path.isdir(os.path.join(group, d))]
        matches = [m for m in matches if f'seed{config.seed}' in os.path.basename(m)]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise FileNotFoundError(f'多个匹配的 run 目录: {matches},请用 --run_dir 指定')

    raise FileNotFoundError(
        f'找不到 run 目录 {cand}。请确认已经完成训练,或直接用 '
        f'--run_dir 指定(例如 data/acmap/ckpts/cifar224/b0-inc5/acmap-average-in21k-seed1993)。'
    )


def load_task_adapter(run_dir, task, config, device):
    ckpt = safe_torch_load(os.path.join(run_dir, f'task{task}.pkl'))
    adapter = build_empty_adapter(config).to(device)
    adapter.load_state_dict(ckpt['state_dict'])
    return adapter


def main():
    parser = argparse.ArgumentParser(description='ACMap C1 诊断: 合并损失分解(事后重放模式)')
    parser.add_argument('--config', type=str, default=os.path.join('exps', 'cifar.yaml'))
    parser.add_argument('--init_cls', type=int, default=0)
    parser.add_argument('--increment', type=int, default=5)
    parser.add_argument('--seed', type=int, default=1993, help='必须与训练时 --seed 一致')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--logger', type=str, default='basic')
    parser.add_argument('--prefix', type=str, default='')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--dataset_dir', type=str, default='dataset')
    parser.add_argument('--ckpts_dir', type=str, default=os.path.join('data', 'acmap', 'ckpts'))
    parser.add_argument('--run_dir', type=str, default=None, help='直接指定某次运行的 checkpoint 目录')
    parser.add_argument(
        '--out_dir', type=str, default=None,
        help='输出目录(默认 diag_out/<dataset>-b<init>i<inc>-s<seed>,不同运行互不覆盖)',
    )
    parser.add_argument(
        '--no_save_merged', action='store_true',
        help='不把重放出的合并适配器快照写入 out_dir/merged',
    )
    parser.add_argument('--max_tasks', type=int, default=None, help='只诊断前 K 个任务(快速预览)')
    parser.add_argument('--batch_size', type=int, default=None, help='评估 batch size(默认用配置)')
    parser.add_argument('--no_fresh_proto', action='store_true', help='跳过合并适配器下重算原型(加速)')
    parser.add_argument(
        '--legacy',
        action='store_true',
        help='按修复前的 off-by-one 任务推进重放(用于旧版代码训练出的 checkpoint);'
        '修复后重新训练的运行不要加此参数',
    )
    args = parser.parse_args()

    config = build_config(args)
    set_random(config.seed)
    device = config.device
    batch_size = args.batch_size if args.batch_size else config.exp.batch_size
    num_workers = config.exp.num_workers

    out_dir = args.out_dir or os.path.join(
        'diag_out', f'{config.exp.dataset}-b{config.init_cls}-i{config.increment}-s{config.seed}'
    )

    if args.legacy:
        def _legacy_next_task(self):
            if self.cur_task < self.num_tasks - 1:
                self.cur_task += 1
                self.known_classes += self.cur_task_size

        Context.next_task = _legacy_next_task
        print('[diag] 使用修复前的 legacy 任务推进逻辑(仅用于旧 checkpoint 重放)')

    # ---- 数据集与类顺序(与训练完全一致) ----
    data_manager = DataManager(
        dataset_name=config.exp.dataset,
        shuffle=config.exp.shuffle,
        seed=config.seed,
        dataset_dir=config.dataset_dir,
    )

    # 模拟训练时的 Context 推进,得到每个任务对应的类范围
    sim_ctx = Context(config=config, logger=NullLogger(), class_order=data_manager.class_order)
    ranges = []
    for _ in range(sim_ctx.num_tasks):
        ranges.append((sim_ctx.known_classes, sim_ctx.total_classes))
        sim_ctx.next_task()
    num_tasks = len(ranges)
    print(f'[diag] dataset={config.exp.dataset} num_tasks={num_tasks}')

    # ---- 定位 checkpoint ----
    run_dir = find_run_dir(config, args)
    ckpt_tasks = sorted(
        int(m.group(1)) for f in os.listdir(run_dir) if (m := re.match(r'task(\d+)\.pkl', f))
    )
    T = min(ckpt_tasks[-1], args.max_tasks) if args.max_tasks else ckpt_tasks[-1]
    print(f'[diag] run_dir={run_dir} checkpoints task1..task{T}')

    # ---- 模型 ----
    dctx = Context(config=config, logger=NullLogger(), class_order=data_manager.class_order)
    net = ACMapNet(context=dctx)
    net.to(device)
    net.eval()

    # ---- 加载全部任务适配器 + 预计算各自训练原型 ----
    print('[diag] loading task adapters ...')
    task_adapters = {t: load_task_adapter(run_dir, t, config, device) for t in range(1, T + 1)}

    protos = {}  # protos[t] = (classes, proto_mat)  任务 t 的训练原型(用 theta_t)
    print('[diag] computing per-task train prototypes ...')
    for t in range(1, T + 1):
        lo, hi = ranges[t - 1]
        ds = data_manager.get_dataset(np.arange(lo, hi), source='train', mode='test')
        ld = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        protos[t] = extract_protos(net, ds, ld, task_adapters[t], device)
        print(f'  task {t}: classes {lo}-{hi - 1}, {protos[t][1].shape[0]} protos')

    # ---- 任务间重叠 ----
    ov, cf = overlap_matrices(task_adapters, config)

    # ---- 每个任务的测试集 loader(预构建,矩阵评估复用) ----
    test_loaders = {}
    for s in range(1, T + 1):
        s_lo, s_hi = ranges[s - 1]
        s_ds = data_manager.get_dataset(np.arange(s_lo, s_hi), source='test', mode='test')
        test_loaders[s] = DataLoader(s_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    os.makedirs(os.path.join(out_dir, 'merged'), exist_ok=True)

    rows = []
    A_mat, Atid_mat = {}, {}  # A_mat[t][s] = 时间 t 模型在任务 s 上的精度
    print('[diag] replaying merge & evaluating ...')
    for t in range(1, T + 1):
        lo, hi = ranges[t - 1]

        # 1) 与 incremental_train 相同的后处理: 载入 theta_t -> merge -> 重建 fc
        net.backbone.cur_adapter.load_state_dict(task_adapters[t].state_dict())
        net.merge_adapters()
        net.update_fc()
        proto_ds = data_manager.get_dataset(np.arange(lo, hi), source='train', mode='test')
        proto_ld = DataLoader(proto_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        net.replace_fc(proto_ds, proto_ld)

        merged = net.merged_adapter_list[-1]
        if not args.no_save_merged:
            torch.save(
                {'tasks': t, 'state_dict': merged.state_dict()},
                os.path.join(out_dir, 'merged', f'task{t}.pkl'),
            )

        # 2) 本任务测试集
        test_ld = test_loaders[t]
        labels = np.asarray(test_ld.dataset.labels)

        # 3) 遗忘矩阵 A[t,s] / A_tid[t,s]: 时间 t 的模型在全部已见任务 s<=t 上的精度
        A_row, Atid_row = {}, {}
        for s in range(1, t + 1):
            a_s, a_tid_s = eval_acmap_net(net, test_loaders[s], ranges, device)
            A_row[s] = a_s
            Atid_row[s] = a_tid_s
        A_mat[t] = A_row
        Atid_mat[t] = Atid_row

        # 4) 当前任务的 ACMap 真实推理口径(矩阵对角线)
        a_merge_acmap, a_merge_tid = A_row[t], Atid_row[t]

        # 5) 各适配器特征
        z_merge = extract_feats(net, test_ld, merged, device)
        z_sep = extract_feats(net, test_ld, task_adapters[t], device)

        # 6) A_sep: theta_t + 本任务原型(oracle)
        a_sep = float((cosine_pred(z_sep, *protos[t]) == labels).mean() * 100)

        # 7) A_route: 全部已见任务适配器 + 原型路由(无 task-id)
        zs, top_scores, top_classes = {}, {}, {}
        for s in range(1, t + 1):
            zs[s] = extract_feats(net, test_ld, task_adapters[s], device)
            sims = torch.nn.functional.normalize(zs[s], dim=1) @ torch.nn.functional.normalize(
                protos[s][1], dim=1
            ).T
            top_scores[s] = sims.max(dim=1).values
            top_classes[s] = sims.argmax(dim=1)
        n = len(labels)
        s_hat = np.argmax(np.stack([top_scores[s].numpy() for s in range(1, t + 1)], axis=1), axis=1) + 1
        preds_route = np.empty(n, dtype=np.int64)
        for i in range(n):
            s = int(s_hat[i])
            preds_route[i] = protos[s][0][top_classes[s][i].item()]
        a_route = float((preds_route == labels).mean() * 100)
        r_route = float((s_hat == t).mean() * 100)

        # 8) A_merge_fresh: 合并 adapter + 当前特征空间重算原型(去掉原型漂移)
        a_merge_fresh = float('nan')
        if not args.no_fresh_proto:
            all_ds = data_manager.get_dataset(np.arange(0, hi), source='train', mode='test')
            all_ld = DataLoader(all_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
            f_classes, f_mat = extract_protos(net, all_ds, all_ld, merged, device)
            a_merge_fresh = float((cosine_pred(z_merge, f_classes, f_mat) == labels).mean() * 100)

        # 9) LMC: 前一合并 adapter 与 theta_t 的线性路径(固定分类器 = protos[t])
        lmc = [float('nan')] * 5
        lmc_barrier = float('nan')
        if t > 1:
            prev_merged = net.merged_adapter_list[-2]
            accs = []
            for lam in [0.0, 0.25, 0.5, 0.75, 1.0]:
                ad = interp_adapter(prev_merged, task_adapters[t], lam, config, device)
                z = extract_feats(net, test_ld, ad, device)
                accs.append(float((cosine_pred(z, *protos[t]) == labels).mean()))
            lmc = accs
            err = [1.0 - a / 100.0 for a in accs]
            lmc_barrier = max(
                err[i] - ((1.0 - lam) * err[0] + lam * err[-1])
                for i, lam in enumerate([0.0, 0.25, 0.5, 0.75, 1.0])
            )

        # 10) 重叠/模长
        ov_cum = float(np.mean([ov[t, s] for s in range(1, t)])) if t > 1 else float('nan')
        cf_cum = float(np.mean([cf[t, s] for s in range(1, t)])) if t > 1 else float('nan')
        norm_task = fro_norm(task_adapters[t])
        norm_merged = fro_norm(merged)

        rows.append(
            {
                'task': t,
                'n_seen': hi,
                'A_sep': a_sep,
                'A_route': a_route,
                'R_route': r_route,
                'A_merge_acmap': a_merge_acmap,
                'A_merge_tid': a_merge_tid,
                'A_merge_fresh': a_merge_fresh,
                'LMC_0': lmc[0],
                'LMC_25': lmc[1],
                'LMC_50': lmc[2],
                'LMC_75': lmc[3],
                'LMC_100': lmc[4],
                'LMC_barrier': lmc_barrier,
                'overlap_avg_cum': ov_cum,
                'cos_flat_avg_cum': cf_cum,
                'norm_task': norm_task,
                'norm_merged': norm_merged,
            }
        )

        print(
            f'  task {t:>2}: A_sep={a_sep:6.2f} A_route={a_route:6.2f} (R={r_route:5.2f}) '
            f'A_merge={a_merge_acmap:6.2f} A_tid={a_merge_tid:6.2f} '
            f'A_fresh={a_merge_fresh if np.isnan(a_merge_fresh) else round(a_merge_fresh, 2)} '
            f'overlap={ov_cum if np.isnan(ov_cum) else round(ov_cum, 3)} '
            f'barrier={lmc_barrier if np.isnan(lmc_barrier) else round(lmc_barrier, 3)}'
        )

        # 与 after_task 相同地推进
        net.freeze()
        dctx.next_task()

    finalize_results(rows, out_dir, ov=ov, cf=cf)
    write_amatrices(out_dir, A_mat, Atid_mat, T)


if __name__ == '__main__':
    main()
