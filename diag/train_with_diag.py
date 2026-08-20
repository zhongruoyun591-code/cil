# -*- coding: utf-8 -*-
"""
C1 诊断(在线模式): 训练过程中直接计算全部诊断指标,不保存任何 checkpoint。
训练逻辑与官方 train.py 完全一致(使用修复后的 Context),差别只有两点:
  1) 不调用 model.save_checkpoint(...) —— 磁盘零 checkpoint;
  2) 每个任务训练完成(merge + replace_fc 之后)立即在内存中运行诊断钩子。

用法(与官方训练相同的参数 + 诊断参数):
  python diag/train_with_diag.py --config exps/imagenet-r.yaml --init_cls 0 --increment 20 --seed 1993

输出(默认 diag_out/):
  diag_results.csv / overlap_matrix.csv / cos_matrix.csv
"""

import argparse
import copy
import os

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from diag_common import (
    NullLogger,
    build_config,
    extract_feats,
    extract_protos,
    cosine_pred,
    interp_adapter,
    fro_norm,
    pairwise_overlaps,
    eval_acmap_net,
    finalize_results,
    write_amatrices,
)

from acmap.utils import factory
from acmap.utils.context import Context
from acmap.utils.data_manager import DataManager
from acmap.utils.toolkit import set_random


def diag_hook(
    model, data_manager, context, task, ranges, task_adapters, protos,
    ov_mat, cf_mat, A_mat, Atid_mat, test_loaders, args, config, device,
):
    """在每个任务训练完成后调用,计算并缓存该任务的诊断指标,返回一行 CSV 数据。"""
    net = model.network
    lo, hi = context.known_classes, context.total_classes
    ranges.append((lo, hi))

    # 保存本任务适配器的内存副本(后续任务的路由/LMC/重叠要用)
    task_adapters[task] = copy.deepcopy(net.backbone.cur_adapter)

    batch_size = args.batch_size if args.batch_size else config.exp.batch_size
    num_workers = config.exp.num_workers

    # ---- 本任务测试集(缓存 loader,后续任务回看时复用) ----
    if task not in test_loaders:
        t_ds = data_manager.get_dataset(np.arange(lo, hi), source='test', mode='test')
        test_loaders[task] = DataLoader(t_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_ld = test_loaders[task]
    labels = np.asarray(test_ld.dataset.labels)

    # ---- 遗忘矩阵 A[t,s] / A_tid[t,s]: 时间 t 模型在全部已见任务 s<=t 上的精度 ----
    A_row, Atid_row = {}, {}
    for s in range(1, task + 1):
        a_s, a_tid_s = eval_acmap_net(net, test_loaders[s], ranges, device)
        A_row[s] = a_s
        Atid_row[s] = a_tid_s
    A_mat[task] = A_row
    Atid_mat[task] = Atid_row

    # ---- A_merge(当前任务口径,矩阵对角线) ----
    a_merge_acmap, a_merge_tid = A_row[task], Atid_row[task]

    # ---- 特征 ----
    merged = net.merged_adapter_list[-1]
    z_merge = extract_feats(net, test_ld, merged, device)
    z_sep = extract_feats(net, test_ld, net.backbone.cur_adapter, device)

    # ---- 本任务训练原型(用 theta_t) ----
    proto_ds = data_manager.get_dataset(np.arange(lo, hi), source='train', mode='test')
    proto_ld = DataLoader(proto_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    protos[task] = extract_protos(net, proto_ds, proto_ld, net.backbone.cur_adapter, device)

    # ---- A_sep: theta_t + 本任务原型(oracle) ----
    a_sep = float((cosine_pred(z_sep, *protos[task]) == labels).mean() * 100)

    # ---- A_route: 全部已见任务适配器 + 原型路由(无 task-id) ----
    zs, top_scores, top_classes = {}, {}, {}
    for s in range(1, task + 1):
        zs[s] = extract_feats(net, test_ld, task_adapters[s], device)
        sims = F.normalize(zs[s], dim=1) @ F.normalize(protos[s][1], dim=1).T
        top_scores[s] = sims.max(dim=1).values
        top_classes[s] = sims.argmax(dim=1)
    n = len(labels)
    s_hat = np.argmax(np.stack([top_scores[s].numpy() for s in range(1, task + 1)], axis=1), axis=1) + 1
    preds_route = np.empty(n, dtype=np.int64)
    for i in range(n):
        s = int(s_hat[i])
        preds_route[i] = protos[s][0][top_classes[s][i].item()]
    a_route = float((preds_route == labels).mean() * 100)
    r_route = float((s_hat == task).mean() * 100)

    # ---- A_merge_fresh: 合并 adapter + 当前特征空间重算原型 ----
    a_merge_fresh = float('nan')
    if not args.no_fresh_proto:
        all_ds = data_manager.get_dataset(np.arange(0, hi), source='train', mode='test')
        all_ld = DataLoader(all_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        f_classes, f_mat = extract_protos(net, all_ds, all_ld, merged, device)
        a_merge_fresh = float((cosine_pred(z_merge, f_classes, f_mat) == labels).mean() * 100)

    # ---- LMC: 前一合并 adapter 与 theta_t 的线性路径(固定分类器 = protos[task]) ----
    lmc = [float('nan')] * 5
    lmc_barrier = float('nan')
    if task > 1:
        prev_merged = net.merged_adapter_list[-2]
        accs = []
        for lam in [0.0, 0.25, 0.5, 0.75, 1.0]:
            ad = interp_adapter(prev_merged, task_adapters[task], lam, config, device)
            z = extract_feats(net, test_ld, ad, device)
            accs.append(float((cosine_pred(z, *protos[task]) == labels).mean()))
        lmc = accs
        err = [1.0 - a / 100.0 for a in accs]
        lmc_barrier = max(
            err[i] - ((1.0 - lam) * err[0] + lam * err[-1])
            for i, lam in enumerate([0.0, 0.25, 0.5, 0.75, 1.0])
        )

    # ---- 重叠(与所有旧任务)与模长 ----
    ov_cum = float('nan')
    cf_cum = float('nan')
    for s in range(1, task):
        ov_mat[task, s], cf_mat[task, s] = pairwise_overlaps(task_adapters[task], task_adapters[s], config)
        ov_mat[s, task], cf_mat[s, task] = ov_mat[task, s], cf_mat[task, s]
    if task > 1:
        ov_cum = float(ov_mat[task, 1:task].mean())
        cf_cum = float(cf_mat[task, 1:task].mean())
    norm_task = fro_norm(task_adapters[task])
    norm_merged = fro_norm(merged)

    row = {
        'task': task,
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

    print(
        f'  [diag] task {task:>2}: A_sep={a_sep:6.2f} A_route={a_route:6.2f} (R={r_route:5.2f}) '
        f'A_merge={a_merge_acmap:6.2f} A_tid={a_merge_tid:6.2f} '
        f'A_fresh={a_merge_fresh if np.isnan(a_merge_fresh) else round(a_merge_fresh, 2)} '
        f'overlap={ov_cum if np.isnan(ov_cum) else round(ov_cum, 3)} '
        f'barrier={lmc_barrier if np.isnan(lmc_barrier) else round(lmc_barrier, 3)}'
    )
    return row


def main():
    parser = argparse.ArgumentParser(description='ACMap 训练 + C1 在线诊断(不保存 checkpoint)')
    parser.add_argument('--config', type=str, default=os.path.join('exps', 'cifar.yaml'))
    parser.add_argument('--init_cls', type=int, default=0)
    parser.add_argument('--increment', type=int, default=5)
    parser.add_argument('--seed', type=int, default=1993)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--logger', type=str, default='basic')
    parser.add_argument('--prefix', type=str, default='')
    parser.add_argument('--debug', action='store_true', help='本脚本本来就不保存 checkpoint,此参数仅为兼容 Config')
    parser.add_argument('--dataset_dir', type=str, default='dataset')
    parser.add_argument('--ckpts_dir', type=str, default=os.path.join('data', 'acmap', 'ckpts'))
    parser.add_argument(
        '--out_dir', type=str, default=None,
        help='输出目录(默认 diag_out/<dataset>-b<init>i<inc>-s<seed>,不同运行互不覆盖)',
    )
    parser.add_argument('--max_tasks', type=int, default=None, help='只训练并诊断前 K 个任务')
    parser.add_argument('--batch_size', type=int, default=None, help='评估 batch size(默认用配置)')
    parser.add_argument('--no_fresh_proto', action='store_true', help='跳过合并适配器下重算原型(加速)')
    args = parser.parse_args()

    config = build_config(args)
    set_random(config.seed)
    device = config.device

    out_dir = args.out_dir or os.path.join(
        'diag_out', f'{config.exp.dataset}-b{config.init_cls}-i{config.increment}-s{config.seed}'
    )

    data_manager = DataManager(
        dataset_name=config.exp.dataset,
        shuffle=config.exp.shuffle,
        seed=config.seed,
        dataset_dir=config.dataset_dir,
    )

    context = Context(config=config, logger=NullLogger(), class_order=data_manager.class_order)
    model = factory.get_model(context=context)
    model.network.to(device)
    print(f'[diag-train] dataset={config.exp.dataset} num_tasks={context.num_tasks} '
          f'increments={context.increments[:3]}...{context.increments[-2:]}')

    rows, ranges = [], []
    task_adapters, protos = {}, {}
    test_loaders, A_mat, Atid_mat = {}, {}, {}
    ov_mat = np.zeros((context.num_tasks + 1, context.num_tasks + 1))
    cf_mat = np.zeros_like(ov_mat)

    for task in range(1, context.num_tasks + 1):
        if args.max_tasks and task > args.max_tasks:
            break

        print(f'[diag-train] Task {task}/{context.num_tasks} '
              f'(classes {context.known_classes}-{context.total_classes - 1}) training ...')

        # 与官方 train.py 相同的训练步骤(内部完成 merge + replace_fc)
        model.incremental_train(data_manager=data_manager)

        # 官方口径的全已见类评估(与 train.py 打印的 top-1 一致)
        cnn_accy, _ = model.eval_task()
        print(f'[diag-train] Task {task} official top-1 (all seen classes): {cnn_accy["top1"]}')

        # 诊断钩子(在 after_task 之前,此时 cur_adapter 仍是 theta_t)
        row = diag_hook(
            model, data_manager, context, task, ranges, task_adapters, protos,
            ov_mat, cf_mat, A_mat, Atid_mat, test_loaders, args, config, device,
        )
        rows.append(row)

        model.after_task()

    finalize_results(rows, out_dir, ov=ov_mat, cf=cf_mat)
    T = len(rows)
    write_amatrices(out_dir, A_mat, Atid_mat, T)


if __name__ == '__main__':
    main()
