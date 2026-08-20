# -*- coding: utf-8 -*-
"""
验证 ACMap 官方 context.py 的 Context.next_task() 是否存在 off-by-one。
方法: 逐字提取官方源码中的 Context 类并执行(只注入 Config/Logger 桩),
      原样模拟 train.py 的主循环,打印每次迭代实际训练/评估的类范围。

用法: python diag/verify_context_fix.py
"""

import os
import re
import sys

sys.stdout.reconfigure(encoding='utf-8')

CONTEXT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src', 'acmap', 'utils', 'context.py')


class ConfigStub:
    def __init__(self, init_cls, increment):
        self.init_cls = init_cls
        self.increment = increment


class LoggerStub:
    pass


def load_context_class(source: str, variant: str):
    """执行官方 Context 类源码。
    variant='buggy'  : 注入修复前的 off-by-one 条件(官方原始行为,历史对照);
    variant='fixed'  : 使用修复后的条件(即当前仓库代码)。
    """
    cls_src = 'class Context' + source.split('class Context')[1]

    if variant == 'buggy':
        cls_src = cls_src.replace('self.cur_task < self.num_tasks:', 'self.cur_task < self.num_tasks - 1:')
        assert 'self.cur_task < self.num_tasks - 1:' in cls_src, 'buggy 注入未生效'
    elif variant == 'fixed':
        cls_src = cls_src.replace('self.cur_task < self.num_tasks - 1:', 'self.cur_task < self.num_tasks:')
    else:
        raise ValueError(variant)

    ns = {'Config': ConfigStub, 'Logger': LoggerStub, 'List': list}
    exec(compile(cls_src, '<official context.py>', 'exec'), ns)
    return ns['Context']


def simulate(ContextCls, init_cls, increment, n_classes):
    config = ConfigStub(init_cls, increment)
    ctx = ContextCls(config=config, logger=LoggerStub(), class_order=list(range(n_classes)))

    print(f'\n--- init_cls={init_cls}, increment={increment}, {n_classes} classes, '
          f'num_tasks={ctx.num_tasks}, increments={ctx.increments} ---')

    trained, evaluated, duplicated = set(), set(), []
    prev = None
    for task in range(1, ctx.num_tasks + 1):
        lo, hi = ctx.known_classes, ctx.total_classes
        rng = tuple(range(lo, hi))
        print(f'  iteration {task:>2}: cur_task={ctx.cur_task:>2} train/eval classes {lo:>3}-{hi-1:>3} '
              f'(size {hi-lo:>2})  {"<-- 与上一次相同!" if rng == prev else ""}')
        if rng == prev:
            duplicated.append(task)
        trained.update(rng)
        evaluated.update(rng)
        prev = rng
        ctx.next_task()  # 等价于 train.py 的 model.after_task()

    missing = sorted(set(range(n_classes)) - trained)
    print(f'  => 共训练 {len(trained)}/{n_classes} 类; 未训练类: {missing if missing else "无"}')
    print(f'  => 重复迭代: {duplicated if duplicated else "无"}')
    return missing, duplicated


def main():
    with open(CONTEXT_PATH, encoding='utf-8') as f:
        source = f.read()

    print('=' * 70)
    print('buggy 变体(官方原始行为, 注入旧条件重建)')
    print('=' * 70)
    for init_cls, inc in [(0, 5), (20, 20)]:
        simulate(load_context_class(source, 'buggy'), init_cls, inc, 100)

    print('\n' + '=' * 70)
    print('修复后(当前仓库代码, cur_task < num_tasks)')
    print('=' * 70)
    for init_cls, inc in [(0, 5), (20, 20)]:
        simulate(load_context_class(source, 'fixed'), init_cls, inc, 100)


if __name__ == '__main__':
    main()
