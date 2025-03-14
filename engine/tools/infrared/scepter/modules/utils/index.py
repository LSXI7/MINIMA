# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import os.path as osp

from tqdm import tqdm

from tools.infrared.scepter.modules.utils.file_system import FS


def init_1level_llfs(list_file,
                     max_lines=1024,
                     index_name='index',
                     delimiter='\n'):
    r"""Construct large-list-file index.
    """
    index_dir = osp.splitext(list_file)[0]
    print(list_file)
    with FS.get_from(list_file, wait_finish=True) as local_path:
        print(local_path)
        _num_split, index_stack, save_files = 0, [], []
        with open(local_path, 'r', buffering=1000000) as f:
            for line in tqdm(f):
                index_stack.append(line.strip())
                if len(index_stack) >= max_lines:
                    save_file = f'{index_dir}/{index_name}/{_num_split + 1:09d}.txt'
                    with FS.put_to(save_file) as cache_path:
                        with open(cache_path, 'w') as f_w:
                            f_w.write('\n'.join(index_stack))
                    index_stack = []
                    _num_split += 1
                    save_files.append(save_file)

    if len(index_stack) > 0:
        save_file = f'{index_dir}/{index_name}/{_num_split + 1:06d}.txt'
        with FS.put_to(save_file) as cache_path:
            with open(cache_path, 'w') as f_w:
                f_w.write('\n'.join(index_stack))
        save_files.append(save_file)

    # output meta-file
    index_file = osp.join(index_dir, f'{index_name}.txt')
    with FS.put_to(index_file) as cache_path:
        with open(cache_path, 'w') as f_w:
            f_w.write('\n'.join(save_files))
    return index_file
