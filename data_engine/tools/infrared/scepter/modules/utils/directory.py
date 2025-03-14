# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import hashlib
import os.path as osp


def osp_path(prefix, data_file):
    if data_file.startswith(prefix):
        return data_file
    else:
        return osp.join(prefix, data_file)


def get_relative_folder(abs_path, keep_index=-1):
    path_tup = abs_path.split('/')[:keep_index]
    return '/'.join(path_tup)


def get_md5(ori_str):
    md5 = hashlib.md5(ori_str.encode()).hexdigest()
    return md5
