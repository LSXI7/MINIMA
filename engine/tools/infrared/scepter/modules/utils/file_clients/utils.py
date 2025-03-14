# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import os
import os.path as osp


def check_if_local_path(path):
    """
    Check if path is a local path, no matter file or directory.

    True:
        file:///home/admin/a.txt (standard path)
        /home/admin/a.txt (standard unix path)
        C:\\Users\\a.txt (standard windows path)
        C:/Users/a.txt (works as well)
        ./a.txt (relative path)
        a.txt (relative path)
    False:
        http://www.aliyun.com/a.txt
        http://www.aliyun.com/a.txt
        oss://aliyun/a.txt

    Args:
        path (str):

    Returns:
        True if path is a local path.
    """
    if path.startswith('file://'):
        return True
    return '://' not in path


def remove_temp_path(path):
    """
    Delete local temp path.

    Args:
        path (str):

    Returns:
    """
    if not osp.exists(path):
        return
    if not osp.isfile(path):
        return
    try:
        os.remove(path)
    except Exception:
        pass
    # warnings.warn(f"remove {path}")
