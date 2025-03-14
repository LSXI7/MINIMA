# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
from tools.infrared.scepter.modules.utils.file_system import FS


def init_env(cfg_general):
    work_dir = cfg_general.WORK_DIR
    file_system = cfg_general.get('FILE_SYSTEM', None)
    if file_system is not None:
        if isinstance(file_system, list):
            for file_sys in file_system:
                _prefix = FS.init_fs_client(file_sys)
        elif file_system is not None:
            _prefix = FS.init_fs_client(file_system)  # noqa
    is_flag = FS.make_dir(work_dir)
    assert is_flag
    return cfg_general


def get_available_memory():
    import psutil
    mem = psutil.virtual_memory()
    return {'total': mem.total, 'available': mem.available}
