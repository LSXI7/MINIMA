# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import logging
import os
import os.path as osp
import shutil
from typing import Optional, Union

from tools.infrared.scepter.modules.utils.config import dict_to_yaml
from tools.infrared.scepter.modules.utils.file_clients.base_fs import BaseFs
from tools.infrared.scepter.modules.utils.file_clients.registry import FILE_SYSTEMS


def is_start_of_line(f, position, delimiter='\n'):
    if position == 0:
        return True
    # Check whether the previous character is EOL
    f.seek(position - 1)
    return f.read(1) == delimiter


def get_next_line_position(f, position):
    # Read the current line till the end
    f.seek(position)
    f.readline()
    # Return a position after reading the line
    return f.tell()


@FILE_SYSTEMS.register_class()
class LocalFs(BaseFs):
    def __init__(self, cfg, logger=None):
        super(LocalFs, self).__init__(cfg, logger=logger)
        self._fs_prefix = os.path.abspath(os.curdir)

    def get_prefix(self) -> str:
        return self._fs_prefix

    def reconstruct_path(self, target_path) -> str:
        if target_path.startswith(self.get_prefix()):
            return target_path
        if target_path.startswith('./') or target_path.startswith('../'):
            return os.path.join(self.get_prefix(),
                                target_path).replace('/./',
                                                     '/').replace('/../', '/')
        if target_path.startswith('/'):
            return target_path
        if target_path.startswith('file://'):
            return os.path.join(self.get_prefix(),
                                target_path[len('file://'):])
        return os.path.join(self.get_prefix(), target_path)

    def support_write(self) -> bool:
        return True

    def support_link(self) -> bool:
        return True

    def map_to_local(self, target_path) -> (str, bool):
        target_path = self.reconstruct_path(target_path)
        return target_path, False

    def get_object_to_local_file(self,
                                 target_path,
                                 local_path=None,
                                 wait_finish=False) -> Optional[str]:
        target_path = self.reconstruct_path(target_path)
        if local_path is not None:
            local_path = self.reconstruct_path(local_path)
            local_path = osp.abspath(local_path)
            if local_path != target_path:
                # copy target_path to local_path
                os.makedirs(osp.dirname(local_path), exist_ok=True)
                try:
                    shutil.copy(target_path, local_path)
                except Exception as e:
                    self.logger.info(f'Copy file failed {e}')
                    return None

            return local_path
        return target_path

    def get_dir_to_local_dir(self,
                             target_path,
                             local_path=None,
                             wait_finish=False,
                             multi_thread=False,
                             timeout=3600,
                             sign_key=None,
                             worker_id=0) -> Optional[str]:
        if not self.isdir(target_path):
            self.logger.info(
                f"{target_path} is not directory or doesn't exist.")
        if not target_path.endswith('/'):
            target_path += '/'
        if local_path is None:
            local_path, is_tmp = self.map_to_local(target_path)
        else:
            is_tmp = False
        local_path = local_path.replace('/./', '/')
        os.makedirs(local_path, exist_ok=True)
        generator = self.walk_dir(target_path)
        for file_name in generator:
            if file_name == target_path or file_name == target_path + '/':
                continue
            local_file_name = os.path.join(
                local_path,
                file_name.split(target_path)[-1]).replace('/./', '/')
            if not self.isdir(file_name):
                self.get_object_to_local_file(file_name,
                                              local_file_name,
                                              wait_finish=wait_finish)
            else:
                self.get_dir_to_local_dir(file_name,
                                          local_file_name,
                                          wait_finish=wait_finish)
        if is_tmp:
            self.add_temp_file(local_path)
        return local_path

    def get_object(self, target_path) -> Optional[bytes]:
        target_path = self.reconstruct_path(target_path)
        try:
            local_data = open(target_path, 'rb').read()
        except Exception as e:
            self.logger.error(f'Read {target_path} error {e}')
            local_data = None
        return local_data

    def get_object_stream(
            self,
            target_path,
            start,
            size=10000,
            delimiter=None) -> (Union[bytes, str, None], Optional[int]):
        target_path = self.reconstruct_path(target_path)
        if not osp.exists(target_path):
            self.logger.error(f'Read {target_path} error: file not exist.')
            return None, None

        file_size = os.path.getsize(target_path)
        start, end = start, min(file_size, start + size)
        if start >= end - 1:
            return None, None

        with open(target_path, 'rb') as f:
            if delimiter is None or end == file_size:
                f.seek(start)
                local_data = f.read(end - start)
                return local_data, end
            f.seek(start)
            local_data = f.read(end - start)
            try:
                total_len = len(local_data)
                offset = 0
                try:
                    sp_data = local_data.split(bytes(delimiter, 'utf-8'))
                # if failed, suppose the bytes is splited error.
                except Exception as e:
                    self.logger.info(
                        f'Return data split error,please check your delimiter {e}'
                    )
                    return None, end
                if not len(sp_data[-1]) == len(local_data):
                    cur_offset = len(sp_data[-1])
                    offset += cur_offset
                    local_data = local_data[:total_len - cur_offset]
                    local_data = local_data[len(sp_data[0]):]
                end = end - offset
            except Exception as e:
                self.logger.info(f'Local data decode error {e}')
            return local_data, end

    def get_object_chunk_list(self,
                              target_path,
                              chunk_num=-1,
                              chunk_size=-1,
                              delimiter=None) -> Optional[list]:
        target_path = self.reconstruct_path(target_path)
        if not osp.exists(target_path):
            self.logger.error(f'Read {target_path} error: file not exist.')
            return None
        file_size = os.path.getsize(target_path)
        if chunk_size < 0 and chunk_num < 0:
            self.logger.error(
                'Suppose chunk size > 0 or chunk num > 0, instead of both < 0')
            return None
        if chunk_size < 0 and chunk_num > 0:
            chunk_size = file_size // chunk_num + 1
        chunk_st_et = []
        # Don't care of the lines info
        if delimiter is None:
            chunk_start = 0
            # Iterate over all chunks and construct arguments for `process_chunk`
            while chunk_start < file_size:
                chunk_end = min(file_size, chunk_start + chunk_size)
                chunk_st_et.append([chunk_start, chunk_end - chunk_start])
                chunk_start = chunk_end
        else:
            with open(target_path, 'rb') as f:
                chunk_start = 0
                offset = 0
                # Iterate over all chunks and construct arguments for `process_chunk`
                while chunk_start < file_size:
                    chunk_end = min(file_size,
                                    chunk_start + chunk_size + offset)
                    quota_st = max(0, chunk_end - 20000)
                    quota_st = max(chunk_start, quota_st)
                    f.seek(quota_st)
                    local_data = f.read(chunk_end - quota_st)
                    offset = 0
                    if not chunk_end == file_size:
                        try:
                            try:
                                sp_data = local_data.split(
                                    bytes(delimiter, 'utf-8'))
                            # if failed, suppose the bytes is splited error.
                            except Exception as e:
                                self.logger.info(
                                    f'Return data split error,please check your delimiter {e}'
                                )
                                return None
                            if not len(sp_data[-1]) == len(local_data):
                                cur_offset = len(sp_data[-1])
                                offset += cur_offset
                            chunk_end = chunk_end - offset
                        except Exception as e:
                            self.logger.info(
                                f'Local data decode error {e}, check your data is supported by str.decode().'
                            )
                        chunk_end = chunk_end - offset
                    chunk_st_et.append([chunk_start, chunk_end - chunk_start])
                    chunk_start = chunk_end
        return chunk_st_et

    def put_object(self, local_data, target_path) -> bool:
        target_path = self.reconstruct_path(target_path)
        with open(target_path, 'w') as f:
            f.write(local_data)
        return True

    def walk_dir(self, file_dir, recurse=True):
        for root, dirs, files in os.walk(file_dir, topdown=recurse):
            sub_files = files + dirs
            for name in sub_files:
                yield os.path.join(root, name)

    def put_object_from_local_file(self, local_path, target_path) -> bool:
        target_path = self.reconstruct_path(target_path)
        local_path = self.reconstruct_path(local_path)
        if local_path != target_path:
            try:
                shutil.copy(local_path, target_path)
            except Exception:
                return False
        return True

    def get_url(self,
                target_path,
                set_public=False,
                skip_check=False,
                lifecycle=3600 * 100):
        return target_path

    def make_dir(self, target_dir) -> bool:
        target_dir = self.reconstruct_path(target_dir)
        if osp.exists(target_dir):
            if osp.isfile(target_dir):
                self.logger.error(f'{target_dir} already exists as a file!')
                return False
            return True
        try:
            os.makedirs(target_dir)
        except Exception as e:
            self.logger.error(e)
            return False
        return True

    def make_link(self, target_link_path, target_path) -> bool:
        target_path = self.reconstruct_path(target_path)
        target_link_path = self.reconstruct_path(target_link_path)
        try:
            if osp.lexists(target_link_path):
                os.remove(target_link_path)
            os.symlink(target_path, target_link_path)
            return True
        except Exception:
            return False

    def remove(self, target_path) -> bool:
        target_path = self.reconstruct_path(target_path)
        if osp.exists(target_path):
            try:
                os.remove(target_path)
            except Exception:
                return False
        return True

    def get_logging_handler(self, target_logging_path):
        dirname = os.path.dirname(target_logging_path)
        if not os.path.exists(dirname):
            os.makedirs(dirname, exist_ok=True)
        return logging.FileHandler(target_logging_path)

    def put_dir_from_local_dir(self,
                               local_dir,
                               target_dir,
                               multi_thread=False) -> bool:
        local_dir = self.reconstruct_path(local_dir)
        target_dir = self.reconstruct_path(target_dir)
        if local_dir == target_dir:
            return True
        # # cp -f local_dir/* target_dir/*
        # if not osp.exists(target_dir):
        #     status = os.system(f'mkdir -p {target_dir}')
        #     if status != 0:
        #         return False
        try:
            shutil.copytree(local_dir, target_dir, symlinks=True)
        except Exception:
            return False
        return True

    def size(self, target_path) -> Optional[int]:
        target_path = self.reconstruct_path(target_path)
        if not osp.exists(target_path):
            self.logger.info(f"File {target_path} doesn't exist.")
            return -1
        file_size = os.path.getsize(target_path)
        return file_size

    def exists(self, target_path) -> bool:
        target_path = self.reconstruct_path(target_path)
        return osp.exists(target_path)

    def isfile(self, target_path) -> bool:
        target_path = self.reconstruct_path(target_path)
        return osp.isfile(target_path)

    def isdir(self, target_path) -> bool:
        target_path = self.reconstruct_path(target_path)
        return osp.isdir(target_path)

    @staticmethod
    def get_config_template():
        return dict_to_yaml('FILE_SYSTEMS',
                            __class__.__name__, {},
                            set_name=True)
