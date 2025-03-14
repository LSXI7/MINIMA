# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import io
import os
import threading
import time
import warnings
from contextlib import contextmanager
from queue import Queue

from tools.infrared.scepter.modules.utils.config import Config
from tools.infrared.scepter.modules.utils.file_clients.base_fs import BaseFs
from tools.infrared.scepter.modules.utils.file_clients.local_fs import LocalFs
from tools.infrared.scepter.modules.utils.file_clients.registry import FILE_SYSTEMS
from tools.infrared.scepter.modules.utils.file_clients.utils import check_if_local_path


class IoString(str):
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class IoBytes(bytes):
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class ReadException(Exception):
    pass


class WriteException(Exception):
    pass


class FileSystem(object):
    def __init__(self):
        self._prefix_to_clients = {}
        self._default_client = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def __del__(self):
        for k, client in self._prefix_to_clients.items():
            client.clear()

    @property
    def support_prefix(self):
        return self._prefix_to_clients

    def init_fs_client(self, cfg=None, logger=None, overwrite=True):
        """ Initialize file system backend
        Supported backend:
            1. Local file system, e.g. /home/admin/work_dir, work_dir_bk/imagenet_pretrain
            2. Aliyun Oss, e.g. oss://bucket_name/work_dir
            3. Http, only support to read content, e.g.
                https://www.google.com.hk/images/branding/googlelogo/2x/googlelogo_color_272x92dp.png
            4. other fs backend...

        Args:
            cfg (list, dict, optional):
                list: list of file system configs to be initialized
                dict: a dict contains file system configs as values or a file system config dict
                optional: Will only use default LocalFs
        """

        fs_cfg = cfg or Config(load=False)
        if not isinstance(fs_cfg, Config):
            raise '{} is not a Config Instance!'.format(fs_cfg)

        if not fs_cfg.have('NAME'):
            raise KeyError(f'{fs_cfg} does not contain key NAME!')

        fs_client = FILE_SYSTEMS.build(fs_cfg, logger=logger)
        _prefix = fs_client.get_prefix()
        if _prefix in self._prefix_to_clients and not overwrite:
            return _prefix
        if _prefix in self._prefix_to_clients:
            warnings.warn(
                'File client {} has already been set, will be replaced by newer config.'
                .format(_prefix))
        self._prefix_to_clients[_prefix] = fs_client
        return _prefix

    def get_fs_client(self, target_path, safe=False) -> BaseFs:
        """ Get the client by input path.
        Every file system has its own identifier, default will use local file system to have a try.
        If copy needed, only do shallow copy.

        Args:
            target_path (str):
            safe (bool): In safe mode, get the copy of the client.
        """
        obj = None

        for prefix in sorted(list(self._prefix_to_clients.keys()),
                             key=lambda a: -len(a)):
            if target_path.startswith(prefix):
                obj = self._prefix_to_clients[prefix]
                break
        if obj is not None:
            if safe:
                return obj.copy()
            else:
                return obj

        if not check_if_local_path(target_path):
            warnings.warn(
                f'{target_path} is not a local path, use LocalFs may cause an error.'
            )
        if self._default_client is None:
            self._default_client = LocalFs(Config(load=False))
        if safe:
            return self._default_client.copy()
        else:
            return self._default_client

    def get_dir_to_local_dir(self,
                             target_path,
                             local_path=None,
                             wait_finish=False,
                             timeout=3600,
                             multi_thread=False,
                             sign_key=None,
                             worker_id=0):
        with self.get_fs_client(target_path) as client:
            local_path = client.get_dir_to_local_dir(target_path,
                                                     local_path=local_path,
                                                     wait_finish=wait_finish,
                                                     timeout=timeout,
                                                     multi_thread=multi_thread,
                                                     sign_key=sign_key,
                                                     worker_id=worker_id)
            if local_path is None:
                raise ReadException(
                    f'Failed to fetch {target_path} to {local_path}')
            return IoString(local_path)

    def add_target_local_map(self, target_dir, local_dir):
        """ Map target directory to local file system directory

        Args:
            target_dir (str): Target directory.
            local_dir (str): Directory in local file system.
        """
        with self.get_fs_client(target_dir, safe=False) as client:
            client.add_target_local_map(target_dir, local_dir)

    def make_dir(self, target_dir):
        """ Make a directory.
                If target_dir is already exists, return True.

                Args:
                    target_dir (str):

                Returns:
                    True if target_dir exists or created.
                """
        with self.get_fs_client(target_dir) as client:
            return client.make_dir(target_dir)

    def exists(self, target_path):
        """ Check if target_path exists.

        Args:
            target_path (str):

        Returns:
            Bool.
        """
        with self.get_fs_client(target_path) as client:
            return client.exists(target_path)

    def map_to_local(self, target_path):
        """ Map target path to local file path. (NO IO HERE).

        Args:
            target_path (str): Target file path.

        Returns:
            A local path and a flag indicates if the local path is a temporary file.
        """
        with self.get_fs_client(target_path) as client:
            local_path, is_tmp = client.map_to_local(target_path)
            return local_path, is_tmp

    def put_dir_from_local_dir(self,
                               local_dir,
                               target_dir,
                               multi_thread=False):
        """ Upload all contents in local_dir to target_dir, keep the file tree.

        Args:
            local_dir (str):
            target_dir (str):

        Returns:
            Bool.
        """
        with self.get_fs_client(target_dir) as client:
            return client.put_dir_from_local_dir(local_dir,
                                                 target_dir,
                                                 multi_thread=multi_thread)

    def walk_dir(self, target_dir, recurse=True):
        """ Iterator to access the files of target dir.
                Args:
                    target_dir (str):

                Returns:
                    Generator.
                """
        with self.get_fs_client(target_dir) as client:
            return client.walk_dir(target_dir, recurse=recurse)

    def is_local_client(self, target_path) -> bool:
        """ Check if the client support read or write to target_path is a LocalFs.

        Args:
            target_path (str):

        Returns:
            Bool.
        """
        with self.get_fs_client(target_path) as client:
            return type(client) is LocalFs

    def put_object_from_local_file(self, local_path, target_path) -> bool:
        with self.get_fs_client(target_path) as client:
            flag = client.put_object_from_local_file(local_path, target_path)
            return flag

    def get_from(self, target_path, local_path=None, wait_finish=False):
        with self.get_fs_client(target_path) as client:
            local_path = client.get_object_to_local_file(
                target_path, local_path=local_path, wait_finish=wait_finish)
            if local_path is None:
                raise ReadException(
                    f'Failed to fetch {target_path} to {local_path}')
            return IoString(local_path)

    def get_url(self,
                target_path,
                set_public=False,
                skip_check=False,
                lifecycle=3600 * 100):
        with self.get_fs_client(target_path) as client:
            output_url = client.get_url(target_path,
                                        set_public=set_public,
                                        skip_check=skip_check,
                                        lifecycle=lifecycle)
            return output_url

    def get_object(self, target_path):
        with self.get_fs_client(target_path) as client:
            local_data = client.get_object(target_path)
            if local_data is None:
                return IoBytes(b'')
            return IoBytes(local_data)

    def put_object(self, local_data, target_path):
        with self.get_fs_client(target_path) as client:
            flg = client.put_object(local_data, target_path)
            return flg

    def delete_object(self, target_path):
        with self.get_fs_client(target_path) as client:
            if self.isfile(target_path):
                flg = client.remove(target_path)
                return flg
            else:
                return False

    def get_batch_objects_from(self, target_path_list, wait_finish=False):
        data_quene = Queue()
        batch_size = 20
        R = threading.Lock()

        def get_one_object(target_path_list):
            for target_path in target_path_list:
                if self.exists(target_path):
                    local_path = self.get_from(target_path,
                                               wait_finish=wait_finish)
                else:
                    local_path = None
                R.acquire()
                try:
                    data_quene.put_nowait([target_path, local_path])
                except Exception:
                    R.release()
                R.release()

        while True:
            batch_list = target_path_list[:4 * batch_size]
            if len(batch_list) < 1:
                break
            target_path_list = target_path_list[4 * batch_size:]
            threading_list = []
            for i in range(batch_size):
                t = threading.Thread(target=get_one_object,
                                     args=(batch_list[i::batch_size], ))
                t.daemon = True
                t.start()
                threading_list.append(t)
            [threading_t.join() for threading_t in threading_list]
            file_dict = {}
            while not data_quene.empty():
                target_path, local_path = data_quene.get_nowait()
                file_dict[target_path] = local_path

            for target_path in batch_list:
                local_path = file_dict.get(target_path, None)
                yield local_path

    def put_batch_objects_to(self,
                             local_path_list,
                             target_path_list,
                             batch_size=20,
                             wait_finish=False):
        data_quene = Queue()
        R = threading.Lock()

        def put_one_object(local_path_list, target_path_list):
            for local_path, target_path in zip(local_path_list,
                                               target_path_list):
                if local_path is None or target_path is None:
                    flg = False
                elif isinstance(local_path, io.BytesIO):
                    flg = FS.put_object(local_path.getvalue(), target_path)
                elif isinstance(local_path, bytes):
                    flg = FS.put_object(local_path, target_path)
                elif self.exists(local_path):
                    local_cache = self.get_from(local_path,
                                                local_path + f'{time.time()}',
                                                wait_finish=wait_finish)
                    flg = self.put_object_from_local_file(
                        local_cache, target_path)
                    try:
                        if os.path.exists(local_cache):
                            os.remove(local_cache)
                    except Exception:
                        pass
                else:
                    flg = False
                R.acquire()
                try:
                    data_quene.put_nowait([local_path, target_path, flg])
                except Exception:
                    R.release()
                R.release()

        while True:
            batch_local_list = local_path_list[:4 * batch_size]
            batch_target_list = target_path_list[:4 * batch_size]
            if len(batch_local_list) < 1:
                break
            local_path_list = local_path_list[4 * batch_size:]
            target_path_list = target_path_list[4 * batch_size:]
            threading_list = []
            for i in range(batch_size):
                t = threading.Thread(target=put_one_object,
                                     args=(
                                         batch_local_list[i::batch_size],
                                         batch_target_list[i::batch_size],
                                     ))
                t.daemon = True
                t.start()
                threading_list.append(t)
            [threading_t.join() for threading_t in threading_list]
            file_dict = {}
            while not data_quene.empty():
                local_path, target_path, flg = data_quene.get_nowait()
                file_dict[local_path] = [local_path, target_path, flg]

            for idx, local_path in enumerate(batch_local_list):
                local_path, target_path, flg = file_dict.get(
                    local_path,
                    [batch_local_list[idx], batch_target_list[idx], False])
                yield local_path, target_path, flg

    def get_object_stream(self,
                          target_path,
                          start,
                          size=10000,
                          delimiter=None):
        with self.get_fs_client(target_path) as client:
            local_data, end = client.get_object_stream(target_path,
                                                       start,
                                                       size=size,
                                                       delimiter=delimiter)
            return local_data, end

    def get_object_chunk_list(self,
                              target_path,
                              chunk_num=1,
                              chunk_size=-1,
                              delimiter=None):
        with self.get_fs_client(target_path) as client:
            chunk_list = client.get_object_chunk_list(target_path,
                                                      chunk_num=chunk_num,
                                                      chunk_size=chunk_size,
                                                      delimiter=delimiter)
            if chunk_list is None:
                raise ReadException(f'Failed to fetch {target_path}')
            return chunk_list

    def size(self, target_path):
        with self.get_fs_client(target_path) as client:
            size = client.size(target_path)
            return size

    def isfile(self, target_path):
        with self.get_fs_client(target_path) as client:
            is_file = client.isfile(target_path)
            return is_file

    def isdir(self, target_path):
        with self.get_fs_client(target_path) as client:
            is_dir = client.isdir(target_path)
            return is_dir

    @contextmanager
    def put_to(self, target_path):
        with self.get_fs_client(target_path) as client:
            local_path, is_tmp = client.map_to_local(target_path)
            if is_tmp:
                client.add_temp_file(local_path)
            if not os.path.exists(os.path.dirname(local_path)):
                os.makedirs(os.path.dirname(local_path))
            yield local_path
            status = client.put_object_from_local_file(local_path, target_path)
            if not status:
                raise WriteException(
                    f'Failed to upload from {local_path} to {target_path}')
            if not isinstance(client, LocalFs):
                try:
                    if os.path.exists(local_path):
                        os.remove(local_path)
                except Exception:
                    pass

    def __repr__(self) -> str:
        s = 'Support prefix list:\n'
        for prefix in sorted(list(self._prefix_to_clients.keys()),
                             key=lambda a: -len(a)):
            s += f'\t{prefix} -> {self._prefix_to_clients[prefix]}\n'
        return s


global FS, DATA_FS, MODEL_FS
# global instance, easy to use
FS = FileSystem()
DATA_FS = FS
MODEL_FS = FS
