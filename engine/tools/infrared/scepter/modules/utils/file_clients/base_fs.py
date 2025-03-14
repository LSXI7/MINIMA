# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import datetime
import os
import os.path as osp
import random
import tempfile
import warnings
from abc import ABCMeta, abstractmethod
from copy import copy
from typing import Optional

from tools.infrared.scepter.modules.utils.config import dict_to_yaml
from tools.infrared.scepter.modules.utils.directory import get_md5
from tools.infrared.scepter.modules.utils.file_clients.utils import remove_temp_path
from tools.infrared.scepter.modules.utils.logger import get_logger


class BaseFs(object, metaclass=ABCMeta):
    para_dict = {
        'TEMP_DIR': {
            'value':
            None,
            'description':
            'default is None, means using system cache dir and auto remove! If you set dir, the data will'
            ' be saved in this temp dir without autoremoving default.'
        },
        'AUTO_CLEAN': {
            'value':
            False,
            'description':
            'when TEMP_DIR is not None, if you set AUTO_CLEAN to True, the data will be clean automatics.'
        }
    }

    def __init__(self, cfg, logger=None):
        self._target_local_mapper = {}
        self._temp_files = set()
        self.cfg = cfg
        self.tmp_dir = cfg.get('TEMP_DIR', None)
        self.auto_clean = cfg.get('AUTO_CLEAN', False)
        if self.tmp_dir is None:
            self.auto_clean = True
        if self.tmp_dir is not None:
            if not os.path.exists(self.tmp_dir):
                try:
                    os.makedirs(self.tmp_dir, exist_ok=True)
                except Exception as e:
                    warnings.warn(
                        f'Create cache folder failed use default cache file{e}!'
                        .format(self.tmp_dir))
                    self.tmp_dir = None
        # checking that the logger exists or not
        if logger is None:
            self.logger = get_logger(name='File System')
        else:
            self.logger = logger

    # Functions without io
    @abstractmethod
    def get_prefix(self) -> str:
        """ Get supported path prefix to determine which handler to use.

        Returns:
            A prefix.
        """
        pass

    @abstractmethod
    def support_write(self) -> bool:
        """ Return flag if this file system supports write operation.

        Returns:
            Bool.
        """
        pass

    @abstractmethod
    def support_link(self) -> bool:
        """ Return if this file system supports create a soft link.

        Returns:
            Bool.
        """
        pass

    def add_target_local_map(self, target_dir, local_dir):
        """ Map target directory to local file system directory

        Args:
            target_dir (str): Target directory.
            local_dir (str): Directory in local file system.
        """
        self._target_local_mapper[target_dir] = local_dir

    def map_to_local(self, target_path, etag='') -> (str, bool):
        """ Map target path to local file path. (NO IO HERE).

        Args:
            target_path (str): Target file path.

        Returns:
            A local path and a flag indicates if the local path is a temporary file.
        """
        for target_dir, local_dir in self._target_local_mapper.items():
            if target_path.startswith(target_dir):
                return osp.join(local_dir, osp.relpath(target_path,
                                                       target_dir)), False
        else:
            return self._make_temporary_file(target_path, etag=etag), True

    def convert_to_local_path(self, target_path, etag='') -> str:
        """ Deprecated. Use map_to_local() function instead.
        """
        warnings.warn(
            'Function convert_to_local_path is deprecated, use map_to_local() function instead.'
        )
        local_path, _ = self.map_to_local(target_path, etag=etag)
        return local_path

    def basename(self, target_path) -> str:
        """ Get file name from target_path

        Args:
            target_path (str): Target file path.

        Returns:
            A file name.
        """
        return osp.basename(target_path)

    # Functions with heavy io
    @abstractmethod
    def get_object_to_local_file(self,
                                 target_path,
                                 local_path=None,
                                 wait_finish=False) -> Optional[str]:
        """ Transfer file object to local file.
        If local_path is not None,
            if path can be searched in local_mapper, download it as a persistent file
            else, download it as a temporary file
        else
            download it as a persistent file

        wait_finish when multi-processing download the same data, set wait_finish as True to avoid conflict

        Args:
            target_path (str): path of object in different file systems
            local_path (Optional[str]): If not None, will write path to local_path.

        Returns:
            Local file path of the object, none means a failure happened.
        """
        pass

    # Functions with heavy io
    @abstractmethod
    def get_object(self, target_path):
        """ Transfer file object to local file.
        If local_path is not None,
            if path can be searched in local_mapper, download it as a persistent file
            else, download it as a temporary file
        else
            download it as a persistent file

        Args:
            target_path (str): path of object in different file systems
            local_path (Optional[str]): If not None, will write path to local_path.

        Returns:
            Local file path of the object, none means a failure happened.
        """
        pass

    @abstractmethod
    def get_object_stream(self, target_path, start, size, delimiter=None):
        """ Transfer file object to local file.
        If local_path is not None,
            if path can be searched in local_mapper, download it as a persistent file
            else, download it as a temporary file
        else
            download it as a persistent file

        Args:
            target_path (str): path of object in different file systems
            start (int): object's start position.
            size (int): object's bytes size.
            delimiter (str): records's delimiter.

        Returns:
            Local file path of the object, none means a failure happened.
        """
        pass

    @abstractmethod
    def put_object_from_local_file(self, local_path, target_path) -> bool:
        """ Put local file to target file system path.

        Args:
            local_path (str): local file path of the object
            target_path (str): target file path of the object

        Returns:
            Bool.
        """
        pass

    @abstractmethod
    def put_object(self, local_data, target_path) -> bool:
        """ Put local file to target file system path.

        Args:
            local_path (binary): local data of the object
            target_path (str): target file path of the object

        Returns:
            Bool.
        """
        pass

    @abstractmethod
    def make_link(self, target_link_path, target_path) -> bool:
        """ Make soft link to target_path.

        Args:
            target_link_path (str):
            target_path (str)

        Returns:
            Bool.
        """
        pass

    @abstractmethod
    def make_dir(self, target_dir) -> bool:
        """ Make a directory.
        If target_dir is already exists, return True.

        Args:
            target_dir (str):

        Returns:
            True if target_dir exists or created.
        """
        pass

    @abstractmethod
    def remove(self, target_path) -> bool:
        """ Remove target file.

        Args:
            target_path (str):

        Returns:
            Bool.
        """
        pass

    @abstractmethod
    def get_logging_handler(self, target_logging_path):
        """ Get logging handler to target logging path.

        Args:
            target_logging_path:

        Returns:
            A handler which has a type of subclass of logging.Handler.
        """
        pass

    @abstractmethod
    def walk_dir(self, file_dir, recurse=True):
        pass

    @abstractmethod
    def put_dir_from_local_dir(self, local_dir, target_dir) -> bool:
        """ Upload all contents in local_dir to target_dir, keep the file tree.

        Args:
            local_dir (str):
            target_dir (str):

        Returns:
            Bool.
        """
        pass

    def _make_temporary_file(self, target_path, etag=''):
        """ Make a temporary file for target_path, which should have the same suffix.

        Args:
            target_path (str):

        Returns:
            A path (str).
        """
        file_name = self.basename(target_path)
        _, suffix = osp.splitext(file_name)
        if self.tmp_dir is None:
            rand_name = '{0:%Y%m%d%H%M%S%f}'.format(
                datetime.datetime.now()) + '_' + ''.join(
                    [str(random.randint(1, 10)) for _ in range(5)])
            # rand_name = get_md5(target_path)
            if suffix:
                rand_name += f'{suffix}'
            tmp_file = osp.join(tempfile.gettempdir(), rand_name)
        else:
            cache_name = '{}{}{}'.format(etag, get_md5(target_path), suffix)
            tmp_file = osp.join(self.tmp_dir, cache_name)
        return tmp_file

    # Functions only for status, light io
    @abstractmethod
    def exists(self, target_path) -> bool:
        """ Check if target_path exists.

        Args:
            target_path (str):

        Returns:
            Bool.
        """
        pass

    @abstractmethod
    def isfile(self, target_path) -> bool:
        """ Check if target_path is a file.

        Args:
            target_path (str):

        Returns:
            Bool.
        """
        pass

    @abstractmethod
    def isdir(self, target_path) -> bool:
        """ Check if target_path is a directory.

        Args:
            target_path (str):

        Returns:
            Bool.
        """

    def add_temp_file(self, tmp_file):
        self._temp_files.add(tmp_file)

    def clear(self):
        """Delete all temp files
        """
        if self.auto_clean:
            for temp_local_file in self._temp_files:
                remove_temp_path(temp_local_file)

    # Functions for context
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # if self.tmp_dir is None or self.auto_clean:
        #     for temp_local_file in self._temp_files:
        #         remove_temp_path(temp_local_file)
        pass

    def __del__(self):
        pass

    def copy(self):
        obj = copy(self)
        obj._temp_files = set(
        )  # A new obj to avoid confusing in multi-thread context.
        return obj

    @staticmethod
    def get_config_template():
        return dict_to_yaml('FILE_SYSTEMS',
                            __class__.__name__,
                            BaseFs.para_dict,
                            set_name=True)
