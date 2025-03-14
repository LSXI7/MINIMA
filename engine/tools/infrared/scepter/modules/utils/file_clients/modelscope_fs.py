# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import os.path as osp
import urllib.parse as parse
import urllib.request
from typing import Optional, Union

from tools.infrared.scepter.modules.utils.file_clients.base_fs import BaseFs
from tools.infrared.scepter.modules.utils.file_clients.registry import FILE_SYSTEMS


@FILE_SYSTEMS.register_class()
class ModelscopeFs(BaseFs):
    para_dict = {
        'RETRY_TIMES': {
            'value': 10,
            'description': 'Retry get object times.'
        }
    }
    para_dict.update(BaseFs.para_dict)

    def __init__(self, cfg, logger):
        super(ModelscopeFs, self).__init__(cfg, logger=logger)
        retry_times = cfg.get('RETRY_TIMES', 10)
        self._retry_times = retry_times
        self._model_id_loaded = set()
        self._model_file_loaded = set()

    def get_prefix(self) -> str:
        return 'ms://'

    def support_write(self) -> bool:
        return False

    def support_link(self) -> bool:
        return False

    def basename(self, target_path) -> str:
        url = parse.unquote(target_path)
        url = url.split('?')[0]
        return osp.basename(url)

    def get_object_to_local_file(self,
                                 target_path,
                                 local_path=None,
                                 wait_finish=False) -> Optional[str]:
        from modelscope.hub.file_download import model_file_download

        key = osp.relpath(target_path, self.get_prefix())

        if '#' in key:
            key, token = key.split('#', 1)
        else:
            token = None

        key, file_path = key.split('@', 1)

        if ':' in key:
            key, revision = key.split(':', 1)
        else:
            revision = None

        if local_path is None:
            local_path, is_tmp = self.map_to_local(key)
        else:
            is_tmp = False

        if revision is not None:
            local_path = local_path + '_' + str(revision)

        retry = 0
        while retry < self._retry_times:
            try:
                model_file = os.path.join(key, file_path)
                if model_file in self._model_file_loaded:
                    local_path = os.path.join(local_path, model_file)
                    if not osp.exists(local_path):
                        self._model_file_loaded.remove(key)
                else:
                    if token is not None:
                        cookies = self.get_modelscope_cookie(token)
                    else:
                        cookies = None
                    local_path = model_file_download(model_id=key,
                                                     revision=revision,
                                                     file_path=file_path,
                                                     cookies=cookies,
                                                     cache_dir=local_path)
                if osp.exists(local_path):
                    break
            except Exception:
                retry += 1

        if retry >= self._retry_times:
            return None
        self._model_file_loaded.add(model_file)
        if is_tmp:
            self.add_temp_file(local_path)
        return local_path

    def get_dir_to_local_dir(self,
                             target_path,
                             local_path=None,
                             wait_finish=False,
                             multi_thread=False,
                             timeout=3600,
                             sign_key=None,
                             worker_id=-1) -> Optional[str]:
        from modelscope.hub.snapshot_download import snapshot_download
        assert target_path.startswith(self.get_prefix())

        key = osp.relpath(target_path, self.get_prefix())

        if '#' in key:
            key, token = key.split('#', 1)
        else:
            token = None

        if '@' not in key:
            key, ret_folder = key.split('@', 1)[0], ''
        else:
            at_level_folder = key.split('@')
            if len(at_level_folder) > 2:
                raise f'Target path should include only one @, but you give {len(at_level_folder)} @.'
            key, ret_folder = at_level_folder

        if ':' in key:
            key, revision = key.split(':', 1)
        else:
            revision = None

        if local_path is None:
            local_path, is_tmp = self.map_to_local(key)
        else:
            is_tmp = False

        if revision is not None:
            local_path = local_path + '_' + str(revision)

        retry = 0
        while retry < self._retry_times:
            try:
                if key in self._model_id_loaded:
                    local_path = os.path.join(local_path, key)
                    if not osp.exists(local_path):
                        self._model_id_loaded.remove(key)
                else:
                    if token is not None:
                        cookies = self.get_modelscope_cookie(token)
                    else:
                        cookies = None
                    local_path = snapshot_download(key,
                                                   revision=revision,
                                                   cookies=cookies,
                                                   cache_dir=local_path)
                if osp.exists(local_path):
                    break
            except Exception:
                retry += 1

        if retry >= self._retry_times:
            return None

        self._model_id_loaded.add(key)
        if is_tmp:
            self.add_temp_file(local_path)
        if not ret_folder == '':
            local_path = os.path.join(local_path, ret_folder)
        return local_path

    def get_modelscope_cookie(self, m_session_id: str):
        from modelscope.hub.utils.utils import get_endpoint
        from modelscope.hub.api import ModelScopeConfig
        from modelscope.hub.errors import raise_on_error
        import requests
        path = f'{get_endpoint()}/api/v1/login'
        r = requests.post(
            path,
            json={'AccessToken': m_session_id},
            headers={'user-agent': ModelScopeConfig.get_user_agent()})
        r.raise_for_status()
        d = r.json()
        raise_on_error(d)
        return r.cookies

    def get_object(self, target_path):
        try:
            local_data = open(self.get_object_to_local_file(target_path),
                              'rb').read()
        except Exception as e:
            self.logger.error(f'Read {target_path} error {e}')
            local_data = None
        return local_data

    def put_object(self, local_data, target_path):
        raise NotImplementedError

    def put_object_from_local_file(self, local_path, target_path) -> bool:
        raise NotImplementedError

    def make_link(self, target_link_path, target_path) -> bool:
        raise NotImplementedError

    def make_dir(self, target_dir) -> bool:
        raise NotImplementedError

    def remove(self, target_path) -> bool:
        raise NotImplementedError

    def get_logging_handler(self, target_logging_path):
        raise NotImplementedError

    def walk_dir(self, file_dir, recurse=True):
        raise NotImplementedError

    def put_dir_from_local_dir(self,
                               local_dir,
                               target_dir,
                               multi_thread=False) -> bool:
        raise NotImplementedError

    def size(self, target_path) -> Optional[int]:
        raise NotImplementedError

    def get_object_chunk_list(self,
                              target_path,
                              chunk_num=1,
                              delimiter=None) -> Optional[list]:
        raise NotImplementedError

    def get_object_stream(
            self,
            target_path,
            start,
            size=10000,
            delimiter=None) -> (Union[bytes, str, None], Optional[int]):
        raise NotImplementedError

    def get_url(self,
                target_path,
                set_public=False,
                skip_check=False,
                lifecycle=3600 * 100):
        return target_path

    def exists(self, target_path) -> bool:
        req = urllib.request.Request(target_path)
        req.get_method = lambda: 'HEAD'

        try:
            urllib.request.urlopen(req)
            return True
        except Exception:
            return False

    def isfile(self, target_path) -> bool:
        # Well for a http url, it should only be a file.
        return True

    def isdir(self, target_path) -> bool:
        return False
