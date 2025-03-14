# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import copy
import json
import logging
import os
import os.path as osp
import queue
import random
import sys
import tempfile
import threading
import time
import warnings
from typing import Optional

from tqdm import tqdm

import oss2
from oss2 import determine_part_size
from oss2.models import PartInfo
from tools.infrared.scepter.modules.utils.config import dict_to_yaml
from tools.infrared.scepter.modules.utils.directory import get_md5
from tools.infrared.scepter.modules.utils.file_clients.base_fs import BaseFs
from tools.infrared.scepter.modules.utils.file_clients.registry import FILE_SYSTEMS


def compute_size(bytes):
    if bytes > 1024 * 1024 * 1024:
        gyte = bytes / (1024 * 1024 * 1024)
        return '{:.2f}G'.format(gyte)
    elif bytes > 1024 * 1024:
        gyte = bytes / (1024 * 1024)
        return '{:.2f}M'.format(gyte)
    elif bytes > 1024:
        gyte = bytes / 1024
        return '{:.2f}K'.format(gyte)
    else:
        return bytes


def upload_process_bar(consumed_bytes, total_bytes):
    sys.stdout.flush()
    percent = 100 * consumed_bytes / total_bytes
    sys.stdout.write('upload {:.2f}% [{}/{}]'.format(
        percent, compute_size(consumed_bytes), compute_size(total_bytes)))
    sys.stdout.flush()
    sys.stdout.write('\r')


def download_process_bar(consumed_bytes, total_bytes):
    sys.stdout.flush()
    percent = 100 * consumed_bytes / total_bytes
    sys.stdout.write('Download {:.2f}% [{}/{}]'.format(
        percent, compute_size(consumed_bytes), compute_size(total_bytes)))
    sys.stdout.flush()
    sys.stdout.write('\r')


def process_msg(msg):
    sys.stdout.write(' ' * 100)
    sys.stdout.flush()
    sys.stdout.write(msg)
    sys.stdout.flush()
    sys.stdout.write('\r')
    sys.stdout.flush()


class OssLoggingHandler(logging.StreamHandler):
    def __init__(self, log_file, cfg):
        super(OssLoggingHandler, self).__init__()
        self.cfg = cfg
        self._log_file = log_file
        self._sessions = {}
        _bucket = self._init_bucket()
        if _bucket.object_exists(self._log_file):
            size = _bucket.get_object_meta(self._log_file).content_length
        else:
            size = 0
        self._pos = _bucket.append_object(self._log_file, size, '')

    def _init_bucket(self):
        endpoint = self.cfg.ENDPOINT
        bucket = self.cfg.BUCKET
        ak = self.cfg.OSS_AK
        sk = self.cfg.OSS_SK
        # session
        session = self._sessions.setdefault(f'{bucket}@{os.getpid()}',
                                            oss2.Session())
        _bucket: oss2.Bucket = oss2.Bucket(oss2.Auth(ak, sk),
                                           endpoint,
                                           bucket,
                                           session=session)
        return _bucket

    def emit(self, record):
        msg = self.format(record) + '\n'
        for _ in range(5):
            _bucket = self._init_bucket()
            try:
                self._pos = _bucket.append_object(self._log_file,
                                                  self._pos.next_position, msg)
                break
            except oss2.exceptions.PositionNotEqualToLength:
                self._pos = _bucket.get_object_meta(
                    self._log_file).content_length
                self._pos = _bucket.append_object(self._log_file,
                                                  self._pos.next_position, msg)
                break
            except Exception:
                continue


@FILE_SYSTEMS.register_class()
class AliyunOssFs(BaseFs):
    para_dict = {
        'ENDPOINT': {
            'value': '',
            'description': 'the oss endpoint'
        },
        'BUCKET': {
            'value': '',
            'description': 'the oss bucket'
        },
        'OSS_AK': {
            'value': '',
            'description': 'the oss ak'
        },
        'OSS_SK': {
            'value': '',
            'description': 'the oss sk'
        },
        'PREFIX': {
            'value': '',
            'description': 'the file system prefix!'
        },
        'WRITABLE': {
            'value': True,
            'description': 'this file system is writable or not!'
        },
        'CHECK_WRITABLE': {
            'value': False,
            'decription': 'check fs is writable or not!'
        },
        'RETRY_TIMES': {
            'value': 10,
            'description': 'for one file retry download or upload times!'
        }
    }
    para_dict.update(BaseFs.para_dict)

    def __init__(self, cfg, logger=None):
        super(AliyunOssFs, self).__init__(cfg, logger=logger)
        prefix = cfg.get('PREFIX', None)
        writable = cfg.get('WRITABLE', True)
        check_writable = cfg.get('CHECK_WRITABLE', False)
        retry_times = cfg.get('RETRY_TIMES', 10)
        bucket = self.cfg.BUCKET
        self._sessions = {}
        _bucket = self._init_bucket()
        self._fs_prefix = f'oss://{bucket}/' + (''
                                                if prefix is None else prefix)
        self._prefix = f'oss://{bucket}/'
        try:
            _bucket.list_objects(max_keys=1)
        except Exception as e:
            warnings.warn(
                f'Cannot list objects in {self._prefix}, please check auth information. \n{e}'
            )
        self._retry_times = retry_times

        self._writable = writable
        if check_writable:
            self._writable = self._test_write(bucket, prefix)

    def _init_bucket(self):
        endpoint = self.cfg.ENDPOINT
        bucket = self.cfg.BUCKET
        ak = self.cfg.OSS_AK
        sk = self.cfg.OSS_SK
        # session
        session = self._sessions.setdefault(f'{bucket}@{os.getpid()}',
                                            oss2.Session())
        _bucket: oss2.Bucket = oss2.Bucket(oss2.Auth(ak, sk),
                                           endpoint,
                                           bucket,
                                           session=session)
        return _bucket

    def _test_write(self, bucket, prefix) -> bool:
        local_tmp_file = osp.join(
            tempfile.gettempdir(),
            f"oss_{bucket}_{'' if prefix is None else prefix}_try_test_write" +
            ''.join([str(random.randint(1, 10) for _ in range(5))]))
        with open(local_tmp_file, 'w') as f:
            f.write('Try to write')
        target_tmp_file = osp.join(self._prefix, osp.basename(local_tmp_file))
        status = self.put_object_from_local_file(local_tmp_file,
                                                 target_tmp_file)
        if status:
            self.remove(target_tmp_file)
        return status

    def get_prefix(self) -> str:
        return self._fs_prefix

    def support_write(self) -> bool:
        return self._writable

    def support_link(self) -> bool:
        return self._writable

    def get_meta(self, target_path):
        key = osp.relpath(target_path, self._prefix)
        retry, wait_retry = 0, 0  # noqa
        try:
            while retry < self._retry_times:
                _bucket = self._init_bucket()
                try:
                    meta = _bucket.get_object_meta(key)
                    etag = meta.etag
                    size = meta.content_length
                    break
                except oss2.exceptions.NoSuchKey as e:
                    warnings.warn(f'Get file meta {e}')
                    return None, None
                except Exception as e:
                    warnings.warn(f'Get file meta {e}')
                    retry += 1
            if retry >= self._retry_times:
                return None, None
        except Exception:
            etag = ''
            size = 100
        return etag, size

    def get_object_to_local_file(self,
                                 target_path,
                                 local_path=None,
                                 wait_finish=False,
                                 worker_id=-1) -> Optional[str]:
        key = osp.relpath(target_path, self._prefix)
        etag, size = self.get_meta(target_path)
        if local_path is None:
            local_path, is_tmp = self.map_to_local(target_path, etag=etag)
        else:
            is_tmp = False

        if wait_finish:
            from tools.infrared.scepter.modules.utils.distribute import we
            wait_times = size / 1024 / 1024
            cur_time = 0
            interval = 30
            while not worker_id == 0 and not osp.exists(
                    local_path) and wait_times > 0:
                if we.share_storage and we.rank == 0:
                    break
                if not we.share_storage and (we.device_id == 0
                                             or we.device_id == 'cpu'):
                    break
                if self.logger is not None:
                    self.logger.info(
                        f'GPU {we.device_id} have waited for {cur_time}s. Status: '
                        f'the data {target_path} is downloaded to {local_path},'
                        f'share storage {we.share_storage}, device {we.device_id}, rank {we.rank}!'
                    )
                time.sleep(interval)
                cur_time += interval
                wait_times -= interval
        if osp.exists(local_path) and osp.getsize(local_path) == size:
            return local_path

        os.makedirs(osp.dirname(local_path), exist_ok=True)
        retry, _ = 0, 0
        temp_file = local_path + '.{}_temp'.format(time.time())

        _bucket = self._init_bucket()
        if not (os.path.exists(local_path)
                and osp.getsize(local_path) == size):
            while retry < self._retry_times:
                if size < 100 * 1024 * 1024:
                    try:
                        if wait_finish:
                            _bucket.get_object_to_file(
                                key,
                                temp_file,
                                progress_callback=download_process_bar)
                        else:
                            _bucket.get_object_to_file(key, temp_file)
                        break
                    except oss2.exceptions.NoSuchKey as e:
                        warnings.warn(f'Download {key} error {e}')
                        return None
                    except Exception as e:
                        warnings.warn(f'{e}')
                        retry += 1
                else:
                    try:
                        _ = self._download_object_multi_part(target_path,
                                                             temp_file,
                                                             chunk_size=50 *
                                                             1024 * 1024)
                        break
                    except Exception as e:
                        retry += 1
                        self.logger.info(
                            'Download file {} error {} retry {} times!'.format(
                                target_path, e, retry))

        if retry >= self._retry_times:
            return None

        try:
            if not os.path.exists(local_path):
                os.rename(temp_file, local_path)
            elif not osp.getsize(local_path) == size:
                os.remove(local_path)
                os.rename(temp_file, local_path)
        except Exception as e:
            warnings.warn(f'Download local path {e}')
        if os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except Exception:
                warnings.warn('Remove temp file error!')

        if is_tmp:
            self.add_temp_file(local_path)
        return local_path

    def _download_object_multi_part(self,
                                    target_path,
                                    local_path,
                                    thread_num=20,
                                    chunk_size=10 * 1024) -> bool:
        '''
        Args:
            target_path:
            local_path:
        Returns:
        '''
        key = osp.relpath(target_path, self._prefix)
        chunk_list = self.get_object_chunk_list(target_path,
                                                chunk_num=-1,
                                                chunk_size=chunk_size)
        if chunk_list is None or len(chunk_list) == 0:
            self.logger.info("File {} isn't exists!".format(target_path))
            return False
        object_size = chunk_list[-1][0] + chunk_list[-1][1]
        slice_queue = queue.Queue()
        ret_slice_queue = queue.Queue()
        R = threading.Lock()
        for chunk_id, chunk in enumerate(chunk_list):
            slice_queue.put_nowait([chunk_id, chunk])
        process_msg(
            f'Split {target_path} to {len(chunk_list)} parts to download!')
        all_part_number = len(chunk_list)
        cache_folder = f'{local_path}_cache'
        os.makedirs(cache_folder, exist_ok=True)
        _bucket = self._init_bucket()

        def download_one_part(key):
            while not slice_queue.empty():
                R.acquire()
                try:
                    if not slice_queue.empty():
                        part_number, chunk = slice_queue.get_nowait()
                        data_status = True
                    else:
                        data_status = False
                except Exception as e:
                    self.logger.info('Get chunk {} error {}'.format(
                        target_path, e))
                    data_status = False
                R.release()
                if not data_status:
                    continue
                temp_part_file = os.path.join(cache_folder, f'{part_number}')
                retry = 0
                while retry < self._retry_times:
                    try:
                        data, end = self.get_object_stream(
                            target_path, chunk[0], chunk[1])
                        with open(temp_part_file, 'wb') as f:
                            f.write(data)
                        R.acquire()
                        try:
                            ret_slice_queue.put_nowait({
                                'part_number':
                                part_number,
                                'status':
                                True,
                                'file_path':
                                temp_part_file
                            })
                        except Exception as e:
                            self.logger.info('Ret status error {}'.format(e))
                        R.release()
                        break
                    except Exception as e:
                        retry += 1
                        self.logger.info(
                            'Download part {} for {} error {} retry {} times!'.
                            format(part_number, key, e, retry))
                if retry >= self._retry_times:
                    R.acquire()
                    try:
                        ret_slice_queue.put_nowait({
                            'part_number': part_number,
                            'status': False,
                            'file_path': None
                        })
                    except Exception as e:
                        self.logger.info('Ret status error {}'.format(e))
                    R.release()
                if ret_slice_queue.qsize() < all_part_number:
                    download_process_bar(chunk_size * ret_slice_queue.qsize(),
                                         object_size)
                else:
                    download_process_bar(object_size, object_size)

        threading_list = []
        for i in range(thread_num):
            t = threading.Thread(target=download_one_part, args=(key, ))
            t.daemon = True
            t.start()
            threading_list.append(t)
        for thread in threading_list:
            thread.join()
        parts = []
        while not ret_slice_queue.empty():
            upload_status = ret_slice_queue.get_nowait()
            if upload_status['status']:
                parts.append(
                    (upload_status['part_number'], upload_status['file_path']))
            else:
                return False

        parts.sort(key=lambda x: x[0])

        with open(local_path, 'wb') as fw:
            for part in tqdm(parts, desc='merge parts to file...'):
                with open(part[1], 'rb') as f:
                    fw.write(f.read())
        try:
            os.system('rm -rf {}'.format(cache_folder))
        except Exception as e:
            self.logger.info('Remove cache error {}'.format(e))
        if _bucket.object_exists(key):
            return True
        else:
            return False

    def check_folder(self, check_file, sign_key=None):
        meta_dict = json.load(open(check_file, 'r'))
        if sign_key in meta_dict:
            etag, size = self.get_meta(sign_key)
            if not etag == meta_dict[sign_key]:
                return False
            else:
                return True
        for key, v in meta_dict.items():
            etag, size = self.get_meta(key)
            if not etag == v:
                return False
        return True

    def _get_dir(self,
                 target_path,
                 local_path,
                 wait_finish=False,
                 meta_dict={}):
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
                etag, size = self.get_meta(file_name)
                if local_file_name in meta_dict and meta_dict[
                        local_file_name] == etag:
                    continue
                process_msg(f'Download {file_name} to {local_file_name}....')
                local_file_name = self.get_object_to_local_file(
                    file_name, local_file_name, wait_finish=wait_finish)
                assert local_file_name is not None
                meta_dict[file_name] = etag
            else:
                meta_dict.update(
                    self._get_dir(file_name,
                                  local_file_name,
                                  meta_dict=copy.deepcopy(meta_dict)))
        return meta_dict

    def _get_dir_multi(self,
                       target_path,
                       local_path,
                       wait_finish=False,
                       meta_dict={}):
        local_path = local_path.replace('/./', '/')
        os.makedirs(local_path, exist_ok=True)
        generator = self.walk_dir(target_path)
        single_file_name = []
        for file_name in generator:
            if file_name == target_path or file_name == target_path + '/':
                continue
            local_file_name = os.path.join(
                local_path,
                file_name.split(target_path)[-1]).replace('/./', '/')
            if not self.isdir(file_name):
                single_file_name.append((file_name, local_file_name))
            else:
                meta_dict.update(
                    self._get_dir_multi(file_name,
                                        local_file_name,
                                        meta_dict=copy.deepcopy(meta_dict)))

        data_quene = queue.Queue()
        batch_size = 20
        R = threading.Lock()

        def get_one_object(target_path_list):
            if isinstance(target_path_list, tuple):
                target_path_list = [target_path_list]
            for target_path, local_path in target_path_list:
                if self.exists(target_path):
                    etag, size = self.get_meta(target_path)
                    if local_path in meta_dict and meta_dict[
                            local_path] == etag:
                        continue
                    process_msg(f'Download {target_path} to {local_path}....')
                    local_path = self.get_object_to_local_file(
                        target_path, local_path, wait_finish=wait_finish)
                    assert local_path is not None
                    meta_dict[target_path] = etag
                else:
                    local_path = None
                R.acquire()
                try:
                    data_quene.put_nowait([target_path, local_path])
                except Exception:
                    R.release()
                R.release()

        while True:
            batch_list = single_file_name[:10 * batch_size]
            if len(batch_list) < 1:
                break
            single_file_name = single_file_name[10 * batch_size:]
            threading_list = []
            for i in range(batch_size):
                cur_batch = batch_list[i::batch_size]
                if isinstance(cur_batch, tuple):
                    cur_batch = [cur_batch]
                t = threading.Thread(target=get_one_object, args=(cur_batch, ))
                t.daemon = True
                t.start()
                threading_list.append(t)
            [threading_t.join() for threading_t in threading_list]
            file_dict = {}
            while not data_quene.empty():
                target_path, local_path = data_quene.get_nowait()
                file_dict[target_path] = local_path
        return meta_dict

    def get_dir_to_local_dir(self,
                             target_path,
                             local_path=None,
                             wait_finish=False,
                             timeout=3600,
                             multi_thread=False,
                             sign_key=None,
                             worker_id=-1) -> Optional[str]:
        if not self.isdir(target_path):
            self.logger.info(
                f"{target_path} is not directory or doesn't exist.")
            return None
        if not target_path.endswith('/'):
            target_path += '/'
        if local_path is None:
            local_path, is_tmp = self.map_to_local(target_path)
        else:
            is_tmp = False
        local_path = local_path.replace('/./', '/')
        os.makedirs(local_path, exist_ok=True)
        check_file = os.path.join(local_path,
                                  f'{get_md5(target_path)}_data_meta.json')
        if wait_finish:
            from tools.infrared.scepter.modules.utils.distribute import we
            wait_times = timeout
            while wait_times > 0 and not (osp.exists(check_file)
                                          and self.check_folder(
                                              check_file, sign_key=sign_key)):
                if we.share_storage and we.rank == 0:
                    break
                if not we.share_storage and (we.device_id == 0
                                             or we.device_id == 'cpu'):
                    break
                if self.logger is not None:
                    self.logger.info(
                        f'GPU {we.device_id} is waiting that '
                        f'the data {target_path} is downloaded to {local_path}!'
                    )
                time.sleep(5)
                wait_times -= 1
        if osp.exists(check_file) and self.check_folder(check_file,
                                                        sign_key=sign_key):
            return local_path

        if osp.exists(check_file):
            meta_dict = json.load(open(check_file, 'r'))
        else:
            meta_dict = {}
        if multi_thread:
            meta_dict = self._get_dir_multi(target_path,
                                            local_path=local_path,
                                            meta_dict=copy.deepcopy(meta_dict))
        else:
            meta_dict = self._get_dir(target_path,
                                      local_path=local_path,
                                      meta_dict=copy.deepcopy(meta_dict))
        json.dump(meta_dict, open(check_file, 'w'))
        if is_tmp:
            self.add_temp_file(local_path)
        return local_path

    def put_object_from_local_file(self, local_path, target_path) -> bool:
        key = osp.relpath(target_path, self._prefix)
        retry = 0
        object_size = os.path.getsize(local_path)
        while retry < self._retry_times:
            _bucket = self._init_bucket()
            if object_size <= 100 * 1024 * 1024:
                try:
                    _bucket.put_object_from_file(key, local_path)
                    if _bucket.object_exists(key):
                        break
                except Exception as e:
                    retry += 1
                    self.logger.info(
                        'Upload file {} error {} retry {} times!'.format(
                            target_path, e, retry))
            else:
                try:
                    flag = self._put_object_multi_part(local_path,
                                                       key,
                                                       object_size=object_size)
                    return flag
                except Exception as e:
                    retry += 1
                    self.logger.info(
                        'Upload file {} error {} retry {} times!'.format(
                            target_path, e, retry))

        if retry >= self._retry_times:
            return False

        return True

    def _put_object_multi_part(self,
                               local_path,
                               key,
                               object_size=-1,
                               thread_num=20) -> bool:
        '''
        Args:
            local_path:
            key:
            object_size:

        Returns:

        '''
        if object_size < 0:
            object_size = os.path.getsize(local_path)
        _bucket = self._init_bucket()
        part_size = determine_part_size(object_size,
                                        preferred_size=10 * 1024 * 1024)
        upload_id = _bucket.init_multipart_upload(key).upload_id
        parts = []
        fileobj = open(local_path, 'rb')
        slice_queue = queue.Queue()
        ret_slice_queue = queue.Queue()
        R = threading.Lock()
        part_number = 1
        offset = 0
        while offset < object_size:
            num_to_upload = min(part_size, object_size - offset)
            slice_queue.put_nowait([part_number, offset, num_to_upload])
            offset += num_to_upload
            part_number += 1
        process_msg(f'Split {key} to {part_number - 1} parts to upload!')
        all_part_number = part_number

        def upload_one_part(key, upload_id):
            while not slice_queue.empty():
                R.acquire()
                try:
                    if not slice_queue.empty():
                        part_number, offset, num_to_upload = slice_queue.get_nowait(
                        )
                        fileobj.seek(offset)
                        raw_data = fileobj.read(num_to_upload)
                        data_status = True
                    else:
                        data_status = False
                except Exception as e:
                    self.logger.info('Seek file {} error {}'.format(
                        local_path, e))
                    data_status = False
                R.release()
                if not data_status:
                    continue
                retry = 0
                while retry < self._retry_times:
                    _bucket = self._init_bucket()
                    try:
                        result = _bucket.upload_part(key, upload_id,
                                                     part_number, raw_data)
                        R.acquire()
                        try:
                            ret_slice_queue.put_nowait({
                                'part_number': part_number,
                                'status': True,
                                'etag': result.etag
                            })
                        except Exception as e:
                            self.logger.info('Ret status error {}'.format(e))
                        R.release()
                        break
                    except Exception as e:
                        retry += 1
                        self.logger.info(
                            'Upload part {} for {} error {} retry {} times!'.
                            format(part_number, key, e, retry))
                if retry >= self._retry_times:
                    R.acquire()
                    try:
                        ret_slice_queue.put_nowait({
                            'part_number': part_number,
                            'status': False,
                            'etag': None
                        })
                    except Exception as e:
                        self.logger.info('Ret status error {}'.format(e))
                    R.release()
                if ret_slice_queue.qsize() < all_part_number - 1:
                    upload_process_bar(part_size * ret_slice_queue.qsize(),
                                       object_size)
                else:
                    upload_process_bar(object_size, object_size)

        threading_list = []
        for i in range(thread_num):
            t = threading.Thread(target=upload_one_part, args=(key, upload_id))
            t.daemon = True
            t.start()
            threading_list.append(t)
        for thread in threading_list:
            thread.join()

        while not ret_slice_queue.empty():
            upload_status = ret_slice_queue.get_nowait()
            if upload_status['status']:
                parts.append(
                    PartInfo(upload_status['part_number'],
                             upload_status['etag']))
            else:
                return False
        _ = _bucket.complete_multipart_upload(key, upload_id, parts)
        try:
            fileobj.close()
        except Exception as e:
            self.logger.info(f'{e}')
        if _bucket.object_exists(key):
            return True
        else:
            return False

    def get_object(self, target_path):
        try:
            retry = 0
            key = osp.relpath(target_path, self._prefix)
            local_data = None
            while retry < self._retry_times:
                _bucket = self._init_bucket()
                try:
                    local_data = _bucket.get_object(key).read()
                    break
                except oss2.exceptions.NoSuchKey as e:
                    self.logger.info(f'Download local path {e}')
                    return None
                except Exception as e:
                    self.logger.info(f'{e}')
                    retry += 1
        except Exception as e:
            self.logger.error(f'Read {target_path} error {e}')
            local_data = None
        return local_data

    def get_object_stream(self,
                          target_path,
                          start,
                          size=10000,
                          delimiter=None):
        end = start + size - 1
        retry = 0
        key = osp.relpath(target_path, self._prefix)
        local_data = None
        _bucket = self._init_bucket()
        if not _bucket.object_exists(key):
            return local_data, end
        meta_data = _bucket.get_object_meta(key)
        content_length = meta_data.content_length
        end = min(end, content_length - 1)

        if start >= content_length - 1:
            return local_data, None

        while retry < self._retry_times:
            _bucket = self._init_bucket()
            try:
                local_data = _bucket.get_object(key, byte_range=(start,
                                                                 end)).read()
                break
            except oss2.exceptions.NoSuchKey as e:
                self.logger.info(f'Download local path {e}')
                return None, end
            except Exception as e:
                self.logger.info(f'{e}')
                retry += 1
        if delimiter is not None and not end == content_length - 1:
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
        return local_data, end + 1

    def get_object_chunk_list(self,
                              target_path,
                              chunk_num=-1,
                              chunk_size=-1,
                              delimiter=None):
        chunk_st_et = []
        key = osp.relpath(target_path, self._prefix)
        _bucket = self._init_bucket()
        if not _bucket.object_exists(key):
            self.logger.info(f'{target_path} is not exists!')
            return chunk_st_et
        if chunk_size < 0 and chunk_num < 0:
            self.logger.error(
                'Suppose chunk size > 0 or chunk num > 0, instead of both < 0')
            return None
        meta_data = _bucket.get_object_meta(key)
        content_length = meta_data.content_length
        if chunk_size < 0 and chunk_num > 0:
            chunk_size = content_length // chunk_num + 1
        # Don't care of the lines info
        if delimiter is None:
            chunk_start = 0
            # Iterate over all chunks and construct arguments for `process_chunk`
            while chunk_start < content_length - 1:
                chunk_end = min(content_length - 1, chunk_start + chunk_size)
                chunk_st_et.append([chunk_start, chunk_end - chunk_start + 1])
                chunk_start = chunk_end + 1
        else:
            chunk_start = 0
            offset = 0
            while chunk_start < content_length - 1:
                chunk_end = min(content_length - 1,
                                chunk_start + chunk_size + offset)
                retry = 0
                while retry < self._retry_times:
                    _bucket = self._init_bucket()
                    try:
                        quota_st = max(0, chunk_end - 20000)
                        quota_st = max(chunk_start, quota_st)
                        local_data = _bucket.get_object(
                            key, byte_range=(quota_st, chunk_end)).read()
                        break
                    except oss2.exceptions.NoSuchKey as e:
                        warnings.warn(f'Download local path {e}')
                        return None
                    except Exception as e:
                        warnings.warn(f'{e}')
                        retry += 1
                if not chunk_end == content_length - 1:
                    try:
                        offset = 0
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
                chunk_st_et.append([chunk_start, chunk_end - chunk_start + 1])
                chunk_start = chunk_end + 1
        return chunk_st_et

    def put_object(self, local_data, target_path):
        key = osp.relpath(target_path, self._prefix)
        retry = 0
        while retry < self._retry_times:
            _bucket = self._init_bucket()
            try:
                _bucket.put_object(key, local_data)
                if _bucket.object_exists(key):
                    break
            except Exception as e:
                retry += 1
                self.logger.info('Upload file {} error {}'.format(
                    target_path, e))

        if retry >= self._retry_times:
            return False

        return True

    def get_url(self,
                target_path,
                set_public=False,
                lifecycle=3600 * 100,
                skip_check=False,
                slash_safe=True):
        key = osp.relpath(target_path, self._prefix)
        _bucket = self._init_bucket()
        if not skip_check and not _bucket.object_exists(key):
            self.logger.info(f'{target_path} is not exists!')
            return None
        retry = 0
        while retry < self._retry_times:
            _bucket = self._init_bucket()
            try:
                output_url = _bucket.sign_url('GET',
                                              key,
                                              lifecycle,
                                              slash_safe=slash_safe)
                _bucket.put_object_acl(key, oss2.OBJECT_ACL_PUBLIC_READ)
                if set_public:
                    output_url = output_url.replace('%2F', '/').split('?')[0]
                return output_url
            except Exception as e:
                retry += 1
                self.logger.info('Upload file {} error {}'.format(
                    target_path, e))
        return None

    def make_link(self, target_link_path, target_path) -> bool:
        if not self.support_link():
            return False
        link_key = osp.relpath(target_link_path, self._prefix)
        target_key = osp.relpath(target_path, self._prefix)
        _bucket = self._init_bucket()
        try:
            _bucket.put_symlink(target_key, link_key)
        except Exception as e:
            print(e)
            return False
        return True

    def make_dir(self, target_dir) -> bool:
        # OSS treat file path as a key, it will create directory automatically when putting a file.
        return True

    def remove(self, target_path) -> bool:
        key = osp.relpath(target_path, self._prefix)
        _bucket = self._init_bucket()
        try:
            _bucket.delete_object(key)
            return True
        except Exception as e:
            print(e)
            return False

    def get_logging_handler(self, target_logging_path):
        oss_key = osp.relpath(target_logging_path, self._prefix)
        _ = self._init_bucket()
        return OssLoggingHandler(oss_key, self.cfg)

    def walk_dir(self, file_dir, recurse=True):
        key = file_dir.replace(self._prefix, '')
        if self.isdir(file_dir) and not file_dir.endswith('/'):
            key += '/'
        if recurse:
            delimiter = ''
        else:
            delimiter = '/'
        _bucket = self._init_bucket()
        for obj in oss2.ObjectIteratorV2(_bucket,
                                         prefix=key,
                                         delimiter=delimiter,
                                         max_keys=1000):
            # if obj.is_prefix():
            #     continue
            if obj.key == key:
                continue
            yield osp.join(self._prefix, obj.key)

    def put_dir_from_local_dir(self,
                               local_dir,
                               target_dir,
                               multi_thread=False) -> bool:
        singe_file_names = []
        for folder, sub_folders, files in os.walk(local_dir):
            for file in files:
                file_abs_path = osp.join(folder, file)
                file_rel_path = osp.relpath(file_abs_path, local_dir)
                target_path = osp.join(target_dir, file_rel_path)
                singe_file_names.append((file_abs_path, target_path))

        if not multi_thread:
            for file_abs_path, target_path in singe_file_names:
                status = self.put_object_from_local_file(
                    file_abs_path, target_path)
                if not status:
                    return False
        else:
            data_quene = queue.Queue()
            R = threading.Lock()
            batch_size = 20

            def put_one_object(target_path_list):
                if isinstance(target_path_list, tuple):
                    target_path_list = [target_path_list]
                for local_path, target_path in target_path_list:
                    if local_path is None or target_path is None:
                        flg = False
                    elif os.path.exists(local_path):
                        flg = self.put_object_from_local_file(
                            local_path, target_path)
                    else:
                        flg = False
                    R.acquire()
                    try:
                        data_quene.put_nowait([local_path, target_path, flg])
                    except Exception:
                        R.release()
                    R.release()

            while True:
                batch_list = singe_file_names[:10 * batch_size]
                if len(batch_list) < 1:
                    break
                singe_file_names = singe_file_names[10 * batch_size:]
                threading_list = []
                for i in range(batch_size):
                    cur_batch = batch_list[i::batch_size]
                    if isinstance(cur_batch, tuple):
                        cur_batch = [cur_batch]
                    t = threading.Thread(target=put_one_object,
                                         args=(cur_batch, ))
                    t.daemon = True
                    t.start()
                    threading_list.append(t)
                [threading_t.join() for threading_t in threading_list]
                while not data_quene.empty():
                    local_path, target_path, flg = data_quene.get_nowait()
                    if not flg:
                        return False
        return True

    def size(self, target_path) -> Optional[int]:
        key = osp.relpath(target_path, self._prefix)
        _bucket = self._init_bucket()
        if not _bucket.object_exists(key):
            self.logger.info(f"File {key} doesn't exist.")
            return -1
        meta_data = _bucket.get_object_meta(key)
        content_length = meta_data.content_length
        return content_length

    def exists(self, target_path) -> bool:
        # if is folder,try list all objects
        _bucket = self._init_bucket()
        target_path = target_path.replace(self._prefix, '')
        try:
            object_exists = _bucket.object_exists(target_path)
        except Exception as e:
            print(e)
            return False
        # if object doesnot exist, suppose it's a folder
        if not object_exists:
            if not target_path.endswith('/'):
                target_path += '/'
            ret_object_list = _bucket.list_objects(target_path,
                                                   max_keys=10).object_list
            return len(ret_object_list) > 0
        return object_exists

    def isfile(self, target_path) -> bool:
        if target_path.endswith('/'):
            return False
        target_path = osp.relpath(target_path, self._prefix)
        _bucket = self._init_bucket()
        try:
            object_exists = _bucket.object_exists(target_path)
        except Exception as e:
            print(e)
            return False
        return object_exists

    def isdir(self, target_path) -> bool:
        if not target_path.endswith('/'):
            target_path += '/'
        return self.exists(target_path)

    @staticmethod
    def get_config_template():
        return dict_to_yaml('FILE_SYSTEMS',
                            __class__.__name__,
                            AliyunOssFs.para_dict,
                            set_name=True)
