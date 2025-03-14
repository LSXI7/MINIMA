# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import copy
import os.path
from numbers import Number

import numpy as np
import torch
from PIL import Image
from tools.infrared.scepter.modules.utils.file_system import FS


def check_legal_type(data):
    if isinstance(data, str) or isinstance(data, Number):
        return True
    elif isinstance(data, dict):
        for k, v in data.items():
            if not check_legal_type(v):
                return False
        return True
    elif isinstance(data, list):
        for v in data:
            if not check_legal_type(v):
                return False
        return True
    else:
        return False


def register_data(probe_data: dict, key_prefix=''):
    ret_data = {}
    dist_data = {}
    for k, v in probe_data.items():
        key = f'{key_prefix}_{k}'
        if isinstance(v, torch.Tensor) or isinstance(v, np.ndarray):
            ret_data[key] = ProbeData(v)
        elif isinstance(v, ProbeData):
            ret_data[key] = v
        else:
            if not check_legal_type(v):
                raise f'The datatype of {key} should be included in [array, tensor, number, str] or the dict or ' \
                      f'list of (number, str); if you want register the list of image, please use ProbeData instance.'
            ret_data[key] = ProbeData(v)
        if ret_data[key].view_distribute:
            dist_data[key] = ret_data[key].distribute

    return ret_data, dist_data


def merge_gathered_probe(all_gathered_data):
    '''
     Merge the gathered data on rank_0.
    Returns:
        The merged data.

    '''
    for key, gathered_data in all_gathered_data.items():
        # Must be the list of ProbeData.
        if isinstance(gathered_data, list):
            for v in gathered_data:
                if not isinstance(v, ProbeData):
                    all_gathered_data[key] = gathered_data
            # Must be the gathered data.
            ret_data = gathered_data[0]
            if not isinstance(ret_data.data,
                              list) and (isinstance(ret_data.data, np.ndarray)
                                         or isinstance(ret_data.data, dict)
                                         or check_legal_type(ret_data.data)):
                new_data = [v.data for v in gathered_data]
                if ret_data.build_label is not None:
                    ret_data.build_label = [
                        v.build_label for v in gathered_data
                    ]
                all_gathered_data[key] = ProbeData(
                    new_data,
                    is_image=ret_data.is_image,
                    build_html=ret_data.build_html,
                    build_label=ret_data.build_label,
                    view_distribute=ret_data.view_distribute)
            elif isinstance(ret_data.data, list):
                if ret_data.build_label is not None:
                    if isinstance(ret_data.build_label, str):
                        ret_data.build_label = [
                            ret_data.build_label for _ in ret_data.data
                        ]
                for v in gathered_data[1:]:
                    ret_data.data += v.data
                    if ret_data.build_label is not None:
                        if isinstance(v.build_label, str):
                            ret_data.build_label.extend(
                                [v.build_label for _ in v.data])
                        ret_data.build_label.extend(v.build_label)
                all_gathered_data[key] = ProbeData(
                    ret_data.data,
                    is_image=ret_data.is_image,
                    build_html=ret_data.build_html,
                    build_label=ret_data.build_label,
                    view_distribute=ret_data.view_distribute)
        else:
            all_gathered_data[key] = gathered_data
    return all_gathered_data


class ProbeData():
    def __init__(self,
                 data,
                 is_image=False,
                 build_html=False,
                 build_label=None,
                 view_distribute=False):
        ''' Probe Data Initialize.
            We only support basic types such as [torch.Tensor, numpy.ndarray, number, str],
            or [dict, list] of [dict, list,
            number, str], or [dict, list] of [tensor, array]
        '''
        data = copy.deepcopy(data)
        self.basic_type = True
        self._distribute_dict = {}
        if view_distribute:
            is_legal = True
            if isinstance(data, str) or isinstance(data, Number):
                is_legal = True
                if data in self._distribute_dict:
                    self._distribute_dict[data] += 1
                else:
                    self._distribute_dict[data] = 1
            elif isinstance(data, list):
                for v in data:
                    if isinstance(v, str) or isinstance(v, Number):
                        is_legal = True
                        if v in self._distribute_dict:
                            self._distribute_dict[v] += 1
                        else:
                            self._distribute_dict[v] = 1
                    else:
                        is_legal = False
            elif isinstance(data, dict):
                for k, v in data.items():
                    if isinstance(v, str) or isinstance(v, Number):
                        is_legal = True
                        n_k = f'{k}_{v}'
                        if n_k in self._distribute_dict:
                            self._distribute_dict[n_k] += 1
                        else:
                            self._distribute_dict[n_k] = 1
                    else:
                        is_legal = False
            else:
                is_legal = False
            if not is_legal:
                print('Unsurpport data type', data)
                assert is_legal
        self.view_distribute = view_distribute
        if isinstance(data, torch.Tensor):
            self.data = data.detach().cpu().numpy()
        elif isinstance(data, np.ndarray):
            self.data = data
        elif isinstance(data, dict):
            for k, v in data.items():
                if not check_legal_type(v):
                    if isinstance(v, torch.Tensor):
                        data[k] = v.detach().cpu().numpy()
                        self.basic_type = False
                    elif isinstance(v, np.ndarray):
                        data[k] = v
                        self.basic_type = False
                    else:
                        raise f'Unsupport data type for {v}'
            self.data = data
        elif isinstance(data, list):
            for idx, v in enumerate(data):
                if not check_legal_type(v):
                    if isinstance(v, torch.Tensor):
                        data[idx] = v.detach().cpu().numpy()
                        self.basic_type = False
                    elif isinstance(v, np.ndarray):
                        data[idx] = v
                        self.basic_type = False
                    else:
                        raise f'Unsupport data type for {v}'
            self.data = data
        elif check_legal_type(data):
            self.data = data
        else:
            raise f'Unsupport data type for {data}'

        self.is_image = is_image
        self.build_html = build_html

        if self.build_html:
            assert build_label is not None
            if isinstance(self.data, str):
                assert isinstance(build_label, str)
            if isinstance(self.data, list):
                assert isinstance(build_label, str) or isinstance(
                    build_label, list)
            if isinstance(self.data, dict):
                assert isinstance(build_label, str) or isinstance(
                    build_label, dict)
        self.build_label = build_label

    def save_image(self, file_prefix, images, image_postfix):
        np_shape = images.shape
        # 4D
        shape_str = '_'.join([str(v) for v in np_shape])
        if len(np_shape) == 4:
            # channel is 1 or 3
            if np_shape[-1] == 1 or np_shape[-1] == 3 or np_shape[-1] == 4:
                if np_shape[-1] == 1:
                    images = images.reshape(images.shape[:-1])
                file_list = []
                for idx in range(np_shape[0]):
                    file_path = file_prefix + f'_probe_{idx}_[{shape_str}].{image_postfix}'
                    with FS.put_to(file_path) as local_path:
                        Image.fromarray(images[idx, ...]).save(local_path)
                    file_list.append(file_path)
                return file_list
            else:
                raise f"Ensure your data's dim is BWHC, and channel is 1 or 3 for {file_prefix}"
        elif len(np_shape) == 3:
            if np_shape[-1] == 1 or np_shape[-1] == 3 or np_shape[-1] == 4:
                if np_shape[-1] == 1:
                    images = images.reshape(images.shape[:-1])
                file_path = file_prefix + f'_probe_[{shape_str}].{image_postfix}'
                with FS.put_to(file_path) as local_path:
                    Image.fromarray(images).save(local_path)
                return file_path
            else:
                images = images.reshape(list(images.shape) + [1])
                return self.save_image(file_prefix, images, image_postfix)
        elif len(np_shape) == 2:
            file_path = file_prefix + f'_probe_[{shape_str}].{image_postfix}'
            with FS.put_to(file_path) as local_path:
                Image.fromarray(images).save(local_path)
            return file_path
        else:
            raise f"Ensure your data's dim is BWHC or WHC or WH, and channel is 1 or 3 for {file_prefix}"

    def save_npy(self, file_prefix, data):
        shape_str = '_'.join([str(v) for v in data.shape])
        file_path = file_prefix + f'_{shape_str}.npy'
        with FS.put_to(file_path) as local_path:
            np.save(local_path, data)
        return file_path

    def save_html(self, html_prefix, ret_data, ret_label):
        height = 600
        with FS.put_to(html_prefix) as local_path:
            with open(local_path, 'w') as f:
                f.writelines('<meta charset="utf-8">\n')
                f.writelines('<style>input{height:' + f'{height}px;' +
                             'opacity:1.0;}</style>\n')
                f.writelines('<br><hr/>\n')
                all_ranks = list()
                for save_id, save_data in enumerate(zip(ret_data, ret_label)):
                    save_path, save_label = save_data
                    one_rank = '<table><tr>'
                    for idx, one_data in enumerate(zip(save_path, save_label)):
                        one_path, one_label = one_data
                        one_label = one_label.replace('<', '&lt;').replace(
                            '>', '&gt;')
                        url = FS.get_url(one_path,
                                         lifecycle=3600 * 365 * 24).replace(
                                             '.oss-internal.aliyun-inc.',
                                             '.oss.aliyuncs.').replace(
                                                 '-internal', '')
                        one_rank += (
                            f'<td align="center"><input type="image" src="{url}" >'
                            f'<br><font size="4"><strong>{save_id}-{idx}|{one_label}<strong></font><br/></td>'
                        )
                    one_rank += '</tr></table><hr/>'
                    all_ranks.append(one_rank)
                f.writelines('\n'.join(all_ranks))
        return html_prefix

    @property
    def distribute(self):
        return self._distribute_dict

    def to_log(self, prefix=None, image_postfix='jpg'):
        if isinstance(self.data, np.ndarray):
            if prefix is None:
                raise 'You should provide the save prefix for array sample.'
            # save jpg
            if self.is_image:
                ret_data = self.save_image(prefix, self.data, image_postfix)
                if isinstance(ret_data, list):
                    ret_data = [ret_data]
                    if self.build_html:
                        ret_label = []
                        if isinstance(self.build_label, str):
                            ret_label.append(
                                [self.build_label for _ in ret_data[0]])
                        else:
                            ret_label.append(self.build_label)
                        if not len(ret_data[0]) == len(ret_label[0]):
                            raise f"The {prefix} label's length should be equal with 1st dim {self.data.shape[0]}."
                        html_prefix = prefix + '_probe.html'
                        html_file = self.save_html(html_prefix, ret_data,
                                                   ret_label)
                        return {'ori_file': ret_data, 'html': html_file}
                    else:
                        return {'ori_file': ret_data}
                else:
                    return ret_data
            else:
                ret_data = self.save_npy(prefix, self.data)
                return ret_data
        elif isinstance(self.data, list):
            if not self.basic_type:
                ret_data = []
                ret_label = []
                for idx, v in enumerate(self.data):
                    prefix_path = os.path.join(prefix, f'{idx}')
                    if self.is_image:
                        ret_images = self.save_image(prefix_path, v,
                                                     image_postfix)
                        ret_data.append(ret_images if isinstance(
                            ret_images, list) else [ret_images])
                        if self.build_html:
                            if isinstance(ret_images, list):
                                if isinstance(self.build_label, str):
                                    ret_label.append(
                                        [self.build_label for _ in ret_images])
                                elif isinstance(self.build_label[idx], list):
                                    assert len(self.build_label[idx]) == len(
                                        ret_images)
                                    ret_label.append(self.build_label[idx])
                                else:
                                    ret_label.append([
                                        self.build_label[idx]
                                        for _ in ret_images
                                    ])
                            else:
                                if isinstance(self.build_label, str):
                                    ret_label.append([self.build_label])
                                else:
                                    ret_label.append([self.build_label[idx]])
                    else:
                        ret_data.append(self.save_npy(prefix_path, v))
                if self.is_image and self.build_html:
                    html_prefix = prefix + '_probe.html'
                    html_file = self.save_html(html_prefix, ret_data,
                                               ret_label)
                    return {'ori_file': ret_data, 'html': html_file}
                else:
                    return {'ori_file': ret_data}
            else:
                return self.data
        elif isinstance(self.data, dict):
            if not self.basic_type:
                ret_data = []
                ret_label = []
                for k, v in self.data:
                    prefix_path = os.path.join(prefix, f'{k}_')
                    if self.is_image:
                        ret_images = self.save_image(prefix_path, v,
                                                     image_postfix)
                        if isinstance(ret_images, list):
                            ret_data.append(ret_images)
                        else:
                            ret_data.append([ret_images])
                        if self.build_html:
                            if isinstance(ret_images, list):
                                if isinstance(self.build_label, str):
                                    ret_label.append(
                                        [self.build_label for _ in ret_images])
                                elif isinstance(self.build_label[k], list):
                                    assert len(
                                        self.build_label[k]) == len(ret_images)
                                    ret_label.append(self.build_label[k])
                                else:
                                    ret_label.append([
                                        self.build_label[k] for _ in ret_images
                                    ])
                            else:
                                if isinstance(self.build_label, str):
                                    ret_label.append([self.build_label])
                                else:
                                    ret_label.append([self.build_label[k]])
                    else:
                        ret_data.append(self.save_npy(prefix_path, v))
                if self.is_image and self.build_html:
                    html_prefix = prefix + '_probe.html'
                    html_file = self.save_html(html_prefix, ret_data,
                                               ret_label)
                    return {'ori_file': ret_data, 'html': html_file}
                else:
                    return {'ori_file': ret_data}
            else:
                return self.data
        else:
            return self.data
