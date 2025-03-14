# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import functools
import os
import pickle
import random
import warnings
from collections import OrderedDict

import numpy as np
import torch
import torch.distributed as dist
from tools.infrared.scepter.modules.utils.model import StdMsg

__all__ = [
    'gather_data', 'we', 'broadcast', 'barrier', 'reduce_scatter', 'reduce',
    'all_reduce', 'send', 'recv', 'isend', 'irecv', 'scatter',
    'shared_random_seed'
]

try:
    from onnxruntime.transformers.benchmark_helper import set_random_seed
except Exception:

    def set_random_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)


def get_dist_info():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    else:
        return 0, 1


def gather_data(data):
    """ Gather tensors and other picklable objects to rank 0.
    Will recursively walk through inner list and dict values.

    Args:
        data (any): Anything.

    Returns:
        A object has same structure with input `data`.
    """
    if not we.is_distributed:
        return data
    if isinstance(data, torch.Tensor):
        return gather_gpu_tensors(data)
    elif isinstance(data, dict):
        # Keep in order, dict type DO NOT guarantee a fixed key order
        keys = sorted(list(data.keys()))
        ret = OrderedDict()
        for key in keys:
            ret[key] = gather_data(data[key])
        return ret
    elif isinstance(data, list):
        return gather_list(data)
    else:
        return gather_picklable(data)


def gather_list(data):
    """ Gather list of picklable objects to a new list on rank 0.
    Will NOT recursively walk through.

    Args:
        data (list): List of picklable things.

    Returns:
        A new flat list.
    """
    rank, _ = get_dist_info()
    list_of_list = gather_picklable(data)
    if rank == 0:
        return sum(list_of_list, [])


def gather_picklable(data):
    """ Gather picklable object to a list on rank 0.
    Will NOT recursively walk through.

    Args:
        data (picklable): Picklable data.

    Returns:
        A list contains data collected.
    """
    from packaging import version
    from torch.version import __version__
    if version.parse(__version__) < version.parse('1.8.0'):
        return _gather_picklable_custom(data)
    else:
        rank, world_size = we.rank, we.world_size
        obj_list = [None for _ in range(world_size)]
        dist.all_gather_object(obj_list, data)
        if rank == 0:
            return obj_list


def _gather_picklable_custom(data):
    """ Custom implementation function to gather picklable object to a list on rank 0.
    If torch version is lower than 1.8.0, use this.

    Args:
        data (picklable): Picklable data.

    Returns:
        A list contains data collected.
    """
    import pickle
    byte_tensor = torch.tensor(bytearray(pickle.dumps(data)),
                               dtype=torch.uint8,
                               device='cuda')
    rank, world_size = we.rank, we.world_size
    shape_tensor = torch.tensor(byte_tensor.shape, device='cuda')
    shape_list = [shape_tensor.clone() for _ in range(world_size)]
    dist.all_gather(shape_list, shape_tensor)
    shape_max = torch.tensor(shape_list).max()

    tensor_send = torch.zeros(shape_max,
                              dtype=byte_tensor.dtype,
                              device='cuda')
    tensor_send[0:shape_tensor[0]] = byte_tensor
    tensor_list = [torch.zeros_like(tensor_send) for _ in range(world_size)]
    dist.all_gather(tensor_list, tensor_send)

    if rank == 0:
        data_out = []
        for tensor_recv, shape_recv in zip(tensor_list, shape_list):
            new_data = pickle.loads(
                tensor_recv[:shape_recv[0]].cpu().numpy().tobytes())
            data_out.append(new_data)
        return data_out


def gather_gpu_tensors(tensor, all_recv=False, is_cat=True):
    """
    Args:
        tensor (torch.Tensor):
        all_recv: Gather tensor to rank 0 and concat it.

    Returns:
        A new tensor.
    """
    assert dist.get_backend() == 'nccl'

    device = tensor.device
    if device.type == 'cpu':
        tensor = tensor.to(we.device_id)

    rank, world_size = we.rank, we.world_size

    shape_tensor = torch.tensor(tensor.shape[0], device='cuda')
    shape_list = [shape_tensor.clone() for _ in range(world_size)]
    dist.all_gather(shape_list, shape_tensor)
    shape_max = torch.tensor(shape_list).max()

    tensor_send = torch.zeros((shape_max, *tensor.shape[1:]),
                              dtype=tensor.dtype,
                              device='cuda')
    tensor_send[0:tensor.shape[0]] = tensor
    tensor_list = [torch.zeros_like(tensor_send) for _ in range(world_size)]
    dist.all_gather(tensor_list, tensor_send)
    if not all_recv:
        if rank == 0:
            if not is_cat:
                return tensor_list, shape_list
            tensors_out = []
            for tensor_recv, shape_recv in zip(tensor_list, shape_list):
                tensors_out.append(tensor_recv[0:shape_recv])
            tensor_out = torch.cat(tensors_out).contiguous()
            if device.type == 'cpu':
                tensor_out = tensor_out.cpu()
            del tensor_list, shape_list
            return tensor_out
        else:
            del tensor_list, shape_list
    else:
        if not is_cat:
            return tensor_list, shape_list
        tensors_out = []
        for tensor_recv, shape_recv in zip(tensor_list, shape_list):
            tensors_out.append(tensor_recv[0:shape_recv])
        tensor_out = torch.cat(tensors_out).contiguous()
        if device.type == 'cpu':
            tensor_out = tensor_out.cpu()
        del tensor_list, shape_list
        return tensor_out


def broadcast(tensor, src, group=None, **kwargs):
    if we.is_distributed:
        return dist.broadcast(tensor, src, group, **kwargs)


def barrier():
    if we.is_distributed:
        dist.barrier()


@functools.lru_cache()
def get_global_gloo_group():
    backend = dist.get_backend()
    assert backend in ['gloo', 'nccl']
    if backend == 'nccl':
        return dist.new_group(backend='gloo')
    else:
        return dist.group.WORLD


def reduce_scatter(output,
                   input_list,
                   op=dist.ReduceOp.SUM,
                   group=None,
                   **kwargs):
    if we.is_distributed:
        return dist.reduce_scatter(output, input_list, op, group, **kwargs)


def all_reduce(tensor, op=dist.ReduceOp.SUM, group=None, **kwargs):
    if we.is_distributed:
        return dist.all_reduce(tensor, op, group, **kwargs)


def reduce(tensor, dst, op=dist.ReduceOp.SUM, group=None, **kwargs):
    if we.is_distributed:
        return dist.reduce(tensor, dst, op, group, **kwargs)


def _serialize_to_tensor(data):
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage)
    return tensor


def _unserialize_from_tensor(recv_data):
    buffer = recv_data.cpu().numpy().tobytes()
    return pickle.loads(buffer)


def send(tensor, dst, group=None, **kwargs):
    if we.is_distributed:
        assert tensor.is_contiguous(
        ), 'ops.send requires the tensor to be contiguous()'
        return dist.send(tensor, dst, group, **kwargs)


def recv(tensor, src=None, group=None, **kwargs):
    if we.is_distributed:
        assert tensor.is_contiguous(
        ), 'ops.recv requires the tensor to be contiguous()'
        return dist.recv(tensor, src, group, **kwargs)


def isend(tensor, dst, group=None, **kwargs):
    if we.is_distributed:
        assert tensor.is_contiguous(
        ), 'ops.isend requires the tensor to be contiguous()'
        return dist.isend(tensor, dst, group, **kwargs)


def irecv(tensor, src=None, group=None, **kwargs):
    if we.is_distributed:
        assert tensor.is_contiguous(
        ), 'ops.irecv requires the tensor to be contiguous()'
        return dist.irecv(tensor, src, group, **kwargs)


def scatter(data, scatter_list=None, src=0, group=None, **kwargs):
    r"""NOTE: only supports CPU tensor communication.
    """
    world_size = we.world_size
    if world_size == 1:
        data.copy_(scatter_list[0])
    if group is None:
        group = get_global_gloo_group()
    return dist.scatter(data, scatter_list, src, group, **kwargs)


def shared_random_seed():
    seed = np.random.randint(2**31)
    all_seeds, _ = gather_gpu_tensors(seed, all_recv=True, is_cat=False)
    return all_seeds[0]


global we


def mp_worker(gpu, ngpus_per_node, cfg, fn, pmi_rank, world_size, work_env):
    rank = pmi_rank * ngpus_per_node + gpu
    work_env.device_id = gpu % ngpus_per_node
    work_env.rank = rank
    dist.init_process_group(backend='nccl', world_size=world_size, rank=rank)
    torch.backends.cudnn.deterministic = cfg.ENV.get('CUDNN_DETERMINISTIC',
                                                     True)
    torch.backends.cudnn.benchmark = cfg.ENV.get('CUDNN_BENCHMARK', False)
    torch.cuda.set_device(work_env.device_id)
    if work_env.logger is not None:
        work_env.logger.info(
            f'Now running in the distributed environment with world size {work_env.world_size}!'
        )
        work_env.logger.info(f'PMI rank {pmi_rank}!')
        work_env.logger.info(f'Nums of gpu {ngpus_per_node}!')
        work_env.logger.info(
            f'Current rank {work_env.rank} current devices num {ngpus_per_node} '
            f'current machine rank {pmi_rank} and all world size {world_size}')

    we.set_env(work_env.get_env())
    fn(cfg)


class Workenv(object):
    def __init__(self):
        self.initialized = False
        self.is_distributed = False
        self.sync_bn = False
        self.rank = 0
        self.world_size = 1
        self.device_id = 0
        self.device_count = 1
        self.seed = 2023
        self.debug = False
        self.use_pl = False
        self.launcher = 'spawn' if torch.cuda.device_count() > 1 else None
        self.data_online = False
        self.share_storage = False

    def init_env(self, config, fn, logger=None):
        # if use pytorch_lightning: then direct use pytorch_lightning.
        config.ENV = config.get('ENV', {})
        self.seed = config.ENV.get('SEED', 2023)
        self.debug = os.environ.get('ES_DEBUG', None) == 'true'
        set_random_seed(self.seed)
        if logger is not None:
            logger.info(f'And running with seed {self.seed}!')
        if config.ENV.get('USE_PL', False):
            self.use_pl = config.ENV.USE_PL
            fn(config)
            return
        if hasattr(config, 'args') and hasattr(config.args, 'launcher'):
            self.launcher = config.args.launcher
        if logger is None:
            self.logger = StdMsg(name='env')
        else:
            self.logger = logger

        self.data_online = os.environ.get('DATA_ONLINE', None) == 'true'
        self.share_storage = os.environ.get('SHARE_STORAGE', None) == 'true'

        if not torch.cuda.is_available():
            self.device_id = 'cpu'
            fn(config)
            return

        if (os.environ.get('WORLD_SIZE') is None or int(os.environ.get('WORLD_SIZE')) == 1) \
                and torch.cuda.device_count() == 1 and not self.launcher == 'dist':
            self.device_id = 0
            fn(config)
            return

        if self.launcher == 'torchrun':
            try:
                torch.multiprocessing.set_start_method('spawn')
            except Exception as e:
                warnings.warn(f'{e}')
            # checking train mode is distributed or not
            if not os.environ.get('WORLD_SIZE') is None:
                if self.logger is not None:
                    self.logger.info(
                        f"Now running in the distributed environment with {os.environ.get('WORLD_SIZE')}!"
                    )
                self.is_distributed = True
            if not self.initialized:
                if self.is_distributed:
                    self.backend = config.ENV.get('BACKEND', 'nccl')
                    self.sync_bn = config.ENV.get('SYNC_BN', False)
                    dist.init_process_group(backend=self.backend)
                    # dist.barrier()
                self.initialized = True
                if dist.is_initialized():
                    self.rank, self.world_size = dist.get_rank(
                    ), dist.get_world_size()
                    if self.logger is not None:
                        self.logger.info(f'And running in rank {self.rank}!')
                        self.logger.info(
                            f"And cuda visible devices {os.environ.get('CUDA_VISIBLE_DEVICES')}!"
                        )
                else:
                    self.rank, self.world_size = 0, 1
                local_devices = os.environ.get(
                    'LOCAL_WORLD_SIZE') or torch.cuda.device_count()
                local_devices = int(local_devices)
                self.device_count = local_devices
                self.device_id = self.rank % local_devices
            self.logger.info(f"We's attributes: \n"
                             f' launcher {self.launcher} \n'
                             f' rank {self.rank} \n'
                             f' world size {self.world_size} \n'
                             f' device_id {self.device_id}')
            torch.cuda.set_device(self.device_id)
            torch.backends.cudnn.deterministic = config.ENV.get(
                'CUDNN_DETERMINISTIC', True)
            torch.backends.cudnn.benchmark = config.ENV.get(
                'CUDNN_BENCHMARK', False)
            fn(config)
        else:
            import torch.multiprocessing as mp
            if 'MASTER_ADDR' not in os.environ:
                os.environ['MASTER_ADDR'] = 'localhost'
            if 'MASTER_PORT' not in os.environ:
                os.environ['MASTER_PORT'] = '14567'
            pmi_rank = int(os.environ.get('RANK', 0))
            pmi_world_size = int(os.environ.get('WORLD_SIZE', 1))
            ngpus_per_node = os.environ.get(
                'LOCAL_WORLD_SIZE') or torch.cuda.device_count()
            ngpus_per_node = int(ngpus_per_node)
            self.device_count = ngpus_per_node
            world_size = ngpus_per_node * pmi_world_size
            self.world_size = world_size
            if self.world_size > 1:
                self.is_distributed = True
                self.initialized = True
            if self.is_distributed:
                self.backend = config.ENV.get('BACKEND', 'nccl')
                self.sync_bn = config.ENV.get('SYNC_BN', False)
            mp.spawn(mp_worker,
                     nprocs=ngpus_per_node,
                     args=(ngpus_per_node, config, fn, pmi_rank, world_size,
                           self))

    def get_env(self):
        return self.__dict__

    def set_env(self, we_env):
        for k, v in we_env.items():
            setattr(self, k, v)
        set_random_seed(self.seed)

    def __str__(self):
        environ_str = f'Now running in the distributed environment with world size {self.world_size}\n!'
        environ_str += f'Current pod have {self.device_count} devices!\n'
        environ_str += f'Current task executes on device {self.device_id}!\n'
        environ_str += f"Current task's global rank is {self.rank} \n"
        environ_str += f"Current task's data online is set {self.data_online} \n"
        environ_str += f"Current task's share storage is set {self.share_storage} \n"
        environ_str += f"Current task's global seed is set {self.seed}"
        return environ_str


we = Workenv()
