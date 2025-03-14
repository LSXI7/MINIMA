# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import io
from io import BytesIO

import onnx
import onnxruntime
import torch
from tools.infrared.scepter.modules.utils.distribute import we
from torch.onnx import OperatorExportTypes

type_map = {
    'float32': torch.float32,
    'float16': torch.float16,
    'int64': torch.int64,
    'int32': torch.int32,
    'int16': torch.int16,
    'int8': torch.int8
}


@torch.no_grad()
def save_develop_model_multi_io(model,
                                input_size,
                                input_type,
                                input_name,
                                output_name,
                                limit,
                                save_onnx_path=None,
                                save_pt_path=None):

    # save aggregation
    rank, word_size = we.rank, we.world_size
    assert isinstance(input_type, list)
    example = []
    dynamic_axes = {}

    for idx, type_name in enumerate(input_type):
        assert type_name in type_map
        torch_type = type_map[type_name]
        size = input_size[idx]
        if 'float' in type_name:
            input_ex = torch.rand(tuple(size)).type(torch_type).to(rank)
        elif 'int' in type_name:
            input_ex = torch.randint(limit[idx][0], limit[idx][1],
                                     tuple(size)).type(torch_type).to(rank)
        example.append(input_ex)
        dynamic_axes[input_name[idx]] = {0: 'batch_size'}

    if word_size > 0:
        save_module = model.module
    else:
        save_module = model

    def _check_eval(module):
        assert not module.training

    save_module.apply(_check_eval)

    if len(example) == 1:
        input_example = example[0]
    else:
        input_example = tuple(example)
    traced_script_module = torch.jit.trace(save_module, input_example)

    for p in traced_script_module.parameters():
        p.requires_grad = False
    if len(example) == 1:
        output = save_module(input_example)
    else:
        output = save_module(*input_example)
    print('Ori output:', output)

    module = None
    if save_pt_path is not None:
        traced_script_module.save(save_pt_path)
        module = torch.jit.load(io.BytesIO(open(save_pt_path, 'rb').read()),
                                map_location=torch.device(rank))
        if len(example) == 1:
            output = module(input_example)
        else:
            output = module(*input_example)
        print('PT output:', output)

    onnx_module = None
    if save_onnx_path is not None:
        # export the model to ONNX
        with torch.autocast(device_type='cpu',
                            enabled=True,
                            dtype=torch.bfloat16):
            with BytesIO() as f:
                torch.onnx.export(
                    save_module,
                    input_example,
                    f,
                    operator_export_type=OperatorExportTypes.ONNX,
                    opset_version=11,
                    input_names=input_name,
                    output_names=output_name,
                    dynamic_axes=dynamic_axes,
                    export_params=True,
                    do_constant_folding=True)
                onnx_model = onnx.load_from_string(f.getvalue())
        onnx.save(onnx_model, save_onnx_path)
        onnx_module = onnxruntime.InferenceSession(
            save_onnx_path, providers=['CUDAExecutionProvider'])
        input_data = {}
        for idx, ex in enumerate(example):
            input_data[input_name[idx]] = ex.detach().cpu().numpy()
        output_tensor = onnx_module.run(output_name, input_data)
        print('ONNX_OUTPUT', output_tensor, output_tensor[0].shape)
    return module, onnx_module
