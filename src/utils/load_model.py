import logging
import os
import torch
from copy import deepcopy


def load_roma(args, test_orginal_megadepth=False):
    import sys
    sys.path.append("./third_party/RoMa/")
    from third_party.RoMa.romatch import roma_outdoor
    from third_party.RoMa.romatch import tiny_roma_v1_outdoor
    if test_orginal_megadepth:
        from src.config.default_for_megadepth_dense import get_cfg_defaults
    else:
        from src.config.default import get_cfg_defaults
    from src.utils.data_io_roma import DataIOWrapper, lower_config
    config = get_cfg_defaults(inference=True)
    config = lower_config(config)
    device = torch.device('cuda:{}'.format(0) if torch.cuda.is_available() else 'cpu')
    if args.ckpt2 == 'large':
        # print('loading large roma model')
        if args.ckpt is not None:
            # print('loading roma model with weights')
            pth_path = args.ckpt
            state_dict = torch.load(pth_path, map_location=device)
            matcher = roma_outdoor(device=device, weights=state_dict)
        else:
            matcher = roma_outdoor(device=device)
    else:
        # print('loading tiny roma model')
        matcher = tiny_roma_v1_outdoor(device=device)
    matcher = DataIOWrapper(matcher, config=config["test"])
    logging.info(config["test"])
    return matcher


def load_loftr(args, test_orginal_megadepth=False):
    from third_party.LoFTR.src.loftr import LoFTR, default_cfg
    if test_orginal_megadepth:
        from src.config.default_for_megadepth_dense import get_cfg_defaults
    else:
        from src.config.default import get_cfg_defaults
    from src.utils.data_io_loftr import DataIOWrapper, lower_config
    config = get_cfg_defaults(inference=True)
    config = lower_config(config)
    # print("default_cfg['coarse']['temp_bug_fix']", default_cfg['coarse']['temp_bug_fix'])
    _default_cfg = deepcopy(default_cfg)
    filename = os.path.basename(args.ckpt)
    if filename != "outdoor_ds.ckpt":
        # print('not using official old model, now change bug')
        _default_cfg['coarse']['temp_bug_fix'] = True  # set to False when using the old ckpt

    _default_cfg['match_coarse']['thr'] = args.thr
    # print('now using thr:', args.thr)
    matcher = LoFTR(config=_default_cfg)
    matcher.load_state_dict(torch.load(args.ckpt)['state_dict'], strict=True)
    matcher = matcher.eval()

    matcher = DataIOWrapper(matcher, config=config["test"])
    logging.info(config["test"])
    return matcher


def load_sp_lg(args, test_orginal_megadepth=False):
    from third_party.LightGlue.lightglue import LightGlue, SuperPoint
    from third_party.LightGlue.lightglue.utils import rbd
    if test_orginal_megadepth:
        from src.config.default_for_megadepth_sparse import get_cfg_defaults
    else:
        from src.config.default import get_cfg_defaults
    from src.utils.data_io_sp_lg import DataIOWrapper, lower_config

    class Matching(torch.nn.Module):
        def __init__(self, sp_conf, lg_conf):
            super().__init__()
            device = torch.device('cuda:{}'.format(0) if torch.cuda.is_available() else 'cpu')
            self.extractor = SuperPoint(**sp_conf).eval().to(device)  # load the feature extractor
            self.matcher = LightGlue(features='superpoint', **lg_conf).eval().to(device)  # load the matcher
            n_layers = lg_conf['n_layers']
            # print(f"n_layers: {n_layers}")
            ckpt_path = args.ckpt
            # rename old state dict entries
            state_dict = torch.load(ckpt_path, map_location=device)
            for i in range(n_layers):
                pattern = f"self_attn.{i}", f"transformers.{i}.self_attn"
                state_dict = {k.replace(*pattern): v for k, v in state_dict.items()}
                pattern = f"cross_attn.{i}", f"transformers.{i}.cross_attn"
                state_dict = {k.replace(*pattern): v for k, v in state_dict.items()}
            self.matcher.load_state_dict(state_dict, strict=False)

        def forward(self, batch):
            # batch = {'image0': image0, 'image1': image1}
            image0 = batch['image0']
            image1 = batch['image1']
            # extract local features
            if test_orginal_megadepth:
                feats0 = self.extractor.extract(image0, resize=None)  # auto-resize the image, disable with resize=None
                feats1 = self.extractor.extract(image1, resize=None)
            else:
                feats0 = self.extractor.extract(image0)  # auto-resize the image, disable with resize=None
                feats1 = self.extractor.extract(image1)

            # match the features
            matches01 = self.matcher({'image0': feats0, 'image1': feats1})
            feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]  # remove batch dimension
            matches = matches01['matches']  # indices with shape (K,2)
            points0 = feats0['keypoints'][matches[..., 0]]  # coordinates in image #0, shape (K,2)
            points1 = feats1['keypoints'][matches[..., 1]]  # coordinates in image #1, shape (K,2)
            matching_scores0 = matches01['matching_scores0']
            matching_scores = matching_scores0[matches[..., 0]]

            return {'matching_scores': matching_scores, 'keypoints0': points0, 'keypoints1': points1}

    sp_conf = {
        "descriptor_dim": 256,
        "nms_radius": 4,
        "max_num_keypoints": 2048,
        "detection_threshold": 0.0005,
        "remove_borders": 4,
    }
    lg_conf = {
        "name": "lightglue",  # just for interfacing
        "input_dim": 256,  # input descriptor dimension (autoselected from weights)
        "descriptor_dim": 256,
        "add_scale_ori": False,
        "n_layers": 9,
        "num_heads": 4,
        "flash": True,  # enable FlashAttention if available.
        "mp": False,  # enable mixed precision
        "depth_confidence": 0.95,  # early stopping, disable with -1
        "width_confidence": 0.99,  # point pruning, disable with -1
        "filter_threshold": 0.1,  # match threshold
        "weights": None,
    }
    matcher = Matching(sp_conf, lg_conf)
    config = get_cfg_defaults(inference=True)
    config = lower_config(config)
    matcher = DataIOWrapper(matcher, config=config["test"])
    logging.info(config["test"])
    return matcher


def load_xoftr(args):
    from third_party.XoFTR.src.xoftr import XoFTR
    from src.config.default import get_cfg_defaults
    from src.utils.data_io import DataIOWrapper, lower_config
    config = get_cfg_defaults(inference=True)
    config = lower_config(config)
    config["xoftr"]["match_coarse"]["thr"] = args.match_threshold
    config["xoftr"]["fine"]["thr"] = args.fine_threshold
    ckpt = args.ckpt
    matcher = XoFTR(config=config["xoftr"])
    matcher = DataIOWrapper(matcher, config=config["test"], ckpt=ckpt)
    logging.info(config["test"])
    return matcher


def load_model(method, args, use_path=True, test_orginal_megadepth=False):
    if use_path:
        matcher = eval(f"load_{method}")(args, test_orginal_megadepth=test_orginal_megadepth)
        return matcher.from_paths
    else:
        matcher = eval(f"load_{method}")(args, test_orginal_megadepth=test_orginal_megadepth)
        return matcher.from_cv_imgs
