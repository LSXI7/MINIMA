from yacs.config import CfgNode as CN

INFERENCE = False

_CN = CN()

##############  ↓  XoFTR Pipeline  ↓  ##############
_CN.XOFTR = CN()
_CN.XOFTR.RESOLUTION = (8, 2)  # options: [(8, 2)]
_CN.XOFTR.FINE_WINDOW_SIZE = 5  # window_size in fine_level, must be odd
_CN.XOFTR.MEDIUM_WINDOW_SIZE = 3  # window_size in fine_level, must be odd

# 1. XoFTR-backbone (local feature CNN) config
_CN.XOFTR.RESNET = CN()
_CN.XOFTR.RESNET.INITIAL_DIM = 128
_CN.XOFTR.RESNET.BLOCK_DIMS = [128, 196, 256]  # s1, s2, s3

# 2. XoFTR-coarse module config
_CN.XOFTR.COARSE = CN()
_CN.XOFTR.COARSE.INFERENCE = INFERENCE
_CN.XOFTR.COARSE.D_MODEL = 256
_CN.XOFTR.COARSE.D_FFN = 256
_CN.XOFTR.COARSE.NHEAD = 8
_CN.XOFTR.COARSE.LAYER_NAMES = ['self', 'cross'] * 4
_CN.XOFTR.COARSE.ATTENTION = 'linear'  # options: ['linear', 'full']

# 3. Coarse-Matching config
_CN.XOFTR.MATCH_COARSE = CN()
_CN.XOFTR.MATCH_COARSE.INFERENCE = INFERENCE
_CN.XOFTR.MATCH_COARSE.D_MODEL = 256
_CN.XOFTR.MATCH_COARSE.THR = 0.3
_CN.XOFTR.MATCH_COARSE.BORDER_RM = 2
_CN.XOFTR.MATCH_COARSE.MATCH_TYPE = 'dual_softmax'  # options: ['dual_softmax']
_CN.XOFTR.MATCH_COARSE.DSMAX_TEMPERATURE = 0.1
_CN.XOFTR.MATCH_COARSE.TRAIN_COARSE_PERCENT = 0.2  # training tricks: save GPU memory
_CN.XOFTR.MATCH_COARSE.TRAIN_PAD_NUM_GT_MIN = 200  # training tricks: avoid DDP deadlock

# 4. XoFTR-fine module config
_CN.XOFTR.FINE = CN()
_CN.XOFTR.FINE.DENSER = False # if true, match all features in fine-level windows
_CN.XOFTR.FINE.INFERENCE = INFERENCE
_CN.XOFTR.FINE.DSMAX_TEMPERATURE = 0.1
_CN.XOFTR.FINE.THR = 0.1
_CN.XOFTR.FINE.MLP_HIDDEN_DIM_COEF = 2 # coef for  mlp hidden dim (hidden_dim = feat_dim * coef)
_CN.XOFTR.FINE.NHEAD_FINE_LEVEL = 8
_CN.XOFTR.FINE.NHEAD_MEDIUM_LEVEL = 7


# 5. XoFTR Losses

_CN.XOFTR.LOSS = CN()
_CN.XOFTR.LOSS.FOCAL_ALPHA = 0.25
_CN.XOFTR.LOSS.FOCAL_GAMMA = 2.0
_CN.XOFTR.LOSS.POS_WEIGHT = 1.0
_CN.XOFTR.LOSS.NEG_WEIGHT = 1.0

# -- # coarse-level
_CN.XOFTR.LOSS.COARSE_WEIGHT = 0.5
# -- # fine-level
_CN.XOFTR.LOSS.FINE_WEIGHT = 0.3
# -- # sub-pixel
_CN.XOFTR.LOSS.SUB_WEIGHT = 1 * 10**4

##############  Dataset  ##############
_CN.DATASET = CN()
# 1. data config
# training and validating
_CN.DATASET.TRAIN_DATA_SOURCE = None  # options: ['ScanNet', 'MegaDepth']
_CN.DATASET.TRAIN_DATA_ROOT = None
_CN.DATASET.TRAIN_POSE_ROOT = None  # (optional directory for poses)
_CN.DATASET.TRAIN_NPZ_ROOT = None
_CN.DATASET.TRAIN_LIST_PATH = None
_CN.DATASET.TRAIN_INTRINSIC_PATH = None
_CN.DATASET.VAL_DATA_SOURCE = None
_CN.DATASET.VAL_DATA_ROOT = None
_CN.DATASET.VAL_POSE_ROOT = None  # (optional directory for poses)
_CN.DATASET.VAL_NPZ_ROOT = None
_CN.DATASET.VAL_LIST_PATH = None    # None if val data from all scenes are bundled into a single npz file
_CN.DATASET.VAL_INTRINSIC_PATH = None
# testing
_CN.DATASET.TEST_DATA_SOURCE = None
_CN.DATASET.TEST_DATA_ROOT = None
_CN.DATASET.TEST_POSE_ROOT = None  # (optional directory for poses)
_CN.DATASET.TEST_NPZ_ROOT = None
_CN.DATASET.TEST_LIST_PATH = None   # None if test data from all scenes are bundled into a single npz file
_CN.DATASET.TEST_INTRINSIC_PATH = None

# 2. dataset config
# general options
_CN.DATASET.MIN_OVERLAP_SCORE_TRAIN = 0.4  # discard data with overlap_score < min_overlap_score
_CN.DATASET.MIN_OVERLAP_SCORE_TEST = 0.0
_CN.DATASET.AUGMENTATION_TYPE = "rgb_thermal"  # options: [None, 'dark', 'mobile']

# MegaDepth options
_CN.DATASET.MGDPT_IMG_RESIZE = 640  # resize the longer side, zero-pad bottom-right to square.
_CN.DATASET.MGDPT_IMG_PAD = True  # pad img to square with size = MGDPT_IMG_RESIZE
_CN.DATASET.MGDPT_DEPTH_PAD = True  # pad depthmap to square with size = 2000
_CN.DATASET.MGDPT_DF = 8

# VisTir options
_CN.DATASET.VISTIR_IMG_RESIZE = 640  # resize the longer side, zero-pad bottom-right to square.
_CN.DATASET.VISTIR_IMG_PAD = False  # pad img to square with size = VISTIR_IMG_RESIZE
_CN.DATASET.VISTIR_DF = 8

# Pretrain dataset options
_CN.DATASET.PRETRAIN_IMG_RESIZE = 640  # resize the longer side, zero-pad bottom-right to square.
_CN.DATASET.PRETRAIN_IMG_PAD = True  # pad img to square with size = PRETRAIN_IMG_RESIZE
_CN.DATASET.PRETRAIN_DF = 8
_CN.DATASET.PRETRAIN_FRAME_GAP = 2 # the gap between video frames of Kaist dataset

##############  Trainer  ##############
_CN.TRAINER = CN()
_CN.TRAINER.WORLD_SIZE = 1
_CN.TRAINER.CANONICAL_BS = 64
_CN.TRAINER.CANONICAL_LR = 6e-3
_CN.TRAINER.SCALING = None  # this will be calculated automatically
_CN.TRAINER.FIND_LR = False  # use learning rate finder from pytorch-lightning

_CN.TRAINER.USE_WANDB = False # use weight and biases

# optimizer
_CN.TRAINER.OPTIMIZER = "adamw"  # [adam, adamw]
_CN.TRAINER.TRUE_LR = None  # this will be calculated automatically at runtime
_CN.TRAINER.ADAM_DECAY = 0.  # ADAM: for adam
_CN.TRAINER.ADAMW_DECAY = 0.1

# step-based warm-up
_CN.TRAINER.WARMUP_TYPE = 'linear'  # [linear, constant]
_CN.TRAINER.WARMUP_RATIO = 0.
_CN.TRAINER.WARMUP_STEP = 4800

# learning rate scheduler
_CN.TRAINER.SCHEDULER = 'MultiStepLR'  # [MultiStepLR, CosineAnnealing, ExponentialLR]
_CN.TRAINER.SCHEDULER_INTERVAL = 'epoch'    # [epoch, step]
_CN.TRAINER.MSLR_MILESTONES = [3, 6, 9, 12]  # MSLR: MultiStepLR
_CN.TRAINER.MSLR_GAMMA = 0.5
_CN.TRAINER.COSA_TMAX = 30  # COSA: CosineAnnealing
_CN.TRAINER.ELR_GAMMA = 0.999992  # ELR: ExponentialLR, this value for 'step' interval

# plotting related
_CN.TRAINER.ENABLE_PLOTTING = True
_CN.TRAINER.N_VAL_PAIRS_TO_PLOT = 128     # number of val/test paris for plotting
_CN.TRAINER.PLOT_MODE = 'evaluation'  # ['evaluation', 'confidence']
_CN.TRAINER.PLOT_MATCHES_ALPHA = 'dynamic'

# geometric metrics and pose solver
_CN.TRAINER.EPI_ERR_THR = 5e-4  # recommendation: 5e-4 for ScanNet, 1e-4 for MegaDepth (from SuperGlue)
_CN.TRAINER.POSE_GEO_MODEL = 'E'  # ['E', 'F', 'H']
_CN.TRAINER.POSE_ESTIMATION_METHOD = 'RANSAC'  # [RANSAC, DEGENSAC, MAGSAC]
_CN.TRAINER.RANSAC_PIXEL_THR = 0.5
_CN.TRAINER.RANSAC_CONF = 0.99999
_CN.TRAINER.RANSAC_MAX_ITERS = 10000
_CN.TRAINER.USE_MAGSACPP = False

# data sampler for train_dataloader
_CN.TRAINER.DATA_SAMPLER = 'scene_balance'  # options: ['scene_balance', 'random', 'normal']
# 'scene_balance' config
_CN.TRAINER.N_SAMPLES_PER_SUBSET = 200
_CN.TRAINER.SB_SUBSET_SAMPLE_REPLACEMENT = True  # whether sample each scene with replacement or not
_CN.TRAINER.SB_SUBSET_SHUFFLE = True  # after sampling from scenes, whether shuffle within the epoch or not
_CN.TRAINER.SB_REPEAT = 1  # repeat N times for training the sampled data
# 'random' config
_CN.TRAINER.RDM_REPLACEMENT = True
_CN.TRAINER.RDM_NUM_SAMPLES = None

# gradient clipping
_CN.TRAINER.GRADIENT_CLIPPING = 0.5

# reproducibility
# This seed affects the data sampling. With the same seed, the data sampling is promised
# to be the same. When resume training from a checkpoint, it's better to use a different
# seed, otherwise the sampled data will be exactly the same as before resuming, which will
# cause less unique data items sampled during the entire training.
# Use of different seed values might affect the final training result, since not all data items
# are used during training on ScanNet. (60M pairs of images sampled during traing from 230M pairs in total.)
_CN.TRAINER.SEED = 66

##############  Pretrain  ##############
_CN.PRETRAIN = CN()
_CN.PRETRAIN.PATCH_SIZE = 64 # patch sıze for masks
_CN.PRETRAIN.MASK_RATIO = 0.5 
_CN.PRETRAIN.MAE_MARGINS = [0, 0.4, 0, 0] # margins not to be masked (up bottom left right)
_CN.PRETRAIN.VAL_SEED = 42 # rng seed to crate the same masks for validation

_CN.XOFTR.PRETRAIN_PATCH_SIZE = _CN.PRETRAIN.PATCH_SIZE 

##############  Test/Inference  ##############
_CN.TEST = CN()
_CN.TEST.IMG0_RESIZE = 640 # resize the longer side
_CN.TEST.IMG1_RESIZE = 640 # resize the longer side
_CN.TEST.DF = 8
_CN.TEST.PADDING = False  # pad img to square with size = IMG0_RESIZE, IMG1_RESIZE
_CN.TEST.COARSE_SCALE = 0.125

def get_cfg_defaults(inference=False):
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    if inference:
        _CN.XOFTR.COARSE.INFERENCE = True
        _CN.XOFTR.MATCH_COARSE.INFERENCE = True
        _CN.XOFTR.FINE.INFERENCE = True
    return _CN.clone()
