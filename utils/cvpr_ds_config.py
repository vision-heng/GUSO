from yacs.config import CfgNode as CN


def lower_config(yacs_cfg):
    if not isinstance(yacs_cfg, CN):
        return yacs_cfg
    return {k.lower(): lower_config(v) for k, v in yacs_cfg.items()}


_CN = CN()


##############  ↓  JAMMA Pipeline  ↓  ##############
_CN.JAMMA = CN()
_CN.RESOLUTION = (8, 2)  # options: [(8, 2), (16, 4)]
_CN.FINE_WINDOW_SIZE = 5  # window_size in fine_level, must be odd

_CN.COARSE = CN()
_CN.COARSE.D_MODEL = 256

_CN.MATCH_COARSE = CN()
_CN.MATCH_COARSE.USE_SM = True
_CN.MATCH_COARSE.THR = 0.1
_CN.MATCH_COARSE.BORDER_RM = 2
_CN.MATCH_COARSE.DSMAX_TEMPERATURE = 0.1
_CN.MATCH_COARSE.SKH_ITERS = 3
_CN.MATCH_COARSE.TRAIN_COARSE_PERCENT = 0.2  # training tricks: save GPU memory
_CN.MATCH_COARSE.TRAIN_PAD_NUM_GT_MIN = 200  # training tricks: avoid DDP deadlock
_CN.MATCH_COARSE.INFERENCE = True

_CN.FINE = CN()
_CN.FINE.D_MODEL = 128
_CN.FINE.DENSER = False # if true, match all features in fine-level windows
_CN.FINE.INFERENCE = True
_CN.FINE.DSMAX_TEMPERATURE = 0.1
_CN.FINE.THR = 0.1


_CN.BACKBONE_TYPE = 'wtconvnext'
_CN.CEM_TYPE = 'mambamixer'
_CN.MATCHING_TYPE = 'hm'



default_cfg = lower_config(_CN)