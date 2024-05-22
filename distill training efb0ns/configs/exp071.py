from types import SimpleNamespace

cfg = SimpleNamespace(**{})

cfg.exp_name = 'exp071'
cfg.apex = True  # [True, False]

######################
# Globals #
######################
cfg.seed = 42
cfg.epochs = 400
cfg.folds = [0]  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
cfg.external = True
cfg.use_sampler = True #False  # [True, False]

######################
# Data #
######################
cfg.train_datadir = "../"#"/home/simon/disk1/Simon/Code/kaggle_competion_list/Birdclef/birdclef-2024/input/" #


######################
# Dataset #
######################
cfg.period = 5 #20 #20  # [5, 10, 20, 30]
cfg.frames = -1  # [-1, 480000, 640000, 960000]

cfg.use_pcen = False
cfg.n_mels = 128  # [64, 128, 224, 256]
cfg.fmin = 300  # [20, 50]
cfg.fmax = 16000  # [14000, 16000]
cfg.n_fft = 2048  # [1024, 2048]
cfg.hop_length = 512  # [320, 512]
cfg.sample_rate = 32000
cfg.secondary_coef = 1.0

cfg.target_columns = ['asbfly', 'ashdro1', 'ashpri1', 'ashwoo2', 'asikoe2', 'asiope1',
                    'aspfly1', 'aspswi1', 'barfly1', 'barswa', 'bcnher', 'bkcbul1',
                    'bkrfla1', 'bkskit1', 'bkwsti', 'bladro1', 'blaeag1', 'blakit1',
                    'blhori1', 'blnmon1', 'blrwar1', 'bncwoo3', 'brakit1', 'brasta1',
                    'brcful1', 'brfowl1', 'brnhao1', 'brnshr', 'brodro1', 'brwjac1',
                    'brwowl1', 'btbeat1', 'bwfshr1', 'categr', 'chbeat1', 'cohcuc1',
                    'comfla1', 'comgre', 'comior1', 'comkin1', 'commoo3', 'commyn',
                    'compea', 'comros', 'comsan', 'comtai1', 'copbar1', 'crbsun2',
                    'cregos1', 'crfbar1', 'crseag1', 'dafbab1', 'darter2', 'eaywag1',
                    'emedov2', 'eucdov', 'eurbla2', 'eurcoo', 'forwag1', 'gargan',
                    'gloibi', 'goflea1', 'graher1', 'grbeat1', 'grecou1', 'greegr',
                    'grefla1', 'grehor1', 'grejun2', 'grenig1', 'grewar3', 'grnsan',
                    'grnwar1', 'grtdro1', 'gryfra', 'grynig2', 'grywag', 'gybpri1',
                    'gyhcaf1', 'heswoo1', 'hoopoe', 'houcro1', 'houspa', 'inbrob1',
                    'indpit1', 'indrob1', 'indrol2', 'indtit1', 'ingori1', 'inpher1',
                    'insbab1', 'insowl1', 'integr', 'isbduc1', 'jerbus2', 'junbab2',
                    'junmyn1', 'junowl1', 'kenplo1', 'kerlau2', 'labcro1', 'laudov1',
                    'lblwar1', 'lesyel1', 'lewduc1', 'lirplo', 'litegr', 'litgre1',
                    'litspi1', 'litswi1', 'lobsun2', 'maghor2', 'malpar1', 'maltro1',
                    'malwoo1', 'marsan', 'mawthr1', 'moipig1', 'nilfly2', 'niwpig1',
                    'nutman', 'orihob2', 'oripip1', 'pabflo1', 'paisto1', 'piebus1',
                    'piekin1', 'placuc3', 'plaflo1', 'plapri1', 'plhpar1', 'pomgrp2',
                    'purher1', 'pursun3', 'pursun4', 'purswa3', 'putbab1', 'redspu1',
                    'rerswa1', 'revbul', 'rewbul', 'rewlap1', 'rocpig', 'rorpar',
                    'rossta2', 'rufbab3', 'ruftre2', 'rufwoo2', 'rutfly6', 'sbeowl1',
                    'scamin3', 'shikra1', 'smamin1', 'sohmyn1', 'spepic1', 'spodov',
                    'spoowl1', 'sqtbul1', 'stbkin1', 'sttwoo1', 'thbwar1', 'tibfly3',
                    'tilwar1', 'vefnut1', 'vehpar1', 'wbbfly1', 'wemhar1', 'whbbul2',
                    'whbsho3', 'whbtre1', 'whbwag1', 'whbwat1', 'whbwoo2', 'whcbar1',
                    'whiter2', 'whrmun', 'whtkin2', 'woosan', 'wynlau1', 'yebbab1',
                    'yebbul3', 'zitcis1']

cfg.bird2id = {b: i for i, b in enumerate(cfg.target_columns)}
cfg.id2bird = {i: b for i, b in enumerate(cfg.target_columns)}
######################
# Augmentation #
######################

cfg.aug_noise            = 0.
cfg.aug_gain             = 0.0
cfg.aug_wave_pitchshift  = 0.0 #効果はあるので入れたいが、重いので実験中は使わない
cfg.aug_wave_shift       = 0.

cfg.aug_spec_xymasking   = 0.
cfg.aug_spec_coarsedrop  = 0.
cfg.aug_spec_hflip       = 0.

##mixup param
cfg.aug_wave_mixup       = 1.0
cfg.aug_spec_mixup       = 0.0
cfg.aug_spec_mixup_prob  = 0.5 #specmixup++をさせる確率
cfg.alpha=0.95

cfg.smoothing_value      = 0.0
# spec_mix_mask_percent = 20

cfg.aug_spec_xymasking = 0.1


######################
# Loaders #
######################
cfg.loader_params = {
    "train": {
        "batch_size": 64,
        "pin_memory": True,
        "num_workers": 12,
        "drop_last": True,
        "shuffle": True if not cfg.use_sampler else False
    },
    "valid": {
        "batch_size": 64,
        "pin_memory": True,
        "num_workers": 12,
        "shuffle": False
    }
}

######################
# Model #
######################
cfg.backbone = 'tf_efficientnet_b0_ns' # 'spnasnet_100' # 'eca_nfnet_l0'
cfg.use_imagenet_weights = True
cfg.pretrained = True
cfg.num_classes = 182
cfg.in_channels = 1
cfg.lr_max = 1e-3
cfg.lr_min = 1e-7
cfg.weight_decay = 1e-6
cfg.max_grad_norm = 10    # 1
cfg.early_stopping = 20

cfg.mixup_p = 1.0


cfg.pretrained_weights = True #False
# cfg.pretrained_path = "/home/simon/disk1/Simon/Code/kaggle_competion_list/Birdclef/birdclef-2024/BirdCLEF-2023-Identify-bird-calls-in-soundscapes-main-20240518T060118Z-001/timm/tf_efficientnet_b0.ns_jft_in1k/pytorch_model.bin"#None # '../models/exp087_eca_nfnet_l0/fold_0_model.bin'
cfg.pretrained_path = "/home/simon/disk1/Simon/Code/kaggle_competion_list/Birdclef/birdclef-2024/BirdCLEF-2023-Identify-bird-calls-in-soundscapes-main-20240518T060118Z-001/BirdCLEF-2023-Identify-bird-calls-in-soundscapes-main/models/exp071_tf_efficientnet_b0_ns/fold_0_model.bin"
# "/home/simon/disk1/Simon/Code/kaggle_competion_list/Birdclef/birdclef-2024/BirdCLEF-2023-Identify-bird-calls-in-soundscapes-main-20240518T060118Z-001/BirdCLEF-2023-Identify-bird-calls-in-soundscapes-main/models/exp071_tf_efficientnet_b0_ns/fold_0_distill0_9766.bin"

cfg.model_output_path = f"../models/{cfg.exp_name}_{cfg.backbone}"
