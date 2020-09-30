
def get_hyperparams(args, test_code=0):
    if not test_code:
        if args.hyperparams_set_index == 1: # DenseNet Hyperparameters
            args.epochs = 90
            args.start_eval_epoch = 0
            args.batch_size = 256
            ### data transform
            args.autoaugment = False
            args.colorjitter = True
            args.change_light = True
            ### optimizer
            args.optimizer = 'SGD'
            args.lr = 0.1
            args.momentum = 0.9
            args.weigh_decay_apply_on_all = True #TODO: weight decay apply on which params
            args.weight_decay = 1e-4
            args.nesterov = True
            ### criterion
            args.labelsmooth = 0
            ### lr scheduler
            args.scheduler = 'multistep'
            args.lr_decay_rate = 0.1
            args.lr_decay_step = 30
            ### Mixup
            args.mixup = 0.0
            return args
        elif args.hyperparams_set_index == 11: # DenseNet Hyperparameters for Big Model (EfficientNet-B5)
            args.epochs = 90
            args.start_eval_epoch = 0
            args.batch_size = 128
            ### data transform
            args.autoaugment = False
            args.colorjitter = True
            args.change_light = True
            ### optimizer
            args.optimizer = 'SGD'
            args.lr = 0.05
            args.momentum = 0.9
            args.weigh_decay_apply_on_all = True  # TODO: weight decay apply on which params
            args.weight_decay = 1e-4
            args.nesterov = True
            ### criterion
            args.labelsmooth = 0
            ### lr scheduler
            args.scheduler = 'multistep'
            args.lr_decay_rate = 0.1
            args.lr_decay_step = 30
            ### Mixup
            args.mixup = 0.0
            return args
        elif args.hyperparams_set_index == 2: # ResNeSt Hyperparameters (EfficientNet-B5)
            args.epochs = 270
            args.start_eval_epoch = 230
            args.batch_size = 256
            ### data transform
            args.autoaugment = True
            args.colorjitter = True
            args.change_light = True
            ### optimizer
            args.optimizer = 'SGD'
            args.lr = 0.1
            args.momentum = 0.9
            args.weigh_decay_apply_on_all = False  # TODO: weight decay apply on which params
            args.weight_decay = 1e-4
            args.nesterov = True
            ### criterion
            args.labelsmooth = 0.1
            ### lr scheduler
            args.scheduler = 'cosine'
            args.warmup_epoch = 0
            ### Mixup
            args.mixup = 0.2
            return args
        elif args.hyperparams_set_index == 4: # RegNet
            args.epochs = 100
            args.start_eval_epoch = 0
            args.batch_size = 1024
            ### data transform
            args.autoaugment = False
            args.colorjitter = False
            args.change_light = True
            ### optimizer
            args.optimizer = 'SGD'
            args.lr = 0.8
            args.momentum = 0.9
            args.weigh_decay_apply_on_all = True  # TODO: weight decay apply on which params
            args.weight_decay = 5e-5
            args.nesterov = True
            ### criterion
            args.labelsmooth = 0
            ### lr scheduler
            args.scheduler = 'cosine'
            args.warmup_epoch = 5
            args.warmup_lr = 0.08 # see RegNet pycls/core/config.py _C.OPTIM.WARMUP_FACTOR = 0.1
            ### Mixup
            args.mixup = 0.0
            return args
        elif args.hyperparams_set_index == 41: # RegNet-Batchsize256/Lr0.1
            args.epochs = 100
            args.start_eval_epoch = 0
            args.batch_size = 1024
            ### data transform
            args.autoaugment = False
            args.colorjitter = False
            args.change_light = True
            ### optimizer
            args.optimizer = 'SGD'
            args.lr = 0.4
            args.momentum = 0.9
            args.weigh_decay_apply_on_all = True  # TODO: weight decay apply on which params
            args.weight_decay = 5e-5
            args.nesterov = True
            ### criterion
            args.labelsmooth = 0
            ### lr scheduler
            args.scheduler = 'cosine'
            args.warmup_epoch = 5
            args.warmup_lr = 0.04 # see RegNet pycls/core/config.py _C.OPTIM.WARMUP_FACTOR = 0.1
            ### Mixup
            args.mixup = 0.0
            return args
        elif args.hyperparams_set_index == 5: # ShuffleNet-V2
            args.epochs = 240
            args.start_eval_epoch = 200
            args.batch_size = 1024
            ### data transform
            args.autoaugment = False
            args.colorjitter = True
            args.change_light = False
            ### optimizer
            args.optimizer = 'SGD'
            args.lr = 0.5
            args.momentum = 0.9
            args.weigh_decay_apply_on_all = False  #
            args.weight_decay = 4e-5
            args.nesterov = True
            ### criterion
            args.labelsmooth = 0.1
            ### lr scheduler
            args.scheduler = 'linear'
            ### Mixup
            args.mixup = 0.0
            return args
        elif args.hyperparams_set_index == 6: # EfficientNet
            args.epochs = 350
            args.start_eval_epoch = 0
            args.batch_size = 1024 # default is bs=4096/lr=0.256
            ### data transform
            args.autoaugment = True
            args.colorjitter = False
            args.change_light = False
            ### optimizer
            args.optimizer = 'RMSprop'
            args.lr = 0.064
            args.lr_decay_rate = 0.97
            args.lr_decay_step = 2.4
            args.momentum = 0.9
            args.weigh_decay_apply_on_all = True
            args.weight_decay = 1e-5
            ### criterion
            args.labelsmooth = 0.1
            ### lr scheduler
            args.scheduler = 'rmsprop_step'
            ### Mixup
            args.mixup = 0.0
            return args
        elif args.hyperparams_set_index == 93: # Mixup
            args.epochs = 200
            args.start_eval_epoch = 0
            args.batch_size = 1024
            ### data transform
            args.autoaugment = False
            args.colorjitter = False
            args.change_light = False
            ### optimizer
            args.optimizer = 'SGD'
            args.lr = 0.4
            args.momentum = 0.9
            args.weigh_decay_apply_on_all = True  # TODO: weight decay apply on which params
            args.weight_decay = 1e-4
            args.nesterov = True
            ### criterion
            args.labelsmooth = 0
            ### lr scheduler
            args.scheduler = 'multistep'
            args.lr_decay_rate = 0.1
            args.lr_decay_step = 60
            args.warmup_epoch = 5
            args.warmup_lr = 0.1
            ### Mixup
            args.mixup = 0.2

            return args
        elif args.hyperparams_set_index == 94: # AutoAugment
            args.epochs = 270
            args.start_eval_epoch = 235
            args.batch_size = 1024
            ### data transform
            args.autoaugment = True
            args.colorjitter = False
            args.change_light = False
            ### optimizer
            args.optimizer = 'SGD'
            args.lr = 0.4
            args.momentum = 0.9
            args.weigh_decay_apply_on_all = True  # TODO: weight decay apply on which params
            args.weight_decay = 1e-4
            args.nesterov = True
            ### criterion
            args.labelsmooth = 0
            ### lr scheduler
            args.scheduler = 'uneven_multistep'
            args.lr_decay_rate = 0.1
            args.lr_milestone = [90, 180, 240]
            ### Mixup
            args.mixup = 0.0

            return args
        elif args.hyperparams_set_index == 95: # SENet
            args.epochs = 100
            args.start_eval_epoch = 0
            args.batch_size = 128
            ### data transform
            args.autoaugment = False
            args.colorjitter = False
            args.change_light = False
            ### optimizer
            args.optimizer = 'SGD'
            args.lr = 0.075
            args.momentum = 0.9
            args.weigh_decay_apply_on_all = True  # TODO: weight decay apply on which params
            args.weight_decay = 1e-4
            args.nesterov = True
            ### criterion
            args.labelsmooth = 0
            ### lr scheduler
            args.scheduler = 'multistep'
            args.lr_decay_rate = 0
            args.lr_decay_step = 30
            ### Mixup
            args.mixup = 0.0

            return args
        elif args.hyperparams_set_index == 96: # SKNet
            args.epochs = 100
            args.start_eval_epoch = 0
            args.batch_size = 256
            ### data transform
            args.autoaugment = False
            args.colorjitter = False
            args.change_light = False
            ### optimizer
            args.optimizer = 'SGD'
            args.lr = 0.1
            args.momentum = 0.9
            args.weigh_decay_apply_on_all = True  # TODO: weight decay apply on which params
            args.weight_decay = 1e-4
            args.nesterov = True
            ### criterion
            args.labelsmooth = 0.1
            ### lr scheduler
            args.scheduler = 'multistep'
            args.lr_decay_rate = 0.1
            args.lr_decay_step = 30
            ### Mixup
            args.mixup = 0.0

            return args
        elif args.hyperparams_set_index == 97: # PyramidNet
            args.epochs = 120
            args.start_eval_epoch = 0
            args.batch_size = 256
            ### data transform
            args.autoaugment = False
            args.colorjitter = False
            args.change_light = False
            ### optimizer
            args.optimizer = 'SGD'
            args.lr = 0.1
            args.momentum = 0.9
            args.weigh_decay_apply_on_all = True  # TODO: weight decay apply on which params
            args.weight_decay = 1e-4
            args.nesterov = True
            ### criterion
            args.labelsmooth = 0
            ### lr scheduler
            args.scheduler = 'uneven_multistep'
            args.lr_decay_rate = 0.1
            args.lr_milestone = [60, 90, 105]
            ### Mixup
            args.mixup = 0.0

            return args
        elif args.hyperparams_set_index == 98: # bLResNet
            args.epochs = 110
            args.start_eval_epoch = 0
            args.batch_size = 256
            ### data transform
            args.autoaugment = False
            args.colorjitter = True
            args.change_light = True
            ### optimizer
            args.optimizer = 'SGD'
            args.lr = 0.1
            args.momentum = 0.9
            args.weigh_decay_apply_on_all = True  # TODO: weight decay apply on which params
            args.weight_decay = 1e-4
            args.nesterov = True
            ### criterion
            args.labelsmooth = 0
            ### lr scheduler
            args.scheduler = 'cosine'
            args.warmup_epoch = 0
            args.warmup_lr = 0
            ### Mixup
            args.mixup = 0.0

            return args

        elif args.hyperparams_set_index == 3: # sarResNet
            args.epochs = 110
            args.start_eval_epoch = 0
            args.batch_size = 256*4
            ### data transform
            args.autoaugment = False
            args.colorjitter = True
            args.change_light = True
            ### optimizer
            args.optimizer = 'SGD'
            args.lr = 0.1 * (args.batch_size // 256)
            args.momentum = 0.9
            args.weigh_decay_apply_on_all = True  # TODO: weight decay apply on which params
            args.weight_decay = 1e-4
            args.nesterov = True
            ### criterion
            args.labelsmooth = 0
            ### lr scheduler
            args.scheduler = 'cosine'
            args.warmup_epoch = 0
            args.warmup_lr = 0
            ### Mixup
            args.mixup = 0.0

            return args
        elif args.hyperparams_set_index == 333: # sarResNet, use label smooth
            args.epochs = 110
            args.start_eval_epoch = 0
            args.batch_size = 256*4
            ### data transform
            args.autoaugment = False
            args.colorjitter = True
            args.change_light = True
            ### optimizer
            args.optimizer = 'SGD'
            args.lr = 0.1 * (args.batch_size // 256)
            args.momentum = 0.9
            args.weigh_decay_apply_on_all = True  # TODO: weight decay apply on which params
            args.weight_decay = 1e-4
            args.nesterov = True
            ### criterion
            args.labelsmooth = 0.1
            ### lr scheduler
            args.scheduler = 'cosine'
            args.warmup_epoch = 0
            args.warmup_lr = 0
            ### Mixup
            args.mixup = 0.0

            return args

        elif args.hyperparams_set_index == 99: # ResAttentionNet
            args.epochs = 106
            args.start_eval_epoch = 0
            args.batch_size = 256
            ### data transform
            args.autoaugment = False
            args.colorjitter = False
            args.change_light = True
            ### optimizer
            args.optimizer = 'SGD'
            args.lr = 0.1
            args.momentum = 0.9
            args.weigh_decay_apply_on_all = True  # TODO: weight decay apply on which params
            args.weight_decay = 1e-4
            args.nesterov = True
            ### criterion
            args.labelsmooth = 0
            ### lr scheduler
            args.scheduler = 'uneven_multistep'
            args.lr_decay_rate = 0.1
            args.lr_milestone = [40, 80, 100]
            ### Mixup
            args.mixup = 0.0

            return args
        elif args.hyperparams_set_index == 100: # RegNet implement EfficientNet
            args.epochs = 100
            args.start_eval_epoch = 80
            args.batch_size = 256
            ### data transform
            args.autoaugment = False
            args.colorjitter = False
            args.change_light = True
            ### optimizer
            args.optimizer = 'SGD'
            args.lr = 0.4 # batchsize=128/lr=0.2 https://github.com/facebookresearch/pycls/blob/master/MODEL_ZOO.md
            args.momentum = 0.9
            args.weigh_decay_apply_on_all = True  # TODO: weight decay apply on which params
            args.weight_decay = 1e-5
            args.nesterov = True
            ### criterion
            args.labelsmooth = 0
            ### lr scheduler
            args.scheduler = 'cosine'
            args.warmup_epoch = 5
            args.warmup_lr = 0.04
            ### Mixup
            args.mixup = 0.0

            return args
        elif args.hyperparams_set_index == 101: # test
            args.epochs = 1
            args.start_eval_epoch = 0
            args.batch_size = 256
            ### data transform
            args.autoaugment = False
            args.colorjitter = False
            args.change_light = True
            ### optimizer
            args.optimizer = 'SGD'
            args.lr = 0.4
            args.momentum = 0.9
            args.weigh_decay_apply_on_all = True  # TODO: weight decay apply on which params
            args.weight_decay = 1e-5
            args.nesterov = True
            ### criterion
            args.labelsmooth = 0
            ### lr scheduler
            args.scheduler = 'cosine'
            args.warmup_epoch = 5
            args.warmup_lr = 0.1
            ### Mixup
            args.mixup = 0.0

            return args
        elif args.hyperparams_set_index == 102: # CondenseNet Original Hyperparameters except epoch=100
            args.epochs = 100
            args.start_eval_epoch = 0
            args.batch_size = 256
            ### data transform
            args.autoaugment = False
            args.colorjitter = False
            args.change_light = False
            ### optimizer
            args.optimizer = 'SGD'
            args.lr = 0.1
            args.momentum = 0.9
            args.weigh_decay_apply_on_all = True  # TODO: weight decay apply on which params
            args.weight_decay = 1e-4
            args.nesterov = True
            ### criterion
            args.labelsmooth = 0
            ### lr scheduler
            args.scheduler = 'cosine'
            args.warmup_epoch = 0
            args.warmup_lr = 0
            ### Mixup
            args.mixup = 0.0

            return args
        elif args.hyperparams_set_index == 103: # CondenseNet Original Hyperparameters except epoch=100 and weightdecay=5e-5
            args.epochs = 100
            args.start_eval_epoch = 0
            args.batch_size = 256
            ### data transform
            args.autoaugment = False
            args.colorjitter = False
            args.change_light = False
            ### optimizer
            args.optimizer = 'SGD'
            args.lr = 0.1
            args.momentum = 0.9
            args.weigh_decay_apply_on_all = True  # TODO: weight decay apply on which params
            args.weight_decay = 5e-5
            args.nesterov = True
            ### criterion
            args.labelsmooth = 0
            ### lr scheduler
            args.scheduler = 'cosine'
            args.warmup_epoch = 0
            args.warmup_lr = 0
            ### Mixup
            args.mixup = 0.0

            return args
        else:
            raise NotImplementedError("The hyperparams set index {} is out of range!"
                                      "Please select from 1 to 5".format(args.hyperparams_set_index))

    else:
        args.epochs = 90
        args.start_eval_epoch = 0
        args.batch_size = 128
        ### data transform
        args.autoaugment = False
        args.colorjitter = True
        args.change_light = True
        ### optimizer
        args.optimizer = 'SGD'
        args.lr = 0.05
        args.momentum = 0.9
        args.weigh_decay_apply_on_all = False  # TODO: weight decay apply on which params
        args.weight_decay = 1e-4
        args.nesterov = True
        ### criterion
        args.labelsmooth = 0
        ### lr scheduler
        args.scheduler = 'multistep'
        args.lr_decay_rate = 0.1
        args.lr_decay_step = 30
        ### Mixup
        args.mixup = 0.0

        return args