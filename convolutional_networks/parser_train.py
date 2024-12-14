import argparse

def parser():
    parser = argparse.ArgumentParser(description="Model training")
    
    parser.add_argument(
        '--gpu', 
        type=str,
        default='0',
        help="GPU to use"
    )
    
    parser.add_argument(
        "--dir",
        type=str,
        default=None,
        required=True,
        help="training directory (default: None)",
    )
    
    parser.add_argument(
        "--dataset", 
        type=str, 
        default="CIFAR10", 
        help="dataset name (default: CIFAR10)"
    )
    
    parser.add_argument(
        "--data_path",
        type=str,
        default="~/datasets/",
        metavar="PATH",
        help="path to datasets location (default: ~/datasets/)",
    )
    
    parser.add_argument(
        "--use_test",
        dest="use_test",
        action="store_true",
        help="use test dataset instead of validation (default: False)",
    )
    
    parser.add_argument(
        "--corrupt_train", 
        type=float, 
        default=None,
        help="train data corruption fraction (default: None --- no corruption)",
    )
    
    parser.add_argument(
        "--split_classes", 
        type=int, 
        default=None,
        help="split classes for CIFAR-10 (default: None --- no split)",
    )

    parser.add_argument(
        "--fbgd",
        dest="fbgd",
        action="store_true",
        help="train with full-batch GD (default: False)",
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        metavar="N",
        help="input batch size (default: 128)",
    )
    
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        metavar="N",
        help="number of workers (default: 4)",
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        required=True,
        metavar="MODEL",
        help="model name (default: None)",
    )
    
    parser.add_argument(
        "--trial",
        type=int,
        default=0,
        help="trial number (default: 0)",
    )
    
    parser.add_argument(
        "--resume_epoch",
        type=int,
        default=-1,
        metavar="CKPT",
        help="checkpoint epoch to resume training from (default: -1 --- start from scratch)",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=1001,
        metavar="N",
        help="number of epochs to train (default: 1001)",
    )
    
    parser.add_argument(
        "--use_data_size",
        type=int,
        default=None,
        help="how many examples to use for training",
    )
    
    parser.add_argument(
        "--save_freq",
        type=int,
        default=None, 
        metavar="N",
        help="save frequency (default: None --- custom saving)",
    )
    
    parser.add_argument(
        "--save_freq_int",
        type=int,
        default=0, 
        metavar="N",
        help="internal save frequency - how many times to save at each epoch (default: None --- custom saving)",
    )
    
    parser.add_argument(
        "--eval_freq",
        type=int,
        default=10,
        metavar="N",
        help="evaluation frequency (default: 10)",
    )
    
    parser.add_argument(
        "--lr_init",
        type=float,
        default=0.01,
        metavar="LR",
        help="initial learning rate (default: 0.01)",
    )
    
    parser.add_argument(
        "--noninvlr",
        type=float,
        default=-1,
        metavar="LR",
        help="learning rate for not scale-inv parameters",
    )
    
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        metavar="M",
        help="SGD momentum (default: 0.9)",
    )
    
    parser.add_argument(
        "--wd", 
        type=float, 
        default=1e-4, 
        help="weight decay (default: 1e-4)"
    )
    
    parser.add_argument(
        "--loss",
        type=str,
        default="CE",
        help="loss to use for training model (default: Cross-entropy)",
    )

    parser.add_argument(
        "--seed", 
        type=int, 
        default=1, 
        metavar="S", 
        help="random seed (default: 1)"
    )
    
    parser.add_argument(
        "--num_channels", 
        type=int, 
        default=64, 
        help="number of channels for resnet"
    )
    
    parser.add_argument(
        "--depth", 
        type=int, 
        default=3, 
        help="depth of convnet"
    )

    parser.add_argument(
        "--scale", 
        type=int, 
        default=25, 
        help="scale of lenet"
    )
    
    parser.add_argument(
        "--no_schedule", 
        action="store_true", 
        help="disable lr schedule"
    )
    
    parser.add_argument(
        "--c_schedule", 
        type=float,
        default=None, 
        help="continuous schedule - decrease lr linearly after 1/4 epochs so that at the end it is x times lower "
    )
    
    parser.add_argument(
        "--d_schedule", 
        type=float,
        default=None, 
        help="discrete schedule - decrease lr x times after each 1/4 epochs "
    )
    
    parser.add_argument(
        "--init_scale", 
        type=float,
        default=-1, 
        help="init norm of the last layer weights "
    )
    
    parser.add_argument(
        "--no_aug", 
        action="store_true", 
        help="disable data augmentation"
    )
    
    parser.add_argument(
        "--fix_si_pnorm", 
        action="store_true", 
        help="set SI-pnorm to init after each iteration"
    )
    
    parser.add_argument(
        "--fix_si_pnorm_value", 
        type=float, 
        default=-1,
        help="custom fixed SI-pnorm value (must go with --fix_si_pnorm flag; default: -1 --- use init SI-pnorm value)",
    )
    
    parser.add_argument(
        "--cosan_schedule", 
        action="store_true", 
        help="cosine anealing schedule"
    )

    args = parser.parse_args()
    return args