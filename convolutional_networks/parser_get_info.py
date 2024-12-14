import argparse

def parser():
    parser = argparse.ArgumentParser(description="Get information from the trained models")

    parser.add_argument(
        "--models_dir", 
        type=str, 
        default=None, 
        required=True, 
        help="checkpoints dir"
    )

    parser.add_argument(
        '--gpu', 
        type=str,
        default='0',
        help="GPU to use"
    )

    parser.add_argument(
        "--save_path", 
        type=str, 
        default=None, 
        required=True, 
        help="path to npz results file"
    )

    parser.add_argument(
        "--update",
        action="store_true",
        help="update file instead of rewriting (default: False)",
    )

    parser.add_argument(
        "--seed", 
        type=int, 
        default=1,
        metavar="S",
        help="random seed (default: 1)"
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
        help="use test dataset instead of train set (default: False)",
    )

    parser.add_argument(
        "--use_aug",
        dest="use_aug",
        action="store_true",
        help="use augmentation (default: False)",
    )

    parser.add_argument(
        "--corrupt_train", 
        type=float, 
        default=None,
        help="train data corruption fraction (default: None --- no corruption)",
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
        default="PreResNet56",
        metavar="MODEL",
        help="model name (default: PreResNet56)",
    )

    parser.add_argument(
        "--num_channels",
        type=int,
        default=64,
        help="number of channels for resnet or convnet"
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
        "--train_mode",
        dest="train_mode",
        action="store_true",
        help="use train mode during evaluation (default: False)",
    )

    parser.add_argument(
        "--prebn_only",
        dest="prebn_only",
        action="store_true",
        help="use only pre-BN parameters (default: False)",
    )

    parser.add_argument(
        "--eval_on_train_subsets",
        action="store_true",
        help="eval model (quality and traces) on train subsets separately (default: False)",
    )

    parser.add_argument(
        "--eval_on_random_subset",
        action="store_true",
        help="eval model on random subset of 10 batches (default: False)",
    )

    #parser.add_argument(
    #    "--rs_size",
    #    type=int, 
    #    default=1000, 
    #    help="random subset max size"
    #)

    parser.add_argument(
        "--eval_model",
        dest="eval_model",
        action="store_true",
        help="eval model quality (default: False)",
    )

    parser.add_argument(
        "--calc_grads",
        dest="calc_grads",
        action="store_true",
        help="calculate batch gradients statistics (default: False)",
    )

    parser.add_argument(
        "--fisher_trace",
        dest="fisher_trace",
        action="store_true",
        help="eval fisher trace (default: False)",
    )
    
    parser.add_argument(
        "--calc_grad_norms",
        dest="calc_grad_norms",
        action="store_true",
        help="calculate batch gradients statistics (default: False)",
    )

    parser.add_argument(
        "--fisher_evals",
        dest="fisher_evals",
        action="store_true",
        help="eval 10 fisher eigenvalues (default: False)",
    )

    parser.add_argument(
        "--hess_trace",
        dest="hess_trace",
        action="store_true",
        help="eval hessian trace (default: False)",
    )

    parser.add_argument(
        "--hess_evals",
        dest="hess_evals",
        action="store_true",
        help="eval 10 hessian eigenvalues (default: False)",
    )

    parser.add_argument(
        "--all_pnorm",
        action="store_true",
        help="calculate norm of each parameter separately (default: False)",
    )
    
    parser.add_argument(
        "--use_data_size",
        type=int,
        default=None,
        help="how many examples to use for training",
    )
    
    parser.add_argument(
        "--save_freq_int",
        type=int,
        default=0, 
        metavar="N",
        help="internal save frequency - how many times to save at each epoch (default: None --- custom saving)",
    )
    
    parser.add_argument(
        "--init_scale", 
        type=float,
        default=-1, 
        help="init norm of the last layer weights "
    )

    args = parser.parse_args()
    return args