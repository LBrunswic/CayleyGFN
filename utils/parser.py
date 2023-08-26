import argparse


def generate_argparser():
    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--gpu",
        help="which gpu for training",
        default=-1,
        dest="GPU",
        type=int,
    )
    parser.add_argument(
        "--folder",
        help="which gpu for training",
        default='_default',
        dest="folder",
        type=str,
    )

    parser.add_argument(
        "--batchsize",
        help="Gflow training: Batch size ",
        default=4,
        dest="BATCH_SIZE",
        type=int,
    )

    parser.add_argument(
        "--initflow",
        help="Gflow traininginitflow Batch size ",
        default=1.,
        dest="initflow",
        type=float,
    )
    parser.add_argument(
        "--epochs",
        help="Gflow training: EPOCHS",
        default=100,
        dest="EPOCHS",
        type=int,
    )
    parser.add_argument(
        "--size",
        help="Gflow training: SIZE",
        default=3,
        dest="SIZE",
        type=int,
    )

    parser.add_argument(
        "--generators",
        help="Gflow training: GENERATORS",
        # default='3_cycles',
        default='trans_cycle_a',
        dest="GENERATORS",
        type=str,
    )
    parser.add_argument(
        "--step_per_epoch",
        help="Gflow training: STEP_PER_EPOCH",
        default=10,
        dest="STEP_PER_EPOCH",
        type=int,
    )

    parser.add_argument(
        "--MLP_depth",
        help="Gflow training: MLP_DEPTH",
        default=2,
        dest="MLP_DEPTH",
        type=int,
    )
    parser.add_argument(
        "--MLP_width",
        help="Gflow training: MLP_WIDTH",
        default=16,
        dest="MLP_WIDTH",
        type=int,
    )
    parser.add_argument(
        "--inverse",
        help="Group generators inverse",
        default=0,
        dest="INVERSE",
        type=int,
    )
    parser.add_argument(
        "--memory_limit",
        help="GPU memory limit",
        default=13,
        dest="MEMORY_LIMIT",
        type=int,
    )
    parser.add_argument(
        "--LR",
        help="Learning rate",
        default=1e-2,
        dest="LR",
        type=float,
    )

    parser.add_argument(
        "--load",
        help="reload weights",
        default=0,
        dest="LOAD",
        type=int,
    )

    parser.add_argument(
        "--scheduler",
        help="scheduler choice",
        default=0,
        dest="SCHEDULER",
        type=int,
    )

    parser.add_argument(
        "--loss",
        help="loss",
        default='pow,2',
        dest="LOSS",
        type=str,
    )
    parser.add_argument(
        "--seed",
        help="seed",
        default=1234,
        dest="SEED",
        type=int,
    )
    parser.add_argument(
        "--batch_memory",
        help="BATCH_MEMORY",
        default=4,
        dest="BATCH_MEMORY",
        type=int,
    )
    parser.add_argument(
        "--heuristic",
        help="Heuristic",
        default='',
        dest="HEURISTIC",
        type=str,
    )
    parser.add_argument(
        "--heuristic_param",
        help="Heuristic",
        default=1e-4,
        dest="HEURISTIC_PARAM",
        type=float,
    )
    parser.add_argument(
        "--heuristic_scale",
        help="Heuristic",
        default=1.,
        dest="heuristic_scale",
        type=float,
    )

    parser.add_argument(
        "--length_cutoff_factor",
        help="length_cutoff_factor",
        default=2,
        dest="length_cutoff_factor",
        type=int,
    )
    parser.add_argument(
        "--reward",
        help="REWARD",
        default='R_first_one',
        dest="REWARD",
        type=str,
    )
    parser.add_argument(
        "--exploration",
        help="exploration factor",
        default=1e-2,
        dest="EXPLORATION",
        type=float,
    )

    parser.add_argument(
        "--bootstrap",
        help="bootstrap",
        default='',
        dest="BOOTSTRAP",
        type=str,
    )
    return parser
