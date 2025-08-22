import argparse
parser = argparse.ArgumentParser(description='PyTorch CodeBase')
parser.add_argument(
        "--cfg",
        help="decide which cfg to use",
        required=False,
        default="/data/ActivityNet/Counting_CodeBase/code/configs/rpc.yaml",
        type=str,
    )
parser.add_argument(
    "opts",
    help="Modify config options using the command-line",
    default=None,
    nargs=argparse.REMAINDER,
)
