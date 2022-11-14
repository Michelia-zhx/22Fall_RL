import argparse

def get_args():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument(
        '--env-name',
        type=str,
        default='MontezumaRevengeNoFrameskip-v0')
    parser.add_argument(
        '--num-stacks',
        type=int,
        default=8)
    parser.add_argument(
        '--num-steps',
        type=int,
        default=250)
    parser.add_argument(
        '--batch-size',
        type=int,
        default=50)
    parser.add_argument(
        '--test-steps',
        type=int,
        default=1000)
    parser.add_argument(
        '--num-frames',
        type=int,
        default=2500)
    parser.add_argument(
        '--beta',
        type=float,
        default=1.0)
    parser.add_argument(
        '--feature-dim',
        type=int,
        default=16)
    parser.add_argument(
        '--lr',
        type=float,
        default=0.05)
    parser.add_argument(
        '--num-epoch',
        type=int,
        default=50)
    parser.add_argument(
        '--lr-rampdown-epochs',
        type=int,
        default=50)
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=0)
    parser.add_argument(
        '--model-dir',
        type=str,
        default='best.pth')

    ## other parameter
    parser.add_argument(
        '--log-interval',
        type=int,
        default=1,
        help='log interval, one log per n updates (default: 10)')
    parser.add_argument(
        '--save-img',
        type=bool,
        default=True)
    parser.add_argument(
        '--save-interval',
        type=int,
        default=10,
        help='save interval, one eval per n updates (default: None)')
    parser.add_argument(
        '--play-game',
        type=bool,
        default=False)
    args = parser.parse_args()


    return args