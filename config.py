import argparse


def load_config():
    parser = argparse.ArgumentParser('Segmentation')

    parser.add_argument('--pretrained', type=bool, default=True)
    parser.add_argument('--n_classes', type=int, default=19)

    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--lr', type=float, default=0.007)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--max_iteration', type=int, default=50000)
    parser.add_argument('--warmup_iteration', type=float, default=1500)

    args = parser.parse_args()

    return args
