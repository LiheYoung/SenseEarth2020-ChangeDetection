import argparse


class Options:
    def __init__(self):
        parser = argparse.ArgumentParser('SenseEarth -- Change Detection')
        parser.add_argument("--data-root", type=str, default="data/datasets")
        parser.add_argument("--batch-size", type=int, default=16)
        parser.add_argument("--val-batch-size", type=int, default=16)
        parser.add_argument("--test-batch-size", type=int, default=16)
        parser.add_argument("--epochs", type=int, default=20)
        parser.add_argument("--lr", type=float, default=0.0003)
        parser.add_argument("--weight-decay", type=float, default=1e-4)
        parser.add_argument("--backbone", type=str, default="resnet34")
        parser.add_argument("--model", type=str, default="fcn")
        parser.add_argument("--lightweight", dest="lightweight", action="store_true")
        parser.add_argument("--load-from", type=str)
        parser.add_argument("--pretrain-from", type=str)
        parser.add_argument("--pretrained", dest="pretrained", action="store_true")
        parser.add_argument("--tta", dest="tta", action="store_true")
        parser.add_argument("--save-mask", dest="save_mask", action="store_true")
        parser.add_argument("--use-pseudo-label", dest="use_pseudo_label", action="store_true")

        self.parser = parser

    def parse(self):
        args = self.parser.parse_args()
        print(args)
        return args
