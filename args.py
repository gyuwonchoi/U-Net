import argparse

parser = argparse.ArgumentParser(
        description="Pytorch Implementation for ResNet")
# parser.add_argument('--batch_size', type=int, default=256)      
parser.add_argument('--mini_batch', type=int, default=1)
parser.add_argument('--layer', type=int, default=20)
parser.add_argument('--epoch', type=int, default=10)
parser.add_argument('--dataset', type=str, default ='cifar10')
parser.add_argument('--lr', type=float, default = 0.1)
parser.add_argument('--mode', type=str, default='resume')
parser.add_argument('--id', type=int, default=0)
