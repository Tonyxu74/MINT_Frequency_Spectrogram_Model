import argparse

parser = argparse.ArgumentParser()

######################## Model parameters ########################

parser.add_argument('--model_name', default='resnet18',
                    help='pretrained model name')
parser.add_argument('--classes', default=5, type=int,
                    help='# of classes')

parser.add_argument('--lr', default=0.0001, type=float,
                    help='learning rate')
parser.add_argument('--weight_decay', default=0.0001, type=float,
                    help='weight decay/weights regularizer for sgd')
parser.add_argument('--beta1', default=0.9, type=float,
                    help='momentum for sgd, beta1 for adam')
parser.add_argument('--beta2', default=0.999, type=float,
                    help='momentum for sgd, beta1 for adam')

parser.add_argument('--num_epoch', default=250, type=int,
                    help='epochs to train for')
parser.add_argument('--start_epoch', default=1, type=int,
                    help='epoch to start training. useful if continue from a checkpoint')

parser.add_argument('--batch_size', default=1, type=int,
                    help='input batch size')
parser.add_argument('--batch_size_eval', default=1, type=int,
                    help='input batch size at eval time')


######################## Image properties (size) ########################

parser.add_argument('--patch_width', default=12, type=int,
                    help='patch size width')
parser.add_argument('--patch_height', default=12, type=int,
                    help='patch size height')
parser.add_argument('--patch_classes', default=8, type=int,
                    help='patch size height')
parser.add_argument('--dx', default=224*5, type=int,
                    help='image crop size width dx')
parser.add_argument('--dy', default=224*5, type=int,
                    help='image crop size height dy')


######################## Folders ########################


args = parser.parse_args()