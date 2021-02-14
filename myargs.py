import argparse

parser = argparse.ArgumentParser()

######################## Model parameters ########################

parser.add_argument('--model_name', default='spectro_conv',
                    help='model name')
parser.add_argument('--pretrained_path', default='./data/model/pretrained_multilayer_conv_38.pt',
                    help='pretrained model weights path')
parser.add_argument('--classes', default=2, type=int,
                    help='# of classes')

parser.add_argument('--lr', default=0.001, type=float,
                    help='learning rate')
parser.add_argument('--weight_decay', default=0.0001, type=float,
                    help='weight decay/weights regularizer for sgd')
parser.add_argument('--beta1', default=0.9, type=float,
                    help='momentum for sgd, beta1 for adam')
parser.add_argument('--beta2', default=0.999, type=float,
                    help='momentum for sgd, beta1 for adam')

parser.add_argument('--cnn_output_features', default=256, type=int,
                    help='size of output features to use for CNN')

parser.add_argument('--num_epochs', default=200, type=int,
                    help='epochs to train for')
parser.add_argument('--start_epoch', default=1, type=int,
                    help='epoch to start training. useful if continue from a checkpoint')

parser.add_argument('--batch_size', default=32, type=int,
                    help='input batch size')
parser.add_argument('--batch_size_eval', default=32, type=int,
                    help='input batch size at eval time')

parser.add_argument('--continue_train', default=False, type=bool,
                    help='continue training from certain epoch?')
parser.add_argument('--pretrain', default=False, type=bool,
                    help='Use pretrained model weights?')
parser.add_argument('--workers', default=0, type=int,
                    help='amount of workers to use when runnning iterator')

######################## Image properties (size) ########################
parser.add_argument('--patch_dims', default=(64, 24, 24), type=tuple,
                    help='total patch size')
parser.add_argument('--seq_length', default=1024, type=int,
                    help='input sequence length')
parser.add_argument('--seq_stride', default=500, type=int,
                    help='input sequence length')
parser.add_argument('--patch_width', default=24, type=int,
                    help='patch size width')
parser.add_argument('--patch_height', default=24, type=int,
                    help='patch size height')
parser.add_argument('--patch_classes', default=64, type=int,
                    help='patch size height')

######################## Folders ########################


args = parser.parse_args()