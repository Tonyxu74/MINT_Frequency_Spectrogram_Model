from torch import nn
from myargs import args
import torch


class Simp_Model(nn.Module):
    def __init__(self):
        super(Simp_Model, self).__init__()
        self.inputsize = args.patch_width * args.patch_height * args.patch_classes  # should be 1152
        self.FC1 = nn.Linear(self.inputsize, 512)
        self.FC2 = nn.Linear(512, 64)
        self.FC3 = nn.Linear(64, args.classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        y = self.relu(self.FC1(x))
        y = self.relu(self.FC2(y))
        y = self.FC3(y)

        return y


class BuildConv(nn.Module):
    """
    Building convolutional model backbones for feature extraction
    from Deep Learning for EEG motor imagery classification based on multi-layer CNNs feature fusion
    https://www.sciencedirect.com/science/article/abs/pii/S0167739X19306077
    """

    def __init__(self, size, input_channels, output_size, fc_input_size):
        super(BuildConv, self).__init__()
        self.size = size
        self.fc_input_size = fc_input_size
        self.input_channels = input_channels
        self.inputsize = args.seq_length  # add this, some kind of seq length of input to this model

        # dims are (batch, 1, channels, time), add primary convolutions
        if size == 1:
            self.add_module('conv_11', nn.Conv2d(1, 50, kernel_size=(1, 30)))

        elif size == 2:
            self.add_module('conv_11', nn.Conv2d(1, 50, kernel_size=(1, 25)))

        elif size == 3:
            self.add_module('conv_11', nn.Conv2d(1, 50, kernel_size=(1, 20)))

        elif size == 4:
            self.add_module('conv_11', nn.Conv2d(1, 50, kernel_size=(1, 10)))

        else:
            raise NotImplementedError('size from 1-4 available only')

        self.add_module('bn_11', nn.BatchNorm2d(50))  # batch norm after every convolution
        self.add_module('conv_12', nn.Conv2d(50, 50, kernel_size=(input_channels, 1)))  # channel conv
        self.add_module('bn_12', nn.BatchNorm2d(50))

        self.add_module('pool_1', nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3)))  # pool in time

        # append intermediate layers, prev size gets previous channels sizes
        prev_size = 50
        for layer_num in range(2, size + 1):
            self.add_module(f'conv_{layer_num}', nn.Conv2d(prev_size, 100, kernel_size=(1, 10)))
            self.add_module(f'bn_{layer_num}', nn.BatchNorm2d(100))
            self.add_module(f'pool_{layer_num}', nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3)))
            prev_size = 100

        self.relu = nn.ReLU()
        # this first number feels like it needs to change based on size of input, adjust later
        self.dropout = nn.Dropout(0.5)  # dropout before fc
        self.fc = nn.Linear(fc_input_size, output_size)

    def forward(self, x):
        # conv_11
        x = self.conv_11(x)
        x = self.bn_11(x)
        x = self.relu(x)

        # conv 12
        x = self.conv_12(x)
        x = self.bn_12(x)
        x = self.relu(x)

        # pool
        x = self.pool_1(x)

        # additional layers
        for layer_num in range(2, self.size + 1):
            x = self._modules[f'conv_{layer_num}'](x)
            x = self._modules[f'bn_{layer_num}'](x)
            x = self.relu(x)
            # print(x.shape)
            x = self._modules[f'pool_{layer_num}'](x)
            # print(x.shape)

        # flatten, pass through fc layers
        # print(x.shape)
        x = torch.flatten(x, start_dim=1)
        x = self.dropout(x)
        x = self.fc(x)

        return x


class MultiLayerModel(nn.Module):
    """
    Multilayer eeg model taking in ***********RAW EEG DATA************ as input
    from Deep Learning for EEG motor imagery classification based on multi-layer CNNs feature fusion
    https://www.sciencedirect.com/science/article/abs/pii/S0167739X19306077
    """

    def __init__(self, input_channels, size_dict):
        super(MultiLayerModel, self).__init__()
        self.inputsize = args.seq_length  # add this, some kind of seq length of input to this model, may need to pass to CNN to get proper FC sizes...
        self.input_channels = input_channels

        self.CNN_1 = BuildConv(1, input_channels, args.cnn_output_features, size_dict[1])
        self.CNN_2 = BuildConv(2, input_channels, args.cnn_output_features, size_dict[2])
        self.CNN_3 = BuildConv(3, input_channels, args.cnn_output_features, size_dict[3])
        self.CNN_4 = BuildConv(4, input_channels, args.cnn_output_features, size_dict[4])

        self.merge_fc1 = nn.Linear(args.cnn_output_features * 4, args.cnn_output_features)
        self.merge_fc2 = nn.Linear(args.cnn_output_features, args.classes)

        # if we're including dropout here, consider removing it for CNN_1 - CNN_4
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def freeze_backbone(self):
        for param in self.CNN_1.parameters():
            param.requires_grad = False
        for param in self.CNN_2.parameters():
            param.requires_grad = False
        for param in self.CNN_3.parameters():
            param.requires_grad = False
        for param in self.CNN_4.parameters():
            param.requires_grad = False

    def forward(self, x):
        # backbone feature extractors
        x1 = self.CNN_1(x)
        x2 = self.CNN_2(x)
        x3 = self.CNN_3(x)
        x4 = self.CNN_4(x)

        # concatenate all these!!
        x = torch.cat((x1, x2, x3, x4), dim=1)

        # merge features
        x = self.dropout(x)
        x = self.merge_fc1(x)
        x = self.relu(x)
        x = self.merge_fc2(x)

        return x


class Conv_Model(nn.Module):
    def __init__(self):
        super(Conv_Model, self).__init__()

        self.conv1 = nn.Conv2d(args.patch_classes, 32, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0, bias=False)
        self.bn4 = nn.BatchNorm2d(256)

        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        self.fc = nn.Linear(256, args.classes)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)

        x = self.pool(x)

        x = x.flatten(1)
        x = self.dropout(x)
        x = self.fc(x)

        return x


# model = Conv_Model()
# input = torch.randn(4, 8, 17, 17)
# print(model(input).shape)
# size_dict = {1: 7850, 2: 4900, 3: 1300, 4: 100}
# input = torch.randn(4, 1, 8, 500)
# model = MultiLayerModel(8, size_dict)
#
# print(model(input).shape)
# print(model)
