from torch import nn
from myargs import args


class Simp_Model(nn.Module):
    def __init__(self):
        super(Simp_Model, self).__init__()
        self.inputsize = args.patch_width * args.patch_size * args.patch_classes  # should be 1152
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
