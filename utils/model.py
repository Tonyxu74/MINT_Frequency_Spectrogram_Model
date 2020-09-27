from torch import nn
from myargs import args


class simple_dnn(nn.Module):
    def __init__(self):
        super(Simp_Model, self).__init__()
        self.inputsize = args.patch_width * args.patch_height * args.patch_classes  # should be 1152
        self.FC1 = nn.Linear(self.inputsize, 512)
        self.FC2 = nn.Linear(512, 64)
        self.FC3 = nn.Linear(64, args.classes)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        #x = x.view(x.size(0), -1)
        y = self.relu(self.FC1(x))
        y = self.relu(self.FC2(y))
        y = self.FC3(y)
        y = self.softmax(y)
        y = y.view(1, y.size(0))

        return y

# new models
# class whatever_model(nn.Module):
#     def __init__(self):
#         #layers, whatever
#     def forward(self, x)
#         #y = f(x), etc
#         return y
