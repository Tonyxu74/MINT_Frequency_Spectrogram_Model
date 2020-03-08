from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

now = datetime.now()
logdir = "runs/" + now.strftime("%Y%m%d-%H%M%S") + "/"

writer = SummaryWriter(logdir)

