from torch import cuda, nn
import torch.nn.functional as F

class CNN(nn.Module):
    use_cuda = cuda.is_available()

    def __init__(self):
        super(CNN, self).__init__() # build nn module
        conv_one = nn.Conv2d(1, 6, 5, 1) # 6@24*24

        pool_one = nn.MaxPool2d(2) # 6@12*12
        conv_two = nn.Conv2d(6, 16, 5, 1) # 16@8*8

        pool_two = nn.MaxPool2d(2) # 16@4*4

        self.conv_module = nn.Sequential(
            conv_one,
            nn.ReLU(),
            pool_one,
            conv_two,
            nn.ReLU(),
            pool_two
        ) # construct learning model

        # fc = fully connection
        fc1 = nn.Linear(16*4*4, 120)
        fc2 = nn.Linear(120, 84)
        fc3 = nn.Linear(84, 10)

        self.fc_module = nn.Sequential(
            fc1,
            nn.ReLU(),
            fc2,
            nn.ReLU(),
            fc3
        )

        if CNN.use_cuda:
            self.conv_module = self.conv_module.cuda()
            self.fc_module = self.fc_module.cuda()

    def forward(self, x):
        #dim = dimension
        out = self.conv_module(x)
        dim = 1

        for d in out.size()[1:]:
            dim = dim * d

        out = out.view(-1, dim)
        out = self.fc_module(out)

        return F.Softmax(out, dim=1)
