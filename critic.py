import torch

class Critic(torch.nn.Module):

    def __init__(self):
        super(Critic, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 128, 4, stride=2, padding=1)
        self.conv1bn = torch.nn.BatchNorm2d(128)
        self.conv2 = torch.nn.Conv2d(128, 256, 4, stride=2, padding=1)
        self.conv2bn = torch.nn.BatchNorm2d(256)
        self.conv3 = torch.nn.Conv2d(256, 512, 4, stride=2, padding=1)
        self.conv3bn = torch.nn.BatchNorm2d(512)
        self.conv4 = torch.nn.Conv2d(512, 1, 4, stride=1, padding=0)   
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):

        x = torch.nn.functional.leaky_relu(self.conv1(x))
        x = self.conv1bn(x)
        x = torch.nn.functional.leaky_relu(self.conv2(x))
        x = self.conv2bn(x)
        x = torch.nn.functional.leaky_relu(self.conv3(x))
        x = self.conv3bn(x)
        x = self.conv4(x)
        x = self.sigmoid(x)
        
        return x
