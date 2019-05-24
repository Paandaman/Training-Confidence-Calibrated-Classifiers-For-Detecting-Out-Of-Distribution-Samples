import torch

class Generator(torch.nn.Module):

    def __init__(self, m):
        super(Generator, self).__init__()
        self.latent_size = m
        self.linear1 = torch.nn.Linear(self.latent_size, 1024*4*4)
        self.conv0bn = torch.nn.BatchNorm2d(1024)
        self.conv1 = torch.nn.ConvTranspose2d(1024, 128, kernel_size = 4, stride=1, padding=0)
        self.conv1bn = torch.nn.BatchNorm2d(128)
        self.conv2 =  torch.nn.ConvTranspose2d(128, 256, kernel_size = 4, stride=2, padding=1)
        self.conv2bn = torch.nn.BatchNorm2d(256)
        self.conv3 = torch.nn.ConvTranspose2d(256, 64, kernel_size = 4, stride=2, padding=1)
        self.conv3bn = torch.nn.BatchNorm2d(64)
        self.conv4 = torch.nn.ConvTranspose2d(64, 3, kernel_size = 7, stride=1, padding=1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):

        x = self.linear1(x)
        x = torch.nn.functional.relu(x)
        x = x.view(-1, 1024, 4, 4) 
        x = self.conv0bn(x)
        x = self.conv1(x)
        x = torch.nn.functional.relu(x)
        x = self.conv1bn(x) 
        x = self.conv2(x)
        x = torch.nn.functional.relu(x)
        x = self.conv2bn(x)
        x = self.conv3(x)
        x = torch.nn.functional.relu(x)
        x = self.conv3bn(x)     
        x = self.conv4(x)
        x = self.sigmoid(x)

        return x
