import torch
import torch.nn.functional as F

class ESPCN(torch.nn.Module):
    def __init__(self,scale_factor):
        super(ESPCN,self).__init__()
        self.num_channels=3
        self.conv1=torch.nn.Conv2d(self.num_channels,64,kernel_size=5,stride=1,padding=5//2)
        self.conv2=torch.nn.Conv2d(64,32,kernel_size=3,stride=1,padding=3//2)
        self.conv3=torch.nn.Conv2d(32,self.num_channels*(scale_factor**2),kernel_size=3,padding=3//2)
        self.hr_img=torch.nn.PixelShuffle(scale_factor)

    def forward(self,x):
        x=torch.tanh(self.conv1(x))
        x=torch.tanh(self.conv2(x))
        x=torch.tanh(self.conv3(x))
        return self.hr_img(x)

if __name__=='__main__':
    model=ESPCN(3)
    test_input=torch.rand((1,3,100,100))
    output=model(test_input)
    print(output.size())
