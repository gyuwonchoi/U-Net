import torch 
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import pytorch_model_summary
 
class Conv3Down(nn.Module):
    def __init__(self, channel):
        super().__init__() 
        
        self.channel = channel
        self.c = self.channel * 2
        self.conv3_double= nn.Conv2d(self.channel, self.c , 3, 1, 0)  
        self.conv3_same= nn.Conv2d(self.c , self.c , 3, 1, 0)  
        
        self.bn = nn.BatchNorm2d(self.c)
        self.relu = nn.ReLU()
        
    def forward(self, x):    
        x = self.conv3_double(x)
        x = self.bn(x)
        x = self.relu(x)
        
        x = self.conv3_same(x)     
        x = self.bn(x)
        x = self.relu(x)
        
        return x
    
class Down(nn.Module):
    def __init__(self, channel):
        super().__init__() 

        self.channel = channel
        self.maxpool = nn.MaxPool2d(2, stride=2)
        self.conv_dwn = Conv3Down(self.channel)
    
    def forward(self, x):
        x = self.conv_dwn(self.maxpool(x))
        return x
    
class Conv3DUp(nn.Module):
    def __init__(self, channel):
        super().__init__() 
        
        self.c = channel // 2
        
        self.conv3_half= nn.Conv2d(channel, self.c , 3, 1, 0)  
        self.conv3_same= nn.Conv2d(self.c , self.c , 3, 1, 0)  
        
        self.bn = nn.BatchNorm2d(self.c)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.conv3_half(x)
        x = self.bn(x)
        x = self.relu(x)
        
        x = self.conv3_same(x)     
        x = self.bn(x)
        x = self.relu(x)
        
        return x
    
class Up(nn.Module):
    def __init__(self, channel):
        super().__init__() 
        
        self.conv_up = Conv3DUp(channel)
        self.upsample = nn.ConvTranspose2d(channel, channel//2, 2, 2) 
    
    def forward(self, x, y):
        x = self.upsample(x)
        x = self._concat(y, x)
        x = self.conv_up(x)

        return x
    
    # https://github.com/ddamddi/UNet-pytorch/blob/bfb1c47147ddeb8a85b3b50a4af06b3a2082d933/model/ops.py#L55
    
    def _concat(self, hres, x):
        _, _, w, h = x.size()
        hres_crop = self._crop(hres, w, h)
        return torch.cat((hres_crop, x), 1)

    def _crop(self, x, tw, th):
        w, h = x.size()[-2:]
        dw = (w-tw) // 2
        dh = (h-th) // 2
        return x[:,:,dw:dw+tw,dh:dh+th]
    
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__() 
        self.layer1 = nn.Sequential(
                    nn.Conv2d(in_channels= 3, out_channels= 64, kernel_size= 3, stride= 1, padding=0), 
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.Conv2d(64, 64, 3, 1, 0), 
                    nn.BatchNorm2d(64),
                    nn.ReLU()
                    )
        
        self.layer2 = Down(64)
        self.layer3 = Down(128)
        self.layer4 = Down(256)
        self.layer5 = Down(512)

        self.layer6 = Up(1024)
        self.layer7 = Up(512)
        self.layer8 = Up(256)
        self.layer9 = Up(128)
        
        self.final = nn.Conv2d(64, 21,  1,  1)

    def forward(self, x):
        
        x1 = self.layer1(x)
        
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x5 = self.layer5(x4)

        x = self.layer6(x5, x4)
        x = self.layer7(x, x3)
        x = self.layer8(x, x2)
        x = self.layer9(x, x1)    
            
        x = self.final(x)
        
        return x
    
if __name__=="__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"    
    print(device)
    
    model = UNet()
    summary(model, (3, 572, 572), device ='cpu')  
    
    # print(pytorch_model_summary.summary(model, torch.zeros(1,3,572,572), show_input = True))
    
