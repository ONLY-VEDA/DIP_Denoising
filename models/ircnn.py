import torch.nn as nn
'''
Padding caculs
o = output
p = padding
k = kernel_size
s = stride
d = dilation
o = [i + 2*p - k - (k-1)*(d-1)]/s + 1
'''

class IRBlock(nn.Module):
    def __init__(self, in_c=64, out_c=64, dilation=1):
        super(IRBlock, self).__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=3, padding=dilation, stride=1, dilation=dilation, bias=False)
        self.batch_norm = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.batch_norm(out)
        out = self.relu(out)
        return out

class IRCNN(nn.Module):

    def __init__(self, in_c=3, out_c=3):
        super(IRCNN, self).__init__()
        dilation_list = [2, 3, 4, 3, 2]
        self.baseBlock = nn.Sequential(
                nn.Conv2d(in_c, 64, kernel_size=3, padding=1, stride=1, bias=False),
                nn.ReLU(inplace=True)
                ) 
        self.layers = self._make_layer(IRBlock, dilation_list)
        self.EndBlock = nn.Sequential(
                nn.Conv2d(64, out_c, kernel_size=3, padding=1, stride=1, bias=False)
                )

    def _make_layer(self, block, dilation_list):
        layers = []
        for i in range(len(dilation_list)):
            layers.append(block(64, 64, dilation_list[i]))

        return nn.Sequential(*layers)

    def forward(self, x):
        y = self.baseBlock(x)
        y = self.layers(y)
        y = self.EndBlock(y)
        return x - y

        
