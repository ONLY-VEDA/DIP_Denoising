import torch.nn as nn

class DnBlock(nn.Module):
    def __init__(self, in_c=64, out_c=64):
        super(DnBlock, self).__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, stride=1, bias=False)
        self.batch_norm = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)
        #self.relu = nn.PReLU(init=0.1)

    def forward(self, x):
        out = self.conv(x)
        out = self.batch_norm(out)
        out = self.relu(out)
        return out

class DnCNN(nn.Module):

    def __init__(self, in_c=3, out_c=3):
        super(DnCNN, self).__init__()
        self.baseBlock = nn.Sequential(
                nn.Conv2d(in_c, 64, kernel_size=3, padding=1, stride=1, bias=False),
                #nn.PReLU(init=0.1)
                nn.ReLU(inplace=True)
                ) 
        self.layers = self._make_layer(DnBlock,15)
        self.EndBlock = nn.Sequential(
                nn.Conv2d(64, out_c, kernel_size=3, padding=1, stride=1, bias=False)
                )

    def _make_layer(self, block, num_blocks):
        layers = []
        for i in range(num_blocks):
            layers.append(block(64, 64))

        return nn.Sequential(*layers)

    def forward(self, x):
        y = self.baseBlock(x)
        y = self.layers(y)
        y = self.EndBlock(y)
        return x - y

        
