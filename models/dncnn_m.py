import torch.nn as nn

class DnmBlock(nn.Module):
    def __init__(self, in_c=64, out_c=64):
        super(DnmBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_c, in_c, kernel_size=3, padding=1, stride=1, groups=in_c, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(in_c)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(in_c, out_c, 1, 1, 0, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(out_c)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.batch_norm1(out)
        out = self.relu1(out)
        out = self.conv2(x)
        out = self.batch_norm2(out)
        out = self.relu2(out)
        return out

class DnCNN_M(nn.Module):

    def __init__(self, in_c=3, out_c=3):
        super(DnCNN_M, self).__init__()
        self.baseBlock = nn.Sequential(
                nn.Conv2d(in_c, 64, kernel_size=3, padding=1, stride=1, bias=False),
                #nn.PReLU(init=0.1)
                nn.ReLU(inplace=True)
                ) 
        self.layers = self._make_layer(DnmBlock,15)
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

        
