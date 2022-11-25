import torch
import torch.nn as nn
import torch.nn.functional as functional
import numpy as np
class Net(nn.Module):
    def __init__(self, angular_in):
        super(Net, self).__init__()
        channel = 64
        self.angRes = angular_in
        self.FeaExtract = InitFeaExtract(channel)

    def forward(self, x_mv):
        #print(x_mv.shape)
        b, n, c, h, w = x_mv.shape
        out = self.FeaExtract(x_mv)

        return out

class InitFeaExtract(nn.Module):
    def __init__(self, channel):
        super(InitFeaExtract, self).__init__()
        self.FEconv = nn.Sequential(
            nn.Conv2d(2, channel, kernel_size=1, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            RB(channel),
            RB(channel),
            RB(channel),
            RB(channel),
            RB(channel),
            nn.Conv2d(channel, 1, kernel_size=1, stride=1, padding=0, bias=False)
            )

    def forward(self, x):
        b, n, r, h, w = x.shape
        x = x.contiguous().view(b, -1, h, w)
        buffer = self.FEconv(x)
        _, c, h, w = buffer.shape
        #buffer = buffer.unsqueeze(1).contiguous().view(b, -1, c, h, w)#.permute(0,2,1,3,4)  # buffer_sv:  B, N, C, H, W

        return buffer
class RB(nn.Module):
    def __init__(self, channel):
        super(RB, self).__init__()
        self.conv01 = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)
        self.conv02 = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.att = SELayer(channel)

    def forward(self, x):
        buffer = self.conv01(x)
        buffer = self.lrelu(buffer)
        buffer = self.att(self.conv02(buffer))
        return buffer + x
class SELayer(nn.Module):
    '''
    Channel Attention
    '''
    def __init__(self, out_ch,g=4):
        super(SELayer, self).__init__()
        self.att_c = nn.Sequential(
                nn.Conv2d(out_ch, out_ch//g, 1, 1, 0),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch//g, out_ch, 1, 1, 0),
                nn.Sigmoid()
            )

    def forward(self,fm):
        ##channel
        fm_pool = functional.adaptive_avg_pool2d(fm, (1, 1))
        att = self.att_c(fm_pool)
        fm = fm * att
        return fm
if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    net = Net(2).cuda()
    from thop import profile
    ##### get input index ######         
    ind_all = np.arange(7*7).reshape(7, 7)        
    delt = (7-1) // (2-1)
    ind_source = ind_all[0:7:delt, 0:7:delt]
    ind_source = torch.from_numpy(ind_source.reshape(-1))
    input = torch.randn(1, 2, 1, 64, 64).cuda()
    total = sum([param.nelement() for param in net.parameters()])
    flops, params = profile(net, inputs=(input,))
    print('   Number of parameters: %.4fM' % (total / 1e6))
    print('   Number of FLOPs: %.4fG' % (flops / 1e9))