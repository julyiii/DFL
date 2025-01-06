import torch
import torch.nn as nn

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1):
        x1 = self.up(x1)
        return self.conv(x1)
    
    

class UpSampler(nn.Module):
    def __init__(self, in_dim, mid_dim, out_dim, bilinear=True):
        super().__init__()
        self.up1 = Up(in_dim, out_dim, bilinear=True)

    def forward(self, x):
        x = self.up1(x)
        return x

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None,reserve_sptial_dim = True):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        if reserve_sptial_dim is True:
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        else:
            #downsample 4倍
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=2,stride=2),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, out_channels, kernel_size=2,stride=2),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True))

    def forward(self, x):
        return self.double_conv(x)
    
    
class SegmHead(nn.Module):
    def __init__(self, in_dim, hidden_dim1, hidden_dim2, class_dim,dense_color=False):
        super().__init__()
        self.dense_color = dense_color
        # upsample features
        self.upsampler = UpSampler(in_dim, hidden_dim1, hidden_dim2)

        segm_net = DoubleConv(hidden_dim2, class_dim)
        segm_net.double_conv = segm_net.double_conv[:4]
        self.segm_net = segm_net
        
        
        if dense_color:
            dense_color_net  = DoubleConv(hidden_dim2,3,reserve_sptial_dim = True)
            dense_color_net.double_conv = dense_color_net.double_conv[:4]
            self.dense_color_net = dense_color_net

    def forward(self, img_feat):
        # feature up sample to 256
        hr_img_feat = self.upsampler(img_feat)
        segm_logits = self.segm_net(hr_img_feat)
        if self.dense_color:
            dense_color = self.dense_color_net(hr_img_feat)
            return {'segm_logits': segm_logits,'dense_color':dense_color}
        return {'segm_logits': segm_logits}
    
#64 —> 32 -> 16 -> 反卷积+平滑upsample -> 32 -> 64 -> 128 ->通道降为33
class SegmHead_more_layer(nn.Module):
    def __init__(self, in_dim, hidden_dim1, hidden_dim2, class_dim):
        super().__init__()
        #downsample features
        self.downsampler = DoubleConv(in_dim, hidden_dim2,hidden_dim1,reserve_sptial_dim = False)
        # upsample features
        self.upsampler1 = UpSampler(hidden_dim2,hidden_dim2, hidden_dim2,bilinear=False)
        self.upsampler2 = UpSampler(hidden_dim2, hidden_dim2,hidden_dim2,bilinear=False) #16->32->64
        #self.upsampler3 = UpSampler(hidden_dim2, hidden_dim2,bilinear=False)

        segm_net = DoubleConv(hidden_dim2, class_dim)
        segm_net.double_conv = segm_net.double_conv[:4]
        self.segm_net = segm_net

    def forward(self, img_feat):
        # feature up sample to 256
        downsample_feat = self.downsampler(img_feat)
        hr_img_feat_1 = self.upsampler1(downsample_feat)
        hr_img_feat_2 = self.upsampler2(hr_img_feat_1)
        segm_logits = self.segm_net(hr_img_feat_2)
        return {'segm_logits': segm_logits}


class SegmNet(nn.Module):
    def __init__(self,dense_color=False):
        super(SegmNet, self).__init__()
        self.segm_head = SegmHead(256, 128, 64, 33,dense_color = dense_color)

    def map2labels(self, segm_hand):
        with torch.no_grad():
            segm_hand = segm_hand.permute(0, 2, 3, 1)
            _, pred_segm_hand = segm_hand.max(dim=3)
            return pred_segm_hand

    def forward(self, img_feat):
        segm_dict = self.segm_head(img_feat)
        segm_logits = segm_dict['segm_logits']

        segm_mask = self.map2labels(segm_logits)

        segm_dict['segm_mask'] = segm_mask
        segm_dict['segm_logits'] = segm_logits
        return segm_dict


