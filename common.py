from torch import nn
from torchvision.datasets import ImageFolder
import torch
from torch.nn import functional as F
'''
# encoder_1 

def get_deep_encoder(out_channels=384):
    print('encoder_1')
    return nn.Sequential(
        # Encoder
        nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),  # [B, 64, H/2, W/2]
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # [B, 128, H/4, W/4]
        nn.ReLU(inplace=True),
        nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # [B, 256, H/8, W/8]
        nn.ReLU(inplace=True),
        nn.Conv2d(256, 384, kernel_size=4, stride=2, padding=1),  # [B, 384, H/16, W/16]
        nn.ReLU(inplace=True),
        nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),  # [B, 384, H/16, W/16]
        nn.ReLU(inplace=True),
        nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),  # [B, 384, H/16, W/16]
        nn.ReLU(inplace=True),
        nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),  # [B, 384, H/16, W/16]
        nn.ReLU(inplace=True),
        nn.Upsample(size=(56, 56), mode='bilinear', align_corners=False),  # Resize to 56x56
    )


# encoder_2

def get_deep_encoder(out_channels=384):
    print('encoder_2')
    return nn.Sequential(
        # Encoder
        nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),  # 256 ->128
        nn.ReLU(inplace=True),
        nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 128 ->64
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 64 -> 32
        nn.ReLU(inplace=True),
        nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 32 -> 16
        nn.ReLU(inplace=True),
        nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),  # 16 -> 16
        nn.ReLU(inplace=True),
        nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),  # 16 -> 16
        nn.ReLU(inplace=True),
        nn.Upsample(size=(56, 56), mode='bilinear', align_corners=False),  # Resize to 56x56
    )


# encoder_3 ->

def get_deep_encoder(out_channels=384):
    print('encoder_3')
    return nn.Sequential(
        # Encoder
        nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),  # [B, 64, H/2, W/2]
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # [B, 128, H/4, W/4]
        nn.ReLU(inplace=True),
        nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # [B, 256, H/8, W/8]
        nn.ReLU(inplace=True),
        nn.Conv2d(256, 384, kernel_size=4, stride=2, padding=1),  # [B, 384, H/16, W/16]
        nn.ReLU(inplace=True),
        nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),  # [B, 384, H/16, W/16]
        nn.ReLU(inplace=True),
        nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),  # [B, 384, H/16, W/16]
        nn.ReLU(inplace=True),
        nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),  # [B, 384, H/16, W/16]
        nn.ReLU(inplace=True),
        nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),  # [B, 384, H/16, W/16]
        nn.ReLU(inplace=True),
        nn.Upsample(size=(56, 56), mode='bilinear', align_corners=False),  # Resize to 56x56
    )
'''
# encoder_4 (ours)
def get_autoencoder(out_channels=384):
    print('encoder_4')
    return nn.Sequential(
        # encoder
        nn.Conv2d(in_channels=3, out_channels=32, kernel_size=4, stride=2,  # 256 -> 128
                  padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4, stride=2,  # 128 -> 64
                  padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2,  # 64 -> 32
                  padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2,  # 32 -> 16
                  padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2,  # 16 -> 8
                  padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=8),  # 8 -> 1
        # decoder
        nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=8),  # 1 -> 8
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=4, stride=2,  # 8 -> 16
                           padding=1),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=4, stride=2,  # 16 -> 32
                           padding=1),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=4, stride=2,  # 32 -> 64
                           padding=1),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=4, stride=2,  # 64 -> 128
                           padding=1),
        nn.ReLU(inplace=True), #  add
        nn.Upsample(size=56, mode='bilinear'),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, # 56 -> 56
                  padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=3, # 56 -> 56
                  stride=1, padding=1)
    )
'''
# encoder_5 - encoder_4 + Dropout(0.2) + add (similar original) / image-level-AUROC : 99.7%, pixel-level-AUROC : 84.94%
def get_autoencoder(out_channels=384):
    print('encoder_5')
    return nn.Sequential(
        # encoder
        nn.Conv2d(in_channels=3, out_channels=32, kernel_size=4, stride=2,  # 256 -> 128
                  padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4, stride=2,  # 128 -> 64
                  padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2,  # 64 -> 32
                  padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2,  # 32 -> 16
                  padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2,  # 16 -> 8
                  padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=8),  # 8 -> 1
        # decoder
        nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=8),  # 1 -> 8
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=4, stride=2,  # 8 -> 16
                           padding=1),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=4, stride=2,  # 16 -> 32
                           padding=1),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=4, stride=2,  # 32 -> 64
                           padding=1),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=4, stride=2,  # 64 -> 128
                           padding=1),
        nn.ReLU(inplace=True), #  add
        nn.Dropout(0.2),   #  add
        nn.Upsample(size=56, mode='bilinear'),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, # 56 -> 56
                  padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=3, # 56 -> 56
                  stride=1, padding=1)
    )

#encoder_6 : use bicubic instead of bilinear / image-level AUROC : 99.83%, pixel-level AUROC : 82.81%
def get_autoencoder(out_channels=384):
    print('encoder_6')
    return nn.Sequential(
        # encoder
        nn.Conv2d(in_channels=3, out_channels=32, kernel_size=4, stride=2, # 256 ->128
                  padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4, stride=2, # 128 ->64
                  padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, # 64 -> 32
                  padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, # 32 -> 16
                  padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, # 16 -> 8
                  padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=8), # 8 -> 1
        # decoder
        nn.Upsample(size=3, mode='bicubic'), # 1 -> 3
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1, # 3 -> 4
                  padding=2),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2), # 0.2 -> 0.3
        nn.Upsample(size=8, mode='bicubic'), # 4 -> 8
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1, # 8 -> 9
                  padding=2),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Upsample(size=15, mode='bicubic'), # # 8 -> 15
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1, # 15 -> 16
                  padding=2),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Upsample(size=32, mode='bicubic'),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1,
                  padding=2),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Upsample(size=63, mode='bicubic'),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1,
                  padding=2),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Upsample(size=127, mode='bicubic'), # ... -> 127
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1, # 127 -> 128
                  padding=2),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        # add layer
        #nn.Upsample(size=255, mode='bilinear'), # ... -> 127
        #nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1, # 255 -> 256
        #          padding=2),
        #nn.ReLU(inplace=True),
        #nn.Dropout(0.2),
        # end
        nn.Upsample(size=56, mode='bilinear'), # 128(256) -> 56
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, # 56 -> 56
                  padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=3, # 56 -> 56
                  stride=1, padding=1)
    )

# encoder_original
def get_autoencoder(out_channels=384):
    print('encoder_original')
    return nn.Sequential(
        # encoder
        nn.Conv2d(in_channels=3, out_channels=32, kernel_size=4, stride=2, # 256 ->128
                  padding=1), 
        nn.ReLU(inplace=True), # EncConv-1
        nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4, stride=2, # 128 ->64
                  padding=1),
        nn.ReLU(inplace=True), # EncConv-2
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, # 64 -> 32
                  padding=1),
        nn.ReLU(inplace=True), # EncConv-3
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, # 32 -> 16
                  padding=1),
        nn.ReLU(inplace=True), # EncConv-4
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, # 16 -> 8
                  padding=1),
        nn.ReLU(inplace=True), # EncConv-5
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=8), # 8 -> 1 # EncConv-6
        # decoder
        nn.Upsample(size=3, mode='bilinear'), # 1 -> 3 # Bilinear-1
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1, # 3 -> 4
                  padding=2),
        nn.ReLU(inplace=True), # DecConv-1
        nn.Dropout(0.2), # 0.2 -> 0.3 #Dropout-1
        nn.Upsample(size=8, mode='bilinear'), # 4 -> 8
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1, # 8 -> 9
                  padding=2),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Upsample(size=15, mode='bilinear'), # # 8 -> 15
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1, # 15 -> 16
                  padding=2),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Upsample(size=32, mode='bilinear'),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1,
                  padding=2),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Upsample(size=63, mode='bilinear'),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1,
                  padding=2),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Upsample(size=127, mode='bilinear'), # ... -> 127
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1, # 127 -> 128
                  padding=2),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        # add layer
        #nn.Upsample(size=255, mode='bilinear'), # ... -> 127
        #nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1, # 255 -> 256
        #          padding=2),
        #nn.ReLU(inplace=True),
        #nn.Dropout(0.2),
        # end
        nn.Upsample(size=56, mode='bilinear'), # 128(256) -> 56
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, # 56 -> 56
                  padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=3, # 56 -> 56
                  stride=1, padding=1)
    )
'''
def get_pdn_small(out_channels=384, padding=False):
    pad_mult = 1 if padding else 0
    return nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=128, kernel_size=4,
                  padding=3 * pad_mult),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult),
        nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4,
                  padding=3 * pad_mult),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult),
        nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3,
                  padding=1 * pad_mult),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=256, out_channels=out_channels, kernel_size=4)
    )

def get_pdn_medium(out_channels=384, padding=False):
    pad_mult = 1 if padding else 0
    return nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=256, kernel_size=4,
                  padding=3 * pad_mult),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult),
        nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4,
                  padding=3 * pad_mult),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult),
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3,
                  padding=1 * pad_mult),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=512, out_channels=out_channels, kernel_size=4),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                  kernel_size=1)
    )

class ImageFolderWithoutTarget(ImageFolder):
    def __getitem__(self, index):
        sample, target = super().__getitem__(index)
        return sample

class ImageFolderWithPath(ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample, target = super().__getitem__(index)
        return sample, target, path

def InfiniteDataloader(loader):
    iterator = iter(loader)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(loader)