import torch
import torch.nn as nn
import torchvision


class UNet(nn.Module):
    """
    References:
    1. https://github.com/GokulKarthik/Deep-Learning-Projects.pytorch/blob/master/\
        6-Image-Segmentation-with-UNet.ipynb
    """
    def __init__(self, image_width, geometry='AABB', with_classification=False):
        """
        Parameters:
        -----------
        geometry : str
            It should be one from {'AABB', 'RBOX', 'QUAD'}
        num_classes : int
            It denotes the number of languages in Indian Scene Text Dectection Dataset
        """
        super(UNet, self).__init__()
        self.image_width = image_width
        self.geometry = geometry
        self.with_classification = with_classification
        if self.with_classification:
            self.num_scores = 5 + 1 # 5 for each language; 1 for background
        else:
            self.num_scores = 1 # 1 for text & background

        self.contracting_11 = self.conv_block(in_channels=3, out_channels=64)
        self.contracting_12 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.contracting_21 = self.conv_block(in_channels=64, out_channels=128)
        self.contracting_22 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.contracting_31 = self.conv_block(in_channels=128, out_channels=256)
        self.contracting_32 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.contracting_41 = self.conv_block(in_channels=256, out_channels=512)
        self.contracting_42 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.middle = self.conv_block(in_channels=512, out_channels=1024)

        self.expansive_11 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=3, \
            stride=2, padding=1, output_padding=1)
        self.expansive_12 = self.conv_block(in_channels=1024, out_channels=512)
        self.expansive_21 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3, \
            stride=2, padding=1, output_padding=1)
        self.expansive_22 = self.conv_block(in_channels=512, out_channels=256)
        self.expansive_31 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, \
            stride=2, padding=1, output_padding=1)
        self.expansive_32 = self.conv_block(in_channels=256, out_channels=128)
        self.expansive_41 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, \
            stride=2, padding=1, output_padding=1)
        self.expansive_42 = self.conv_block(in_channels=128, out_channels=64)

        self.score_map = nn.Conv2d(in_channels=64, out_channels=self.num_scores, kernel_size=1)
        if self.geometry == 'AABB': 
            out_channels = 4
        elif self.geometry == 'RBOX':
            out_channels = 4
            self.angle_map = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1)
        elif self.geometry == 'QUAD':
            out_channels = 8
        else:
            raise NotImplemetedError()
        self.geometry_map = nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=1)


    def conv_block(self, in_channels, out_channels):
        block = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, \
            kernel_size=3, stride=1, padding=1),
                            nn.ReLU(),
                            nn.BatchNorm2d(num_features=out_channels),
                            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, \
                                kernel_size=3, stride=1, padding=1),
                            nn.ReLU(),
                            nn.BatchNorm2d(num_features=out_channels))
        return block
    

    def forward(self, X):
        contracting_11_out = self.contracting_11(X) # [-1, 64, h, w]
        contracting_12_out = self.contracting_12(contracting_11_out) # [-1, 64, h/2, w/2]
        contracting_21_out = self.contracting_21(contracting_12_out) # [-1, 128, h/2, w/2]
        contracting_22_out = self.contracting_22(contracting_21_out) # [-1, 128, h/4, w/4]
        contracting_31_out = self.contracting_31(contracting_22_out) # [-1, 256, h/4, w/4]
        contracting_32_out = self.contracting_32(contracting_31_out) # [-1, 256, h/8, w/8]
        contracting_41_out = self.contracting_41(contracting_32_out) # [-1, 512, h/8, w/8]
        contracting_42_out = self.contracting_42(contracting_41_out) # [-1, 512, h/16, w/16]

        middle_out = self.middle(contracting_42_out) # [-1, 1024, h/16, w/16]

        expansive_11_out = self.expansive_11(middle_out) # [-1, 512, h/8, w/8]
        expansive_12_out = self.expansive_12(torch.cat((expansive_11_out, contracting_41_out), \
            dim=1)) # [-1, 1024, h/8, w/8] -> [-1, 512, h/8, w/8]
        expansive_21_out = self.expansive_21(expansive_12_out) # [-1, 256, h/4, w/4]
        expansive_22_out = self.expansive_22(torch.cat((expansive_21_out, contracting_31_out), \
            dim=1)) # [-1, 512, h/4, w/4] -> [-1, 256, h/4, w/4]
        expansive_31_out = self.expansive_31(expansive_22_out) # [-1, 128, h/2, w/2]
        expansive_32_out = self.expansive_32(torch.cat((expansive_31_out, contracting_21_out), \
            dim=1)) # [-1, 256, h/2, w/2] -> [-1, 128, h/2, w/2]
        expansive_41_out = self.expansive_41(expansive_32_out) # [-1, 64, h, w]
        expansive_42_out = self.expansive_42(torch.cat((expansive_41_out, contracting_11_out), \
            dim=1)) # [-1, 128, h, w] -> [-1, 64, h, w]
        
        score_out = self.score_map(expansive_42_out) # [-1, -2, h, w]
        geometry_out = torch.relu(self.geometry_map(expansive_42_out))
        if self.geometry == 'RBOX':
            angle_out = torch.sigmoid(self.angle_map(expansive_42_out)) # [-1, -2, h, w]
            angle_out = (angle_out - 0.5) * math.pi / 2
        else:
            angle_out = None
        
        return score_out, geometry_out, angle_out
    
    
class EAST(nn.Module):


    def __init__(self, image_width, geometry='AABB', with_classification=False):
        """
        Parameters:
        -----------
        geometry : str
            It should be one from {'AABB', 'RBOX', 'QUAD'}
        num_classes : int
            It denotes the number of languages in Indian Scene Text Dectection Dataset
        """
        super(EAST, self).__init__()

        self.image_width = image_width
        self.geometry = geometry
        self.with_classification = with_classification
        if self.with_classification:
            self.num_scores = 5 + 1 # 5 for each language; 1 for background
        else:
            self.num_scores = 1 # 1 for text & background

        ## Feature Extraction Essentials
        # conv1
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=1)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/2

        # conv2
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/4

        # conv3
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/8

        # conv4
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_3 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/16

        # conv5
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_3 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True) # 1/32


        ## Feature Merging Essentials
        layer1 = nn.Sequential(nn.Conv2d(1024, 256, 1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
                               nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))

        layer2 = nn.Sequential(nn.Conv2d(512, 128, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
                               nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True))

        layer3 = nn.Sequential(nn.Conv2d(256, 32, 1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
                               nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True))

        layer4 = nn.Sequential(nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True))

        self.feature_convs = nn.ModuleList([layer1, layer2, layer3, layer4])

        self.unpool = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        ## Output Layer Essentials
        self.score_map = nn.Conv2d(in_channels=32, out_channels=self.num_scores, kernel_size=1)
        if self.geometry == 'AABB': 
            out_channels = 4
        elif self.geometry == 'RBOX':
            out_channels = 4
            self.angle_map = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1)
        elif self.geometry == 'QUAD':
            out_channels = 8
        else:
            raise NotImplemetedError()
        self.geometry_map = nn.Conv2d(in_channels=32, out_channels=out_channels, kernel_size=1)

        self._init_weights()

        vgg16 = torchvision.models.vgg16(pretrained=True)

        self.copy_params_from_vgg16(vgg16)

        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

                    
    def copy_params_from_vgg16(self, vgg16):
        features = [
            self.conv1_1, self.relu1_1,
            self.conv1_2, self.relu1_2,
            self.pool1,
            self.conv2_1, self.relu2_1,
            self.conv2_2, self.relu2_2,
            self.pool2,
            self.conv3_1, self.relu3_1,
            self.conv3_2, self.relu3_2,
            self.conv3_3, self.relu3_3,
            self.pool3,
            self.conv4_1, self.relu4_1,
            self.conv4_2, self.relu4_2,
            self.conv4_3, self.relu4_3,
            self.pool4,
            self.conv5_1, self.relu5_1,
            self.conv5_2, self.relu5_2,
            self.conv5_3, self.relu5_3,
            self.pool5,
        ]

        for l1, l2 in zip(vgg16.features, features):
            if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                assert l1.weight.size() == l2.weight.size()
                assert l1.bias.size() == l2.bias.size()
                l2.weight.data = l1.weight.data
                l2.bias.data = l1.bias.data

                
    def forward(self, x):
        h = x
        h = self.relu1_1(self.conv1_1(h))
        h = self.relu1_2(self.conv1_2(h))
        h = self.pool1(h)

        h = self.relu2_1(self.conv2_1(h))
        h = self.relu2_2(self.conv2_2(h))
        h = self.pool2(h)
        pool2 = h

        h = self.relu3_1(self.conv3_1(h))
        h = self.relu3_2(self.conv3_2(h))
        h = self.relu3_3(self.conv3_3(h))
        h = self.pool3(h)
        pool3 = h

        h = self.relu4_1(self.conv4_1(h))
        h = self.relu4_2(self.conv4_2(h))
        h = self.relu4_3(self.conv4_3(h))
        h = self.pool4(h)
        pool4 = h

        h = self.relu5_1(self.conv5_1(h))
        h = self.relu5_2(self.conv5_2(h))
        h = self.relu5_3(self.conv5_3(h))
        h = self.pool5(h)
        pool5 = h


        f = [pool5, pool4, pool3, pool2]
        g = [None, None, None, None]
        h = [None, None, None, None]

        for i in range(4):

            if i == 0:
                h[i] = f[i]
            else:
                concat = torch.cat([g[i - 1], f[i]], dim=1)
                h[i] = self.feature_convs[i - 1](concat)

            if i <= 2:
                g[i] = self.unpool(h[i])
            else:
                g[i] = self.feature_convs[i](h[i])


        score_out = self.score_map(g[3]) # [-1, -2, h, w]
        geometry_out = torch.relu(self.geometry_map(g[3]))
        if self.geometry == 'RBOX':
            angle_out = torch.sigmoid(g[3]) # [-1, -2, h, w]
            angle_out = (angle_out - 0.5) * math.pi / 2
        else:
            angle_out = None


        return score_out, geometry_out, angle_out