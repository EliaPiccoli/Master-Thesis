import torch.nn as nn
import numpy as np

class VideoObjectSegmentationModel(nn.Module):
    def __init__(self, K):
        super().__init__()
        
        self.K = K
        self.final_conv_size = 7 * 7 * 64 # check
        self.cnn = nn.Sequential(
            nn.Conv2d(2, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU()
        )
        self.fc = nn.Linear(self.final_conv_size, 512)

        # object_camera_translation
        self.obj_trans = nn.Linear(512, self.K*2) # reshape [ ? x K x 2]

        # objective_masks

        
    
    def forward(input):
        # [ BS x C x H x W ]
        pass