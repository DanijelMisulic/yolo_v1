#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 29 21:31:52 2022

@author: danijelmisulic
"""

import torch
import torch.nn as nn

#kernel size, number of filters, stride, padding
    
arch_config = [
    (7, 64, 2, 3),
    "M",
    (3, 192, 1, 1),
    "M",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1)
]


class CNN_block(nn.Module):
    
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CNN_block, self).__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, bias = False, **kwargs)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        #slope = 0.1
        self.leaky_rely = nn.LeakyReLU(0.1)
        
        
    def forward(self, x):
        return self.leaky_rely(self.batch_norm(self.conv(x)))
    
    
class Yolo_v1(nn.Module):
    
    def __init__(self, in_channels = 3, **kwargs):
        super(Yolo_v1, self).__init__()
        
        self.architecture = arch_config
        self.in_channels = in_channels
        self.darknet = self.create_conv_layers(self.architecture)
        self.fully_connected_layers = self.create_fully_connected_layers(**kwargs)
        
        
    def forward(self, x):
        x = self.darknet(x)
        return self.fully_connected_layers(torch.flatten(x, start_dim = 1))
    
    
    def create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels
        
        for el in architecture:
            if type(el) == tuple:
                layers += [CNN_block(in_channels, out_channels = el[1], kernel_size = el[0], 
                                    stride = el[2], padding = el[3])]
                in_channels = el[1]
            
            elif type(el) == str:
                layers += [nn.MaxPool2d(kernel_size = 2, stride = 2)]
                
            elif type(el) == list:
                conv1 = el[0]
                conv2 = el[1]
                num_repeats = el[2]
                
                for _ in range(num_repeats):
                    layers += [CNN_block(in_channels, out_channels = conv1[1], 
                                         kernel_size = conv1[0], stride = conv1[2],
                                         padding = conv1[3]
                                         )
                                         ]

                    layers += [CNN_block(in_channels = conv1[1], out_channels = conv2[1], 
                                         kernel_size = conv2[0], stride = conv2[2],
                                         padding = conv2[3]
                                         )
                                         ] 
                    
                    in_channels = conv2[1]
            
        return nn.Sequential(*layers)
    
    
    def create_fully_connected_layers(self, split_size, num_boxes, num_classes):
        
        return nn.Sequential(nn.Flatten(),
                             nn.Linear(1024 * split_size * split_size, 496),
                             nn.Dropout(0.0),
                             nn.LeakyReLU(0.1),
                             nn.Linear(496, split_size * split_size * (num_classes + num_boxes * 5))
                             )
                
    
if __name__ == "__main__":    
    model = Yolo_v1(split_size = 7, num_boxes = 2, num_classes = 20)
    x = torch.randn(2, 3, 448, 448)
    print(model(x).shape)
    
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

