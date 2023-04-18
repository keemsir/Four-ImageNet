import torch
import torch.nn as nn
import torch.nn.functional as F

from efficientnet_pytorch import EfficientNet



class EffNetNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Define Feature part (IMAGE)
        # self.features = EfficientNet.from_pretrained('efficientnet-b0')
        self.features = EfficientNet.from_name('efficientnet-b2')
        
        # (CSV)
        self.csv = nn.Sequential(nn.Linear(17, 250),
                                 nn.BatchNorm1d(250),
                                 nn.ReLU(),
                                 nn.Dropout(p=0.2),
                                 
                                 nn.Linear(250, 250),
                                 nn.BatchNorm1d(250),
                                 nn.ReLU(),
                                 nn.Dropout(p=0.2))
        
        # Define Classification part
        self.classification = nn.Sequential(nn.Linear(1408 + 250, 1)) # 1408

        # 1x1 convolution
        self.con1by1 = nn.Sequential(nn.Conv2d(264, 1, kernel_size=1, stride=1, padding="same"), nn.ReLU())



    def forward(self, image, meta, prints=False):
        
        if prints: print('Input Image shape:', image.shape, '\n'+
                         'Input metadata shape:', meta.shape)
        
        # Image CNN
        image = self.con1by1(image)
        image = self.features.extract_features(image)
        image = F.avg_pool2d(image, image.size()[2:]).reshape(-1, 1408) # 1408
        if prints: print('Features Image shape:', image.shape)
        
        # CSV FNN
        meta = self.csv(meta)
        if prints: print('Meta Data:', meta.shape)
        
        # Concatenate layers from image with layers from csv_data
        image_meta_data = torch.cat((image, meta), dim=1)
        if prints: print('Concatenated Data:', image_meta_data.shape)
        
        # CLASSIF
        out = self.classification(image_meta_data)
        if prints: print('Out shape:', out.shape)
        
        return out



class EffNetNetwork_image(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Define Feature part (IMAGE)
        # self.features = EfficientNet.from_pretrained('efficientnet-b0')
        self.features = EfficientNet.from_name('efficientnet-b2')
        
        # (CSV)
        # self.csv = nn.Sequential(nn.Linear(17, 250),
        #                          nn.BatchNorm1d(250),
        #                          nn.ReLU(),
        #                          nn.Dropout(p=0.2),
                                 
        #                          nn.Linear(250, 250),
        #                          nn.BatchNorm1d(250),
        #                          nn.ReLU(),
        #                          nn.Dropout(p=0.2))
        
        # Define Classification part
        self.classification = nn.Sequential(nn.Linear(1408, 1, bias=False)) # 1408
    
        # 1x1 convolution
        self.con1by1 = nn.Sequential(nn.Conv2d(264, 1, kernel_size=1, stride=1, padding="same", bias=False), nn.ReLU())



    def forward(self, image, meta, prints=False):
        
        if prints: print('Input Image shape:', image.shape, '\n'+
                         'Input metadata shape:', meta.shape)
        
        # Image CNN
        print("1", image.size())
        image = self.con1by1(image)
        print("2", image.size())
        image = self.features.extract_features(image)
        print("3", image.size())
        image = F.avg_pool2d(image, image.size()[2:]).reshape(-1, 1408) # 1408
        print("4", image.size())
        if prints: print('Features Image shape:', image.shape)
        
        # CSV FNN
        # meta = self.csv(meta)
        # if prints: print('Meta Data:', meta.shape)
        
        # Concatenate layers from image with layers from csv_data
        # image_meta_data = torch.cat((image, meta), dim=1)
        # if prints: print('Concatenated Data:', image_meta_data.shape)
        
        # CLASSIF
        # out = self.classification(image_meta_data)
        out = self.classification(image)
        if prints: print('Out shape:', out.shape)
        
        return out



class EffNetNetwork_meta(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Define Feature part (IMAGE)
        # self.features = EfficientNet.from_pretrained('efficientnet-b0')
        self.features = EfficientNet.from_name('efficientnet-b2')
        
        # (CSV)
        self.csv = nn.Sequential(nn.Linear(17, 250),
                                 nn.BatchNorm1d(250),
                                 nn.ReLU(),
                                 nn.Dropout(p=0.2)
                                 ,
                                 nn.Linear(250, 250),
                                 nn.BatchNorm1d(250),
                                 nn.ReLU(),
                                 nn.Dropout(p=0.2)
                                 )
        
        # Define Classification part
        self.classification = nn.Sequential(nn.Linear(250, 1)) # 1408

        # 1x1 convolution
        # self.con1by1 = nn.Sequential(nn.Conv2d(264, 1, kernel_size=1, stride=1, padding="same"), nn.ReLU())



    def forward(self, image, meta, prints=False):
        
        if prints: print('Input Image shape:', image.shape, '\n'+
                         'Input metadata shape:', meta.shape)
        
        # Image CNN
        # image = self.con1by1(image)
        # image = self.features.extract_features(image)
        # image = F.avg_pool2d(image, image.size()[2:]).reshape(-1, 1408) # 1408
        # if prints: print('Features Image shape:', image.shape)
        
        # CSV FNN
        meta = self.csv(meta)
        if prints: print('Meta Data:', meta.shape)
        
        # Concatenate layers from image with layers from csv_data
        # image_meta_data = torch.cat((image, meta), dim=1)
        # if prints: print('Concatenated Data:', image_meta_data.shape)
        
        # CLASSIF
        out = self.classification(meta)
        if prints: print('Out shape:', out.shape)
        
        return out
