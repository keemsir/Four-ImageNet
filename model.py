import torch
import torch.nn as nn
import torch.nn.functional as F

from efficientnet_pytorch import EfficientNet
from torchvision.models import resnet50



class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        self.keep_prob = 0.7 # 0.5
        # self.images = images
        # self.meta = meta
        # L1 ImgIn shape=(?, 28, 28, 1)
        #    Conv     -> (?, 28, 28, 32)
        #    Pool     -> (?, 14, 14, 32)
        self.layer1 = nn.Sequential(
            nn.Conv2d(264, 1, kernel_size=1, stride=1, padding="same"),
            nn.ReLU()
            #nn.MaxPool2d(kernel_size=2, stride=2)
            )
        # L2 ImgIn shape=(?, 14, 14, 32)
        #    Conv      ->(?, 14, 14, 64)
        #    Pool      ->(?, 7, 7, 64)
        self.layer2 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding="same"),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1))
        # L3 ImgIn shape=(?, 7, 7, 64)
        #    Conv      ->(?, 7, 7, 128)
        #    Pool      ->(?, 4, 4, 128)
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding="same"),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1))

        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding="same"),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1))

        # L4 FC 4x4x128 inputs -> 625 outputs
        self.fc1 = nn.Linear(29 * 29 * 128, 1000, bias=False) # self.fc1 = nn.Linear(65 * 65 * 128, 1000, bias=False)
        nn.init.xavier_uniform_(self.fc1.weight)

        self.layer5 = nn.Sequential(
            self.fc1,
            nn.BatchNorm1d(1000),
            nn.ReLU(),
            nn.Dropout(p=1 - self.keep_prob))
        # L5 Final FC 625 inputs -> 10 outputs (1)
        # self.fc2 = nn.Linear(1000, 100, bias=False)
        # nn.init.xavier_uniform_(self.fc2.weight)
        self.meta_csv = nn.Sequential(
            nn.Linear(17, 500),
            nn.BatchNorm1d(500),
            nn.ReLU(),
            nn.Dropout(p=1 - self.keep_prob))

        self.fc2 = nn.Linear(1500, 1, bias=False)
        nn.init.xavier_uniform_(self.fc2.weight)
        # self.concat_layer = torch.cat((self.fc2, ), dim=1)
        
        # self.fc3 = nn.Linear(100, 1, bias=False)
        # self.fc3 = nn.Linear(100, 1, bias=False)
        # nn.init.xavier_uniform_(self.fc3.weight)
    

    def forward(self, x, y):
        # print("x:", x.shape)
        out = self.layer1(x)
        # print("layer1:", out.shape)
        out = self.layer2(out)
        # print("layer2:", out.shape)
        out = self.layer3(out)
        # print("layer3:", out.shape)
        out = self.layer4(out)
        # print("layer4:", out.shape)
        out = out.view(out.size(0), -1)   # Flatten them for FC
        # print("Flatten:", out.shape)
        out = self.layer5(out)
        out_csv = self.meta_csv(y)
        # print("layer5:", out.shape)
        out = torch.cat((out, out_csv), dim=1)
        out = self.fc2(out)
        # print("fc2:", out.shape)
        # out = self.concat_layer(y)
        # out = torch.cat((out, y), dim=1)
        # print("concat_layer:", out.shape)
        # out = self.fc3(out)
        # print("fc3:", out.shape)
        
        return out


class CNN_img(nn.Module):

    def __init__(self):
        super(CNN_img, self).__init__()
        self.keep_prob = 0.7 # 0.5
        # self.images = images
        # self.meta = meta
        # L1 ImgIn shape=(?, 28, 28, 1)
        #    Conv     -> (?, 28, 28, 32)
        #    Pool     -> (?, 14, 14, 32)
        self.layer1 = nn.Sequential(
            nn.Conv2d(264, 1, kernel_size=1, stride=1, padding="same"),
            nn.ReLU()
            #nn.MaxPool2d(kernel_size=2, stride=2)
            )
        # L2 ImgIn shape=(?, 14, 14, 32)
        #    Conv      ->(?, 14, 14, 64)
        #    Pool      ->(?, 7, 7, 64)
        self.layer2 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding="same"),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1))
        # L3 ImgIn shape=(?, 7, 7, 64)
        #    Conv      ->(?, 7, 7, 128)
        #    Pool      ->(?, 4, 4, 128)
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding="same"),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1))

        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding="same"),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1))

        # L4 FC 4x4x128 inputs -> 625 outputs
        self.fc1 = nn.Linear(29 * 29 * 128, 1000, bias=False) # self.fc1 = nn.Linear(65 * 65 * 128, 1000, bias=False)
        nn.init.xavier_uniform_(self.fc1.weight)

        self.layer5 = nn.Sequential(
            self.fc1,
            nn.BatchNorm1d(1000),
            nn.ReLU(),
            nn.Dropout(p=1 - self.keep_prob))
        # L5 Final FC 625 inputs -> 10 outputs (1)
        # self.fc2 = nn.Linear(1000, 100, bias=False)
        # nn.init.xavier_uniform_(self.fc2.weight)
        self.meta_csv = nn.Sequential(
            nn.Linear(17, 500),
            nn.BatchNorm1d(500),
            nn.ReLU(),
            nn.Dropout(p=1 - self.keep_prob))

        self.fc2 = nn.Linear(1000, 1, bias=False)
        nn.init.xavier_uniform_(self.fc2.weight)

        # self.concat_layer = torch.cat((self.fc2, ), dim=1)
        
        # self.fc3 = nn.Linear(100, 1, bias=False)
        # self.fc3 = nn.Linear(100, 1, bias=False)
        # nn.init.xavier_uniform_(self.fc3.weight)
    

    def forward(self, x, y):
        # print("x:", x.shape)
        out = self.layer1(x)
        # print("layer1:", out.shape)
        out = self.layer2(out)
        # print("layer2:", out.shape)
        out = self.layer3(out)
        # print("layer3:", out.shape)
        out = self.layer4(out)
        # print("layer4:", out.shape)
        out = out.view(out.size(0), -1)   # Flatten them for FC
        # print("Flatten:", out.shape)
        out = self.layer5(out)
        # out_csv = self.meta_csv(y)
        # print("layer5:", out.shape)
        # out = torch.cat((out, out_csv), dim=1)
        out = self.fc2(out)
        # print("fc2:", out.shape)
        # out = self.concat_layer(y)
        # out = torch.cat((out, y), dim=1)
        # print("concat_layer:", out.shape)
        # out = self.fc3(out)
        # print("fc3:", out.shape)
        
        return out



class CNN_meta(nn.Module):

    def __init__(self):
        super(CNN_meta, self).__init__()
        self.keep_prob = 0.7 # 0.5
        # self.images = images
        # self.meta = meta
        # L1 ImgIn shape=(?, 28, 28, 1)
        #    Conv     -> (?, 28, 28, 32)
        #    Pool     -> (?, 14, 14, 32)
        self.layer1 = nn.Sequential(
            nn.Conv2d(264, 1, kernel_size=1, stride=1, padding="same"),
            nn.ReLU()
            #nn.MaxPool2d(kernel_size=2, stride=2)
            )
        # L2 ImgIn shape=(?, 14, 14, 32)
        #    Conv      ->(?, 14, 14, 64)
        #    Pool      ->(?, 7, 7, 64)
        self.layer2 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding="same"),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1))
        # L3 ImgIn shape=(?, 7, 7, 64)
        #    Conv      ->(?, 7, 7, 128)
        #    Pool      ->(?, 4, 4, 128)
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding="same"),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1))

        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding="same"),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1))

        # L4 FC 4x4x128 inputs -> 625 outputs
        self.fc1 = nn.Linear(29 * 29 * 128, 1000, bias=False) # self.fc1 = nn.Linear(65 * 65 * 128, 1000, bias=False)
        nn.init.xavier_uniform_(self.fc1.weight)

        self.layer5 = nn.Sequential(
            self.fc1,
            nn.BatchNorm1d(1000),
            nn.ReLU(),
            nn.Dropout(p=1 - self.keep_prob))
        # L5 Final FC 625 inputs -> 10 outputs (1)
        # self.fc2 = nn.Linear(1000, 100, bias=False)
        # nn.init.xavier_uniform_(self.fc2.weight)
        self.meta_csv = nn.Sequential(
            nn.Linear(17, 500),
            nn.BatchNorm1d(500),
            nn.ReLU(),
            nn.Dropout(p=1 - self.keep_prob))


        self.csv = nn.Sequential(nn.Linear(17, 250),
                            nn.BatchNorm1d(250),
                            nn.ReLU(),
                            nn.Dropout(p=0.2),
                            
                            nn.Linear(250, 250),
                            nn.BatchNorm1d(250),
                            nn.ReLU(),
                            nn.Dropout(p=0.2))


        self.fc2 = nn.Linear(250, 1, bias=False)
        nn.init.xavier_uniform_(self.fc2.weight)
        # self.concat_layer = torch.cat((self.fc2, ), dim=1)
        
        # self.fc3 = nn.Linear(100, 1, bias=False)
        # self.fc3 = nn.Linear(100, 1, bias=False)
        # nn.init.xavier_uniform_(self.fc3.weight)
    

    def forward(self, x, y):
        # print("x:", x.shape)
        # out = self.layer1(x)
        # print("layer1:", out.shape)
        # out = self.layer2(out)
        # print("layer2:", out.shape)
        # out = self.layer3(out)
        # print("layer3:", out.shape)
        # out = self.layer4(out)
        # print("layer4:", out.shape)
        # out = out.view(out.size(0), -1)   # Flatten them for FC
        # print("Flatten:", out.shape)
        # out = self.layer5(out)
        # out_csv = self.meta_csv(y)
        # print("layer5:", out.shape)
        # out = torch.cat((out, out_csv), dim=1)
        out = self.csv(y)
        out = self.fc2(out)
        # print("fc2:", out.shape)
        # out = self.concat_layer(y)
        # out = torch.cat((out, y), dim=1)
        # print("concat_layer:", out.shape)
        # out = self.fc3(out)
        # print("fc3:", out.shape)
        
        return out



class CNN_100(nn.Module):

    def __init__(self):
        super(CNN_100, self).__init__()
        self.keep_prob = 0.7 # 0.5
        # self.images = images
        # self.meta = meta
        # L1 ImgIn shape=(?, 28, 28, 1)
        #    Conv     -> (?, 28, 28, 32)
        #    Pool     -> (?, 14, 14, 32)
        self.layer1 = nn.Sequential(
            nn.Conv2d(264, 1, kernel_size=1, stride=1, padding="same"),
            nn.ReLU()
            #nn.MaxPool2d(kernel_size=2, stride=2)
            )
        # L2 ImgIn shape=(?, 14, 14, 32)
        #    Conv      ->(?, 14, 14, 64)
        #    Pool      ->(?, 7, 7, 64)
        self.layer2 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding="same"),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1))
        # L3 ImgIn shape=(?, 7, 7, 64)
        #    Conv      ->(?, 7, 7, 128)
        #    Pool      ->(?, 4, 4, 128)
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding="same"),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1))

        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding="same"),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1))

        # L4 FC 4x4x128 inputs -> 625 outputs
        self.fc1 = nn.Linear(29 * 29 * 128, 1000, bias=False) # self.fc1 = nn.Linear(65 * 65 * 128, 1000, bias=False)
        nn.init.xavier_uniform_(self.fc1.weight)

        self.layer5 = nn.Sequential(
            self.fc1,
            nn.BatchNorm1d(1000),
            nn.ReLU(),
            nn.Dropout(p=1 - self.keep_prob))
        # L5 Final FC 625 inputs -> 10 outputs (1)
        # self.fc2 = nn.Linear(1000, 100, bias=False)
        # nn.init.xavier_uniform_(self.fc2.weight)

        self.fc2 = nn.Linear(1000, 100, bias=False)
        nn.init.xavier_uniform_(self.fc2.weight)


        self.fc3 = nn.Linear(117, 1, bias=False)
        nn.init.xavier_uniform_(self.fc2.weight)
        # self.concat_layer = torch.cat((self.fc2, ), dim=1)
        
        # self.fc3 = nn.Linear(100, 1, bias=False)
        # self.fc3 = nn.Linear(100, 1, bias=False)
        # nn.init.xavier_uniform_(self.fc3.weight)
    

    def forward(self, x, y):
        # print("x:", x.shape)
        out = self.layer1(x)
        # print("layer1:", out.shape)
        out = self.layer2(out)
        # print("layer2:", out.shape)
        out = self.layer3(out)
        # print("layer3:", out.shape)
        out = self.layer4(out)
        # print("layer4:", out.shape)
        out = out.view(out.size(0), -1)   # Flatten them for FC
        # print("Flatten:", out.shape)
        out = self.layer5(out)
        # print("layer5:", out.shape)
        out = self.fc2(out)
        # print("fc2:", out.shape)
        # out = self.concat_layer(y)
        out = torch.cat((out, y), dim=1)
        # print("concat_layer:", out.shape)
        out = self.fc3(out)
        # print("fc3:", out.shape)
        
        return out



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


class ResNet50Network(nn.Module):
    def __init__(self):
        super().__init__()

        # Define Feature part (IMAGE)
        self.features = resnet50(pretrained=False) # 1000 neurons out
        # (metadata)
        self.csv = nn.Sequential(nn.Linear(17, 500),
                                 nn.BatchNorm1d(500),
                                 nn.ReLU(),
                                 nn.Dropout(p=0.2))
        
        # Define Classification part
        self.classification = nn.Linear(1000 + 500, 1)

        # 1x1 convolution
        self.con1by1 = nn.Sequential(nn.Conv2d(264, 1, kernel_size=1, stride=1, padding="same"), nn.ReLU())



    def forward(self, image, meta, prints=False):
        if prints: print('Input Image shape:', image.shape, '\n'+
                         'Input metadata shape:', meta.shape)
        
        # Image CNN
        image = self.con1by1(image)
        image = self.features(image)
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
