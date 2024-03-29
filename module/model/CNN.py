import torch
import torch.nn as nn
import torch.nn.functional as F

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
            nn.Conv2d(264, 1, kernel_size=1, stride=1, padding="same", bias=False),
            nn.ReLU()
            #nn.MaxPool2d(kernel_size=2, stride=2)
            )
        # L2 ImgIn shape=(?, 14, 14, 32)
        #    Conv      ->(?, 14, 14, 64)
        #    Pool      ->(?, 7, 7, 64)
        self.layer2 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding="same", bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1))
        # L3 ImgIn shape=(?, 7, 7, 64)
        #    Conv      ->(?, 7, 7, 128)
        #    Pool      ->(?, 4, 4, 128)
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding="same", bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1))

        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding="same", bias=False),
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
        # nn.init.xavier_uniform_(self.fc2.weight)

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


class CNN_V2(nn.Module):

    def __init__(self):
        super(CNN_V2, self).__init__()
        self.keep_prob = 0.8 # 0.5
        # shape: [?, 132, 113, 113]
        self.layer1 = nn.Sequential(
            nn.Conv2d(264, 132, kernel_size=3, stride=1, padding="same"),
            nn.BatchNorm2d(132),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
            )

        # shape: [?, 64, 57, 57]
        self.layer2 = nn.Sequential(
            nn.Conv2d(132, 64, kernel_size=3, stride=1, padding="same"),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1))

        # shape: [?, 32, 29, 29]
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding="same"),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1))

        # shape: [?, 16, 15, 15]
        self.layer4 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding="same"),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1))

        # shape: [?, 8, 8, 8]
        self.layer5 = nn.Sequential(
            nn.Conv2d(16, 8, kernel_size=3, stride=1, padding="same"),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1))

        # Layer6 FC 8x8x8 inputs -> 256 outputs
        self.fc1 = nn.Linear(8 * 8 * 8, 256, bias=False)
        nn.init.xavier_uniform_(self.fc1.weight)

        self.fc2 = nn.Linear(512, 256, bias=False)
        nn.init.xavier_uniform_(self.fc2.weight)

        self.fc3 = nn.Linear(256, 128, bias=False)
        nn.init.xavier_uniform_(self.fc3.weight)

        self.fc4 = nn.Linear(128, 64, bias=False)
        nn.init.xavier_uniform_(self.fc4.weight)

        self.fc5 = nn.Linear(64, 1, bias=False)
        nn.init.xavier_uniform_(self.fc5.weight)


        self.layer6 = nn.Sequential(
            self.fc1,
            nn.BatchNorm1d(256),
            nn.ReLU(),
            # nn.Dropout(p=1 - self.keep_prob)
            )

        self.meta_csv = nn.Sequential(
            nn.Linear(17, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=1 - self.keep_prob)
            )


        self.layer7 = nn.Sequential(
            self.fc2,
            # nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=1 - self.keep_prob))

        self.layer8 = nn.Sequential(
            self.fc3,
            # nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=1 - self.keep_prob))

        self.layer9 = nn.Sequential(
            self.fc4,
            # nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(p=1 - self.keep_prob))

        self.layer10 = nn.Sequential(
            self.fc5,
            # nn.BatchNorm1d(1)
            nn.ReLU())


    def forward(self, img, emr):
        out = self.layer1(img)
        # print(out.shape)
        out = self.layer2(out)
        # print(out.shape)
        out = self.layer3(out)
        # print(out.shape)
        out = self.layer4(out)
        # print(out.shape)
        out = self.layer5(out)
        # print(out.shape)

        # Flatten them for FC
        out = out.view(out.size(0), -1)
        # print(out.shape)
        out = self.layer6(out)
        out_csv = self.meta_csv(emr)
        out = torch.cat((out, out_csv), dim=1)
        # out = self.fc2F(out)
        # print(out.shape)
        out = self.layer7(out)
        # # print(out.shape)
        out = self.layer8(out)
        # # print(out.shape)
        out = self.layer9(out)
        # # print(out.shape)
        out = self.layer10(out)
        # print(out.shape)

        return out


class CNN_img_V2(nn.Module):

    def __init__(self):
        super(CNN_img_V2, self).__init__()
        self.keep_prob = 0.7 # 0.5

        self.layer1 = nn.Sequential(
            nn.Conv2d(264, 1, kernel_size=1, stride=1, padding="same"),
            nn.ReLU()
            #nn.MaxPool2d(kernel_size=2, stride=2)
            )

        self.layer2 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding="same"),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1))

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

        # Layer4 FC 29x29x128 inputs -> 1000 outputs
        self.fc1 = nn.Linear(29 * 29 * 128, 1000, bias=True)
        nn.init.xavier_uniform_(self.fc1.weight)

        self.fc2 = nn.Linear(1000, 128, bias=True)
        nn.init.xavier_uniform_(self.fc2.weight)

        self.fc3 = nn.Linear(128, 64, bias=True)
        nn.init.xavier_uniform_(self.fc3.weight)

        self.fc4 = nn.Linear(64, 32, bias=True)
        nn.init.xavier_uniform_(self.fc4.weight)

        self.fc5 = nn.Linear(32, 1, bias=True)
        nn.init.xavier_uniform_(self.fc5.weight)


        self.layer5 = nn.Sequential(
            self.fc1,
            nn.BatchNorm1d(1000),
            nn.ReLU(),
            nn.Dropout(p=1 - self.keep_prob))
        
        self.layer6 = nn.Sequential(
            self.fc2,
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=1 - self.keep_prob))

        self.layer7 = nn.Sequential(
            self.fc3,
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=1 - self.keep_prob))

        self.layer8 = nn.Sequential(
            self.fc4,
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(p=1 - self.keep_prob))

        self.layer9 = nn.Sequential(
            self.fc5,
            nn.BatchNorm1d(1))
    

    def forward(self, img, y):
        out = self.layer1(img)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        # Flatten them for FC
        out = out.view(out.size(0), -1)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)

        return out


class CNN_img_V2w0(nn.Module):

    def __init__(self):
        super(CNN_img_V2w0, self).__init__()
        self.keep_prob = 0.7 # 0.5

        self.layer1 = nn.Sequential(
            nn.Conv2d(264, 1, kernel_size=1, stride=1, padding="same"),
            nn.ReLU()
            #nn.MaxPool2d(kernel_size=2, stride=2)
            )

        self.layer2 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding="same"),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1))

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

        # Layer4 FC 29x29x128 inputs -> 1000 outputs
        self.fc1 = nn.Linear(29 * 29 * 128, 1000, bias=True)
        nn.init.xavier_uniform_(self.fc1.weight)

        self.fc2F = nn.Linear(1000, 1, bias=False)

        self.fc2 = nn.Linear(1000, 128, bias=True)
        # nn.init.xavier_uniform_(self.fc2.weight)

        self.fc3 = nn.Linear(128, 64, bias=True)
        # nn.init.xavier_uniform_(self.fc3.weight)

        self.fc4 = nn.Linear(64, 32, bias=True)
        # nn.init.xavier_uniform_(self.fc4.weight)

        self.fc5 = nn.Linear(32, 1, bias=True)
        # nn.init.xavier_uniform_(self.fc5.weight)


        self.layer5 = nn.Sequential(
            self.fc1,
            nn.BatchNorm1d(1000),
            nn.ReLU(),
            nn.Dropout(p=1 - self.keep_prob))
        
        self.layer6 = nn.Sequential(
            self.fc2,
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=1 - self.keep_prob))

        self.layer7 = nn.Sequential(
            self.fc3,
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=1 - self.keep_prob))

        self.layer8 = nn.Sequential(
            self.fc4,
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(p=1 - self.keep_prob))

        self.layer9 = nn.Sequential(
            self.fc5)
    

    def forward(self, img, y):
        out = self.layer1(img)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        # Flatten them for FC
        out = out.view(out.size(0), -1)
        out = self.layer5(out)
        out = self.fc2F(out)
        # out = self.layer6(out)
        # out = self.layer7(out)
        # out = self.layer8(out)
        # out = self.layer9(out)

        return out


class CNN_img_V3(nn.Module):

    def __init__(self):
        super(CNN_img_V3, self).__init__()
        self.keep_prob = 0.7 # 0.5

        # shape: [?, 132, 113, 113]
        self.layer1 = nn.Sequential(
            nn.Conv2d(264, 132, kernel_size=3, stride=1, padding="same"),
            nn.BatchNorm2d(132),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
            )

        # shape: [?, 64, 57, 57]
        self.layer2 = nn.Sequential(
            nn.Conv2d(132, 64, kernel_size=3, stride=1, padding="same"),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1))

        # shape: [?, 32, 29, 29])
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding="same"),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1))

        # shape: [?, 16, 15, 15])
        self.layer4 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding="same"),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1))

        # shape: [?, 8, 8, 8])
        self.layer5 = nn.Sequential(
            nn.Conv2d(16, 8, kernel_size=3, stride=1, padding="same"),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1))

        # Layer6 FC 8x8x8 inputs -> 256 outputs
        self.fc1 = nn.Linear(8 * 8 * 8, 256, bias=True)
        nn.init.xavier_uniform_(self.fc1.weight)

        self.fc2 = nn.Linear(256, 128, bias=True)
        nn.init.xavier_uniform_(self.fc2.weight)

        self.fc3 = nn.Linear(128, 64, bias=True)
        nn.init.xavier_uniform_(self.fc3.weight)

        self.fc4 = nn.Linear(64, 32, bias=True)
        nn.init.xavier_uniform_(self.fc4.weight)

        self.fc5 = nn.Linear(32, 1, bias=True)
        nn.init.xavier_uniform_(self.fc5.weight)


        self.layer6 = nn.Sequential(
            self.fc1,
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=1 - self.keep_prob))
        
        self.layer7 = nn.Sequential(
            self.fc2,
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=1 - self.keep_prob))

        self.layer8 = nn.Sequential(
            self.fc3,
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=1 - self.keep_prob))

        self.layer9 = nn.Sequential(
            self.fc4,
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(p=1 - self.keep_prob))

        self.layer10 = nn.Sequential(
            self.fc5,
            nn.BatchNorm1d(1))
    

    def forward(self, img, y):
        out = self.layer1(img)
        # print(out.shape)
        out = self.layer2(out)
        # print(out.shape)
        out = self.layer3(out)
        # print(out.shape)
        out = self.layer4(out)
        # print(out.shape)
        out = self.layer5(out)
        # print(out.shape)

        # Flatten them for FC
        out = out.view(out.size(0), -1)
        # print(out.shape)
        out = self.layer6(out)
        # print(out.shape)
        out = self.layer7(out)
        # print(out.shape)
        out = self.layer8(out)
        # print(out.shape)
        out = self.layer9(out)
        # print(out.shape)
        out = self.layer10(out)
        # print(out.shape)

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


        self.fc2 = nn.Sequential(nn.Linear(250, 1, bias=False))
        # nn.init.xavier_uniform_(self.fc2.weight)
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
