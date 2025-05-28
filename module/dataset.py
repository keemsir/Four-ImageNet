import pandas as pd

import torch
from torch.utils.data import Dataset
from albumentations import (ToFloat, Normalize, VerticalFlip, HorizontalFlip, Compose, Resize, 
                            RandomBrightnessContrast, HueSaturationValue, Blur, GaussNoise, Rotate, 
                            RandomResizedCrop, Cutout, ShiftScaleRotate, ToGray, ElasticTransform, GridDistortion)
from albumentations.pytorch import ToTensorV2

# from _warnings import warn


class MI_Dataset(Dataset):
    def __init__(self, dataframe: pd, mode: str, image_select: str, target: str): # IMG_select: []
        self.dataframe = dataframe
        self.mode = mode # "train", "val"
        self.selector = image_select
        self.target = target

        augmentation_p = 0.7 # default: 0.5

        # Data Augmentation
        if mode == "train":
            self.transform = Compose([
                                    Resize(height=224, width=224),
                                    ShiftScaleRotate(rotate_limit=90, scale_limit=[0.8, 1.2]),
                                    HorizontalFlip(p = augmentation_p),
                                    VerticalFlip(p = augmentation_p),
                                    ElasticTransform(p = augmentation_p),
                                    GridDistortion(p = augmentation_p),

                                    # RandomBrightnessContrast(p = augmentation_p),
                                    GaussNoise(p = augmentation_p),
                                    Blur(p = augmentation_p),
                                    ToTensorV2()])
                                    # RandomAffine((-90, 90), translate=(0.2, 0.2))
                                    #RandomPerspective(distortion_scale=0.7, p=0.7),
        elif mode == "val":
            self.transform = Compose([Resize(height=224, width=224),
                                    ToTensorV2()])
        else:
            print(f"{self.mode} is error")

        # self.csv_data = np.array(self.dataframe.iloc[index][csv_columns].values)
        # self.y_data = torch.FloatTensor(y_data)

    def __len__(self):
        return len(self.dataframe)


    def __getitemback__(self, index):
        patient_num = self.dataframe["Unit No"][index]
        
        pet_path = f'./DB/tensor/pet/{patient_num}.pt'
        rtct_path = f'./DB/tensor/rtct/{patient_num}.pt'
        rtdose_path = f'./DB/tensor/rtdose/{patient_num}.pt'
        rtst_path = f'./DB/tensor/rtst_data/{patient_num}.pt'
        
        pet_image = torch.load(pet_path).numpy()
        rtct_image = torch.load(rtct_path).numpy()
        rtdose_image = torch.load(rtdose_path).numpy()
        rtst_image = torch.load(rtst_path).numpy()
        
        # ["mSUV", "sex", "age", "ecog", "smoking", "pys", "vet", "patho", "egfr", "alk", "stage", "cont_ln", "scn", "rtmod", "concht_reg", "tv_f", "ctv_f"]
        csv_data = torch.FloatTensor(self.dataframe.iloc[index][3:-3])
        if self.target == 'os':
            target_data = torch.FloatTensor(self.dataframe.iloc[index][-1:]) # index:::-1: = os
        elif self.target == 'pfs':
            target_data = torch.FloatTensor(self.dataframe.iloc[index][-2:-1]) # index:::-2:-1 = pfs
        elif self.target == 'lrpfs':
            target_data = torch.FloatTensor(self.dataframe.iloc[index][-3:-2]) # index:::-3:-2 = lrpfs
        else:
            print("Target must select from 'os', 'pfs', 'lrpfs'")
        # target_data = torch.FloatTensor(self.dataframe.iloc[index][-2:-1]) # index:::-2:-1 = pfs
        # target_data = torch.FloatTensor(self.dataframe.iloc[index][-3:])
        
        pet_image = self.transform(image=pet_image)['image']
        rtct_image = self.transform(image=rtct_image)['image']
        rtdose_image = self.transform(image=rtdose_image)['image']
        rtst_image = self.transform(image=rtst_image)['image']
        
        # pet_image = FF.resize(pet_image.permute(2,0,1), 224)
        # rtct_image = FF.resize(rtct_image.permute(2,0,1), 224)
        # rtdose_image = FF.resize(rtdose_image.permute(2,0,1), 224)
        # rtst_image = FF.resize(rtst_image.permute(2,0,1), 224)
        
        transf_image = torch.cat((pet_image, rtct_image, rtdose_image, rtst_image), dim=0)
        transf_image = transf_image.type(torch.FloatTensor)
        
        return (transf_image, csv_data, target_data)



    def __getitem__(self, index):
        patient_num = self.dataframe["Unit No"][index]
        
        if self.selector == None:
            pet_path = f'./DB/tensor/pet/{patient_num}.pt'
            pet_image = torch.load(pet_path).numpy()
            pet_image = self.transform(image=pet_image)['image']

            rtct_path = f'./DB/tensor/rtct/{patient_num}.pt'
            rtct_image = torch.load(rtct_path).numpy()
            rtct_image = self.transform(image=rtct_image)['image']

            rtdose_path = f'./DB/tensor/rtdose/{patient_num}.pt'
            rtdose_image = torch.load(rtdose_path).numpy()
            rtdose_image = self.transform(image=rtdose_image)['image']

            rtst_path = f'./DB/tensor/rtst_data/{patient_num}.pt'
            rtst_image = torch.load(rtst_path).numpy()
            rtst_image = self.transform(image=rtst_image)['image']

            transf_image = torch.cat((pet_image, rtct_image, rtdose_image, rtst_image), dim=0)
            transf_image = transf_image.type(torch.FloatTensor)
        
        
        elif self.selector == "pet":
            pet_path = f'./DB/tensor/pet/{patient_num}.pt'
            pet_image = torch.load(pet_path).numpy()
            pet_image = self.transform(image=pet_image)['image']

            # transf_image = torch.cat((pet_image), dim=0)
            transf_image = pet_image.type(torch.FloatTensor)

        elif self.selector == "rtct":
            rtct_path = f'./DB/tensor/rtct/{patient_num}.pt'
            rtct_image = torch.load(rtct_path).numpy()
            rtct_image = self.transform(image=rtct_image)['image']

            # transf_image = torch.cat((rtct_image), dim=0)
            transf_image = rtct_image.type(torch.FloatTensor)

        elif self.selector == "rtdose":
            rtdose_path = f'./DB/tensor/rtdose/{patient_num}.pt'
            rtdose_image = torch.load(rtdose_path).numpy()
            rtdose_image = self.transform(image=rtdose_image)['image']

            # transf_image = torch.cat((rtdose_image), dim=0)
            transf_image = rtdose_image.type(torch.FloatTensor)

        elif self.selector == "rtst":
            rtst_path = f'./DB/tensor/rtst_data/{patient_num}.pt'
            rtst_image = torch.load(rtst_path).numpy()
            rtst_image = self.transform(image=rtst_image)['image']

            # transf_image = torch.cat((rtst_image), dim=0)
            transf_image = rtst_image.type(torch.FloatTensor)


        elif self.selector == "petrtst":
            pet_path = f'./DB/tensor/pet/{patient_num}.pt'
            pet_image = torch.load(pet_path).numpy()
            pet_image = self.transform(image=pet_image)['image']

            rtst_path = f'./DB/tensor/rtst_data/{patient_num}.pt'
            rtst_image = torch.load(rtst_path).numpy()
            rtst_image = self.transform(image=rtst_image)['image']

            transf_image = torch.cat((pet_image, rtst_image), dim=0)
            transf_image = transf_image.type(torch.FloatTensor)

        elif self.selector == "petct":
            pet_path = f'./DB/tensor/pet/{patient_num}.pt'
            pet_image = torch.load(pet_path).numpy()
            pet_image = self.transform(image=pet_image)['image']

            rtct_path = f'./DB/tensor/rtct/{patient_num}.pt'
            rtct_image = torch.load(rtct_path).numpy()
            rtct_image = self.transform(image=rtct_image)['image']

            transf_image = torch.cat((pet_image, rtct_image), dim=0)
            transf_image = transf_image.type(torch.FloatTensor)

        elif self.selector == "petdose":
            pet_path = f'./DB/tensor/pet/{patient_num}.pt'
            pet_image = torch.load(pet_path).numpy()
            pet_image = self.transform(image=pet_image)['image']

            rtdose_path = f'./DB/tensor/rtdose/{patient_num}.pt'
            rtdose_image = torch.load(rtdose_path).numpy()
            rtdose_image = self.transform(image=rtdose_image)['image']

            transf_image = torch.cat((pet_image, rtdose_image), dim=0)
            transf_image = transf_image.type(torch.FloatTensor)

        elif self.selector == "ctdose":
            rtct_path = f'./DB/tensor/rtct/{patient_num}.pt'
            rtct_image = torch.load(rtct_path).numpy()
            rtct_image = self.transform(image=rtct_image)['image']

            rtdose_path = f'./DB/tensor/rtdose/{patient_num}.pt'
            rtdose_image = torch.load(rtdose_path).numpy()
            rtdose_image = self.transform(image=rtdose_image)['image']

            transf_image = torch.cat((rtct_image, rtdose_image), dim=0)
            transf_image = transf_image.type(torch.FloatTensor)

        elif self.selector == "ctst":
            rtct_path = f'./DB/tensor/rtct/{patient_num}.pt'
            rtct_image = torch.load(rtct_path).numpy()
            rtct_image = self.transform(image=rtct_image)['image']

            rtst_path = f'./DB/tensor/rtst_data/{patient_num}.pt'
            rtst_image = torch.load(rtst_path).numpy()
            rtst_image = self.transform(image=rtst_image)['image']

            transf_image = torch.cat((rtct_image, rtst_image), dim=0)
            transf_image = transf_image.type(torch.FloatTensor)

        elif self.selector == "dosest":
            rtdose_path = f'./DB/tensor/rtdose/{patient_num}.pt'
            rtdose_image = torch.load(rtdose_path).numpy()
            rtdose_image = self.transform(image=rtdose_image)['image']

            rtst_path = f'./DB/tensor/rtst_data/{patient_num}.pt'
            rtst_image = torch.load(rtst_path).numpy()
            rtst_image = self.transform(image=rtst_image)['image']

            transf_image = torch.cat((rtdose_image, rtst_image), dim=0)
            transf_image = transf_image.type(torch.FloatTensor)

        else:
            print("Image must select from 'pet', 'rtct', 'rtdose', 'rtst'")


        # ["mSUV", "sex", "age", "ecog", "smoking", "pys", "vet", "patho", "egfr", "alk", "stage", "cont_ln", "scn", "rtmod", "concht_reg", "tv_f", "ctv_f"]
        csv_data = torch.FloatTensor(self.dataframe.iloc[index][3:-3])
        if self.target == 'os':
            target_data = torch.FloatTensor(self.dataframe.iloc[index][-1:]) # index:::-1: = os
        elif self.target == 'pfs':
            target_data = torch.FloatTensor(self.dataframe.iloc[index][-2:-1]) # index:::-2:-1 = pfs
        elif self.target == 'lrpfs':
            target_data = torch.FloatTensor(self.dataframe.iloc[index][-3:-2]) # index:::-3:-2 = lrpfs
        else:
            print("Target must select from 'os', 'pfs', 'lrpfs'")


        return (transf_image, csv_data, target_data)

