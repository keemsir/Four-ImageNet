import pandas as pd
import numpy as np

import torch

from sklearn.metrics import mean_squared_error


def create_csv(path): # './ori_meta.csv'
    # path=path
    return pd.read_csv(path)


def acc_rmse(outputs, values, device):

    predictions = torch.FloatTensor([]).to(device)
    groundtruths = torch.FloatTensor([]).to(device)

    pred = (torch.cat((predictions, outputs), 0)).cpu().detach().numpy()
    gt = (torch.cat((groundtruths, values), 0)).cpu().detach().numpy()

    return np.sqrt(mean_squared_error(pred, gt))



def add_in_file(text, f):
    with open(f"./model/{VERSION}/logs.txt", "a+") as f:
        print(text, file=f)
    print(text)


def text_read(NUM: str):
    with open(f"./model/{NUM}/logs.txt", "r", encoding="UTF8") as f:
        rd = f.read()
    return rd


def acc_sampler_weight(csv, col: str) -> list:
    temp_list = []
    for x in range(0, 90, 10):
        start = x
        end = x + 10
        temp = csv[(csv[col] >= start) & (csv[col] < end)]
        temp_list.append(len(temp))
    return temp_list


def post_weight(train_csv, r, c, sampler_WEIGHT):
    #r: csv's rows
    #o: selection column
    # sampler_WEIGHT: using def acc_sampler_weight

    if train_csv[r][c] <= 10:
        output = (sampler_WEIGHT[0])
    elif train_csv[r][c] <= 20:
        output = (sampler_WEIGHT[1])
    elif train_csv[r][c] <= 30:
        output = (sampler_WEIGHT[2])
    elif train_csv[r][c] <= 40:
        output = (sampler_WEIGHT[3])
    elif train_csv[r][c] <= 50:
        output = (sampler_WEIGHT[4])
    elif train_csv[r][c] <= 60:
        output = (sampler_WEIGHT[5])
    elif train_csv[r][c] <= 70:
        output = (sampler_WEIGHT[6])
    elif train_csv[r][c] <= 80:
        output = (sampler_WEIGHT[7])
    elif train_csv[r][c] <= 90:
        output = (sampler_WEIGHT[8])
    return output


