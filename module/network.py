import os
import numpy as np
import pandas as pd
import random
import datetime as dtime
from time import time

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import WeightedRandomSampler

import matplotlib
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold

from module.dataset import MI_Dataset
from module.model import CNN, EffNetNetwork, ResNet50Network, CNN_img, CNN_img_V2, CNN_img_V3, CNN_img_V2w0, CNN_img_V3w0, CNN_meta, EffNetNetwork_image, EffNetNetwork_meta, CNN_V2, CNN_V3, CNN_meta_V2, CNN_img_V3_sg, CNN_img_V3_2img, CNN_V2_2img
from module.utils import acc_rmse, acc_sampler_weight, post_weight
from nn_utils import path_utils
from util_msg import tele_msg, tele_img


class NetworkTrainer(object):
    def __init__(self, deterministic=True, network=None, TARGET=None, model_VER=None, fold_SEED=None, select_FOLD=None, select_image=None, CUDA_device=None, img_Merge=False):

        if deterministic:
            np.random.seed(1988)
            torch.manual_seed(1988)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(1988)
            cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True


        ############################################################### optional
        self.NOW = dtime.datetime.now()

        if model_VER is not None:
            self.VERSION = model_VER
        else:
            self.VERSION = f'{self.NOW.year:0>4}{self.NOW.month:0>2}{self.NOW.day:0>2}{self.NOW.hour:0>2}{self.NOW.minute:0>2}'


        if fold_SEED is not None:
            self.fold_SEED = fold_SEED
        else:
            self.fold_SEED = random.randrange(1,999999999)


        if select_FOLD is not None:
            self.select_FOLD = select_FOLD
        else:
            self.select_FOLD = [0,1,2,3,4]

        if select_image is not None:
            self.select_image = select_image
        else:
            self.select_image = None

        ###############################################################
        self.network = network
        self.optimizer = None
        self.lr_scheduler = None

        ###############################################################

        self.fold = None
        self.loss = None
        self.criterion = None
        self.dataset_directory = None

        ###############################################################
        self.dataset = None
        self.dataset_tr = None
        self.dataset_val = None

        ###############################################################
        self.PATIENCE = 100
        self.patience_f = None
        self.max_num_eopchs = 500
        self.minimum_number_of_epochs = 50

        ###############################################################
        self.all_tr_losses = []
        # self.all_val_losses = []
        self.all_tr_rmse = []
        self.all_val_rmse = []

        # self.train_losses = 0
        # self.train_acc = []
        # self.valid_acc = []

        self.best_rmse = None
        self.best_tr_rmse = None

        # self.log_file = None
        self.deterministic = deterministic

        ###############################################################
        # self.save_every = 50
        # self.save_best_checkpoint = True
        # self.save_final_checkpoint = True

        ############################################################### dataloader
        self.path_csv = './ori_meta.csv'
        self.path_csv0 = ''

        ############################################################### parameter setting
        self.learning_rate = 0.005
        self.epoch = 0
        self.epochs = 500
        self.batch_size = 128 # 128
        # self.NOW = dtime.datetime.now()
        self.DURATION = None
        self.final_logs = None
        # self.VERSION = f'{self.NOW.year:0>4}{self.NOW.month:0>2}{self.NOW.day:0>2}{self.NOW.hour:0>2}{self.NOW.minute:0>2}'
        self.output_folder = os.path.join("./model/", f"{self.VERSION}")
        self.save_path = None
        self.progress_savepath = None

        ############################################################### parameter setting
        self.TARGET = TARGET # "os", "pfs", "lrpfs"
        self.model = None
        # self.device = None

        if not torch.cuda.is_available():
            print("WARNING!!! You are attempting to run training on a CPU (torch.cuda.is_available() is False). This can be VERY VERY slow!")
            self.device = 'cpu'

        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
            self.device = 'cuda'

        ############################################################### message token info (Telegram)
        self.telegram_TOKEN = '6028104593:AAHxXAgQghw_cW2WsuG7pdXl9fiYFUAbHJI'
        self.telegram_ID = 6261233157

        ############################################################### etc parameter setting
        # self.train_csv = None
        # self.valid_csv = None
        # self.sampler_w = None
        # self.class_weight_tr = None
        # self.class_weight_val = None
        # self.train_data = None
        # self.valid_data = None
        # self.sampler_tr = None
        # self.sampler_val = None
        # self.train_loader = None
        # self.valid_loader = None

        ############################################################### for prediction

        self.CUDA_device = CUDA_device

        self.img_Merge = img_Merge


    def dataloader_csv(self):
        return pd.read_csv(self.path_csv)



    def plot_progress(self):
        font = {'weight': 'normal', 'size': 18}
        matplotlib.rc('font', **font)
        self.progress_savepath = f'{self.output_folder}/Fold{self.fold+1}_progress.png'

        fig = plt.figure(figsize=(30, 24))
        ax = fig.add_subplot(111)
        ax2 = ax.twinx()
        
        x_values = list(range(self.epoch + 1))

        ax.plot(x_values, self.all_tr_losses, color='b', ls='-', label="loss_tr")

        ax2.plot(x_values, self.all_tr_rmse, color='g', ls='-', label="rmse_tr, train=True")
        ax2.plot(x_values, self.all_val_rmse, color='r', ls='-', label="rmse_val, train=False")

        ax.set_xlabel("epoch")
        ax.set_ylim(0, 100)
        ax.set_ylabel("loss")
        
        ax2.set_ylim(0, 50)
        ax2.set_ylabel("rmse")
        
        ax.legend()
        ax2.legend(loc=9)
        
        # fig.savefig(f"./model/{VERSION}/Fold{fold+1}_progress.png")
        # fig.savefig(os.path.join(self.output_folder, f"/Fold{self.fold+1}_progress.png"))
        fig.savefig(self.progress_savepath)
        
        plt.close()


    def print_log_file(self, text, f):
        with open(f"{self.output_folder}/logs_{self.network}_{self.TARGET}_{self.select_image}.txt", "a+") as f:
            print(text, file=f)
        print(text)


    # def acc_rmse(self, outputs, values):


    #     predictions = torch.FloatTensor([]).to(self.device)
    #     groundtruths = torch.FloatTensor([]).to(self.device)

    #     pred = (torch.cat((predictions, outputs), 0)).cpu().detach().numpy()
    #     gt = (torch.cat((groundtruths, values), 0)).cpu().detach().numpy()

    #     return np.sqrt(mean_squared_error(pred, gt))


    def run_training(self):
        if not torch.cuda.is_available():
            print("WARNING!!! You are attempting to run training on a CPU (torch.cuda.is_available() is False). This can be VERY slow!")
            self.device = 'cpu'
        
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
            self.device = 'cuda'
        
        path_utils.maybe_mkdir_p(self.output_folder)

        f = open(os.path.join(self.output_folder, f"logs_{self.network}_{self.TARGET}_{self.select_image}.txt"), "w+")

        self.print_log_file(f'Network Architecture: {self.network}', f)
        self.print_log_file(f'CUDA_device: {self.CUDA_device}', f)
        self.print_log_file(f'Version: {self.VERSION}', f)
        self.print_log_file(f'SEED number: {self.fold_SEED}', f)
        self.print_log_file(f'Learning rate: {self.learning_rate}', f)
        self.print_log_file(f'Epochs: {self.epochs}', f)
        self.print_log_file(f'Batch size: {self.batch_size}', f)
        self.print_log_file(f'Target: {self.TARGET}', f)
        self.print_log_file(f'Used image: {self.select_image}', f)
        self.print_log_file(f'Merged image: {self.img_Merge}', f)



        kfold = KFold(n_splits=5, shuffle=True, random_state=self.fold_SEED)

        test_csv = self.dataloader_csv()

        for self.fold, (train_idx, val_idx) in enumerate(kfold.split(test_csv)):
            # fold passes
            if self.fold not in self.select_FOLD:
                print(f"Passes {self.fold} Fold")

            else:
                # Dataloader
                train_csv = test_csv.iloc[train_idx].reset_index(drop=True)
                valid_csv = test_csv.iloc[val_idx].reset_index(drop=True)
                sampler_w = acc_sampler_weight(csv=train_csv, col=self.TARGET)

                class_weight_tr = [len(train_csv)/post_weight(train_csv=train_csv, r=self.TARGET, c=i, sampler_WEIGHT=sampler_w) for i in range(len(train_csv))]
                class_weight_val = [len(valid_csv)/post_weight(train_csv=train_csv, r=self.TARGET, c=i, sampler_WEIGHT=sampler_w) for i in range(len(valid_csv))]

                train_data = MI_Dataset(dataframe=train_csv, mode="train", image_select=self.select_image, target=self.TARGET) # ["pet", "rtct", "rtdose", "rtst"]
                valid_data = MI_Dataset(dataframe=valid_csv, mode="val", image_select=self.select_image, target=self.TARGET) # ["pet", "rtct", "rtdose", "rtst"]

                sampler_tr = WeightedRandomSampler(class_weight_tr, int(len(train_data)))
                sampler_val = WeightedRandomSampler(class_weight_val, int(len(valid_data)))

                train_loader = torch.utils.data.DataLoader(train_data, batch_size=self.batch_size, sampler=sampler_tr, pin_memory=True)
                valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=self.batch_size, pin_memory=True) # , sampler=sampler_val

                # print log
                self.print_log_file(f"\n\n---------- Fold: {self.fold+1} ----------", f)

                self.all_tr_losses = []
                self.all_tr_rmse = []
                self.all_val_rmse = []

                self.patience_f = self.PATIENCE
 
                self.best_rmse = None
                self.best_tr_rmse = None
                
                # Select network architecture
                if self.network == 'CNN':
                    self.model = CNN().to(self.device)
                elif self.network == 'CNN_V2':
                    self.model = CNN_V2().to(self.device)
                elif self.network == 'CNN_V2_2img':
                    self.model = CNN_V2_2img().to(self.device)
                elif self.network == 'CNN_V3':
                    self.model = CNN_V3().to(self.device)
                elif self.network == 'CNN_img':
                    self.model = CNN_img().to(self.device)
                elif self.network == 'CNN_img_V2':
                    self.model = CNN_img_V2().to(self.device)
                elif self.network == 'CNN_img_V2w0':
                    self.model = CNN_img_V2w0().to(self.device)
                elif self.network == 'CNN_img_V3':
                    self.model = CNN_img_V3().to(self.device)
                elif self.network == 'CNN_img_V3_sg':
                    self.model = CNN_img_V3_sg().to(self.device)
                elif self.network == 'CNN_img_V3_2img':
                    self.model = CNN_img_V3_2img().to(self.device)
                elif self.network == 'CNN_img_V3w0':
                    self.model = CNN_img_V3w0().to(self.device)
                elif self.network == 'CNN_meta':
                    self.model = CNN_meta().to(self.device)
                elif self.network == 'CNN_meta_V2':
                    self.model = CNN_meta_V2().to(self.device)
                elif self.network == 'effnet':
                    self.model = EffNetNetwork().to(self.device)
                elif self.network == 'effnet_img':
                    self.model = EffNetNetwork_image().to(self.device)
                elif self.network == 'effnet_meta':
                    self.model = EffNetNetwork_meta().to(self.device)
                elif self.network == 'resnet':
                    self.model = ResNet50Network().to(self.device)
                
                # Optimization setting
                self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

                # Scheduler setting (current: False)
                self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=self.optimizer, lr_lambda=lambda epoch: 0.95 ** epoch, last_epoch=-1, verbose=False)

                # loss function setting
                self.criterion = torch.nn.SmoothL1Loss(beta=14.0).to(self.device) # 13.0
                # self.criterion = torch.nn.MSELoss().to(self.device)

                for self.epoch in range(self.epochs):

                    start_time = time()
                    train_losses = 0

                    train_acc = []
                    valid_acc = []
                    
                    # ==== TRAIN MODE ==== #
                    self.model.train()
                    for X, M, Y in train_loader:
                        X = X.to(self.device)
                        M = M.to(self.device)
                        Y = Y.to(self.device)

                        self.optimizer.zero_grad()

                        outputs = self.model(X, M)
                        self.loss = self.criterion(outputs, Y)
                        self.loss.backward()
                        self.optimizer.step()

                        train_losses += self.loss.item()
                        train_acc.append(acc_rmse(outputs, Y, self.device))


                    # ==== EVALUATION MODE ==== #
                    self.model.eval()
                    with torch.no_grad():
                        for X, M, Y in valid_loader:
                            X = X.to(self.device)
                            M = M.to(self.device)
                            Y = Y.to(self.device)
                            
                            outputs = self.model(X, M)
                            
                            # print("Y: ",Y)
                            # print("outputs: ",outputs)
                            valid_acc.append(acc_rmse(outputs, Y, self.device))
                            # print("valid_acc: ",valid_acc)
                    
                    
                    # Log
                    self.all_tr_losses.append(train_losses/len(train_loader))
                    self.all_tr_rmse.append(np.mean(train_acc))
                    self.all_val_rmse.append(np.mean(valid_acc))
                    # print("all_val_rmse: ", self.all_val_rmse)

                    # PRINT INFO
                    self.DURATION = str(dtime.timedelta(seconds=time() - start_time))[:7]
                    self.final_logs = '{} | [Epoch: {:>4}] loss = {:>.9}, train_rmse:{:>.5}, validation_rmse:{:>.5}.'.format(self.DURATION, self.epoch+1, self.all_tr_losses[-1], self.all_tr_rmse[-1], self.all_val_rmse[-1])

                    self.print_log_file(self.final_logs, f)

                    # ----------- plt show save ----------- #
                    self.plot_progress()


                    # self.save_path = os.path.join((self.output_folder, 'Fold{}_Epoch{}_ValRMSE_{:.2f}.pth'.format(self.fold+1, self.epoch+1, np.min(valid_acc))))
                    self.save_path = '{}/Fold{}_Epoch{}_ValRMSE_{:.2f}.pth'.format(self.output_folder, self.fold+1, self.epoch+1, self.all_val_rmse[-1]) # mean < min (self.all_val_rmse[-1]), np.mean(valid_acc)

                    if self.epoch > self.minimum_number_of_epochs: # default: 100
                        
                        # Update best_rmse of validation dataset
                        if not self.best_rmse:
                            self.best_rmse = self.all_val_rmse[-1]
                            self.best_tr_rmse = self.all_tr_rmse[-1]
                            torch.save(self.model.state_dict(), self.save_path)
                            self.model_named = self.save_path
                            continue

                        if self.all_val_rmse[-1] < self.best_rmse:
                            self.best_rmse = self.all_val_rmse[-1]
                            self.best_tr_rmse = self.all_tr_rmse[-1]
                            self.patience_f = self.PATIENCE
                            torch.save(self.model.state_dict(), self.save_path)
                            os.remove(self.model_named)
                            self.model_named = self.save_path
                            
                        else:
                            self.patience_f = self.patience_f - 1
                            if self.patience_f == 0:
                                stop_logs = f'Early stopping (no improvement since {self.PATIENCE} models) | Best RMSE: {self.best_rmse}'
                                self.print_log_file(stop_logs, f)
                                print(stop_logs)
                                break

                try:
                    tele_msg(TOKEN=self.telegram_TOKEN, ID=self.telegram_ID, MSG=f"Training Completed.(Fold: {self.fold+1}/5) :thumbsup: \n final_logs -> {self.final_logs}, best train RMSE -> {self.best_tr_rmse}, best validation RMSE -> {self.best_rmse}")
                    tele_img(TOKEN=self.telegram_TOKEN, ID=self.telegram_ID, IMG_PATH=self.progress_savepath)
                except Exception as e:
                    print("Exception in telegram", e)

                torch.cuda.empty_cache()



    def predict(self, SEED):
        # Path setting
        model_path = os.path.join('./model', f'{self.model_ver}')
        pth_path = path_utils.subfiles(model_path, suffix='pth')

        valid_acc = []

        # Select network architecture
        if self.network == 'CNN':
            self.model = CNN().to(self.device)
        elif self.network == 'effnet':
            self.model = EffNetNetwork().to(self.device)
        elif self.network == 'resnet':
            self.model = ResNet50Network().to(self.device)
        else:
            print(f"'{self.network}' does not exist. Select one of 'CNN', 'effnet' and 'resnet'")
            # break

        csv = self.dataloader_csv()

        valid_data = MI_Dataset(dataframe=csv, mode="val", image_select=["pet", "rtct", "rtdose", "rtst_data"], target=self.TARGET)
        valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=128, shuffle=False, pin_memory=True)


        for pth in pth_path:
            self.model.load_state_dict(torch.load(pth))


        self.model.eval()

        with torch.no_grad():
            for X, M, Y in valid_loader:
                X = X.to(self.device)
                M = M.to(self.device)
                Y = Y.to(self.device)
                
                outputs = self.model(X, M)

                valid_acc.append(acc_rmse(outputs, Y, self.device))


        torch.cuda.empty_cache()

        # output
        print(valid_acc)

