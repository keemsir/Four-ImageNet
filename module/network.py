import numpy as np
import pandas as pd
import os
import random
import datetime as dtime

import torch
import torch.backends.cudnn as cudnn

from module.dataset import MI_Dataset
from module.model import CNN, EffNetNetwork, ResNet50Network
from module.utils import acc_rmse
from nn_utils import path_utils



class NetworkTrainer(object):
    def __init__(self, deterministic=True, fold_SEED=None, network=None):
        
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
        

        if fold_SEED is not None:
            self.fold_SEED = fold_SEED
        else:
            self.fold_SEED = random.randrange(1,999999999)

        ###############################################################
        self.network = str(network)
        self.optimizer = None
        self.lr_scheduler = None
        
        ###############################################################
        self.output_folder = None
        self.save_path = None
        self.fold = None
        self.loss = None
        self.criterion = None
        self.dataset_directory = None

        ###############################################################
        self.dataset = None
        self.dataset_tr = None
        self.dataset_val = None

        ###############################################################
        self.PATIENCE = 150
        self.max_num_eopchs = 500

        ###############################################################
        self.all_tr_losses = []
        self.all_val_losses = []
        self.train_acc = []
        self.valid_acc = []

        self.all_tr_rmse = []
        self.all_val_rmse = []

        self.train_losses = 0
        self.best_rmse = None

        # self.log_file = None
        self.deterministic = deterministic

        ###############################################################
        self.save_every = 50
        self.save_best_checkpoint = True
        self.save_final_checkpoint = True

        ############################################################### dataloader
        self.path_csv = './ori_meta.csv'

        ############################################################### parameter setting
        self.learning_rate = 0.001
        self.epoch = 0
        self.epochs = 500
        self.batch_size = 128
        self.NOW = dtime.datetime.now()
        self.VERSION = f'{self.NOW.year:0>4}{self.NOW.month:0>2}{self.NOW.day:0>2}{self.NOW.hour:0>2}{self.NOW.minute:0>2}'
        self.DURATION = None
        self.final_logs = None

        ############################################################### parameter setting
        self.TARGET = None # "os", "pfs", "lrpfs"
        self.device = None
        self.model = None


    def dataloader_csv(self):
        return pd.read_csv(self.path_csv)


    def split(self):
        pass


    def plot_progress(self):
        font = {'weight': 'normal', 'size': 18}
        matplotlib.rc('font', **font)

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
        fig.savefig(os.path.join(self.output_folder, f"{self.VERSION}/Fold{self.fold+1}_progress.png"))
        
        plt.close()


    def print_log_file(self, text, f):
        with open(f"./logs/logs_{self.VERSION}.txt", "a+") as f:
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
        

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            self.device = 'cuda'
        
        path_utils.maybe_mkdir_p(self.output_folder)

        f = open(os.path.join(self.output_folder, f"{VERSION}/logs.txt"), "w+")

        print_log_file(f'Version: {self.VERSION}', f)
        print_log_file(f'SEED number: {self.fold_SEED}', f)
        print_log_file(f'learning rate: {self.learning_rate}', f)
        print_log_file(f'epochs: {self.epochs}', f)
        print_log_file(f'batch size: {self.batch_size}', f)

        for fold, (train_idx, val_idx) in enumerate(kfold.split(dataloader_csv())):

            # Dataloader
            train_csv = test_csv.iloc[train_idx].reset_index(drop=True)
            valid_csv = test_csv.iloc[val_idx].reset_index(drop=True)

            sampler_w = acc_sampler_weight(csv=train_csv, col=self.TARGET)

            class_weight_tr = [len(train_csv)/post_weight(train_csv=train_csv, r=self.TARGET, c=i, sampler_WEIGHT=sampler_w) for i in range(len(train_csv))]
            class_weight_val = [len(valid_csv)/post_weight(train_csv=train_csv, r=self.TARGET, c=i, sampler_WEIGHT=sampler_w) for i in range(len(valid_csv))]

            train_data = MI_Dataset(dataframe=train_csv, mode="train", image_select=["pet", "rtct", "rtdose", "rtst_data"])
            valid_data = MI_Dataset(dataframe=valid_csv, mode="val", image_select=["pet", "rtct", "rtdose", "rtst_data"])

            sampler_tr = WeightedRandomSampler(class_weight_tr, int(len(train_data)))
            sampler_val = WeightedRandomSampler(class_weight_val, int(len(valid_data)))

            train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=sampler_tr, pin_memory=True)
            valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, sampler=sampler_val, pin_memory=True)

            # print log
            print_log_file(f"\n\n---------- Fold: {fold+1} ----------", f)

            # Create Instances
            if network == 'CNN':
                self.model = CNN().to(self.device)
            elif network == 'effnet':
                self.model = EffNetNetwork().to(self.device)
            elif network == 'resnet':
                self.model = ResNet50Network().to(self.device)
            
            # optimization setting
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

            self.criterion = torch.nn.SmoothL1Loss(beta=15.0).to(self.device)

            for self.epoch in range(self.epochs):

                start_time = time()
                
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

                    self.train_losses += self.loss.item()
                    self.train_acc.append(acc_rmse(outputs, Y, self.device))


                # ==== EVALUATION MODE ==== #
                self.model.eval()
                with torch.no_grad():
                    for X, M, Y in valid_loader:
                        X = X.to(self.device)
                        M = M.to(self.device)
                        Y = Y.to(self.device)
                        
                        outputs = self.model(X, M)

                        self.valid_acc.append(acc_rmse(outputs, Y, self.device))

                
                # Log
                self.all_tr_losses.append(self.train_losses/len(train_loader))
                self.all_tr_rmse.append(np.mean(self.train_acc))
                self.all_val_rmse.append(np.mean(self.valid_acc))

                # PRINT INFO
                self.DURATION = str(dtime.timedelta(seconds=time() - start_time))[:7]
                self.final_logs = '{} | [Epoch: {:>4}] loss = {:>.9}, train_rmse:{:>.5}, validation_rmse:{:>.5}.'.format(self.DURATION, self.epoch+1, self.all_tr_losses[-1], self.all_tr_rmse[-1], self.all_val_rmse[-1])

                print_log_file(self.final_logs, f)

                # ----------- plt show save ----------- #
                self.plot_progress()


                self.save_path = os.path.join((self.output_folder, 'Fold{}_Epoch{}_ValRMSE_{:.2f}.pth'.format(self.fold+1, self.epoch+1, np.min(self.valid_acc))))

                # Update best_rmse
                if not self.best_rmse:
                    self.best_rmse = valid_acc
                    torch.save(model.state_dict(), self.save_path)
                    model_named = self.save_path
                    continue

                if valid_acc < self.best_rmse:
                    self.best_rmse = valid_acc
                    self.patience_f = self.PATIENCE
                    torch.save(model.state_dict(), self.save_path)
                    os.remove(model_named)
                    model_named = self.save_path
                    # os.remove(model_named)
                    
                else:
                    self.patience_f = self.patience_f - 1
                    if self.patience_f == 0:
                        stop_logs = f'Early stopping (no improvement since {PATIENCE} models) | Best RMSE: {best_rmse}'
                        add_in_file(stop_logs, f)
                        print(stop_logs)
                        break

            slack_msg("xoxb-4205210278229-4205226757301-ij1wfwNZPfoFlIe7e6kvn6zC", "D046TRYDNCQ", f"Training Completed.(GPU:0) :thumbsup: final_logs -> {self.final_logs}, best validation rmse: {np.min(self.all_val_rmse)}")
            torch.cuda.empty_cache()

