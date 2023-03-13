import argparse
import os

# from module.dataset import MI_Dataset
from module.network import NetworkTrainer
# from module.model import CNN, EffNetNetwork, ResNet50Network
# from module.utils import acc_rmse, create_csv



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("network")
    # parser.add_argument("fold", help="all or 1")
    parser.add_argument("--fold_seed", default=None)
    parser.add_argument("--cuda_device", required=False, default="0", help="Allocate memory to GPU")

    args = parser.parse_args()


    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device

    network = args.network
    fold_seed = args.fold_seed



    # device = 'cuda' if torch.cuda.is_available() else 'cpu'


    # if fold == 'all':
    #     pass
    # else:
    #     fold = int(fold)
    # trainer = NetworkTrainer(network=network)
    # trainer(network=network, fold_SEED=fold_seed)
    # trainer.run_training()
    NetworkTrainer()






if __name__ == "__main__":
    main()
