import argparse
import os

from module.network import NetworkTrainer

# from module.dataset import MI_Dataset
# from module.model import CNN, EffNetNetwork, ResNet50Network
# from module.utils import acc_rmse, acc_sampler_weight, post_weight
# from nn_utils import path_utils
# from util_msg import slack_msg




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("network", help="example: 'CNN', 'effnet', 'resnet'")
    parser.add_argument("target", help="example: 'os', 'pfs', 'lrpfs'")

    parser.add_argument("-i", "--input_folder", required=True)
    parser.add_argument("-o", "--output_folder", required=True)

    parser.add_argument("--cuda_device", required=False, default="0", help="Allocate memory to GPU")

    args = parser.parse_args()


    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device


    network = str(args.network)
    target = str(args.target)

    input_folder = args.input_folder
    output_folder = args.output_folder


    trainer = NetworkTrainer(network=network, TARGET=target)
    trainer.predict()


if __name__ == "__main__":
    main()


class inference(NetworkTrainer()):


    pass
