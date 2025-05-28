import argparse
import os

from module.network import NetworkTrainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("network", help="example: 'CNN', 'effnet', 'resnet' + ('_img', '_meta') ")
    parser.add_argument("target", help="example: 'os', 'pfs', 'lrpfs'")


    parser.add_argument("--select_fold", default=None, help="example: 12")
    parser.add_argument("--select_img", default=None, help="example: pet, rtct, rtdose, rtst")
    parser.add_argument("--fold_seed", default=None, help="example: 159593293")
    parser.add_argument("--model_version", default=None, help="example: 202304040444")
    parser.add_argument("--cuda_device", required=False, default="0", help="Allocate memory to GPU")

    # merge option (multi-modality) 24.10
    parser.add_argument("--img_merge", required=False, default=False, help="Multi-modality image to merge")

    args = parser.parse_args()


    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device


    network = str(args.network)
    target = str(args.target)


    ##
    if args.model_version is not None:
        model_version = int(args.model_version)
    else:
        model_version = args.model_version

    ##
    if args.fold_seed is not None:
        fold_seed = int(args.fold_seed)
    else:
        fold_seed = args.fold_seed

    ##
    if args.select_fold is not None:
        select_fold = list(map(int, args.select_fold))
    else:
        select_fold = args.select_fold

    ##
    if args.select_img is not None:
        select_img = str(args.select_img)
    else:
        select_img = None

    ####

    cuda_device = args.cuda_device

    img_merge = args.img_merge


    trainer = NetworkTrainer(network=network, TARGET=target, model_VER=model_version, fold_SEED=fold_seed, select_FOLD=select_fold, select_image=select_img, CUDA_device=cuda_device, img_Merge=img_merge)
    trainer.run_training()


if __name__ == "__main__":
    main()
