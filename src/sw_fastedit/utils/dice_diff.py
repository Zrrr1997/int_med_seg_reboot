import os
import argparse


import numpy as np




def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dice_path_1", required=False, help="Path to the first output dir")
    parser.add_argument("--dice_path_2", required=False, help="Path to the second output dir")


    args = parser.parse_args()
    return args




def main():


    

    args = parse_args()
    diffs = []
    dice_1_mean = []
    dice_2_mean = []
    for dice_pth in [os.path.join(args.dice_path_1, el) for el in os.listdir(args.dice_path_1) if 'npy' in el]:
        dice_1 = np.load(dice_pth, allow_pickle=True)
        dice_2 = np.load(dice_pth.replace(args.dice_path_1, args.dice_path_2), allow_pickle=True)
        dice_1_mean.append(dice_1)
        dice_2_mean.append(dice_2)
        diff = np.mean(dice_1 - dice_2)
        diffs.append(diff)
    print(np.mean(np.abs(diffs)))
    print(np.mean(np.array(dice_1), axis=1))
    print(np.mean(np.array(dice_2), axis=1))




if __name__ == "__main__":
    main()
