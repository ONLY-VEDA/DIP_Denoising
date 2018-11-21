import numpy as np
from PIL import Image
import argparse

from utils import cal_psnr

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ori_img",dest='ori')
    parser.add_argument("--dst_img",dest='dst')
    args = parser.parse_args()

    ori = np.array(Image.open(args.ori))
    dst = np.array(Image.open(args.dst))
    psnr = cal_psnr(ori,dst)
    print(psnr)


