import argparse
import glob
from PIL import Image
import PIL
import random
import lmdb
from utils import *

# the pixel value range is '0-255'(uint8 ) of training data

# macro
DATA_AUG_TIMES = 1  # transform a sample to a different sample for DATA_AUG_TIMES times

parser = argparse.ArgumentParser(description='')
parser.add_argument('--src_dir', dest='src_dir', default='./data/SIDD_Small', help='dir of data')
parser.add_argument('--save_dir', dest='save_dir', default='./data/SIDD_Small/train', help='dir of patches')
parser.add_argument('--patch_size', dest='pat_size', type=int, default=256, help='patch size')
parser.add_argument('--stride', dest='stride', type=int, default=256, help='stride')
parser.add_argument('--step', dest='step', type=int, default=0, help='step')
parser.add_argument('--sigma', dest='sigma', type=float, default=25, help='noise level')
parser.add_argument('--batch_size', dest='bat_size', type=int, default=32, help='batch size')
# check output arguments
parser.add_argument('--from_file', dest='from_file', default="./data/img_clean_pats.npy", help='get pic from file')
parser.add_argument('--num_pic', dest='num_pic', type=int, default=10, help='number of pic to pick')
parser.add_argument('--list_path', dest='list_path', default='./data/SIDD_Small/train.txt', help='train data list')
args = parser.parse_args()

def generate_patches_list(isDebug=False):
    global DATA_AUG_TIMES
    count = 0
    filepaths = open(args.list_path).readlines()
    if isDebug:
        filepaths = filepaths[:10]
    print("number of training data %d" % len(filepaths))
    
    scales = [1]
    
    # open 10GB file map
    env = lmdb.open(args.save_dir,map_size=int(32*1e9))
    #ctx = env.begin(write=True)

    count = 0
    # generate patches
    for i in range(len(filepaths)):
        ctx = env.begin(write=True)
        print("processing %d"%i)
        noise_path,gt_path = filepaths[i].split('#')
        noise_img = Image.open(noise_path)  # convert RGB to gray
        gt_img = Image.open(gt_path.strip('\n'))
        for s in range(len(scales)):
            newsize = (int(noise_img.size[0] * scales[s]), int(noise_img.size[1] * scales[s]))
            noise_img_s = noise_img.resize(newsize, resample=PIL.Image.BICUBIC)  # do not change the original img
            gt_img_s = gt_img.resize(newsize, resample=PIL.Image.BICUBIC)
            noise_img_s = np.array(np.transpose(noise_img_s,(2,0,1)), dtype='uint8')
            gt_img_s = np.array(np.transpose(gt_img_s,(2,0,1)), dtype='uint8')
            
            #print(noise_img_s.shape, gt_img_s.shape)
            for j in range(DATA_AUG_TIMES):
                _, im_h, im_w = noise_img_s.shape
                for x in range(0 + args.step, im_h - args.pat_size, args.stride):
                    for y in range(0 + args.step, im_w - args.pat_size, args.stride):
                        aug_choice = random.randint(0,7)
                        noise = data_augmentation(noise_img_s[:,x:x + args.pat_size, y:y + args.pat_size],aug_choice)
                        clean = data_augmentation(gt_img_s[:,x:x + args.pat_size, y:y + args.pat_size],aug_choice)
                        shape = np.array(noise.shape,dtype=np.int32)
                        #print(noise.shape,shape)
                        shape_suc = ctx.put(bytes("shape"+str(count),encoding='utf8'),shape.tobytes())
                        noise_suc = ctx.put(bytes("noise"+str(count),encoding='utf8'),noise.tobytes())
                        clean_suc = ctx.put(bytes("clean"+str(count),encoding='utf8'),clean.tobytes())
                        assert shape_suc == noise_suc == clean_suc == True
                        print("finishied patch :%d"%count)
                        count += 1
        ctx.commit()
    ctx = env.begin(write=True)
    len_suc = ctx.put(bytes('length',encoding='utf8'),bytes(str(count),encoding='utf8'))
    assert len_suc == True
    ctx.commit()

def generate_patches_from_file(isDebug=False):
    global DATA_AUG_TIMES
    count = 0
    filepaths = glob.glob(args.src_dir + '/*.png')
    print("number of training data %d" % len(filepaths))
    
    scales = [1, 0.9, 0.8, 0.7]
    
    # open 16GB file map
    env = lmdb.open(args.src_dir + '/train',map_size=int(1.6*1e10))

    count = 0
    # generate patches
    for i in range(len(filepaths)):
        ctx = env.begin(write=True)
        print("processing %d"%i)
        gt_img = Image.open(filepaths[i])  # convert RGB to gray
        for s in range(len(scales)):
            newsize = (int(gt_img.size[0] * scales[s]), int(gt_img.size[1] * scales[s]))
            gt_img_s = gt_img.resize(newsize, resample=PIL.Image.BICUBIC)
            gt_img_s =  np.reshape(np.array(gt_img_s, dtype="uint8"),
                               (1, gt_img_s.size[0], gt_img_s.size[1]))  # extend one dimension
            noise = np.random.normal(0, args.sigma / 255.0, gt_img_s.shape)
            noise_img_s = np.clip(255 * (gt_img_s.astype(np.float32)/255.0 + noise), 0, 255).astype('uint8')

            for j in range(DATA_AUG_TIMES):
                _, im_h, im_w = noise_img_s.shape
                for x in range(0 + args.step, im_h - args.pat_size, args.stride):
                    for y in range(0 + args.step, im_w - args.pat_size, args.stride):
                        aug_choice = random.randint(0,7)
                        noise = data_augmentation(noise_img_s[:,x:x + args.pat_size, y:y + args.pat_size],aug_choice)
                        clean = data_augmentation(gt_img_s[:,x:x + args.pat_size, y:y + args.pat_size],aug_choice)
                        shape = np.array(noise.shape,dtype=np.int32)
                        #print(noise.shape,shape)
                        shape_suc = ctx.put(bytes("shape"+str(count),encoding='utf8'),shape.tobytes())
                        noise_suc = ctx.put(bytes("noise"+str(count),encoding='utf8'),noise.tobytes())
                        clean_suc = ctx.put(bytes("clean"+str(count),encoding='utf8'),clean.tobytes())
                        assert shape_suc == noise_suc == clean_suc == True
                        print("finishied patch :%d"%count)
                        count += 1
        ctx.commit()
    ctx = env.begin(write=True)
    len_suc = ctx.put(bytes('length',encoding='utf8'),bytes(str(count),encoding='utf8'))
    assert len_suc == True
    ctx.commit()


def generate_test_from_file():
    count = 0
    filepaths = glob.glob(args.src_dir + '/*.png')
    print("number of testing data %d" % len(filepaths))
    
    # open 16GB file map
    env = lmdb.open(args.src_dir + '/test',map_size=int(1.6*1e10))

    count = 0
    # generate patches
    for i in range(len(filepaths)):
        ctx = env.begin(write=True)
        print("processing %d"%i)
        gt_img = Image.open(filepaths[i])  # convert RGB to gray

        gt_img = gt_img.resize((256, 256), resample=PIL.Image.BICUBIC)
        gt_img =  np.reshape(np.array(gt_img, dtype="uint8"),
                            (1, gt_img.size[0], gt_img.size[1]))  # extend one dimension
        noise = np.random.normal(0, args.sigma / 255.0, gt_img.shape)
        noise_img = np.clip(255 * (gt_img.astype(np.float32)/255.0 + noise), 0, 255).astype('uint8')

        shape = np.array(noise_img.shape,dtype=np.int32)
        #print(noise.shape,shape)
        shape_suc = ctx.put(bytes("shape"+str(count),encoding='utf8'),shape.tobytes())
        noise_suc = ctx.put(bytes("noise"+str(count),encoding='utf8'),noise_img.tobytes())
        clean_suc = ctx.put(bytes("clean"+str(count),encoding='utf8'),gt_img.tobytes())
        assert shape_suc == noise_suc == clean_suc == True
        print("finishied patch :%d"%count)
        count += 1
        ctx.commit()

    ctx = env.begin(write=True)
    len_suc = ctx.put(bytes('length',encoding='utf8'),bytes(str(count),encoding='utf8'))
    assert len_suc == True
    ctx.commit()


if __name__ == '__main__':
    # generate_patches_list()
    # generate_patches_from_file()
    generate_test_from_file()
