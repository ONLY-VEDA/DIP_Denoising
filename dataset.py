import os
import logging
import lmdb
import numpy as np
import torch
from torch.utils.data import Dataset

class SIDDDataset(Dataset):

    def __init__(self, lmdb_path, transform=None):
        if not os.path.exists(lmdb_path):
            logging.error("Training data lmdb not found at:{}".format(lmdb_path))
        env = lmdb.open(lmdb_path)
        self.ctx = env.begin()
        self.transform = transform

    def __len__(self):
        len_bytes = self.ctx.get(bytes("length",encoding='utf8'))
        return int(len_bytes.decode('utf8'))
        

    def __getitem__(self, idx):
        shape = np.frombuffer(self.ctx.get(bytes('shape'+str(idx),encoding='utf8')), dtype=np.int32)
        noise = np.frombuffer(self.ctx.get(bytes('noise'+str(idx),encoding='utf8')), dtype=np.uint8).reshape(shape)
        clean = np.frombuffer(self.ctx.get(bytes('clean'+str(idx),encoding='utf8')), dtype=np.uint8).reshape(shape)

        sample = {
                "noise": (noise.astype(np.float32) ) / 255,
                "clean": (clean.astype(np.float32) ) / 255
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

def sidd_test():
    lmdb_path = "./data/SIDD_Small/train"

    sidd_db = SIDDDataset(lmdb_path)

    for i in range(len(sidd_db)):
        sample = sidd_db[i]
        print(i, sample['noise'].shape, sample['clean'].shape)


if __name__ == "__main__":
    sidd_test()
