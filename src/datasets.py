from torchvision.io import decode_image
import matplotlib.pyplot as plt
import os
from PIL import Image
import numpy as np


class S2WDataset(torch.utils.data.Dataset):
    def __init__(self, root, target_shape, is_A, is_val=False):
        super().__init__()
        self.root = root
        self.target_shape = target_shape
        self.is_val = is_val
        if is_A:
            self.train = "trainA"
            self.test = "testA"
        else:
            self.train = "trainB"
            self.test = "testB"

        if not is_val:
            #self.df = pd.read_csv(os.path.join(root, "metadata.csv"))
            self.images = os.listdir(os.path.join(root, self.train))
        else:
            self.images = os.listdir(os.path.join(root, self.test))

        self.pics = len(self.images)

    def __len__(self):
        return self.pics

    def get(self, idx, pic_dir):
        img_path = os.path.join(self.root, pic_dir, self.images[idx])

        # Load the image
        img = Image.open(img_path).resize((self.target_shape, self.target_shape))

        return np.asarray(img, dtype=np.float32) / 128.0 - 1.0, {}

    def get_train(self, idx):
        return self.get(idx, self.train)

    def get_val(self, idx):
        return self.get(idx, self.test)

    def __getitem__(self, idx):
        if self.is_val:
            return self.get_val(idx)
        return self.get_train(idx)
