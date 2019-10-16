import glob
import torch
import torch.utils.data as data_utils
import numpy as np
from PIL import Image
import os
from matplotlib import pyplot as plt


def get_images():
    images = glob.glob("/home/mgazizova/*jpg")
    paths = list()
    labels = list()
    for image in images:
        file_path, file_name = os.path.split(image)
        paths.append(file_path)
        labels.append(file_name)
    return labels, paths


class ImageData(data_utils.Dataset):
    def __init__(self, width=256, height=256, transform=None):
        self.width = width
        self.height = height
        self.transform = transform
        labels, paths = get_images()     # y is a list of labels, x is a list of file paths
        self.y = labels
        self.x = paths

    def __getitem__(self, index):       #преобразование в тензор?
        img = Image.open(self.x[index]+"/"+ self.y[index])  # use pillow to open a file
        img = img.resize((self.width, self.height))  # resize the file to 256x256
        img = img.convert('RGB')  # convert image to RGB channel
        if self.transform is not None:
            img = self.transform(img)

        img = np.asarray(img).transpose(-1, 0, 1)   # we have to change the dimensions from width x height x channel (WHC) to channel x width x height (CWH)
        img = img/255
        img = torch.from_numpy(np.asarray(img))     # create the image tensor
        return img

    def __len__(self):
        return len(self.x)


data = ImageData()
dataloader = data_utils.DataLoader(data, batch_size=10, shuffle=True, num_workers=1)
imgs = next(iter(dataloader))
for img in imgs:
    a = img.numpy()
    plt.imshow(imgs.numpy()[1, 2])
    plt.show()