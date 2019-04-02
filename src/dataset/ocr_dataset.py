from mxnet.gluon import data
import cv2
from skimage.transform import rescale, resize
from skimage import color,io
import os
class OCRDataset(data.Dataset):
    def __init__(self,root):
        self.root = root
        self.data = []
        self.label = []

        for root1,dir,files in os.walk(self.root):
            for file in files:
                image_path = root1+file
                #image = cv2.imread(image_path)
                image = io.imread(image_path)
                image = resize(image, (64, 128),
                       anti_aliasing=True)
                image = cv2.normalize(image,None,alpha=0,beta=1,norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                #image = mx.image.imread(image_path)
                label = file.split("=")[-1].replace(".jpg", "")
                self.data.append(image)

                self.label.append(label)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]
    def __len__(self):
        return len(self.data)