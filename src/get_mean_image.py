import cv2
import numpy as np
import os

def get_mean_pixel(img_dir):
    """
    iter the img_dir get the image,computer the mean pixel of the image, sum of mean pixel
    divide the count
    :param img_dir: the image dir
    :return: mean pixel
    """
    sum_r = 0
    sum_g = 0
    sum_b = 0
    count = 0
    for root, dir, img_name in os.walk(img_dir):
        for img in img_name:
            img_path = os.path.join(root, img)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # img=cv2.resize(img,(224,224))
            sum_r = sum_r + img[:, :, 0].mean()
            sum_g = sum_g + img[:, :, 1].mean()
            sum_b = sum_b + img[:, :, 2].mean()
            count = count + 1
    sum_r = sum_r / count
    sum_g = sum_g / count
    sum_b = sum_b / count
    img_mean = [sum_r, sum_g, sum_b]
    print(img_mean)
    return img_mean

#[180.40889436908472, 171.7787512809523, 173.99838366125906]
if __name__=="__main__":
    img_dir = "E:/project/OCR/data/train/"
    mean_pixel = get_mean_pixel(img_dir)