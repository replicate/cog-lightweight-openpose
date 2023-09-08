import math
import numpy as np
import cv2


def normalize(img, img_mean, img_scale):
    img = np.array(img, dtype=np.float32)
    img = (img - img_mean) * img_scale
    return img


def pad_width(img, stride, pad_value, min_dims):
    height, width, _ = img.shape
    height = min(min_dims[0], height)

    min_dims[0] = math.ceil(min_dims[0] / float(stride)) * stride
    min_dims[1] = max(min_dims[1], width)
    min_dims[1] = math.ceil(min_dims[1] / float(stride)) * stride
    
    pad = []
    pad.append(int(math.floor((min_dims[0] - height) / 2.0)))
    pad.append(int(math.floor((min_dims[1] - width) / 2.0)))
    pad.append(int(min_dims[0] - height - pad[0]))
    pad.append(int(min_dims[1] - width - pad[1]))
    
    padded_img = cv2.copyMakeBorder(
        img, pad[0], pad[2], pad[1], pad[3], cv2.BORDER_CONSTANT, value=pad_value
    )
    return padded_img, pad


def preprocess(
    image, 
    image_size=256, 
    img_mean=[128, 128, 128],
    img_scale=1/256,
    pad_value=(0, 0, 0),
    stride=8,
):
    height, width, _ = image.shape
    scale = image_size / height
    img_mean = np.array(img_mean, np.float32)
    img_scale = np.float32(img_scale)

    image = cv2.resize(image, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    image = normalize(image, img_mean, img_scale)
    min_dims = [image_size, max(image.shape[1], image_size)]
    image, pad = pad_width(image, stride, pad_value, min_dims)

    return image, pad, scale


def get_alpha(rate=30, cutoff=1):
    tau = 1 / (2 * math.pi * cutoff)
    te = 1 / rate
    return 1 / (1 + tau / te)


class LowPassFilter:
    def __init__(self):
        self.x_previous = None

    def __call__(self, x, alpha=0.5):
        if self.x_previous is None:
            self.x_previous = x
            return x
        x_filtered = alpha * x + (1 - alpha) * self.x_previous
        self.x_previous = x_filtered
        return x_filtered
    

class OneEuroFilter:
    def __init__(self, freq=15, mincutoff=1, beta=0.05, dcutoff=1):
        self.freq = freq
        self.mincutoff = mincutoff
        self.beta = beta
        self.dcutoff = dcutoff
        self.filter_x = LowPassFilter()
        self.filter_dx = LowPassFilter()
        self.x_previous = None
        self.dx = None

    def __call__(self, x):
        if self.dx is None:
            self.dx = 0
        else:
            self.dx = (x - self.x_previous) * self.freq

        dx_smoothed = self.filter_dx(self.dx, get_alpha(self.freq, self.dcutoff))
        cutoff = self.mincutoff + self.beta * abs(dx_smoothed)
        x_filtered = self.filter_x(x, get_alpha(self.freq, cutoff))
        self.x_previous = x
        return x_filtered