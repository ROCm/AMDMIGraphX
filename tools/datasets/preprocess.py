import cv2
import numpy as np


def process_image(pil_image, shape=(224, 224), means=None):
    # OpenCV default is BGR, the image is in RGB
    image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    # But we need it to be in RGB, not sure how to import and not convert
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    w, h = shape
    image = resize_with_aspectratio(image, h, w, inter_pol=cv2.INTER_AREA)
    image = center_crop(image, h, w)
    image = np.asarray(image, dtype="float32")
    # Normalize image.
    if means:
        means = np.array(means, dtype=np.float32)
        image -= means
    # RGB is channel_last (HWC), transpose to channel_first (CHW)
    image = image.transpose([2, 0, 1])
    image = image[np.newaxis, :]
    return image


def center_crop(img, out_height, out_width):
    height, width, _ = img.shape
    left = int((width - out_width) / 2)
    right = int((width + out_width) / 2)
    top = int((height - out_height) / 2)
    bottom = int((height + out_height) / 2)
    img = img[top:bottom, left:right]
    return img


def resize_with_aspectratio(img,
                            out_height,
                            out_width,
                            scale=87.5,
                            inter_pol=cv2.INTER_LINEAR):
    height, width, _ = img.shape
    new_height = int(100.0 * out_height / scale)
    new_width = int(100.0 * out_width / scale)
    if height > width:
        w = new_width
        h = int(new_height * height / width)
    else:
        h = new_height
        w = int(new_width * width / height)
    img = cv2.resize(img, (w, h), interpolation=inter_pol)
    return img
