import numpy as np
import cv2

def gamma_transform(img, gamma):
    is_gray = img.ndim == 2 or img.shape[1] == 1
    if is_gray:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    illum = hsv[..., 2] / 255.
    illum = np.power(illum, gamma)
    v = illum * 255.
    v[v > 255] = 255
    v[v < 0] = 0
    hsv[..., 2] = v.astype(np.uint8)
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    if is_gray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def calculate_average_brightness(img):
    # 三色通道的平均值
    B = img[..., 0].mean()
    G = img[..., 1].mean()
    R = img[..., 2].mean()

    # 显示亮度
    brightness = 0.299 * R + 0.587 * G + 0.114 * B
    return brightness, B, G, R