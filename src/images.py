import numpy as np
import utils
import cv2
import os


def preprocess(img):
    img = img[..., ::-1]
    img = img[np.newaxis, :, :, :]
    img -= np.array([123.68, 116.779, 103.939]).reshape((1, 1, 1, 3))
    return img


def postprocess(img):
    img += np.array([123.68, 116.779, 103.939]).reshape((1, 1, 1, 3))
    img = img[0]
    img = np.clip(img, 0, 255).astype('uint8')
    img = img[..., ::-1]
    return img


def normalize(weights):
    denom = sum(weights)
    if denom > 0.:
        return [float(i) / denom for i in weights]
    else:
        return [0.] * len(weights)


def get_content_image(content_img):
    path = os.path.join('./', content_img)
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    utils.check_image(img, path)
    img = img.astype(np.float32)
    h, w, d = img.shape
    mx = 512
    if h > w and h > mx:
        w = (float(mx) / float(h)) * w
        img = cv2.resize(img, dsize=(int(w), mx), interpolation=cv2.INTER_AREA)
    if w > mx:
        h = (float(mx) / float(w)) * h
        img = cv2.resize(img, dsize=(mx, int(h)), interpolation=cv2.INTER_AREA)
    img = preprocess(img)
    return img


def get_style_images(content_img, style_images):
    _, ch, cw, cd = content_img.shape
    style_imgs = []
    for style_fn in style_images:
        path = os.path.join('./', style_fn)
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        utils.check_image(img, path)
        img = img.astype(np.float32)
        img = cv2.resize(img, dsize=(cw, ch), interpolation=cv2.INTER_AREA)
        img = preprocess(img)
        style_imgs.append(img)
    return style_imgs


def get_noise_image(noise_ratio, content_img):
    np.random.seed(10)
    noise_img = np.random.uniform(
            -20., 20., content_img.shape).astype(np.float32)
    img = noise_ratio * noise_img + (1. - noise_ratio) * content_img
    return img


def convert_to_original_colors(content_img, stylized_img, color_convert_type='yuv'):
    content_img = postprocess(content_img)
    stylized_img = postprocess(stylized_img)
    if color_convert_type == 'yuv':
        cvt_type = cv2.COLOR_BGR2YUV
        inv_cvt_type = cv2.COLOR_YUV2BGR
    elif color_convert_type == 'ycrcb':
        cvt_type = cv2.COLOR_BGR2YCR_CB
        inv_cvt_type = cv2.COLOR_YCR_CB2BGR
    elif color_convert_type == 'luv':
        cvt_type = cv2.COLOR_BGR2LUV
        inv_cvt_type = cv2.COLOR_LUV2BGR
    elif color_convert_type == 'lab':
        cvt_type = cv2.COLOR_BGR2LAB
        inv_cvt_type = cv2.COLOR_LAB2BGR
    content_cvt = cv2.cvtColor(content_img, cvt_type)
    stylized_cvt = cv2.cvtColor(stylized_img, cvt_type)
    c1, _, _ = cv2.split(stylized_cvt)
    _, c2, c3 = cv2.split(content_cvt)
    merged = cv2.merge((c1, c2, c3))
    dst = cv2.cvtColor(merged, inv_cvt_type).astype(np.float32)
    dst = preprocess(dst)
    return dst
