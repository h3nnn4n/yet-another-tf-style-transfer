import numpy as np
import struct
import errno
import cv2
import os

import images


def read_image(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    check_image(img, path)
    img = img.astype(np.float32)
    img = images.preprocess(img)
    return img


def write_image(path, img):
    img = images.postprocess(img)
    cv2.imwrite(path, img)


def check_image(img, path):
    if img is None:
        raise OSError(errno.ENOENT, "No such file", path)


def maybe_make_directory(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def read_flow_file(path):
    with open(path, 'rb') as f:
        struct.unpack('4s', f.read(4))[0]
        w = struct.unpack('i', f.read(4))[0]
        h = struct.unpack('i', f.read(4))[0]
        flow = np.ndarray((2, h, w), dtype=np.float32)
        for y in range(h):
            for x in range(w):
                flow[0, y, x] = struct.unpack('f', f.read(4))[0]
                flow[1, y, x] = struct.unpack('f', f.read(4))[0]
    return flow


def read_weights_file(path):
    lines = open(path).readlines()
    header = list(map(int, lines[0].split(' ')))
    w = header[0]
    h = header[1]
    vals = np.zeros((h, w), dtype=np.float32)
    for i in range(1, len(lines)):
        line = lines[i].rstrip().split(' ')
        vals[i-1] = np.array(list(map(np.float32, line)))
        vals[i-1] = list(map(lambda x: 0. if x < 255. else 1., vals[i-1]))
        weights = np.dstack([vals.astype(np.float32)] * 3)
    return weights


def write_image_output(output_img, content_img, style_imgs, init_img, output_name):
    out_dir = './'
    maybe_make_directory(out_dir)
    img_path = os.path.join(out_dir, output_name)
    content_path = os.path.join(out_dir, 'content.png')
    init_path = os.path.join(out_dir, 'init.png')

    write_image('out.png', output_img)
    write_image(img_path, output_img)
    write_image(content_path, content_img)
    write_image(init_path, init_img)

    index = 0
    for indes, style_img in enumerate(style_imgs):
        path = os.path.join(out_dir, 'style_' + str(index) + '.png')
        write_image(path, style_img)
        index += 1
