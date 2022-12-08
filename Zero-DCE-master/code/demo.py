import argparse
import re
from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image
from model import DCENet
import torchvision
import gc
import time

import utils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--testDir', type=str, required=True, help='path to test images')
    parser.add_argument('--ckpt', type=str, required=True, help='path to *_best_model.pth')
    parser.add_argument('--outDir', type=str, required=True, help='path to save output')
    parser.add_argument('--imFormat', type=str, default='png',
                        help='file extension, which will be considered, default to image file named *.jpg')

    args = parser.parse_args()
    return args


def make_grid(nrow, ncol, h, w, hspace, wspace):
    grid = np.ones(
        (nrow * h + hspace * (nrow - 1), ncol * w + (ncol - 1) * wspace, 3),
        dtype=np.float32
    )
    return grid


def read_image(fp, h, w):
    fp = str(fp)
    img = cv2.imread(fp)
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.
    img = cv2.resize(img, (w, h))
    return img


def putText(im, text, pos, color, size=1, scale=2, font=cv2.FONT_HERSHEY_SIMPLEX):
    im = cv2.putText(im, text, pos, font, size, color, scale)
    return im


def row_arrange(wspace, images):
    n = len(images)
    h, w, c = images[0].shape
    row = np.ones((h, (n - 1) * wspace + n * w, c))
    curr_w = 0
    for image in images:
        row[:, curr_w:curr_w + w, :] = image
        curr_w += w + wspace
    return row

time_ini = time.time()

args = parse_args()

device = torch.device('cuda:0')

model = DCENet(return_results=[4, 8])
model.load_state_dict(torch.load(args.ckpt, map_location='cuda:0')['model'])
model.to(device)


inPath = Path(args.testDir)
outPath = Path(args.outDir)
outPath.mkdir(parents=True, exist_ok=True)

r = re.compile(args.imFormat, re.IGNORECASE)  # assume images are in JPG


num_images = 0
for file in inPath.glob('*'):
    if r.search(str(file)):
        img = Image.open(file)

        img = torch.from_numpy(np.array(img))
        img = img.float().div(255)
        img = img.permute((2, 0, 1)).contiguous()
        img = img.unsqueeze(0)
        img = img.to(device)

        results, Astack = model(img)
        enhanced_image = results[1]
        torchvision.utils.save_image(enhanced_image, outPath.joinpath(file.name))
        torch.cuda.empty_cache()
        del enhanced_image
        del img
        del results
        gc.collect()
        num_images += 1
        print(num_images)

time_fim = time.time()
time_final = time_fim - time_ini
    
end = "..\\demo-output2\\"
pasta = "tempo_normal.txt"

dst = end+pasta

arquivo = open(dst, "a")
arquivo.write("Tempo de processamento: " + str(time_final))
arquivo.close()