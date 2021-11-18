import glob
import math
import os
import random
import shutil
import subprocess
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
from tqdm import tqdm



img = cv2.imread('images/a_000261.jpg')
tl = round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
color = (0, 128, 255)
c1, c2 = ((0.498*416, 0.505*416), (0.995*416, 0.596*416))
cv2.rectangle(img, (1, 87), (415, 335), color, 2)
cv2.imwrite('1.jpg', img)
