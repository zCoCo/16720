#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Included according to suggestion in @496 on Piazza.

@author: connorcolombo
"""

from helper import *
from util import *
from submission import *

import cv2
import matplotlib
from matplotlib import pyplot as plt
import numpy as np

im1 = cv2.cvtColor(cv2.imread('../data/im1.png'), cv2.COLOR_BGR2RGB)
im2 = cv2.cvtColor(cv2.imread('../data/im2.png'), cv2.COLOR_BGR2RGB)

corresp = np.load('../data/some_corresp.npz')
pts1 = corresp['pts1']
pts2 = corresp['pts2']

M1 = max(im1.shape)
M2 = max(im2.shape)
M = max(M1,M2)

F = eightpoint(pts1, pts2, M)

Ks = np.load('../data/intrinsics.npz')
K1 = Ks['K1']
K2 = Ks['K2']

E = essentialMatrix(F, K1, K2)

print((E,F,M))