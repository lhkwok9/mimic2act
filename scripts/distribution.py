#!/usr/bin/env python
"""For checking the distribution of the initial position of Can
paste the result to excel and plot"""
import os
import numpy as np

data_folder = "/home/jk/dvrk2robosuite/src/dvrk2robosuite/data_collection/PickPlaceCan_Jul18_original/"

for ep in os.listdir(data_folder):
    np_folder = data_folder + ep + '/' + sorted(os.listdir(data_folder+ep))[2]
    data = np.load(np_folder)
    print(data['states'][0][31:33])