import cv2
import argparse
import h5py
import os
import numpy as np

vidcap = cv2.VideoCapture(0)
has_frame,frame = vidcap.read()

import thermal_monitor as tm
print('Visualizing estimation result. Press Ctrl + C to stop.')
visualizer = tm.visualizer.Visualizer()
visualizer.run(has_frame, frame)

