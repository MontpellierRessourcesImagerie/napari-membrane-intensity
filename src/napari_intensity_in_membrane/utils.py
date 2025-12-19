import numpy as np
import tifffile
import random

def keep_labels(data, points):
    labels_list = list(set([data[*p] for p in points.astype(np.uint16)]))
    output = np.zeros_like(data)
    for label in labels_list:
        output[data == label] = label
    return output

def get_integrated_intensity(intensity, label):
    rdm_nbr = random.randint(0, 1000)
    tifffile.imwrite(f"/tmp/exported/intensity-{rdm_nbr}.tif", intensity)
    tifffile.imwrite(f"/tmp/exported/label-{rdm_nbr}.tif", label)
    mask = (label > 0)
    return np.sum(intensity[mask])