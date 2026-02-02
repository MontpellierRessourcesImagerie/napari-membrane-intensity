import numpy as np
import tifffile
import random

def keep_labels(data, points):
    labels_list = list(set([data[p[0], p[1], p[2]] for p in points.astype(np.uint16)]))
    output = np.zeros_like(data)
    for label in labels_list:
        output[data == label] = label
    return output

def get_integrated_intensity(intensity, label):
    mask = (label > 0)
    return np.sum(intensity[mask])

def merge_labels(labels, shapes):
    unique = np.unique(labels)
    mapping = {i: i for i in unique}
    for shape in shapes:
        bundle = set()
        for p in shape:
            z, y, x = map(int, p)
            label_at_point = labels[z, y, x]
            bundle.add(label_at_point)
        bundle.discard(0)
        min_lbl = min(bundle)
        for lbl in bundle:
            mapping[lbl] = min_lbl
    output = np.zeros_like(labels)
    for orig_lbl, new_lbl in mapping.items():
        output[labels == orig_lbl] = new_lbl
    return output
