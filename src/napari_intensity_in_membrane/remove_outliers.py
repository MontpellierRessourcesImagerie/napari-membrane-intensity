from pathlib import Path
import tifffile
import numpy as np
from scipy.ndimage import (binary_erosion, gaussian_filter, 
                           binary_opening, binary_dilation)
from skimage.measure import regionprops
from skimage.morphology import diamond, disk
from napari_intensity_in_membrane.utils import get_integrated_intensity

class RemoveOutlierIntensities:
    def __init__(self):
        self.label_maps = None
        self.intensity_channel = None
        self.axes = 'TYX'
        self.factor = 2.0
        self.opening = 1
        self.dilation = 3
        self.discarded_mask = None

    def get_discarded_mask(self):
        return self.discarded_mask

    def get_opening_size(self):
        return self.opening
    
    def set_opening_size(self, size):
        if size <= 0:
            raise ValueError("The opening size cannot be negative")
        self.opening = size

    def get_dilation_iterations(self):
        return self.dilation
    
    def set_dilation_iterations(self, iterations):
        if iterations < 0:
            raise ValueError("The number of dilation iterations cannot be negative")
        self.dilation = iterations
    
    def get_factor(self):
        return self.factor
    
    def set_factor(self, factor):
        if factor <= 1e-3:
            raise ValueError("The factor cannot be negative or zero")
        self.factor = factor

    def get_label_maps(self):
        return self.label_maps
    
    def set_label_maps(self, data):
        if data.ndim != len(self.axes):
            raise ValueError("The candidate label maps are not compatible with the current axes.")
        self.label_maps = data

    def override_label_maps(self, labels, axes):
        self.label_maps = None
        self.set_axes(axes)
        self.set_label_maps(labels)

    def get_axes(self):
        return self.axes
    
    def set_axes(self, axes):
        self.check_axes(axes)
        self.axes = axes
    
    def get_intensity_channel(self):
        return self.intensity_channel
    
    def set_intensity_channel(self, data):
        if data.ndim != len(self.axes):
            raise ValueError("The candidate intensity channel is not compatible with the current axes.")
        self.intensity_channel = data

    def override_intensity_channel(self, channel, axes):
        self.intensity_channel = None
        self.set_axes(axes)
        self.set_intensity_channel(channel)
    
    def check_axes(self, axes):
        ordered = ['T', 'Z', 'Y', 'X']
        valid = set(ordered)
        candidate = set([a for a in axes])
        if len(axes) != len(candidate):
            raise ValueError("The candidate axes contain duplicates.")
        if len(axes) != len(valid.intersection(candidate)):
            raise ValueError("The candidate axes contain unknown elements.")
        if (self.intensity_channel is not None) and (len(candidate) != self.intensity_channel.ndim):
            raise ValueError("The candidate axes are not compatible with the current data.")
    
    def remove_outlier_intensities(self):
        if self.intensity_channel is None:
            raise ValueError("Intensity channel has not been set.")
        if self.label_maps is None:
            raise ValueError("Label maps have not been set.")
        mask = np.zeros_like(self.intensity_channel, dtype=bool)
        fp = diamond(1)
        for t in range(len(self.intensity_channel)):
            all_props = regionprops(self.label_maps[t], self.intensity_channel[t])
            for props in all_props:
                if props.label == 0:
                    continue
                mean_intensity = props.mean_intensity
                stddev = np.std(props.intensity_image[props.image])
                threshold = mean_intensity + self.factor * stddev
                too_bright = props.intensity_image > threshold
                too_bright = binary_opening(too_bright, structure=np.ones((2*self.opening+1, 2*self.opening+1)))
                too_bright = binary_dilation(too_bright, structure=fp, iterations=self.dilation)
                y1, x1, y2, x2 = map(int, props.bbox)
                mask[t][y1:y2, x1:x2] = too_bright
        return mask
    
    def run(self):
        if self.intensity_channel is None:
            raise ValueError("Intensity channel has not been set.")
        if self.label_maps is None:
            raise ValueError("Label maps have not been set.")
        self.discarded_mask = self.remove_outlier_intensities()