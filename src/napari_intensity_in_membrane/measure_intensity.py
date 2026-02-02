from pathlib import Path
import tifffile
import numpy as np
from scipy.ndimage import (binary_erosion, gaussian_filter, 
                           binary_opening, binary_dilation)
from skimage.measure import regionprops
from skimage.morphology import diamond, disk
from napari_intensity_in_membrane.utils import get_integrated_intensity

class MeasureMembraneIntensity:
    def __init__(self):
        self.label_maps = None
        self.axes = 'TYX'
        self.intensity_channel = None
        self.membrane_thickness = 4
        self.rings = None
        self.inner = None
        self.results = None
        self.discarded_mask = None

    def get_discarded_mask(self):
        return self.discarded_mask
    
    def set_discarded_mask(self, mask):
        if mask.ndim != len(self.axes):
            raise ValueError("The candidate discarded mask is not compatible with the current axes.")
        self.discarded_mask = mask

    def override_discarded_mask(self, mask, axes):
        self.discarded_mask = None
        self.set_axes(axes)
        self.set_discarded_mask(mask)

    def get_results(self):
        return self.results

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

    def get_membrane_thickness(self):
        return self.membrane_thickness
    
    def set_membrane_thickness(self, thickness):
        if thickness <= 0:
            raise ValueError("The membrane thickness cannot be negative or zero")
        self.membrane_thickness = thickness

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
        
    def labels_to_outlines(self):
        if self.label_maps is None:
            raise ValueError("Label maps have not been set.")
        buffer_rings = np.zeros_like(self.label_maps)
        buffer_inner = np.zeros_like(self.label_maps)
        for t in range(len(self.label_maps)):
            print(f"Measuring frame: {t+1}")
            unique_vals = np.unique(self.label_maps[t])
            for value in unique_vals:
                if value == 0:
                    continue
                mask = (self.label_maps[t] == value).astype(np.uint8)
                eroded = binary_erosion(mask, iterations=self.membrane_thickness)
                rings = (mask - eroded) * value
                inner = eroded * value
                buffer_inner[t] = buffer_inner[t] | inner
                buffer_rings[t] = buffer_rings[t] | rings
        return buffer_rings, buffer_inner
    
    def measure_intensities(self):
        if self.rings is None:
            raise ValueError("Outlines have not been computed.")
        if self.inner is None:
            raise ValueError("Inner labels have not been computed.")
        if self.intensity_channel is None:
            raise ValueError("Intensity channel has not been set.")
        # For each label from each time point, measures the mean intensity, the area, and the sum of intensities.
        results = []
        for t in range(len(self.rings)):
            ring_labels = self.rings[t]
            inner_labels = self.inner[t]
            intensities = self.intensity_channel[t]
            measures_ring = regionprops(ring_labels, intensities)
            measures_inner = regionprops(inner_labels, intensities)
            results_t = {}
            for prop in measures_ring:
                lbl = prop.label
                if int(lbl) == 0:
                    continue
                mean_ring = prop.mean_intensity
                integrated_ring = get_integrated_intensity(prop.intensity_image, prop.image)
                area_ring = prop.area
                results_t[int(lbl)] = (mean_ring, integrated_ring, area_ring)
            for prop in measures_inner:
                lbl = prop.label
                if int(lbl) == 0:
                    continue
                if lbl not in results_t:
                    continue
                mean_inner = prop.mean_intensity
                integrated_inner = get_integrated_intensity(prop.intensity_image, prop.image)
                area_inner = prop.area
                mean_ring, integrated_ring, area_ring = results_t[int(lbl)]
                results_t[int(lbl)] = (mean_ring, integrated_ring, area_ring, mean_inner, integrated_inner, area_inner)
            filtered = {k: v for k, v in results_t.items() if len(v) == 6}
            results.append(filtered)
        self.results = results
    
    def run(self):
        if self.label_maps is None:
            raise ValueError("Label maps have not been set.")
        if self.intensity_channel is None:
            raise ValueError("Intensity channel has not been set.")
        if self.discarded_mask is None:
            raise ValueError("Outlier intensity mask has not been computed.")
        outlines, inner = self.labels_to_outlines()
        outlines = outlines * (1 - self.discarded_mask)
        inner = inner * (1 - self.discarded_mask)
        self.rings = outlines
        self.inner = inner
        self.measure_intensities()
