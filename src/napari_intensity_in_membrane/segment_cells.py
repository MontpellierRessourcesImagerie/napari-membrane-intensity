from cellpose import models, core
from cellpose.io import logger_setup
import tifffile
from pathlib import Path
from termcolor import cprint
import numpy as np
import os

class SegmentCellsWorker(object):
   
    def __init__(self):
        # Should we try to use the GPU?
        self.use_gpu = True
        # Axis for the input data
        self.axes = 'TYX'
        # Image to be segmented
        self.segmentation_channel = None
        # Median diameter of objects to segment
        self.objs_diameter = 30
        # Anisotropy between the Z and the YX axes for 3D.
        self.anisotropy = 1.0
        # Name of the model to use
        self.model_name = "cyto3"
        # Label maps resulting of the segmentation
        self.label_maps = []

    def get_gpu(self):
        return self.use_gpu

    def set_gpu(self, gpu):
        self.use_gpu = gpu and core.use_gpu()

    def get_axes(self):
        return self.axes

    def set_axes(self, axes):
        self.check_axes(axes)
        self.axes = axes

    def get_segmentation_channel(self):
        return self.segmentation_channel

    def set_segmentation_channel(self, data):
        if data.ndim != len(self.axes):
            raise ValueError("The candidate channel is not compatible with the current axes.")
        self.segmentation_channel = data

    def override_segmentation_channel(self, channel, axes):
        self.segmentation_channel = None
        self.set_axes(axes)
        self.set_segmentation_channel(channel)

    def get_objects_diameter(self):
        return self.objs_diameter

    def set_objects_diameter(self, median_diam):
        if median_diam <= 0:
            raise ValueError("The median diameter cannot be negative or zero")
        self.objs_diameter = median_diam

    def get_anisotropy(self):
        return self.anisotropy

    def set_anisotropy(self, ani):
        if ani <= 1e-3:
            raise ValueError("The anisotropy factor cannot be negative or zero.")
        self.anisotropy = ani

    def get_model_name(self):
        return self.model_name
    
    def set_model_name(self, name):
        self.model_name = name
        if name.startswith("CP_"):
            self.model_name = os.path.join(os.path.dirname(__file__), 'models', name)

    def check_axes(self, axes):
        ordered = ['T', 'Z', 'Y', 'X']
        valid = set(ordered)
        candidate = set([a for a in axes])
        if len(axes) != len(candidate):
            raise ValueError("The candidate axes contain duplicates.")
        if len(axes) != len(valid.intersection(candidate)):
            raise ValueError("The candidate axes contain unknown elements.")
        if (self.segmentation_channel is not None) and (len(candidate) != self.segmentation_channel.ndim):
            raise ValueError("The candidate axes are not compatible with the current data.")
        last_pos = -1
        for a in axes:
            curr_pos = ordered.index(a)
            if curr_pos < last_pos:
                raise ValueError("The candidate axes are not in the correct order.")
            last_pos = curr_pos

    def inference(self, model, imIn, callback=None):
        t_axis = self.axes.index('T') if 'T' in self.axes else 0
        n_frame = imIn.shape[t_axis] if 'T' in self.axes else 1
        all_masks = []
        for t in range(n_frame):
            t_point = imIn[t] if 'T' in self.axes else imIn
            masks, _, _ = model.eval(
                t_point,
                do_3D='Z' in self.axes,
                diameter=self.objs_diameter,
                anisotropy=self.anisotropy
            )
            all_masks.append(masks)
            cprint(f"Segmented frame {t + 1} / {n_frame}", 'green', attrs=['bold'])
            if callback is not None:
                callback(t + 1, n_frame)
        return np.array(all_masks)

    def run(self, callback=None):
        if self.axes is None or len(self.axes) < 2:
            raise ValueError("The axes have not been set or are invalid.")
        if self.segmentation_channel is None:
            raise ValueError("The segmentation channel has not been set.")
        
        logger_setup()
        model = models.CellposeModel(
            gpu=self.use_gpu, 
            model_type=self.model_name
        )

        self.label_maps = self.inference(model, self.segmentation_channel, callback=callback)

if __name__ == "__main__":
    folder = Path("/home/clement/Documents/projects/2219-intensity-membrane/20250718_WT_max projections")
    bf_name = "pos1 t4_TL.tif"
    img_path = folder / bf_name
    worker = SegmentCellsWorker()
    img = tifffile.imread(img_path)
    output_folder = folder / "results"

    worker.set_axes('TYX')
    worker.set_segmentation_channel(img)
    worker.set_objects_diameter(35)
    worker.set_anisotropy(1.0)
    worker.set_model_name("cyto3")
    worker.set_gpu(True)
    worker.run()
    for i, label_map in enumerate(worker.label_maps):
        out_path = output_folder / f"{bf_name[:-4]}_labels_t{i}.tif"
        tifffile.imwrite(out_path, label_map.astype('uint16'))