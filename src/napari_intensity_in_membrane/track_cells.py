from pathlib import Path
import tifffile
import numpy as np
import pandas as pd
from skimage.measure import regionprops
from skimage.segmentation import clear_border
import trackpy as tp

class TrackCellsWorker(object):
    def __init__(self):
        # List of label maps, one per frame
        self.label_maps = None
        # Axes present in each label map ('YX' or 'ZYX')
        self.axes = 'TYX'
        # DataFrame of detections
        self.detections = pd.DataFrame()
        # DataFrame of linked tracks
        self.linked = None
        # Searching radius (in pixels) for linking
        self.search_range = 50
        # Memory (in frames) for linking
        self.memory = 2
        # Use velocity for linking
        self.use_velocity = True
        # Labels consistent through time
        self.tracked_labels = None
        # Should we merge the neighboring cells?
        self.merge_neighbors = False
        # Should we remove cells that are not present in all frames?
        self.remove_incomplete = False
        # Dictionary used to keep track of pairs
        self.pairs = {}

    def get_merge_neighbors(self):
        return self.merge_neighbors
    
    def set_merge_neighbors(self, mn):
        self.merge_neighbors = bool(mn)

    def get_remove_incomplete(self):
        return self.remove_incomplete

    def set_remove_incomplete(self, ri):
        self.remove_incomplete = bool(ri)

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

    def get_linked_tracks(self):
        return self.linked
    
    def get_search_range(self):
        return self.search_range
    
    def set_search_range(self, sr):
        if sr <= 0:
            raise ValueError("The search range cannot be negative or zero")
        self.search_range = sr

    def get_memory(self):
        return self.memory
    
    def set_memory(self, mem):
        if mem < 0:
            raise ValueError("The memory cannot be negative")
        self.memory = mem

    def get_use_velocity(self):
        return self.use_velocity
    
    def set_use_velocity(self, uv):
        self.use_velocity = bool(uv)

    def get_tracked_labels(self):
        return self.tracked_labels

    def check_axes(self, axes):
        ordered = ['T', 'Z', 'Y', 'X']
        valid = set(ordered)
        candidate = set([a for a in axes])
        if len(axes) != len(candidate):
            raise ValueError("The candidate axes contain duplicates.")
        if len(axes) != len(valid.intersection(candidate)):
            raise ValueError("The candidate axes contain unknown elements.")
        if (self.label_maps is not None) and (len(candidate) != self.label_maps.ndim):
            raise ValueError("The candidate axes are not compatible with the current data.")
        last_pos = -1
        for a in axes:
            curr_pos = ordered.index(a)
            if curr_pos < last_pos:
                raise ValueError("The candidate axes are not in the correct order.")
            last_pos = curr_pos

    def labels_to_detections(self, label_list):
        rows = []
        for t, lab in enumerate(label_list):
            for rp in regionprops(clear_border(lab)):
                cy, cx = rp.centroid
                rows.append({
                    "frame"      : t,
                    "x"          : float(cx),
                    "y"          : float(cy),
                    "size"       : rp.area,
                    "orig_label" : int(rp.label),
                    "diameter"   : rp.axis_major_length
                })
        self.detections = pd.DataFrame(rows)

    def link_tracks(self):
        predictor = tp.predict.NearestVelocityPredict() if self.use_velocity else tp
        self.linked = predictor.link_df(
            self.detections,
            search_range=self.search_range,
            memory=self.memory
        )
        self.linked["particle"] = self.linked["particle"].astype(int) + 1

    def relabel_with_tracks(self):
        if self.linked is None:
            raise ValueError("No linked tracks to relabel.")
        if self.label_maps is None:
            raise ValueError("Label maps have not been set.")
        out = [np.zeros_like(self.label_maps[t], dtype=np.uint16) for t in range(len(self.label_maps))]
        for t, g in self.linked.groupby("frame"):
            lab = self.label_maps[t]
            out_t = out[t]
            for _, row in g.iterrows():
                track_id = int(row["particle"])
                orig_label = int(row["orig_label"])
                if orig_label == 0:
                    continue
                out_t[lab == orig_label] = track_id

        self.tracked_labels = np.array(out)

    def save_linked_tracks(self, path):
        if self.linked is None:
            raise ValueError("No linked tracks to save.")
        if path is None:
            raise ValueError("The output path is not valid.")
        self.linked.to_csv(path, index=False)

    def isolate_full_tracks(self): # labels present on the last frame
        if self.linked is None:
            raise ValueError("No linked tracks to process.")
        if self.axes[0] != 'T':
            return # nothing to do if no time axis
        complete_labels = set(self.linked[self.linked["frame"] == (self.label_maps.shape[0] - 1)]["particle"].unique())
        self.linked = self.linked[self.linked["particle"].isin(complete_labels)].reset_index(drop=True)
    
    def run(self):
        if self.label_maps is None:
            raise ValueError("Label maps have not been set.")
        self.labels_to_detections(self.label_maps)
        self.link_tracks()
        if self.remove_incomplete:
            self.isolate_full_tracks()
        
        self.relabel_with_tracks()

if __name__ == "__main__":
    folder = Path("/home/clement/Documents/projects/2219-intensity-membrane/20250718_WT_max projections/results")
    image_names = sorted([f for f in folder.iterdir() if f.suffix == ".tif" and "labels" in f.name])
    label_maps = np.array([tifffile.imread(f) for f in image_names])
    worker = TrackCellsWorker()

    worker.set_axes('TYX')
    worker.set_search_range(50)
    worker.set_memory(2)
    worker.set_use_velocity(True)
    worker.set_label_maps(label_maps)
    worker.set_merge_neighbors(True)
    worker.set_remove_incomplete(True)
    relabeled = worker.run()
    
    output_folder = folder / "tracked"
    tracked = worker.get_tracked_labels()
    out_path = output_folder / "tracked.tif"
    tifffile.imwrite(out_path, tracked.astype(np.uint16))
    out_csv = output_folder / "linked_tracks.csv"
    worker.save_linked_tracks(out_csv)