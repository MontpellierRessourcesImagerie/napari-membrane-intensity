import os
from qtpy.QtWidgets import (QWidget, QVBoxLayout, QSpinBox,
                            QGroupBox, QHBoxLayout, QLabel, 
                            QComboBox, QCheckBox, QLineEdit, 
                            QPushButton, QFileDialog, QDoubleSpinBox
)
from qtpy.QtCore import Qt, QThread

import napari
from napari.utils import progress

import tifffile
import numpy as np

from napari.layers import Points, Shapes, Labels, Image
from napari_intensity_in_membrane.segment_cells import SegmentCellsWorker
from napari_intensity_in_membrane.track_cells import TrackCellsWorker
from napari_intensity_in_membrane.measure_intensity import MeasureMembraneIntensity
from napari_intensity_in_membrane.remove_outliers import RemoveOutlierIntensities
from napari_intensity_in_membrane.qt_workers import (QtSegmentCells, QtTrackCells, 
                                                     QtMeasureMembranes, QtRemoveOutlierIntensities)
from napari_intensity_in_membrane.results_table import FrameWiseResultsTable
from napari_intensity_in_membrane.utils import keep_labels, merge_labels

NEUTRAL             = "--------"
SEGMENTATION_SUFFIX = "-labeled"
INNER_SUFFIX        = "-inner"
MEMBRANE_SUFFIX     = "-membrane"
OUTLIERS_SUFFIX     = "-outliers"

class IntensitiesInMembraneWidget(QWidget):
    def __init__(self, napari_viewer: "napari.Viewer"):
        super().__init__()
        self.viewer = napari_viewer
        self.channel_pools = set([]) # List of combo boxes containing layer names, to be refreshed when layers change
        
        self.scw = SegmentCellsWorker()
        self.tcw = TrackCellsWorker()
        self.miw = MeasureMembraneIntensity()
        self.roi = RemoveOutlierIntensities()

        self.init_ui()
        self.viewer.layers.events.inserted.connect(lambda e: self.refresh_layer_names())
        self.viewer.layers.events.removed.connect(lambda e: self.refresh_layer_names())
        self.viewer.layers.events.reordered.connect(lambda e: self.refresh_layer_names())
        self.qt_worker = None
        self.qt_thread = None
        self.rt = None

    def init_segmentation_ui(self, layout):
        seg_group = QGroupBox("Segment Cells")
        seg_layout = QVBoxLayout()
        seg_group.setLayout(seg_layout)
        layout.addWidget(seg_group)

        h_layout = QHBoxLayout()
        h_layout.addWidget(QLabel("Segmentation channel"))
        self.segmentation_channel_combo = QComboBox()
        self.channel_pools.add((self.segmentation_channel_combo, Image))
        h_layout.addWidget(self.segmentation_channel_combo)
        seg_layout.addLayout(h_layout)

        h_layout = QHBoxLayout()
        h_layout.addWidget(QLabel("Cells diameter (px)"))
        self.cell_diameter_spinbox = QSpinBox()
        self.cell_diameter_spinbox.setValue(45)
        h_layout.addWidget(self.cell_diameter_spinbox)
        seg_layout.addLayout(h_layout)

        h_layout = QHBoxLayout()
        h_layout.addWidget(QLabel("Model"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(self.probe_models() + ["cyto3", "nuclei"])
        h_layout.addWidget(self.model_combo)
        seg_layout.addLayout(h_layout)

        seg_layout.addSpacing(10)

        self.btn_run = QPushButton("Run segmentation")
        self.btn_run.clicked.connect(self.launch_segmentation)
        seg_layout.addWidget(self.btn_run)

    def init_tracking_ui(self, layout):
        track_group = QGroupBox("Track Cells")
        track_layout = QVBoxLayout()
        track_group.setLayout(track_layout)
        layout.addWidget(track_group)

        self.track_btn = QPushButton("Track cells")
        self.track_btn.clicked.connect(self.launch_tracking)
        track_layout.addWidget(self.track_btn)

        h_layout = QHBoxLayout()

        self.keep_points_layer_combo = QComboBox()
        self.channel_pools.add((self.keep_points_layer_combo, Points))
        h_layout.addWidget(self.keep_points_layer_combo)
        self.keep_labels_btn = QPushButton("Keep labels")
        self.keep_labels_btn.clicked.connect(self.launch_keep_labels)
        h_layout.addWidget(self.keep_labels_btn)
        track_layout.addLayout(h_layout)

        h_layout = QHBoxLayout()

        self.merge_cells_layer_combo = QComboBox()
        self.channel_pools.add((self.merge_cells_layer_combo, Shapes))
        h_layout.addWidget(self.merge_cells_layer_combo)
        self.merge_labels_btn = QPushButton("Merge cells")
        self.merge_labels_btn.clicked.connect(self.launch_merge_labels)
        h_layout.addWidget(self.merge_labels_btn)
        track_layout.addLayout(h_layout)

    def init_measurement_ui(self, layout):
        measure_group = QGroupBox("Measure in membranes")
        measure_layout = QVBoxLayout()
        measure_group.setLayout(measure_layout)
        layout.addWidget(measure_group)

        h_layout = QHBoxLayout()
        h_layout.addWidget(QLabel("Intensity channel"))
        self.cb_intensity_channel = QComboBox()
        self.channel_pools.add((self.cb_intensity_channel, Image))
        h_layout.addWidget(self.cb_intensity_channel)
        measure_layout.addLayout(h_layout)

        h_layout = QHBoxLayout()
        h_layout.addWidget(QLabel("Membrane thickness (px)"))
        self.le_thickness = QSpinBox()
        self.le_thickness.setValue(5)
        h_layout.addWidget(self.le_thickness)
        measure_layout.addLayout(h_layout)

        measure_layout.addSpacing(10)

        h_layout = QHBoxLayout()
        h_layout.addWidget(QLabel("Opening size (px)"))
        self.le_opening_size = QSpinBox()
        self.le_opening_size.setValue(1)
        h_layout.addWidget(self.le_opening_size)
        measure_layout.addLayout(h_layout)

        h_layout = QHBoxLayout()
        h_layout.addWidget(QLabel("Dilation iterations"))
        self.le_dilation_size = QSpinBox()
        self.le_dilation_size.setValue(3)
        h_layout.addWidget(self.le_dilation_size)
        measure_layout.addLayout(h_layout)

        h_layout = QHBoxLayout()
        h_layout.addWidget(QLabel("Intensity factor"))
        self.le_factor = QDoubleSpinBox()
        self.le_factor.setValue(2.0)
        h_layout.addWidget(self.le_factor)
        measure_layout.addLayout(h_layout)

        measure_layout.addSpacing(10)

        h_layout = QHBoxLayout()

        self.remove_outlier_btn = QPushButton("Remove outliers")
        self.remove_outlier_btn.clicked.connect(self.launch_remove_outlier_intensities)
        h_layout.addWidget(self.remove_outlier_btn)

        self.measure_btn = QPushButton("Measure intensities")
        self.measure_btn.clicked.connect(self.launch_measurement)
        h_layout.addWidget(self.measure_btn)
        measure_layout.addLayout(h_layout)

    def add_end_section(self, layout):
        h_layout = QHBoxLayout()

        self.save_layers_button = QPushButton("Save layers")
        h_layout.addWidget(self.save_layers_button)
        self.save_layers_button.clicked.connect(self.save_layers)

        self.clear_state_button = QPushButton("Clear all")
        h_layout.addWidget(self.clear_state_button)
        self.clear_state_button.clicked.connect(self.clear_state)

        layout.addLayout(h_layout)

    def init_ui(self):
        layout = QVBoxLayout()
        self.setLayout(layout)
        self.init_segmentation_ui(layout)
        self.init_tracking_ui(layout)
        self.init_measurement_ui(layout)
        self.add_end_section(layout)
        self.refresh_layer_names()

    def probe_models(self):
        folder = os.path.join(os.path.dirname(__file__), 'models')
        models = [m for m in os.listdir(folder) if os.path.isfile(os.path.join(folder, m))]
        return models
    
    def _set_combo_safely(self, combo: QComboBox, text: str):
        idx = combo.findText(text)
        if idx >= 0:
            combo.setCurrentIndex(idx)
        else:
            # fall back to neutral
            combo.setCurrentIndex(0)

    def _get_layer_names(self):
        try:
            return [ly.name for ly in self.viewer.layers]
        except Exception:
            return []

    def _populate_layer_combo(self, combo: QComboBox, neutral=NEUTRAL, layer_type=None):
        current = combo.currentText()
        combo.blockSignals(True)
        combo.clear()
        combo.addItem(neutral)
        for name in self._get_layer_names():
            if layer_type is None or isinstance(self.viewer.layers[name], layer_type):
                combo.addItem(name)
        # restore selection if still available
        self._set_combo_safely(combo, current)
        combo.blockSignals(False)

    def refresh_layer_names(self):
        """Call this to refresh all comboboxes with current viewer layers."""
        for combo, layer_type in self.channel_pools:
            self._populate_layer_combo(combo, neutral=NEUTRAL, layer_type=layer_type)

    def get_segmentation_image(self):
        name = self.segmentation_channel_combo.currentText()
        if name == NEUTRAL:
            return None
        if name not in self.viewer.layers:
            return None
        layer = self.viewer.layers[name]
        return layer.data
    
    def get_intensity_image(self):
        name = self.cb_intensity_channel.currentText()
        if name == NEUTRAL:
            return None
        if name not in self.viewer.layers:
            return None
        layer = self.viewer.layers[name]
        return layer.data
    
    def get_labeled_cells(self):
        name = self.segmentation_channel_combo.currentText() + SEGMENTATION_SUFFIX
        if name not in self.viewer.layers:
            return None
        layer = self.viewer.layers[name]
        return layer.data
    
    def get_outliers_mask(self):
        name = self.segmentation_channel_combo.currentText() + OUTLIERS_SUFFIX
        if name not in self.viewer.layers:
            return None
        layer = self.viewer.layers[name]
        return layer.data
    
    def set_active_ui(self, status):
        self.segmentation_channel_combo.setEnabled(status)
        self.cb_intensity_channel.setEnabled(status)
        self.cell_diameter_spinbox.setEnabled(status)
        self.le_thickness.setEnabled(status)
        self.le_factor.setEnabled(status)
        self.btn_run.setEnabled(status)
        self.model_combo.setEnabled(status)
        self.keep_points_layer_combo.setEnabled(status)
        self.keep_labels_btn.setEnabled(status)
        self.track_btn.setEnabled(status)
        self.measure_btn.setEnabled(status)
        self.clear_state_button.setEnabled(status)
        self.merge_cells_layer_combo.setEnabled(status)
        self.merge_labels_btn.setEnabled(status)
        self.remove_outlier_btn.setEnabled(status)
        self.le_opening_size.setEnabled(status)
        self.le_dilation_size.setEnabled(status)
        self.save_layers_button.setEnabled(status)

    def launch_segmentation(self):
        data = self.get_segmentation_image()
        if data is None:
            print("No segmentation channel selected")
            return
        self.scw.set_axes('TYX')
        self.scw.set_segmentation_channel(data)
        self.scw.set_objects_diameter(self.cell_diameter_spinbox.value())
        self.scw.set_gpu(True)
        self.scw.set_model_name(self.model_combo.currentText())

        self.qt_worker = QtSegmentCells(self.scw)
        self.qt_thread = None
        self.qt_thread = QThread()
        self.qt_worker.moveToThread(self.qt_thread)
        self.qt_thread.started.connect(self.qt_worker.run)
        self.qt_worker.finished.connect(self.qt_thread.quit)
        self.qt_worker.finished.connect(self.qt_worker.deleteLater)
        self.qt_thread.finished.connect(self.finish_segment_cells)
        self.qt_thread.start()
        self.set_active_ui(False)

    def finish_segment_cells(self):
        if self.scw is None:
            print("No segmentation worker available")
            return
        label_maps = self.scw.label_maps
        if label_maps is None or len(label_maps) == 0:
            print("No label maps generated")
            return
        layer_name = self.segmentation_channel_combo.currentText() + "-labeled"
        self.viewer.add_labels(label_maps, name=layer_name)
        print("Segmentation finished and added to viewer.")
        self.set_active_ui(True)

    def launch_tracking(self):
        labels_name = self.segmentation_channel_combo.currentText() + SEGMENTATION_SUFFIX
        labels = self.viewer.layers[labels_name].data
        save_layer_name = "backup-" + labels_name
        if save_layer_name not in self.viewer.layers:
            self.viewer.add_labels(labels.copy(), name=save_layer_name, visible=False)
        self.viewer.layers[labels_name].data = self.viewer.layers[save_layer_name].data.copy()

        data = self.get_labeled_cells()
        if data is None:
            print("No segmentation channel selected")
            return
        self.tcw.set_axes('TYX')
        self.tcw.set_search_range(self.cell_diameter_spinbox.value())
        self.tcw.set_remove_incomplete(True)
        self.tcw.set_label_maps(data)

        self.qt_worker = QtTrackCells(self.tcw)
        self.qt_thread = None
        self.qt_thread = QThread()
        self.qt_worker.moveToThread(self.qt_thread)
        self.qt_thread.started.connect(self.qt_worker.run)
        self.qt_worker.finished.connect(self.qt_thread.quit)
        self.qt_worker.finished.connect(self.qt_worker.deleteLater)
        self.qt_thread.finished.connect(self.finish_tracking_cells)
        self.qt_thread.start()
        self.set_active_ui(False)

    def launch_keep_labels(self):
        points_name = self.keep_points_layer_combo.currentText()
        labels_name = self.segmentation_channel_combo.currentText() + SEGMENTATION_SUFFIX
        if points_name not in self.viewer.layers:
            print("No points layer selected")
            return
        if labels_name not in self.viewer.layers:
            print("No labeled segmentation layer available")
            return
        points = self.viewer.layers[points_name].data
        labels = self.viewer.layers[labels_name].data
        self.viewer.layers[labels_name].data = keep_labels(labels, points)

    def launch_merge_labels(self):
        shapes_name = self.merge_cells_layer_combo.currentText()
        labels_name = self.segmentation_channel_combo.currentText() + SEGMENTATION_SUFFIX
        if shapes_name not in self.viewer.layers:
            print("No shapes layer selected")
            return
        if labels_name not in self.viewer.layers:
            print("No labeled segmentation layer available")
            return
        shapes_layer = self.viewer.layers[shapes_name]
        labels = self.viewer.layers[labels_name].data
        if not all(s_type in ['line', 'path'] for s_type in shapes_layer.shape_type):
            print("All shapes must be of type 'line' or 'path' to merge labels")
            return
        shapes = shapes_layer.data
        if len(shapes) == 0: # there must be objects to merge
            print("No shapes to merge labels")
            return
        self.viewer.layers[labels_name].data = merge_labels(labels, shapes)

    def finish_tracking_cells(self):
        if self.tcw is None:
            print("No tracking worker available")
            return
        linked = self.tcw.get_linked_tracks()
        if linked is None:
            print("No linked tracks generated")
            return
        name = self.segmentation_channel_combo.currentText() + SEGMENTATION_SUFFIX
        self.viewer.layers[name].data = self.tcw.get_tracked_labels()
        print(f"Tracking finished. Found {linked['particle'].nunique()} tracks.")
        self.set_active_ui(True)

    def launch_remove_outlier_intensities(self):
        labeled = self.get_labeled_cells()
        if labeled is None:
            print("No segmentation channel selected")
            return
        intensities = self.get_intensity_image()
        if intensities is None:
            print("No intensity channel selected")
            return
        self.roi.set_axes('TYX')
        self.roi.set_label_maps(labeled)
        self.roi.set_intensity_channel(intensities)
        self.roi.set_opening_size(self.le_opening_size.value())
        self.roi.set_dilation_iterations(self.le_dilation_size.value())

        self.qt_worker = QtRemoveOutlierIntensities(self.roi)
        self.qt_thread = None
        self.qt_thread = QThread()
        self.qt_worker.moveToThread(self.qt_thread)
        self.qt_thread.started.connect(self.qt_worker.run)
        self.qt_worker.finished.connect(self.qt_thread.quit)
        self.qt_worker.finished.connect(self.qt_worker.deleteLater)
        self.qt_thread.finished.connect(self.finish_remove_outliers)
        self.qt_thread.start()
        self.set_active_ui(False)

    def finish_remove_outliers(self):
        if self.roi is None:
            print("No outlier removal worker available")
            return
        discard_mask = self.roi.get_discarded_mask()
        if discard_mask is None:
            print("No outlier mask generated")
            return
        seg_name = self.segmentation_channel_combo.currentText()
        seg_layer = self.viewer.layers[seg_name]
        
        name = seg_name + OUTLIERS_SUFFIX
        if name in self.viewer.layers:
            self.viewer.layers[name].data = discard_mask
        else:
            self.viewer.add_labels(discard_mask, name=name, scale=seg_layer.scale)

        print("Outlier removal finished.")
        self.set_active_ui(True)

    def launch_measurement(self):
        labeled = self.get_labeled_cells()
        if labeled is None:
            print("No segmentation channel selected")
            return
        intensities = self.get_intensity_image()
        if intensities is None:
            print("No intensity channel selected")
            return
        outliers = self.get_outliers_mask()
        if outliers is None:
            print("No outlier mask available")
            return
        self.miw.set_axes('TYX')
        self.miw.set_label_maps(labeled)
        self.miw.set_membrane_thickness(self.le_thickness.value())
        self.miw.set_intensity_channel(intensities)
        self.miw.set_discarded_mask(outliers)

        self.qt_worker = QtMeasureMembranes(self.miw)
        self.qt_thread = None
        self.qt_thread = QThread()
        self.qt_worker.moveToThread(self.qt_thread)
        self.qt_thread.started.connect(self.qt_worker.run)
        self.qt_worker.finished.connect(self.qt_thread.quit)
        self.qt_worker.finished.connect(self.qt_worker.deleteLater)
        self.qt_thread.finished.connect(self.finish_measure_membranes)
        self.qt_thread.start()
        self.set_active_ui(False)

    def finish_measure_membranes(self):
        if self.miw is None:
            print("No measurement worker available")
            return
        rings = self.miw.rings
        if rings is None:
            print("No membrane rings generated")
            return
        inner = self.miw.inner
        if inner is None:
            print("No inner masks generated")
            return
        seg_name = self.segmentation_channel_combo.currentText()
        seg_layer = self.viewer.layers[seg_name]
        
        name = seg_name + MEMBRANE_SUFFIX
        if name in self.viewer.layers:
            self.viewer.layers[name].data = rings
        else:
            self.viewer.add_labels(rings, name=name, scale=seg_layer.scale)

        name = seg_name + INNER_SUFFIX
        if name in self.viewer.layers:
            self.viewer.layers[name].data = inner
        else:
            self.viewer.add_labels(inner, name=name, scale=seg_layer.scale)

        self.open_results_table()
        print("Measurement finished. Results:")
        self.set_active_ui(True)

    def open_results_table(self):
        data = self.miw.get_results()
        if data is None:
            print("No results to display")
            return
        if self.rt is not None:
            self.rt.close()
        self.rt = FrameWiseResultsTable(data, name=self.cb_intensity_channel.currentText())
        self.rt.setWindowTitle("Membrane Intensities Results")
        self.rt.show()

    def clear_state(self):
        self.scw = SegmentCellsWorker()
        self.tcw = TrackCellsWorker()
        self.miw = MeasureMembraneIntensity()
        self.roi = RemoveOutlierIntensities()
        self.rt  = None
        self.viewer.layers.clear()
        self.refresh_layer_names()

    def save_layers(self):
        folder = QFileDialog.getExistingDirectory(self, "Select folder to save layers")
        if not folder:
            print("No folder selected")
            return
        for layer in self.viewer.layers:
            path = os.path.join(folder, layer.name + ".tif")
            tifffile.imwrite(path, layer.data)
        print(f"Layers saved to folder: {folder}")


def launch_test_procedure():
    import tifffile as tiff

    viewer = napari.Viewer()
    widget = IntensitiesInMembraneWidget(viewer)
    viewer.window.add_dock_widget(widget)

    print("--- Workflow: Small data ---")

    napari.run()

if __name__ == "__main__":
    launch_test_procedure()