# Intensity in membrane

- This Napari plugin measures the intensity in cell membranes over time without accounting for outliers (strong spots).
- The objects are segmented using Cellpose 3 (custom models allowed) and tracked over time with TrackPy.
- The membranes are isolated and local maxima are excluded before measurements.
- **Limitation:** This plugin was developed for 2D+t images; it does not work for 3D.

## How to use it?

### 1. Setup

- Open Napari and import one of your images in the viewer.
- From the "Plugins" menu, open the "Intensity in membranes" plugin.

### 2. Segment your cells

- In "Segmentation channel", indicate the image on which the segmentation will be performed. It can be a channel different from the one you will measure on.
- In "Cell diameter", indicate the approximate diameter (in number of pixels) of a cell.
- In the "Model" dropdown menu, choose the most suitable model for your images. You have two choices:
  1. Use a pretrained Cellpose model such as "cyto3" or "nuclei".
  2. Use a custom model. To do so, create a folder named "models" in the source folder and download your models there. The path looks like `/where/you/downloaded/napari-intensity-in-membrane/src/napari_intensity_in_membrane/models/`. Some models trained on yeast cells are available:
     - [CP_2026-01-15-yeasts](https://github.com/MontpellierRessourcesImagerie/napari-membrane-intensity/releases/download/v0.0.1/CP_2026-01-15-yeasts)
     - [CP_20251209_153303](https://github.com/MontpellierRessourcesImagerie/napari-membrane-intensity/releases/download/v0.0.1/CP_20251209_153303)
- Click "Run segmentation"; after a few seconds, you should get your labeled cells in a new labels layer.

### 3. Tracking and curation

- Start by clicking "Track cells" so that your segmented cells are tracked (labels remain consistent across time).
- If you want to work on only a few cells, you can use a points layer to indicate the ones you want and use the "Keep labels" button.
- If you need to merge some cells together (mother and daughter, for example), you can use lines from a shape layer and click "Merge cells".
- If you made a mistake, you can click "Track cells" again to reset the state.

### 4. Measure intensities

#### a. Designate data

- In "Intensity channel", indicate the channel in which you want to measure the intensities.
- In "Membrane thickness", indicate (in number of pixels) the thickness of the membranes (to erode the cells).

#### b. Outlier removal

To remove outliers, we start by searching for local maxima. The corresponding mask goes through a morphological opening to remove small pixels. The mask is then dilated to remove a ring around each maximum, ensuring that the surrounding gradient is removed as well. Maxima are identified based on a factor of the mean intensity across the whole cell ("a pixel is part of a maximum if it is N times higher than the local mean").

- Set "Opening size" to remove pixels belonging to false positives (the larger the value, the larger the islands that will be removed).
- Set the "Dilation size" to remove a ring around each maximum.
- Set the "Intensity factor" to set the local threshold used to identify maxima.
- Click "Remove outliers" (as many times as needed while tuning the settings).
- Finally, click "Measure intensities" to open a new window containing a results table.