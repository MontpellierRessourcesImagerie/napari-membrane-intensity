import os
import re
from ij import IJ
from pprint import pprint
from ij.gui import GenericDialog
from java.awt import Panel, FlowLayout, Label, Choice, GridLayout
from ij.plugin import Concatenator

identity_idx   = 1
time_index_idx = 2
dye_idx        = 3
item_regex     = re.compile(r"^(.+)[ _-]+t([0-9]{1,2})[ _-]+(.+)\.tif$")

# In a folder, searches for all the tif files having tN in their name between two separators
def probe_folder(folder_path):
	content = os.listdir(folder_path)
	items = {}
	for file in content:
		if "merged" in file:
			continue
		m = item_regex.match(file)
		if m is None:
			continue
		identity   = m.group(identity_idx)
		time_index = m.group(time_index_idx)
		dye        = m.group(dye_idx)

		items.setdefault(identity, {})
		items[identity].setdefault(dye, [])
		items[identity][dye].append((int(time_index), file))
	return items

def ask_user(items):
    gd = GenericDialog("Merge time series")

    # First pass: count how many rows we need
    rows = []
    for identity, dyes in items.items():
        for dye, images in dyes.items():
            imgs_sorted = sorted(images)
            if imgs_sorted[0][0] != 0:
                continue
            rows.append((identity, dye, imgs_sorted))

    if not rows:
        return None

    n_rows = len(rows)
    n_cols = 4  # identity | dye | first image | choice

    panel = Panel(GridLayout(n_rows, n_cols, 10, 4))

    choices = []

    # Second pass: populate the grid
    for identity, dye, imgs_sorted in rows:
        panel.add(Label(identity))
        panel.add(Label(dye))
        panel.add(Label(imgs_sorted[0][1]))

        choice = Choice()
        for s in imgs_sorted[1:]:
            choice.add(s[1])
        choice.select(imgs_sorted[-1][1])

        panel.add(choice)
        choices.append((identity, dye, imgs_sorted[0][1], choice))

    gd.addPanel(panel)
    gd.showDialog()

    if gd.wasCanceled():
        return None

    # Read back selections
    result = {}
    for identity, dye, t0, choice in choices:
        result.setdefault(identity, {})[dye] = (t0, choice.getSelectedItem())

    return result

def merge_images(folder, items):
	ctn = Concatenator()
	for identity, dyes in items.items():
		for dye, images in dyes.items():
			if (images[0] is None) or (images[1] is None):
				print("Skipping " + identity + " for " + dye)
				continue
			new_name = identity + "-" + dye + "_merged.tif"
			t_0 = IJ.openImage(os.path.join(folder, images[0]))
			t_N = IJ.openImage(os.path.join(folder, images[1]))
			res = ctn.concatenate(t_0, t_N, False)
			output_path = os.path.join(folder, new_name)
			IJ.saveAs(res, "TIFF", output_path)
			res.close()

if __name__ == "__main__":
	folder_path = "/home/clement/Documents/projects/2219-intensity-membrane/20250722_MUT_max projections"
	items = probe_folder(folder_path)
	results = ask_user(items)
	merge_images(folder_path, results)
	
	