# Detecting HER2 with Deep Learning

Code for Kluger Lab project to classify breast cancer tissue into HER2+ or HER2- using CNNs. (Work in progress)

## Requirements:

* Python 3.4 or higher
* Numpy
* PIL
* matplotlib
* tensorflow-hub
* OpenSlide

Openslide can be installed by following the instructions at https://openslide.org/download/.  All
other libraries can be installed via pip.

## Usage:

### Constructing a data set:

This project assumes that the user has available a collection of whole-slide images (WSIs) of breast tissue that
have undergone H&E staining.  We also assume that these slides have had their tumor regions identified via
annotations using the open-source software QuPath.

Starting from QuPath, the user should first open their qpproj file encompassing their annotations and then
run the extract_polygons.groovy script from QuPath's automation interface.  Doing so will generate a folder
for each slide containing CSV files with the coordinates of that slide's annotations.

After doing so, the function `construct_training_dataset` from `slide_utils.py` should be run to
create a training dataset of non-overlapping patches from the slides.  The user will need to change
`annotation_base_path` in `construct_annotation_path_list` as well as provide the correct path for
`top_level_directory` in `construct_training_dataset`.  

TODO: Waiting to receive true labels for dataset

### Training the model

TODO: Use the `retrain.py` TF script to retrain Inception
