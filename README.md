# Detecting HER2 with Deep Learning

Code for Kluger Lab project to classify breast cancer tissue into HER2+ or HER2- using CNNs. (Work in progress)

## Requirements:

* Python 3.4 or higher
* Numpy
* PIL
* matplotlib
* tensorflow
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
run the `extract_polygons.groovy` script from QuPath's automation interface.  Doing so will generate a folder
for each slide containing CSV files with the coordinates of that slide's annotations.

After doing so, the function `construct_training_dataset` from `slide_utils.py` should be run to
create a training dataset of non-overlapping patches from the slides.  The user will need to change
`annotation_base_path` in `construct_annotation_path_list` as well as provide the correct path for
`top_level_directory` in `construct_training_dataset`.  The user will also need to provide a CSV
containing the true labels for each slide (`label_file` argument), as well as potentially edit the lines of code in
`construct_training_dataset` that extract the labels from the CSV.

Running `construct_training_dataset` will save patches to

`output_dir/her2_classification/slide_name`

where `her2_classification` is either `her2_pos` or `her2_neg` depending on the true value in the label file.

### (re-)Training the model

To retrain a pre-trained model pretrained model (ex. Inceptionv3, ResNet50) to classify slides.  We can use
the script `retrain_patient_level.py`.  This script will divide the slides into training, validation, and test
sets at a ratio of 80%/10%/10%.  An example command with this script is below:

`python3 retrain_patient_level.py --image_dir=/data/ethan/hne_patches_tumor_only --tfhub_module=https://tfhub.dev/google/imagenet/resnet_v2_152/feature_vector/1 --how_many_training_steps=1000`

The parameter `image_dir` must point to the directory containing the `her2_pos` and `her2_neg` folders (which themselves contain the patches
from each slide belonging to their respective categories).  `tfhub_module` must point to a pretrained module hosted on tfhub.dev.  Finally,
`how_many_training_steps` controls how many epochs we'll use to train our model.

Once the model is trained it's contents will be saved as `/tmp/output_graph.pb`.  Our module will also save a pickle
file named `testing_slides`.  We will need this to keep track of which slides were in the test set for evaluating our 
model later.

Note to self: Current trainng command is 

`python3 retrain_patient_level.py --image_dir=/data/ethan/hne_patches_tumor_stroma_interface/ --tfhub_module=https://tfhub.dev/google/imagenet/resnet_v2_152/feature_vector/1 --how_many_training_steps=300 --random_crop=5 --random_scale=5 --random_brightness=5 --flip_left_right`

### Evaluating the Model

We can evaluate the performance of our model using `test_network.py`.  This script will attempt to classify slides as
HER2-positive or HER2-negative by passing all patches for a given slide through the network, obtaining the classification
for each patch, then using a majority vote to determine the final class for the slide  
