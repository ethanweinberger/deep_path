import argparse
import math
import sys
import math
import pickle
import os
import numpy as np
import tensorflow as tf
import pandas as pd
import pickle
import shutil
import matplotlib.pyplot as plt
import constants
from sklearn.metrics import auc
from scipy import interp
from utils.slide_utils import get_slide_thumbnail
from utils.file_utils import load_pickle_from_disk
from utils.file_utils import write_pickle_to_disk
from PIL import Image

from collections import namedtuple
Vote_Container = namedtuple("Vote_Container", 
        ["percent_pos_votes",
        "slide_name",
        "slide_category"])
Confidence_Container = namedtuple("Confidence_Container",
        ["patch_path",
        "confidence"])

def load_graph(model_file):
    graph = tf.Graph()
    graph_def = tf.GraphDef()
    
    with open(model_file, "rb") as f:
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def)

    return graph

def read_tensor_from_image_file(file_name,
                                input_height=224,
                                input_width=224):
    input_name = "file_reader"
    file_reader = tf.read_file(file_name, input_name)
    if not (file_name.endswith(".jpg") or file_name.endswith(".jpeg")):
       raise IOError("Images must be provided in jpeg format!") 

    image_reader = tf.image.decode_jpeg(
        file_reader, channels=3, name="jpeg_reader")

    image_as_float = tf.image.convert_image_dtype(image_reader, tf.float32)
    image_4d = tf.expand_dims(image_as_float, 0) 
    resize_shape = tf.stack([input_height, input_width])
    resized_shape_as_int = tf.cast(resize_shape, dtype=tf.int32)
    resized_image = tf.image.resize_bilinear(image_4d, resized_shape_as_int)
   
    return resized_image

def load_bottleneck(patch_path):
    split_path = patch_path.split("/") 
    category = split_path[4]
    image_name = split_path[-1]
    bottleneck_suffix = "_https~tfhub.dev~google~imagenet~resnet_v2_50~feature_vector~1.txt"
    bottleneck_path = os.path.join(constants.BOTTLENECK_DIR, category, image_name + bottleneck_suffix)
    
    with open(bottleneck_path, 'r') as bottleneck_file:
        bottleneck_string = bottleneck_file.read()
        bottleneck_values = [float(x) for x in bottleneck_string.split(',')]

    return bottleneck_values
    
def classify_whole_slides(
        test_slide_list,
        model_file, 
        label_file,
        input_layer, 
        output_layer,
        histogram_folder,
        fold_number):
    """
    Runs all the patches for each test slide as specified in test_slide_list 
    through the trained network, and saves the percentage of patches
    classified as positive.  Also saves to disk a histogram for each slide
    of the distribution of HER2+ patch confidences.

    Args:
        model_file (String): Location of saved network model file
        test_slide_list_path (String): Path to pickle of list of slides held out for testing the network
        label_file (String): Path to csv file with outcome labels
        input_layer (String): Name of input layer in saved model file
        output_layer (String): Name of output layer in saved moddel file
        histogram_folder (String): Directory in which to save histogram plots
        fold_number (Int): Current fold in k-fold validation that we're using
    Returns:
        vote_container_list (Vote_Container list): List of Vote_Container
            objects containing the number of positive and negative votes
            for each slide
    """

    graph = load_graph(model_file)
    bottleneck_input_name  = "import/input/BottleneckInputPlaceholder"
    output_name = "import/final_result" 
    output_operation = graph.get_operation_by_name(output_name)

    if os.path.exists(constants.PATCH_CONFIDENCE_FOLD_SUBFOLDER(fold_number)):
        shutil.rmtree(constants.PATCH_CONFIDENCE_FOLD_SUBFOLDER(fold_number))

    os.makedirs(constants.PATCH_CONFIDENCE_FOLD_SUBFOLDER(fold_number))

    vote_container_list = []
    confidence_container_list = []
    patch_name_to_confidence_map = {}
    
    with tf.Session(graph = graph) as sess:
        test_slide_list = load_pickle_from_disk(test_slide_list) 
        test_slide_list_blanks_removed = []
        
        df = pd.read_csv(label_file)
        pos_slide_confidence_lists = {}
        neg_slide_confidence_lists = {}

        pos_slide_name_list = []
        neg_slide_name_list = []

        for test_slide in test_slide_list:
            slide_name = os.path.basename(test_slide)
            slide_label_row = df.loc[df.patient_code == slide_name]
            if slide_label_row.empty:
                continue

            slide_label = slide_label_row.iloc[0].aggressive_chemo
            if slide_label == "YES":
                slide_category = "received_treatment"
            elif slide_label == "NO":
                slide_category = "no_treatment"
            else:
                continue

            test_slide_list_blanks_removed.append(test_slide)
 
            num_large_cell_votes = 0
            num_small_cell_votes = 0
            large_tumor_cell_confidences = []

            small_cell_patch_dir = os.path.join(constants.SMALL_CELL_PATCHES, slide_name)
            large_cell_patch_dir = os.path.join(constants.LARGE_CELL_PATCHES, slide_name)

            small_cell_patches = os.listdir(small_cell_patch_dir)
            small_cell_patches_full_path = [os.path.join(small_cell_patch_dir, x) for x in small_cell_patches]

            large_cell_patches = os.listdir(large_cell_patch_dir)
            large_cell_patches_full_path = [os.path.join(large_cell_patch_dir, x) for x in large_cell_patches]

            full_test_slide_patches = small_cell_patches_full_path + large_cell_patches_full_path

            if len(full_test_slide_patches) == 0:
                continue

            print("Classifying patches for slide " + slide_name)
            for patch_path in full_test_slide_patches:
                bottleneck = load_bottleneck(patch_path)		
                results = sess.run(output_operation.outputs[0], {
                    input_operation.inputs[0]: [bottleneck],
                })

                results = np.squeeze(results)
                if results[0] > results[1]:
                    num_large_cell_votes += 1
                else:
                    num_small_cell_votes += 1

                large_tumor_cell_confidences.append(results[0])
                conf_container = Confidence_Container(
                        patch_path=os.path.join(patch_path),
                        confidence=results[0])
                patch_name_to_confidence_map[patch_path] = results[0] 
                confidence_container_list.append(conf_container)

            if slide_category == "received_treatment":
                pos_slide_confidence_lists[slide_name] = large_tumor_cell_confidences
                pos_slide_name_list.append(slide_name)
            elif slide_category == "no_treatment":
                neg_slide_confidence_lists[slide_name] = large_tumor_cell_confidences
                neg_slide_name_list.append(slide_name)

            total_votes = num_large_cell_votes + num_small_cell_votes
            percent_pos_votes = num_large_cell_votes / total_votes
    
            current_slide_vote_container = Vote_Container(
                    percent_pos_votes=percent_pos_votes,
                    slide_name=slide_name,
                    slide_category=slide_category)

            vote_container_list.append(current_slide_vote_container)
        
    confidence_container_list.sort(key = lambda x: x.confidence)

    write_pickle_to_disk(os.path.join(constants.TEST_SLIDE_FOLDER, "testing_slide_list_" + str(fold_number)),
        test_slide_list_blanks_removed)
    write_pickle_to_disk(constants.CONFIDENCE_CONTAINER_LIST(fold_number), confidence_container_list)
    write_pickle_to_disk(constants.PATCH_NAME_TO_CONFIDENCE_MAP(fold_number), patch_name_to_confidence_map)
    write_pickle_to_disk(constants.POS_SLIDE_CONFIDENCE_LISTS(fold_number), pos_slide_confidence_lists)
    write_pickle_to_disk(constants.NEG_SLIDE_CONFIDENCE_LISTS(fold_number), neg_slide_confidence_lists)

    return vote_container_list


def draw_confidence_histograms(fold_number, slide_category, num_graphs_per_row=3):
    """
    Given a list of lists of confidence values, turns
    each list into a histogram and displays them
    on a subplot in the same figure.

    Args:
        confidence_lists (List of float lists): List of confidence value
            lists
        histogram_folder (String): Top-level directory for storing histogram plots
        slide_names (String list): List of slide names 
        slide_category (String): String indicating whether slide was her2+ or her2-
        num_graphs_per_row (Int): Integer indicating the number of histograms to display per row
    Returns:
        None (output saved to disk)
    """
    if slide_category == "pos":
        confidence_lists_path = constants.POS_SLIDE_CONFIDENCE_LISTS(fold_number)
    elif slide_category == "neg":
        confidence_lists_path = constants.NEG_SLIDE_CONFIDENCE_LISTS(fold_number)
    else:
        print("Invalid parameter provided for slide_category.  Acceptable options are 'pos' or 'neg'")

    confidence_lists = load_pickle_from_disk(confidence_lists_path)
    fig, axes = plt.subplots(nrows=math.ceil(len(confidence_lists.keys())/3),
            ncols=num_graphs_per_row)
    axes = axes.flatten()
    plt.xticks([0.25, 0.5, 0.75])
    i = 0
    for key, value in confidence_lists.items():
        ax = axes[i]
        ax.set_ylabel("Frequency")
        ax.set_xlabel("Confidence")
        ax.set_xlim(left=-0.05, right=1.05)
        ax.set_ylim(bottom=0, top=3.5)
        ax.yaxis.set_tick_params(labelleft=True)
        ax.xaxis.set_tick_params(labelbottom=True)
        ax.hist(value, bins=15, density=True)
        ax.set_title(key)
        i += 1

    #Hides any unused spots in the grid
    for i in range(len(confidence_lists), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()

    if not os.path.exists(constants.HISTOGRAM_SUBFOLDER(fold_number)):
        os.makedirs(constants.HISTOGRAM_SUBFOLDER(fold_number))
    filepath = os.path.join(constants.HISTOGRAM_SUBFOLDER(fold_number),
            slide_category + "_histogram_composite.png")
    fig.savefig(filepath)

    plt.close()
    

def compute_roc_points(vote_container_list):
    """
    Given a list of Vote_Containers, computes the
    false positive rates and true positive rates at thresholds
    determined by the pos_vote_percentages in each container

    Args:
        vote_container_list (Vote_Container list): List of Vote_Containers
            containing the percentage of positive votes for each test slide
    Returns:
        roc_point_lists: Tuple containing the x (false positive rate) and y
            (true positive rate) values for our ROC curve
    """
    threshold_list = []
    num_total_negatives = 0
    num_total_positives = 0

    for container in vote_container_list:
        threshold = container.percent_pos_votes
        threshold_list.append(threshold)

        if container.slide_category == "no_treatment":
            num_total_negatives += 1
        elif container.slide_category == "received_treatment":
            num_total_positives += 1
        else:
            raise ValueError("Invalid Slide Category")

    roc_point_list = []
    false_positive_rate_list = []
    true_positive_rate_list = []
    threshold_list.sort(reverse=True)

    for threshold in threshold_list:
        num_false_positives = 0
        num_true_positives  = 0

        for container in vote_container_list:
            if container.slide_category == "no_treatment" and container.percent_pos_votes > threshold:
                num_false_positives += 1
            elif container.slide_category == "received_treatment" and container.percent_pos_votes > threshold:
                num_true_positives += 1

        if num_total_negatives > 0:
            false_positive_rate = num_false_positives / num_total_negatives
        else:
            false_positive_rate = 0
        false_positive_rate_list.append(false_positive_rate)
            
        if num_total_positives > 0:
            true_positive_rate = num_true_positives / num_total_positives             
        else:
            true_positive_rate = 1.0
        true_positive_rate_list.append(true_positive_rate)
    
    false_positive_rate_list.append(1.0)
    true_positive_rate_list.append(1.0)
    roc_point_lists = (false_positive_rate_list, true_positive_rate_list)
    return roc_point_lists

def draw_roc_curve(vote_container_list, output_name = "roc_curve"):
    """
    Given a list of Vote_Containers, this function draws an ROC curve 
    for a single classifier and saves the resulting file to disk
    
    Args:
        vote_container_list (Vote_Container list): List of Vote_Containers
            containing the percentage of positive votes for each test slide
        output_name (String): Name for our output graph file
    Returns:
        None (output saved to disk)
        
    """
    (false_positive_rate_list, true_positive_rate_list) = compute_roc_points(vote_container_list)
    roc_auc = auc(false_positive_rate_list, true_positive_rate_list) 

    plt.plot(false_positive_rate_list, true_positive_rate_list, color='darkorange', 
            lw=lw, label="ROC Curve (area = %0.2f)" % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.savefig(output_name + ".png")

def draw_kfold_roc_curve():
    """
    Given a list of Vote_Container list representing the results
    of different folds in k-fold cross validation, this function draws
    both the individual ROC curves for each fold as well as an
    averaged curve

    Args:
        None
    Returns:
        None (output saved to disk)

    """
    fold_vote_container_lists = load_pickle_from_disk(constants.FOLD_VOTE_CONTAINER_LISTS_PATH)

    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    aucs = []
    
    plt.figure()
    i = 0
    for vote_container_list in fold_vote_container_lists:
        (fpr_list, tpr_list) = compute_roc_points(vote_container_list)
        tprs.append(interp(mean_fpr, fpr_list, tpr_list))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr_list, tpr_list)
        aucs.append(roc_auc)
    
        plt.plot(fpr_list, tpr_list, lw=1, alpha=1,
             label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
        i += 1

    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
         label='Luck', alpha=.8)

     
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                 label=r'$\pm$ 1 std. dev.')
    
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('K-fold Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    
    plt.savefig("k-fold_roc.png")


def display_confidence_container(confidence_container):
    """
    Given a confidence container, overwrites the current
    figure to instead display the container's image
    and confidence value

    Args:
        confidence_container (Confidence_Container): Container to display

    Returns:
        None (output is visual)
    """

    img = Image.open(confidence_container.patch_path)
    confidence = confidence_container.confidence

    plt.title("Confidence: " + str(confidence))
    plt.imshow(img)
    plt.draw()

def visualize_confidence_containers(fold_number):
    """
    Given the path to a list of confidence containers,
    allows the user to visualize patches with their
    associated confidences.

    Args:
        fold_number (Int): Which fold we're testing against
    Returns:
        None (output is visual)
    """

    confidence_container_list = load_pickle_from_disk(constants.CONFIDENCE_CONTAINER_LIST(fold_number))

    if len(confidence_container_list) == 0:
        print("No containers found in confidence container list. Exiting visualization.")
        sys.exit()

    index = 0 
    current_container = confidence_container_list[index]
    img = Image.open(current_container.patch_path)
    confidence = current_container.confidence

    def on_key(event):
        nonlocal index
        if event.key == 'left':
            if index - 1 >= 0:
                index -= 1
                current_container = confidence_container_list[index]
                display_confidence_container(current_container)
            else:
                print("Reached beginning of patches: Can't go back any further")
        elif event.key == 'right':
            if index < len(confidence_container_list) - 1:
                index += 1
                current_container = confidence_container_list[index]
                display_confidence_container(current_container)
            else:
                print("Reached end of patches: Can't go any farther forward")
        elif (event.key == '0' or event.key == '1' or event.key == '2' or event.key == '3'
                or event.key == '4' or event.key == '5' or event.key == '6' or event.key == '7'
                or event.key == '8' or event.key == '9'):
            index = len(confidence_container_list) // 10 * int(event.key)
            current_container = confidence_container_list[index]
            display_confidence_container(current_container)
        elif event.key == '-':
            index = len(confidence_container_list) - 1
            current_container = confidence_container_list[index]
            display_confidence_container(current_container)

    fig, ax = plt.subplots()
    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.title("Confidence: " + str(confidence))
    plt.imshow(img)
    plt.show()

def create_extent_from_thumbnail(thumbnail):
    """
    Creates an extent tuple.  This tuple is passed to
    matplotlib to make it possible to overlay one
    image on top of another.

    Args:
        thumbnail (numpy array): numpy array representing our thumbnail image
    Returns:
        extent (tuple): Tuple of the form (xmin, xmax, ymin, ymax)

    """
    xmin = 0
    xmax = thumbnail.shape[1]
    ymin = 0
    ymax = thumbnail.shape[0]
    extent = (xmin, xmax, ymin, ymax)
    return extent


def transparent_cmap(cmap, N=255):
    "Copy colormap and set alpha values"

    mycmap = cmap
    mycmap._init()
    mycmap._lut[:,-1] = np.append(np.linspace(0.8, 0, (N+4)//2), np.linspace(0, 0.8, math.ceil((N+4)/2)))
    return mycmap

def display_slide(slide_patches_path, slide_name_to_patches_map, slide_name_to_tile_dims_map, 
        patch_name_to_coords_map, patch_name_to_confidence_map, slide_top_level_dir, first_time=False,
        fold_number=0):
    """
    Helper function for two_dim_confidence_visualization.
    Given the path to a slide, this function updates 
    the current plot with the confidence heatmap of
    the given slide.

    Args:
        slide_patches_path (String): Path to slide patches for heatmap
        slide_name_to_patches_map (Dict): Mapping from slide names to patch names
        slide_name_to_tile_dims_map (Dict): Mapping from slide names to dimensions of tile space
        patch_name_to_coords_map (Dict): Mapping from patch names to their coordinates in tile space
        patch_name_to_confidence_map (Dict): Mapping from patch name to confidence value
        slide_top_level_path (String): Path to directory containing slide folders
        first_time (bool): Indicates whether we need to run setup code for the plot
    Returns:
        None (output is visual)
    """

    slide_name = os.path.basename(slide_patches_path)
    
    if slide_name not in slide_name_to_patches_map.keys():
        return

    patches_list = slide_name_to_patches_map[slide_name]
    tiled_dims = slide_name_to_tile_dims_map[slide_name]

    confidence_array = np.full(tiled_dims, 0.5)

    for patch in patches_list:
        if "stroma" in patch:
            continue
        patch_coords = patch_name_to_coords_map[patch]
        patch_confidence = patch_name_to_confidence_map[patch + ".jpg"]
        confidence_array[patch_coords] = patch_confidence

    slide_path = os.path.join(slide_top_level_dir, slide_name + ".svs")
    thumbnail = get_slide_thumbnail(slide_path, tiled_dims[0]*100, tiled_dims[1]*100)
    thumbnail = np.array(thumbnail)
    extent = create_extent_from_thumbnail(thumbnail)

    cmap = transparent_cmap(plt.cm.seismic)

    plt.imshow(thumbnail, extent=extent)
    plt.title(slide_name)
    plt.imshow(confidence_array, alpha=1.0, extent=extent, cmap=cmap)

    if first_time:
        cbar = plt.colorbar(ticks=[0, 1])
        cbar.ax.set_yticklabels(["Small Cell", "Large Cell"])
        #cbar.set_label("Confidence", labelpad=15, rotation=270)

    plt.clim(vmin=0, vmax=1.0)
    #plt.show() if first_time else plt.draw()
    plt.savefig(os.path.join(constants.HEATMAP_SUBFOLDER(fold_number),slide_name))


def two_dim_confidence_visualization(fold_number):
    """
    Function to visualize distribution of confidence values
    over a slide.
    """
    patch_name_to_coords_map     = load_pickle_from_disk(constants.PATCH_NAME_TO_COORDS_MAP)
    slide_name_to_tile_dims_map  = load_pickle_from_disk(constants.SLIDE_NAME_TO_TILE_DIMS_MAP)
    slide_name_to_patches_map    = load_pickle_from_disk(constants.SLIDE_NAME_TO_PATCHES_MAP)
    patch_name_to_confidence_map = load_pickle_from_disk(constants.PATCH_NAME_TO_CONFIDENCE_MAP(fold_number))
    test_slide_list              = load_pickle_from_disk(os.path.join(constants.TEST_SLIDE_FOLDER,
        constants.TEST_SLIDE_LIST + "_" + str(fold_number)))

    if os.path.exists(constants.HEATMAP_SUBFOLDER(fold_number)):
        shutil.rmtree(constants.HEATMAP_SUBFOLDER(fold_number))
    
    os.makedirs(constants.HEATMAP_SUBFOLDER(fold_number))

    first_time = True
    for slide_patches_path in test_slide_list:
        display_slide(slide_patches_path, slide_name_to_patches_map, slide_name_to_tile_dims_map,
                patch_name_to_coords_map, patch_name_to_confidence_map, constants.SLIDE_FILE_DIRECTORY, first_time,
                fold_number)
        if first_time:
            first_time = False
    
def create_visualization_helper_files():
    """
    Main function for this file.  Uses our trained networks to
    classify each slide patch and saves the results of these
    classifications to disk for later use

    Args:
        None
    Returns:
        None (output saved to disk)
    """
    testing_slide_lists = os.listdir(constants.TEST_SLIDE_FOLDER)
    model_files = os.listdir(constants.MODEL_FILE_FOLDER)

    testing_slide_lists.sort()
    model_files.sort()

     
    fold_vote_container_lists = []
    i = 0
    for (testing_slide_list, model_file) in zip(testing_slide_lists, model_files):
        
        vote_container_list = classify_whole_slides(
            os.path.join(constants.TEST_SLIDE_FOLDER, testing_slide_list),
            os.path.join(constants.MODEL_FILE_FOLDER, model_file),
            constants.LABEL_FILE,
            constants.INPUT_LAYER,
            constants.OUTPUT_LAYER,
            constants.HISTOGRAM_FOLDER + "/fold_" + str(i),
            i)

        fold_vote_container_lists.append(vote_container_list)
        i += 1
    write_pickle_to_disk(constants.FOLD_VOTE_CONTAINER_LISTS_PATH, fold_vote_container_lists)
