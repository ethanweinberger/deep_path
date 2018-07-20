import argparse
import sys
import math
import pickle
import os
import numpy as np
import tensorflow as tf
import pandas as pd
import pickle
import shutil
import curses
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from scipy import interp
from slide_utils import get_slide_thumbnail
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

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
    
def classify_whole_slides(
        test_slide_list,
        model_file, 
        label_file, 
        input_layer, 
        output_layer,
        histogram_folder,
        patch_confidence_dir):
    """
    Runs all the patches for each test slide as specified in test_slide_list 
    through the trained network, and saves the percentage of patches
    classified as positive.  Also saves to disk a histogram for each slide
    of the distribution of HER2+ patch confidences.

    Args:
        model_file (String): Location of saved network model file
        label_file (String): Location of csv file with true slide labels
        test_slide_list (String list): List of slides held out for testing the network
        input_layer (String): Name of input layer in saved model file
        output_layer (String): Name of output layer in saved moddel file
        histogram_folder (String): Directory in which to save histogram plots
        patch_confidence_dir (String): Directory in which to save patch confidences
    Returns:
        vote_container_list (Vote_Container list): List of Vote_Container
            objects containing the number of positive and negative votes
            for each slide
    """

    graph = load_graph(model_file)
    input_name = "import/" + input_layer
    output_name = "import/" + output_layer
    input_operation = graph.get_operation_by_name(input_name)
    output_operation = graph.get_operation_by_name(output_name)
    pos_folder = os.path.join(histogram_folder, "pos")
    neg_folder = os.path.join(histogram_folder, "neg")

    if os.path.exists(histogram_folder):
        shutil.rmtree(histogram_folder)
    
    os.makedirs(histogram_folder)
    os.makedirs(pos_folder)
    os.makedirs(neg_folder)

    if os.path.exists(patch_confidence_dir):
        shutil.rmtree(patch_confidence_dir)

    os.makedirs(patch_confidence_dir)

    vote_container_list = []
    confidence_container_list = []
    patch_name_to_confidence_map = {}
    
    with tf.Session(graph = graph) as sess:
        with open(test_slide_list, "rb") as fp:
            test_slide_list = pickle.load(fp)
        
        slides_classified_correctly = 0
        slides_examined = 0
        num_positives_classified_correctly = 0
        num_false_positives = 0

        num_true_positives = 0
        num_true_negatives = 0
        
        df = pd.read_csv(label_file)
        pos_slide_confidence_lists = []
        neg_slide_confidence_lists = []

        pos_slide_name_list = []
        neg_slide_name_list = []

        for test_slide in test_slide_list:
            slide_name = os.path.basename(test_slide)
            print("Classifying patches for slide " + slide_name)
            slide_label_row = df.loc[df.specnum_formatted == slide_name]
            if slide_label_row.empty:
                continue

            slide_label = slide_label_row.iloc[0].her2_ihc
            if slide_label == 0.0 or slide_label == 1.0:
                slide_category = "neg"
            elif slide_label == 2.0 or slide_label == 3.0:
                slide_category = "pos"
            else:
                continue
 
            num_neg_votes = 0
            num_pos_votes = 0
            her2_plus_confidences = []
            for file_name in os.listdir(test_slide):
                patch_name = os.path.join(test_slide, file_name)
                img_tensor = read_tensor_from_image_file(patch_name)
                img = sess.run(img_tensor)
                results = sess.run(output_operation.outputs[0], {
                    input_operation.outputs[0]: img 
                })
                results = np.squeeze(results)
                if results[0] > results[1]:
                    num_neg_votes += 1
                else:
                    num_pos_votes += 1
                her2_plus_confidences.append(results[1])
                conf_container = Confidence_Container(
                        patch_path=os.path.join(test_slide, file_name),
                        confidence=results[1])
                patch_name_to_confidence_map[patch_name] = results[1] 
                confidence_container_list.append(conf_container)

            if slide_category == "pos":
                pos_slide_confidence_lists.append(her2_plus_confidences)
                pos_slide_name_list.append(slide_name)
            elif slide_category == "neg":
                neg_slide_confidence_lists.append(her2_plus_confidences)
                neg_slide_name_list.append(slide_name)

            plt.figure()
            plt.hist(her2_plus_confidences, bins=15, density=True)
            plt.xlim(xmin=0, xmax=1.0)
            plt.ylim(bottom=0, top=3.5)
            plt.title(slide_name + "patch HER2+ confidence distribution")
            plt.xlabel("Value")
            plt.ylabel("Frequency")

            histogram_file_path = os.path.join(histogram_folder, slide_category, slide_name)
            plt.savefig(histogram_file_path + ".png")
            plt.close()

            total_votes = num_pos_votes + num_neg_votes
            percent_pos_votes = num_pos_votes / total_votes
    
            current_slide_vote_container = Vote_Container(
                    percent_pos_votes=percent_pos_votes,
                    slide_name=slide_name,
                    slide_category=slide_category)

            vote_container_list.append(current_slide_vote_container)
        
        draw_confidence_histograms(pos_slide_confidence_lists, histogram_folder, pos_slide_name_list, "pos")
        draw_confidence_histograms(neg_slide_confidence_lists, histogram_folder, neg_slide_name_list, "neg")
    
    confidence_container_list.sort(key = lambda x: x.confidence)
    conf_container_list_filename = os.path.join(patch_confidence_dir, "confidences")
    patch_name_to_conf_map_filename = os.path.join(patch_confidence_dir, "map")
    with open(conf_container_list_filename, "wb") as fp:
        pickle.dump(confidence_container_list, fp)
    with open(patch_name_to_conf_map_filename, "wb") as fp:
        pickle.dump(patch_name_to_confidence_map, fp)
    return vote_container_list

def draw_confidence_histograms(confidence_lists, histogram_folder, slide_names, slide_category, num_graphs_per_row=3):
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
    fig, axes = plt.subplots(nrows=math.ceil(len(confidence_lists)/3),
            ncols=num_graphs_per_row)
    axes = axes.flatten()
    plt.xticks([0.25, 0.5, 0.75])
    for i in range(len(confidence_lists)):
        ax = axes[i]
        ax.set_ylabel("Frequency")
        ax.set_xlabel("Confidence")
        ax.set_xlim(left=-0.05, right=1.05)
        ax.set_ylim(bottom=0, top=3.5)
        ax.yaxis.set_tick_params(labelleft=True)
        ax.xaxis.set_tick_params(labelbottom=True)
        ax.hist(confidence_lists[i], bins=15, density=True)
        ax.set_title(slide_names[i])

    #Hides any unused spots in the grid
    for i in range(len(confidence_lists), len(axes)):
        axes[i].set_visible(False)
    filepath = os.path.join(histogram_folder, slide_category)
    plt.tight_layout()
    fig.savefig(filepath + ".png")
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

    #So that our ROC Curve won't miss the top right corner
    threshold_list.append(0.0)
    for container in vote_container_list:
        threshold = container.percent_pos_votes
        threshold_list.append(threshold)

        if container.slide_category == "neg":
            num_total_negatives += 1
        elif container.slide_category == "pos":
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
            if container.slide_category == "neg" and container.percent_pos_votes > threshold:
                num_false_positives += 1
            elif container.slide_category == "pos" and container.percent_pos_votes > threshold:
                num_true_positives += 1

        false_positive_rate = num_false_positives / num_total_negatives
        false_positive_rate_list.append(false_positive_rate)
            
        true_positive_rate = num_true_positives / num_total_positives             
        true_positive_rate_list.append(true_positive_rate)
    
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

def draw_roc_curve_kfold(fold_vote_container_lists):
    """
    Given a list of Vote_Container list representing the results
    of different folds in k-fold cross validation, this function draws
    both the individual ROC curves for each fold as well as an
    averaged curve

    Args:
        fold_vote_container_lists (List of Vote_Container lists): List of 
            Vote_Container lists, each of which represents the results
            of testing the model on one of our k-folds
    Returns:
        None (output saved to disk)
    """
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
    
        plt.plot(fpr_list, tpr_list, lw=1, alpha=0.3,
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

def visualize_confidence_containers(confidence_container_list_path):
    """
    Given the path to a list of confidence containers,
    allows the user to visualize patches with their
    associated confidences.

    Args:
        confidence_container_list_path (String): Path to confidence
            container list pickle
    Returns:
        None (output is visual)
    """

    with open(confidence_container_list_path, "rb") as fp:
        confidence_container_list = pickle.load(fp)

        if len(confidence_container_list) == 0:
            print("No containers found in confidence container list. Exiting visualization.")
            return

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

def display_slide(slide_patches_path, slide_name_to_patches_map, slide_name_to_tile_dims_map, 
        patch_name_to_coords_map, patch_name_to_confidence_map, slide_top_level_dir, first_time=False):
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
    slide_category_folder = os.path.basename(os.path.dirname(slide_patches_path))
    slide_category = "HER2+" if slide_category_folder == "her2_pos" else "HER2-"

    patches_list = slide_name_to_patches_map[slide_name]
    tiled_dims = slide_name_to_tile_dims_map[slide_name]

    confidence_array = np.zeros(tiled_dims)

    for patch in patches_list:
        patch_coords = patch_name_to_coords_map[patch]
        patch_confidence = patch_name_to_confidence_map[patch + ".jpg"]
        
        confidence_array[patch_coords] = patch_confidence

    slide_path = os.path.join(slide_top_level_dir, slide_name, "Scan1", slide_name + "_Scan1.qptiff")
    thumbnail = get_slide_thumbnail(slide_path, tiled_dims[0]*100, tiled_dims[1]*100)
    thumbnail = np.array(thumbnail)
    extent = create_extent_from_thumbnail(thumbnail)

    plt.imshow(thumbnail, extent=extent)
    plt.title(slide_name + " (" + slide_category + ")")
    plt.imshow(confidence_array, alpha=.2, extent=extent)

    if first_time:
        cbar = plt.colorbar()
        cbar.set_label("Confidence", labelpad=15, rotation=270)

    plt.clim(vmin=0, vmax=1.0)
    plt.show() if first_time else plt.draw()

def two_dim_confidence_visualization_interp(confidence_directory, fold_directory, test_slide_list_path,
        slide_top_level_dir="/data/ethan/Breast_Deep_Learning/Polaris/263/"):
    """
    Function to visualize distribution of confidence values
    over a slide.
    """
    with open("patch_name_to_coords_map", "rb") as fp:
        patch_name_to_coords_map = pickle.load(fp)
    with open("slide_name_to_tile_dims_map", "rb") as fp:
        slide_name_to_tile_dims_map = pickle.load(fp)
    with open("slide_name_to_patches_map", "rb") as fp:
        slide_name_to_patches_map = pickle.load(fp)
    with open(os.path.join(confidence_directory, fold_directory, "map"), "rb") as fp:
        patch_name_to_confidence_map = pickle.load(fp)
    with open(test_slide_list_path, "rb") as fp:
        test_slide_list = pickle.load(fp)
 
    def on_key(event):
        nonlocal index
        if event.key == 'left':
            if index - 1 >= 0:
                index -= 1
                slide_patches_path = test_slide_list[index]
                display_slide(slide_patches_path, slide_name_to_patches_map, slide_name_to_tile_dims_map,
                        patch_name_to_coords_map, patch_name_to_confidence_map, slide_top_level_dir)
            else:
                print("Reached beginning of slides: Can't go back any further")
        elif event.key == 'right':
            if index < len(test_slide_list) - 1:
                index += 1
                slide_patches_path = test_slide_list[index]
                display_slide(slide_patches_path, slide_name_to_patches_map, slide_name_to_tile_dims_map,
                        patch_name_to_coords_map, patch_name_to_confidence_map, slide_top_level_dir)
            else:
                print("Reached end of slides: Can't go any farther forward")

    fig = plt.figure(frameon = False)
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    index = 0
    slide_patches_path = test_slide_list[index]
    display_slide(slide_patches_path, slide_name_to_patches_map, slide_name_to_tile_dims_map,
            patch_name_to_coords_map, patch_name_to_confidence_map, slide_top_level_dir, first_time=True)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_file_folder", help="model file to load", default="./output_graph_files")
    parser.add_argument("--input_layer", help="name of input layer", default="Placeholder")
    parser.add_argument("--output_layer", help="name of output layer", default="final_result")
    parser.add_argument("--label_file", help="label file to load", default="/data/ethan/Breast_Deep_Learning/labels.csv")
    parser.add_argument("--test_slide_folder", help="pickle containing list of test slides", default="./testing_slide_lists")
    parser.add_argument("--histogram_folder", help="folder to store patch confidence histograms", default="./histograms")
    parser.add_argument("--patch_confidence_folder", help="folder to store patch confidence values", default="./patch_confidences")
    args = parser.parse_args()
    
    testing_slide_lists = os.listdir(args.test_slide_folder)
    model_files = os.listdir(args.model_file_folder)

    testing_slide_lists.sort()
    model_files.sort()

    """ 
    fold_vote_container_lists = []
    i = 0
    for (testing_slide_list, model_file) in zip(testing_slide_lists, model_files):
        
        vote_container_list = classify_whole_slides(
            os.path.join(args.test_slide_folder, testing_slide_list),
            os.path.join(args.model_file_folder, model_file),
            args.label_file,
            args.input_layer,
            args.output_layer,
            args.histogram_folder + "/fold_" + str(i),
            args.patch_confidence_folder + "/fold_" + str(i))

        fold_vote_container_lists.append(vote_container_list)
        i += 1
    draw_roc_curve_kfold(fold_vote_container_lists) 
    """
    #two_dim_confidence_visualization("patch_confidences", "fold_0", os.path.join(args.test_slide_folder, testing_slide_lists[0]))
    two_dim_confidence_visualization_interp("patch_confidences", "fold_0", os.path.join(args.test_slide_folder, testing_slide_lists[0]))
    #visualize_confidence_containers("./patch_confidences/fold_0/confidences")
    
