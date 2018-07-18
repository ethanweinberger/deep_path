import numpy as np
import pandas as pd
from PIL import Image
import shutil

import openslide
import csv
from openslide.deepzoom import DeepZoomGenerator
from matplotlib.path import Path
from patch import Patch
import augmentation
import os
import sys

def load_slide(path, save_thumbnail=False):
    """
    Function for opening slide images

    Args:
        path: Path to the image file
        save_thumbnail: If true, save thumbnail of the image. Used for testing.

    Returns:
        OpenSlide object

    """

    osr = openslide.OpenSlide(path)
    if save_thumbnail:
        im = osr.get_thumbnail((200, 200))
        im.save('test.jpg')
    return osr

def get_patches_from_slide(slide, tile_size=1024, overlap=0, limit_bounds=False):
    """ 
    Splits an OpenSlide object into nonoverlapping patches

    Args:
        slide: OpenSlide object
        tile_size: Width and height of a single tile
        overlap: Number of extra pixels to add to each interior edge of a tile
        limit_bounds: If True, renders only non-empty slide region
    Returns:
        Array of patches
    """

    tiles = DeepZoomGenerator(slide, tile_size=tile_size, overlap=overlap, 
                limit_bounds=limit_bounds) 

    level = len(tiles.level_tiles) - 1
    x_tiles, y_tiles = tiles.level_tiles[level] #Note: Highest level == Highest resolution
    x, y = 0, 0
    count, batch_count = 0, 0
    patches = []
    while y < y_tiles:
        while x < x_tiles:
            new_patch_img = np.array(tiles.get_tile(level, (x,y)), dtype=np.uint8)
            new_patch_coords = tiles.get_tile_coordinates(level, (x,y))
            if np.shape(new_patch_img) == (tile_size, tile_size, 3):
                new_patch = Patch(new_patch_img, new_patch_coords)
                patches.append(new_patch)
                count += 1
            x += 1
        y += 1
        x = 0
    return patches

def construct_training_dataset(top_level_directory="/data/ethan/Breast_Deep_Learning/Polaris/263/", 
        file_extension="qptiff", 
        output_dir="/data/ethan/hne_patches_tumor_stroma_interface/",
        label_file="/data/ethan/Breast_Deep_Learning/labels.csv", 
        annotations_only=True):
    """
    Recursively searches for files of the given slide file format starting at
    the provided top level directory.  As slide files are found, they are broken
    up into nonoverlapping patches that can be used to train our model

    Args:
        top_level_directory (String): Location of the top-level directory, within which
                                      lie all of our files
        file_extension (String): File extension for slide files
        output_dir (String): Folder in which patch files will be saved
        label_file (String): CSV file containing the true labels for each slide
        annotations_only (Boolean): When true, only saves patches that have at least one corner within an annotation path
    Returns:
        None (Patches saved to disk)
    """
    

    her2_pos_folder = os.path.join(output_dir, "her2_pos")
    her2_neg_folder = os.path.join(output_dir, "her2_neg")

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    os.makedirs(output_dir)
    os.makedirs(her2_pos_folder)
    os.makedirs(her2_neg_folder)

    df = pd.read_csv(label_file)

    for root, dirnames, filenames in os.walk(top_level_directory):
        for filename in filenames:
            if filename.endswith(file_extension):
                full_path = os.path.join(root, filename)
                slide_name = os.path.splitext(os.path.basename(full_path))[0].split("_")[0]

                print("Splitting " + slide_name)
                slide_label_row = df.loc[df.specnum_formatted == slide_name]

                if slide_label_row.empty:
                    continue
                
                slide_label = slide_label_row.iloc[0].her2_ihc
                if slide_label == 0.0 or slide_label == 1.0:
                    class_folder = her2_neg_folder
                elif slide_label == 2.0 or slide_label == 3.0:
                    class_folder = her2_pos_folder
                else:
                    continue #In case we have an empty or malformed cell

                slide_dir = os.path.join(class_folder, slide_name)
                os.makedirs(slide_dir)
                slide = load_slide(full_path)
                path_list = construct_annotation_path_list(slide_name)

                patches = get_patches_from_slide(slide)
                counter = 0

                for patch in patches:
                    if (annotations_only and patch_in_paths(patch, path_list)) or not annotations_only: 
                        patch.save_img_to_disk(slide_dir + "/" + slide_name + "_" + str(counter))
                        counter += 1
                print("Total patches for " + slide_name + ": " + str(counter))
                
def construct_annotation_path_list(slide_name, annotation_base_path="/data/ethan/Breast_Deep_Learning/annotation_csv_files/"):
    """
    Given the name of a slide, returns a list of polygons representing the annotations
    drawn on that slide.

    Args:
        slide_name (String): Name of scanned slide
        annotation_base_path (String): Path to top level directory containing slide annotations
    Returns:
        path_list (Path list): List of Path objects representing the annotations on the given slide
    """

    #TODO: Can I get rid of the _Scan1 somehow?
    full_annotation_dir = annotation_base_path + slide_name + "_Scan1"
    annotation_list = []
    
    for filename in os.listdir(full_annotation_dir):
        if filename.endswith(".csv"):
            annotation_file = full_annotation_dir + "/" + filename 
            current_annotation = read_annotation(annotation_file)
            annotation_list.append(current_annotation)
    
    path_list = list(map(construct_annotation_path, annotation_list))
    return path_list
    
def read_annotation(csv_path):
    """
    Loads the coordinates of an annotation created with QuPath
    and stored in a csv file

    Args:
        csv_path (str): Path to csv file containing annotation

    Returns:
        vertex_list (Nx2 numpy array): Nx2 array containing all 
                                       vertices in the annotaiton
    """

    f = open(csv_path)
    reader = csv.reader(f)
    row_count = sum(1 for line in reader)
    vertex_list = np.zeros((row_count, 2))
    f.close()   

    f = open(csv_path)
    reader = csv.reader(f)
    current_row = 0
    for row in reader:
        vertex_list[current_row,] = row
        current_row += 1

    return vertex_list

def construct_annotation_path(vertices):
    """
    Constructs a matplotlib Path object that represents the polygon
    with provided vertices

    Args:
        vertices (Nx2 numpy array): vertices of our polygon

    Returns:
        path (Path object): Path object representing our polygon
    
    """
    polygon = Path(vertices) 
    return polygon

def patch_in_paths(patch, path_list):
    """
    Utility function to check if a given patch object is contained within
    any of the annotation paths in path_list

    Args:
        patch (Patch): Patch object that we want to check
        path_list (Path list): List of annotation paths for a slide
    Returns:
        in_path (Boolean): True if patch contained within one of the paths in path_list
    """

    in_path = False
    for path in path_list:
        if patch.on_annotation_boundary(path):
            in_path = True

    return in_path
