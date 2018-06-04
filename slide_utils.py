import numpy as np
from PIL import Image

import openslide
import csv
from openslide.deepzoom import DeepZoomGenerator
from Annotation import Annotation
from Vertex import Vertex
from matplotlib.path import Path

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

def get_patches_from_slide(slide, tile_size=512, overlap=0, limit_bounds=False):
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
            new_tile = np.array(tiles.get_tile(level, (x,y)), dtype=np.uint8)
            if np.shape(new_tile) == (tile_size, tile_size, 3):
                patches.append(new_tile)
                count += 1

            x += 1
        y += 1
        x = 0
    return patches

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

def construct_polygon(vertices):
    """
    Constructs a matplotlib Path object that represents the polygon
    with provided vertices

    Args:
        vertices (Nx2 numpy array): vertices of our polygon

    Returns:
        polygon (Path object): Path object representing our polygon
    
    """
    
    return Path(vertices)
    



