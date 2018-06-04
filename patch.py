# Class for slide patches

from enum import Enum
from matplotlib.path import Path

class Patch(object):

    class Patch_Vertex(Enum):
        TOP_LEFT     = 1
        TOP_RIGHT    = 2
        BOTTOM_LEFT  = 3
        BOTTOM_RIGHT = 4

    def __init__(self, img, coords):
        """
        Initializer for Patch object

        Args:
            img (numpy uint8 array): Image contents of our patch
            coords: Coordinates of the image as returned by openslide DeepZoomGenerator's
                    get_tile_coordinates method (i.e., of the format 
                    ((x_topLeft, y_topLeft), level, (width, height))).

        Returns:
            Patch object
        
        """

        self.img = img
        
        self.top_left_vertex = coords[0]
        top_left_x = self.top_left_vertex[0]
        top_left_y = self.top_left_vertex[1]

        patch_dimensions = coords[2]
        patch_width = patch_dimensions[0]
        patch_height = patch_dimensions[1]

        self.top_right_vertex = (top_left_x + patch_width, top_left_y)
        self.bottom_left_vertex = (top_left_x, top_left_y + patch_height)
        self.bottom_right_vertex = (top_left_x + patch_width, top_left_y + patch_height)

    def vertex_in_annotation(self, patch_vertex, annotation):
        """
        Checks to see if a given patch vertex is contained within a provided
        annotation

        Args:
            patch_vertex (Patch_Vertex): Enum representing which vertex of our patch we want to check
            annotation: (matplotlib.path.Path object): Path object representing the polygonal region enclosed
                                                       by a QuPath annotation
        
        Returns:
            in_annotation (bool): Is the given vertex in our polygon?

        """

        in_annotation = False
        if patch_vertex == Patch_Vertex.TOP_LEFT:
            in_annotation = annotation.contains_point(self.top_left_vertex)

        elif patch_vertex == Patch_Vertex.TOP_RIGHT:
            in_annotation = annotation.contains_point(self.top_right_vertex)

        elif patch_vertex == Patch_Vertex.BOTTOM_LEFT:
            in_annotation = annotation.contains_point(self.bottom_left_vertex)

        elif patch_vertex == Patch_Vertex.BOTTOM_RIGHT:
            in_annotation = annotation.contains_point(self.bottom_right_vertex)
        else:
            raise TypeError("Invalid vertex type provided to vertex_in_annotation")

        return in_annotaiton


