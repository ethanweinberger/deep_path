# Class for representing an annotation created in QuPath

class Annotation(object):

    def __init__(self, vertices):
        """
        Initializer for Annotation class

        Args:
            vertices (Vertex list): list of vertices that defines the annotation
        Returns:
            Annotation object

        """

        self.vertices = vertices
