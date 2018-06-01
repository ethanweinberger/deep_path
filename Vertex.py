# Class for representing a vertex in a QuPath annotation

class Vertex(object):

    def __init__(self, x, y):
        """
        Initializer for Vertex class

        Args:
            x (float): x-coordinate of vertex
            y (float): y-coordinate of vertex
        Returns:
            Vertex object
    
        """

        self.x = x
        self.y = y
