class Point:
    """
    The Point class represents a coordinate in 2D space.
    This is useful for finding out where an agent lies in the QuadTree.
    """

    def __init__(self, x, y):
        """
        Initializes a Point object with given x and y coordinates.

        :param x: The x-coordinate.
        :param y: The y-coordinate.
        """
        self.x = x
        self.y = y

    def get_x(self):
        """
        Get the x-coordinate of the Point.

        :return: The x-coordinate.
        """
        return self.x

    def get_y(self):
        """
        Get the y-coordinate of the Point.

        :return: The y-coordinate.
        """
        return self.y

    def __str__(self):
        """
        Return a string representation of the Point in the format [x, y].

        :return: A string representation of the Point.
        """
        return "[" + str(self.x) + " , " + str(self.y) + "]"
