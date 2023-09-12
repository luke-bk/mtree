class Point1D:
    """
    The Point1D class represents a point in 1D space.
    """

    def __init__(self, x):
        """
        Initializes a Point1D object with the given x coordinate.

        :param x: The x-coordinate.
        """
        self.x = x

    def get_x(self):
        """
        Get the x-coordinate of the Point1D.

        :return: The x-coordinate.
        """
        return self.x

    def __str__(self):
        """
        Return a string representation of the Point1D.

        :return: A string representation of the Point1D.
        """
        return str(self.x)
