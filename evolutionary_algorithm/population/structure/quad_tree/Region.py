class Region:
    """
    The Region class represents a quad in 2D space, denoted by four points.

    x1 = 0, y1 = 0, x2 = 3, y2 = 3
    [0,0][1,0][2,0][3,0]
    [0,1][1,1][2,1][3,1]
    [0,2][1,2][2,2][3,2]
    [0,3][1,3][2,3][3,3]

    This is useful for defining quads.
    """

    def __init__(self, x1, y1, x2, y2):
        """
        Initializes a Region object with top-left and bottom-right coordinates.

        :param x1: The top-left x-coordinate.
        :param y1: The top-left y-coordinate.
        :param x2: The bottom-right x-coordinate.
        :param y2: The bottom-right y-coordinate.
        :raises ValueError: If (x1, y1) is not strictly less than (x2, y2).
        """
        if x1 >= x2 or y1 >= y2:
            raise ValueError("(x1, y1) should be strictly less than (x2, y2)")
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    def get_quadrant(self, quadrant_index):
        """
        Get a quadrant of the Region based on the specified quadrant index.

        :param quadrant_index: An integer representing the quadrant (0=SW, 1=NW, 2=NE, 3=SE).
        :return: A new Region representing the specified quadrant.
        """
        quadrant_width = (self.x2 - self.x1) // 2
        quadrant_height = (self.y2 - self.y1) // 2

        if quadrant_index == 0:
            return Region(self.x1, self.y1, self.x1 + quadrant_width, self.y1 + quadrant_height)
        elif quadrant_index == 1:
            return Region(self.x1, self.y1 + quadrant_height, self.x1 + quadrant_width, self.y2)
        elif quadrant_index == 2:
            return Region(self.x1 + quadrant_width, self.y1 + quadrant_height, self.x2, self.y2)
        elif quadrant_index == 3:
            return Region(self.x1 + quadrant_width, self.y1, self.x2, self.y1 + quadrant_height)
        else:
            return None

    def contains_point(self, point):
        """
        Check if the Region contains a given Point.

        :param point: A Point object.
        :return: True if the Region contains the Point, otherwise False.
        """
        return self.x1 <= point.get_x() < self.x2 and self.y1 <= point.get_y() < self.y2

    def does_overlap(self, test_region):
        """
        Check if the Region overlaps with another Region.

        :param test_region: Another Region object.
        :return: True if the Regions overlap, otherwise False.
        """
        if test_region.get_x2() < self.x1 or test_region.get_x1() > self.x2:
            return False
        if test_region.get_y2() < self.y1 or test_region.get_y1() > self.y2:
            return False
        return True

    def __str__(self):
        """
        Return a string representation of the Region.

        :return: A string representation of the Region.
        """
        return f"[Region (x1={self.x1}, y1={self.y1}), (x2={self.x2}, y2={self.y2})]"

    def get_x1(self):
        """
        Get the top-left x-coordinate of the Region.

        :return: The top-left x-coordinate.
        """
        return self.x1

    def get_y1(self):
        """
        Get the top-left y-coordinate of the Region.

        :return: The top-left y-coordinate.
        """
        return self.y1

    def get_x2(self):
        """
        Get the bottom-right x-coordinate of the Region.

        :return: The bottom-right x-coordinate.
        """
        return self.x2

    def get_y2(self):
        """
        Get the bottom-right y-coordinate of the Region.

        :return: The bottom-right y-coordinate.
        """
        return self.y2
