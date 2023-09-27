class Region1D:
    """
    The Region1D class represents a region in 1D space.
    """

    def __init__(self, x1, x2):
        """
        Initializes a Region1D object with the left and right coordinates.

        :param x1: The left x-coordinate.
        :param x2: The right x-coordinate.
        :raises ValueError: If x1 is not strictly less than x2.
        """
        # if x1 >= x2:
        #     raise ValueError("x1 should be strictly less than x2")
        self.x1 = x1
        self.x2 = x2

    def get_subregion(self, subregion_index):
        """
        Get a subregion of the Region1D based on the specified subregion index.

        :param subregion_index: An integer representing the subregion (0=left, 1=right).
        :return: A new Region1D representing the specified subregion.
        """
        region_width = self.x2 - self.x1

        if subregion_index == 0:
            # Return the left half of the region
            return Region1D(self.x1, self.x1 + region_width // 2)
        elif subregion_index == 1:
            # Return the right half of the region
            return Region1D(self.x1 + region_width // 2 + 1, self.x2)
        else:
            return None  # Invalid subregion index

    def contains_point(self, point):
        """
        Check if the Region1D contains a given Point1D.

        :param point: A Point1D object.
        :return: True if the Region1D contains the Point1D, otherwise False.
        """
        return self.x1 <= point.get_x() < self.x2

    def does_overlap(self, test_region):
        """
        Check if the Region1D overlaps with another Region1D.

        :param test_region: Another Region1D object.
        :return: True if the Region1D overlaps with the other Region1D, otherwise False.
        """
        return not (test_region.get_x2() <= self.x1 or test_region.get_x1() >= self.x2)

    def get_x1(self):
        """
        Get the left x-coordinate of the Region1D.

        :return: The left x-coordinate.
        """
        return self.x1

    def get_x2(self):
        """
        Get the right x-coordinate of the Region1D.

        :return: The right x-coordinate.
        """
        return self.x2

    def __str__(self):
        """
        Return a string representation of the Region1D.

        :return: A string representation of the Region1D.
        """
        return f"[(x1={self.x1}), (x2={self.x2})]"
