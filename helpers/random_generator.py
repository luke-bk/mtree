import numpy as np


class RandomGenerator:
    def __init__(self, seed=None):
        """
        Initialize a random number generator.

        Args:
            seed (int, optional): Seed value for reproducible random numbers.
        """
        self.seed = seed
        self.random_generator = np.random.default_rng(seed)

    def randint(self, low, high, size=None):
        """
        Generate random integers between 'low' (inclusive) and 'high' (exclusive).

        Args:
            low (int): Lower bound of the random integers.
            high (int): Upper bound of the random integers.
            size (int or tuple of int, optional): Output shape. If None, a single random integer is generated.

        Returns:
            int or ndarray of int: Random integers.
        """
        return self.random_generator.integers(low, high, size)

    def uniform(self, low, high, size=None):
        """
        Generate random numbers from a uniform distribution between 'low' and 'high'.

        Args:
            low (float): Lower bound of the uniform distribution.
            high (float): Upper bound of the uniform distribution.
            size (int or tuple of int, optional): Output shape. If None, a single random number is generated.

        Returns:
            float or ndarray of float: Random numbers from a uniform distribution.
        """
        return self.random_generator.uniform(low, high, size)

    def choice(self, a, size=None, replace=True, p=None):
        """
        Generate a random sample from a given array.

        :param a: Input array from which to sample.
        :param size: Number of samples to generate. If None, a single sample is generated.
        :param replace: Whether sampling is done with replacement (True) or without replacement (False).
        :param p: Probabilities associated with each entry in 'a'. If None, all entries have uniform probabilities.
        :return: A random sample from the input array.
        """
        return self.random_generator.choice(a, size=size, replace=replace, p=p)

    def shuffle(self, data):
        """
        Shuffle the elements in a list randomly.

        Args:
            data (list): The list to be shuffled.

        Returns:
            list: The shuffled list.
        """
        shuffled_data = data.copy()  # Create a copy to avoid modifying the original list
        self.random_generator.shuffle(shuffled_data)  # Shuffle the copied list
        return shuffled_data

    def random(self):
        return self.random_generator.random()
