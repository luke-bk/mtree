from evolutionary_algorithm.genetic_operators import MutationOperators


class BinaryTree:
    def __init__(self, random_generator, region, level, parent, child_number, population, max_depth):
        """
        Initialize a QuadTree1D instance.

        :param random_generator: The random generator for the algorithm.
        :param region: The region represented by this QuadTree1D.
        :param level: The level of the QuadTree.
        :param parent: The parent QuadTree1D.
        :param child_number: The child number of this QuadTree1D (0 for left, 1 for right).
        :param pop: The subpopulation associated with this QuadTree1D.
        :param max_depth: The maximum depth to which the tree can expand.
        """
        self.random_generator = random_generator  # Use this to generate all random for controlled experiments
        self.region = region  # A 1D region to represent where in the overall string this node represents
        self.level = level  # What depth of the binary tree
        self.is_leaf = True  # Determine whether this is a leaf node of the tree
        self.is_partial_leaf = False  # Determine whether this is a leaf node of the tree
        self.has_split = False  # Has this population split in the past
        self.is_extinct = False  # Has this population been merged
        self.parent = parent  # A link the parent node
        self.child_number = child_number
        self.population = population  # Every node represents a subpopulaton
        self.children = [None, None]  # Binary tree structure, left and right children
        self.max_depth = max_depth  # The maximum depth the tree can split to

    def select_for_split(self, generation):
        """
        Select a leaf node for splitting.

        :param generation: The generation.
        :return: True if the QuadTree1D is selected for splitting, False otherwise.
        """
        leaf_quads = self.get_leaf(None)  # We only split the leaf nodes of the tree

        if len(leaf_quads) > 1:  # checks if it is not the root node
            return leaf_quads[self.random_generator.randint(0, len(leaf_quads))].split(generation)
        else:  # We are the root node, there's only one to split
            return leaf_quads[0].split(generation)

    def select_for_merge(self, name):
        """
        Select a leaf node for merging.

        :param name: The name of the node to merge.
        :return: True if the node is selected for merging, False otherwise.
        """
        leaf_quads = self.get_leaf(None)  # We only merge the leaf nodes of the tree

        for node in leaf_quads:
            if node.population.get_name() == name:
                return node.merge()

        return False

    def split(self, generation):
        """
        Split the QuadTree1D.

        :param generation: The generation.
        :return: True if the node is split, False otherwise.
        """
        if not self.has_split and self.is_leaf and not self.is_extinct and self.level < self.max_depth:
            self.create_children(generation)
            self.is_leaf = False
            self.has_split = True

            return True

        return False

    def merge(self):
        """
        Merge the QuadTree1D.

        :return: True if the QuadTree1D is merged, False otherwise.
        """
        if self.is_leaf and self.parent is not None:  # It is not the root node
            self.is_extinct = True  # We are merging, so this leaf is now extinct
            self.parent.children.remove(self)  # Delete self from tree

            if self.parent.has_children():  # The parent DOES have children
                self.parent.is_partial_leaf = True  # If there is still children, it is a partial leaf
                self.parent.population.chromosomes = self.population.chromosomes  # Parent gets the child's chromosomes of same length
            else:  # The parent doesn't have children
                self.parent.is_leaf = True  # Is a true leaf
                # Parent needs to merge two chromosomes to create one of same length (is left or right chromosome?)
                for x, child in enumerate(self.parent.population.chromosomes):
                    child.merge_chromosome(self.population.chromosomes[x], self.child_number)
                # Fill rest of population with mutated clones

                for i in range(len(self.parent.population.chromosomes)):
                    child_clone = self.parent.population.chromosomes[i].clone()
                    MutationOperators.perform_bit_flip_mutation(self.random_generator, child_clone)
                    self.parent.population.add_chromosome(child_clone)
            return True

        return False

    def create_children(self, generation):
        """
        Create child nodes (children) for the QuadTree1D.

        :param generation: The generation.
        """
        # Select the parents who will survive the split (cloned copies and split to right size)
        surviving_parent_chromosomes = self.population.split_population(generation, 0)

        # Clear out the parent population
        self.population.chromosomes.clear()

        # Create the two child nodes in the tree, populating them with the cut in half parent chromosomes
        for i in range(2):
            region = self.region.get_subregion(i)
            self.children[i] = BinaryTree(
                self.random_generator, region,
                self.level + 1, self,
                i, surviving_parent_chromosomes[i],
                self.max_depth
            )

    def search_point(self, search_region, matches, point):
        """
        Search for points within a search_region.

        :param search_region: The search region.
        :param matches: The list of matching QuadTree1D instances.
        :param point: The point to search for.
        :return: A list of matching QuadTree1D instances.
        """
        if matches is None:
            matches = []

        if not self.region.does_overlap(search_region):
            return matches

        if search_region.contains_point(point) and self.is_leaf and not self.is_extinct:
            matches.append(self)

        if self.has_children() and search_region.contains_point(point):
            for child in self.children:
                if child is not None:
                    child.search_point(search_region, matches, point)

        return matches

    def get_leaf(self, matches):
        """
        Get the leaf QuadTree1D instances.

        :param matches: The list to store leaf BinaryTree instances.
        :return: A list of leaf BinaryTree instances.
        """
        if matches is None:
            matches = []

        if self.is_leaf or self.is_partial_leaf and not self.is_extinct:
            matches.append(self)

        if self.has_children():
            for child in self.children:
                child.get_leaf(matches)

        return matches

    def get_leaf_populations(self):
        """
        Get the populations associated with leaf BinaryTree instances.

        :return: A list of populations.
        """
        subpopulations = []
        matches = []

        if self.is_leaf and not self.is_extinct:
            matches.append(self)

        if self.has_children():
            for child in self.children:
                child.get_leaf(matches)

        for quad_tree in matches:
            subpopulations.append(quad_tree.population)

        return subpopulations

    def has_children(self):
        """
        Check if the QuadTree1D has children.

        :return: True if the QuadTree1D has children, False otherwise.
        """
        return any(child is not None for child in self.children)

    def print_tree(self, depth_indicator=""):
        """
        Print the QuadTree1D in a tree-like structure.

        :param depth_indicator: The depth indicator for tree indentation.
        :return: A string representing the tree structure.
        """
        str = ""

        if depth_indicator == "":
            str += f"{self.level} --> {self.region} is leaf {self.is_leaf} is extinct {self.is_extinct}, " \
                   f"name: {self.population.get_name()}, pop size: {len(self.population.chromosomes)}\n"

        for i, child in enumerate(self.children):
            if child is not None:
                str += f"{depth_indicator}\t{child.level}: {i} --> {child.region} is leaf {child.is_leaf} " \
                       f"is extinct {child.is_extinct}, name: {child.population.get_name()}, " \
                       f"pop size: {len(child.population.chromosomes)}\n"
                str += child.print_tree(depth_indicator + "\t")

        return str

    def print_self(self):
        """
        Print the self.

        :return: The QuadTree information as a string.
        """
        return f"{self.population.get_name()}: Q{self.region} len {len(self.population.chromosomes)}"

    # Other methods can remain the same

    def get_region(self):
        """
        Get the region represented by the BinaryTree.

        :return: The region as a Region1D object.
        """
        return self.region
