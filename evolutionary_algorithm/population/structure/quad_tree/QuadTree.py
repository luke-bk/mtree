class QuadTree:
    """
    The QuadTree class represents a quad in 2D space, denoted by 4 points.
    This is useful for defining quads.
    """

    MAX_DEPTH = 2

    def __init__(self,
                 random_generator, area,
                 level, quad,
                 parent, child_number,
                 pop, generation,
                 best_fitness):
        """
        Initializes a QuadTree instance.
        :param random_generator: The random_generator for the whole algorithm.
        :param area: The region represented by this QuadTree.
        :param level: The level of the QuadTree.
        :param quad: The quadrant number (0 to 3).
        :param parent: The parent QuadTree.
        :param child_number: The child number of this QuadTree.
        :param pop: The subpopulation.
        :param generation: The generation.
        :param best_fitness: The current best fitness found.
        """
        self.random_generator = random_generator
        self.area = area
        self.level = level
        self.is_leaf = True
        self.has_split = False
        self.is_partial_leaf = False
        self.children = 0
        self.is_extinct = False
        self.parent = parent
        self.child_number = child_number
        self.quad = quad
        self.pop = pop
        self.quad_trees = []

        # if self.parent is not None:
        #     self.pop.parent = self.parent.pop
        #
        # if generation > 1:
        #     self.pop.truncate(population_size, state)

        # self.pop.level = level
        self.name = self.pop.get_name()
        # self.pop.name = self.name
        # self.pop.merge_counter = 0
        # self.pop.has_improved = False
        # self.pop.area = area
        # self.pop.created = generation
        # self.pop.best_current_score_before_evaluation = 0 if generation == 0 else best_fitness

        # self.pop.is_leaf = self.is_leaf

        # for ind in self.pop.individuals:
        #     ind.name = self.pop.name

    def select_for_split(self, generation, population):
        """
        Select a QuadTree for splitting.

        :param generation: The generation.
        :param population: The population.
        :param state: The EvolutionState.
        :return: True if the QuadTree is selected for splitting, False otherwise.
        """
        leaf_quads = self.get_leaf(None)

        if len(leaf_quads) > 1:
            return leaf_quads[self.random_generator.randint(len(leaf_quads))].split(generation,
                                                                                 population)
        else:
            return leaf_quads[0].split(generation, population)

    def select_for_merge(self, name, population):
        """
        Select a QuadTree for merging.

        :param name: The name of the QuadTree to merge.
        :param population: The population.
        :return: True if the QuadTree is selected for merging, False otherwise.
        """
        leaf_quads = self.get_leaf(None)

        for quad_tree in leaf_quads:
            if quad_tree.name == name:
                return quad_tree.merge(population)

        return False

    def split(self, generation, population):
        """
        Split the QuadTree.

        :param generation: The generation.
        :param population: The population.
        :return: True if the QuadTree is split, False otherwise.
        """
        if not self.has_split and self.is_leaf and not self.is_extinct and self.level < self.MAX_DEPTH:
            self.create_quadrants(generation, population)
            self.is_leaf = False
            self.has_split = True

            # for subpop in population.subpops:
            #     if subpop.name == self.name:
            #         self.pop = subpop
            #         break
            #
            # for subpop in population.subpops:
            #     if self.parent is not None and subpop.name == self.parent.name:
            #         subpop.individuals.clear()

            return True

        return False

    def merge(self, population):
        """
        Merge the QuadTree.

        :param population: The population.
        :return: True if the QuadTree is merged, False otherwise.
        """
        if self.is_leaf and self.parent is not None:
            self.is_extinct = True
            self.parent.is_partial_leaf = True
            self.parent.children -= 1

            if self.parent.children == 0:
                self.parent.is_partial_leaf = False
                self.parent.is_leaf = True

            # for subpop in population.subpops:
            #     if subpop.name == self.name:
            #         self.pop = subpop

            if self.parent.children == 3:
                population.subpops.append(self.parent.population)

            # self.parent.population_size += self.population_size
            # self.parent.pop.initial_size += self.population_size
            # temp_sub = None

            # for subpop in population.subpops:
            #     if subpop.name == self.parent.name:
            #         temp_sub = self.pop.deep_clone(self.pop)
            #
            #         for ind in temp_sub.individuals:
            #             subpop.individuals.append(ind)

            # # self.population_size = 0
            # self.parent.quad_trees.remove(self)
            # self.quad_trees = self.get_leaf(None)

            return True

        return False

    def create_quadrants(self, generation, population):
        """
        Create child quadrants.

        :param generation: The generation.
        :param population: The population.
        """
        child_population_size = len(population.chromosomes) // 4

        # for subpop in population.subpops:
        #     if subpop.name == self.name:
        #         self.pop = subpop
        #         break

        for i in range(4):
            region = self.area.get_quadrant(i)
            self.quad_trees.append(
                QuadTree(self.random_generator, region,
                         self.level + 1, i,
                         self, i,
                         self.pop.deep_clone(self.pop.get_name() + str(i), generation, self.pop), generation,
                         self.pop.best_fitness_at_creation)
            )

        self.children = 4

    def search_point_beta(self, search_region, matches, point):
        """
        Search for points within a region.

        :param search_region: The search region.
        :param matches: The list of matches.
        :param point: The point to search for.
        :return: A list of matching QuadTrees.
        """
        if matches is None:
            matches = []

        if matches and len(matches) == 1:
            return matches

        if not self.area.does_overlap(search_region):
            return matches

        if search_region.contains_point(point) and self.is_leaf and not self.is_extinct:
            matches.append(self)
        elif search_region.contains_point(point) and self.is_partial_leaf and not self.is_extinct:
            matches.append(self)

        if self.quad_trees and search_region.contains_point(point):
            for i in range(len(self.quad_trees)):
                self.quad_trees[i].search_point_beta(self.quad_trees[i].area, matches, point)

        return matches

    def get_leaf(self, matches):
        """
        Get the leaf QuadTrees.

        :param matches: The list of matches.
        :return: A list of leaf QuadTrees.
        """
        if matches is None:
            matches = []

        if self.is_leaf or (self.is_partial_leaf and not self.is_extinct):
            matches.append(self)

        if self.quad_trees:
            for i in range(len(self.quad_trees)):
                self.quad_trees[i].get_leaf(matches)

        return matches

    def get_leaf_populations(self):
        """
        Get the populations associated with leaf QuadTrees.

        :return: A list of populations.
        """
        subpopulations = []
        matches = []

        if (self.is_leaf or self.is_partial_leaf) and not self.is_extinct:
            matches.append(self)

        if self.quad_trees:
            for i in range(len(self.quad_trees)):
                self.quad_trees[i].get_leaf(matches)

        for a in matches:
            subpopulations.append(a.pop)

        return subpopulations

    def print_tree(self, depth_indicator=""):
        """
        Print the QuadTree in a tree-like structure.

        :param depth_indicator: The depth indicator for tree indentation.
        :return: A string representing the tree structure.
        """
        str = ""

        if depth_indicator == "":
            str += f"{self.level} --> {self.area} is leaf {self.is_leaf} is partial leaf {self.is_partial_leaf} " \
                   f"is extinct is {self.is_extinct}, name: {self.pop.get_name()}, pop size: " \
                   f"{len(self.pop.chromosomes)}\n"

        for i in range(len(self.quad_trees)):
            str += f"{depth_indicator}\t{self.quad_trees[i].level}: Q{self.quad_trees[i].child_number} --> " \
                   f"{self.quad_trees[i].area} is leaf {self.quad_trees[i].is_leaf} is partial leaf " \
                   f"{self.quad_trees[i].is_partial_leaf}, is extinct is {self.quad_trees[i].is_extinct}, name: " \
                   f"{self.quad_trees[i].population.get_name()}, pop size: {len(self.quad_trees[i].population.chromosomes)}\n"
            str += self.quad_trees[i].print_tree(depth_indicator + "\t")

        return str

    def is_leaf(self):
        """
        Check if the QuadTree is a leaf.

        :return: True if the QuadTree is a leaf, False otherwise.
        """
        return self.is_leaf

    def set_leaf(self, leaf):
        """
        Set the QuadTree as a leaf.

        :param leaf: True if the QuadTree is a leaf, False otherwise.
        """
        self.is_leaf = leaf
        self.pop.is_leaf = self.is_leaf

    def get_quad_trees(self):
        """
        Get the child QuadTrees.

        :return: A list of child QuadTrees.
        """
        return self.quad_trees

    def get_quad(self):
        """
        Get the current QuadTree.

        :return: The QuadTree.
        """
        return self

    def get_depth(self):
        """
        Get the depth of the QuadTree.

        :return: The depth.
        """
        return self.level

    def print_search_traverse_path(self):
        """
        Print the search traversal path.

        :return: The search traversal path as a string.
        """
        return str(self.search_traverse_path)

    def print_quad_tree(self):
        """
        Print the QuadTree information.

        :return: The QuadTree information as a string.
        """
        return f"{self.level}: Q{self.quad} --> {self.area} is extinct {self.is_extinct} child number: {self.level} {self.child_number}"

    def get_area(self):
        """
        Get the region represented by the QuadTree.

        :return: The region as a Region object.
        """
        return self.area

    def set_area(self, area):
        """
        Set the region represented by the QuadTree.

        :param area: The region as a Region object.
        """
        self.area = area
