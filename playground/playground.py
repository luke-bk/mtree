# Define a class called Population to represent a population node in a tree.
class Population:
    # Constructor for Population class, takes a name and an optional parent.
    def __init__(self, name, parent=None):
        # Store the name of the population.
        self.name = name
        # Store a reference to the parent population (default is None for root).
        self.parent = parent

    # Method to split the current population into two child populations.
    def split(self):
        # Generate names for the two children by appending "0" and "1" to the current name.
        child1_name = self.name + "0"
        child2_name = self.name + "1"
        # Create child Population objects with their names and a reference to their parent (self).
        child1 = Population(child1_name, self)
        child2 = Population(child2_name, self)
        # Return the two child Population objects.
        return child1, child2

    # Placeholder for a merge method (not implemented in this code).
    def merge(self):
        pass


# Define a class called PopulationManager to manage populations and their state.
class PopulationManager:
    # Constructor for PopulationManager class, initializes active and inactive lists.
    def __init__(self):
        # List to store active populations.
        self.active = []
        # List to store inactive populations.
        self.inactive = []

    # Method to add a population to the active list.
    def add_population(self, population):
        # Append the given population object to the active list.
        self.active.append(population)

    # Method to split a population, replacing it with its children in the active list.
    def split_population(self, population):
        # Check if the population is in the active list.
        if population in self.active:
            # Get the index of the population in the active list.
            index = self.active.index(population)
            # Remove the population from the active list.
            self.active.remove(population)
            # Split the population into two children.
            child1, child2 = population.split()
            # Insert the children at the same index as the original population.
            self.active.insert(index, child2)
            self.active.insert(index, child1)
            # Add the original population to the inactive list.
            self.inactive.append(population)
        else:
            # Raise an error if the population is not found in the active list.
            raise ValueError("Population not found in the active list.")

    # Method to merge a population, moving it from the inactive list to the active list.
    def merge_population(self, population):
        # Get the index of the population in the active list.
        index = self.active.index(population)
        # Check if the population is in the inactive list.
        if population.parent in self.inactive:
            # Remove the parent population from the inactive list.
            self.inactive.remove(population.parent)
            # Add the parent population back to the active list.
            self.active.insert(index, population.parent)

        # Remove the population from the active list.
        self.inactive.append(population)
        # Add the population back to inactive list.
        self.active.remove(population)


# Example usage:
manager = PopulationManager()
root_population = Population("0")
manager.add_population(root_population)
print("-----------start-------------")
# Check the current active and inactive populations
print("Active populations:", [p.name for p in manager.active])
print("Inactive populations:", [p.name for p in manager.inactive])

# Split the root population
# child1, child2 = root_population.split()
manager.split_population(root_population)
print("-----------split-------------")

# Check the current active and inactive populations
print("Active populations:", [p.name for p in manager.active])
print("Inactive populations:", [p.name for p in manager.inactive])
manager.split_population(manager.active[0])
print("-----------split-------------")

# Check the current active and inactive populations
print("Active populations:", [p.name for p in manager.active])
print("Inactive populations:", [p.name for p in manager.inactive])

# Merge one of the children back into the active list
manager.merge_population(manager.active[1])
print("-----------merge-------------")


# Check the current active and inactive populations
print("Active populations:", [p.name for p in manager.active])
print("Inactive populations:", [p.name for p in manager.inactive])
