def update_elite(leaf_node, complete_solution, sub_solution_index):
    """
    Update the elite for a population within a leaf node and perform elitism.

    Args:
        leaf_node: The leaf node containing the population.
        complete_solution: The complete solution.
        sub_solution_index: The index where to insert in the complete solution.

    Returns:
        None
    """
    current_best = leaf_node.population.get_chromosome_with_max_fitness()

    if leaf_node.population.elite is None or leaf_node.population.elite.get_fitness() <= current_best.get_fitness():
        leaf_node.population.elite = current_best
        leaf_node.population.elite_collaborators = complete_solution
        leaf_node.population.index_in_collaboration = sub_solution_index

    # Perform elitism by adding the elite individual to the population
    leaf_node.population.chromosomes[-1] = leaf_node.population.elite