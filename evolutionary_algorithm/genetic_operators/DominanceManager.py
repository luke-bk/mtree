def modify_dominance_top_and_bottom_50_percent(random_genrator, population, dom_increase_factor, dom_decrease_factor):
    """
    Modify the dominance values of the top and bottom 50% of individuals' expressed genes in the population.

    Args:
        population (Population): The population to modify.
        dom_increase_factor (float): The factor by which to increase the dominance values for the top 50%.
        dom_decrease_factor (float): The factor by which to decrease the dominance values for the bottom 50%.
    Returns:
        None
    """
    # Sort the population by fitness from highest to lowest
    population.sort(key=lambda x: x.get_fitness(), reverse=True)

    # Calculate the number of individuals in the top and bottom 50%
    total_individuals = len(population)
    top_50_percent = int(total_individuals * 0.5)

    # Increase dominance values for the top 50%
    for i in range(top_50_percent):
        chromosome = population[i]  # Get the chromosome at index 'i' from the sorted population
        highest_dominance_genes = chromosome.get_highest_dominance_genes()  # Get genes with the highest dominance

        for gene in highest_dominance_genes:
            current_dominance = gene.get_dominance()  # Get the current dominance value of the gene
            new_dominance = current_dominance + (
                    current_dominance * dom_increase_factor)  # Increase dominance by a factor
            new_dominance = min(max(new_dominance, 0), 1)  # Ensure new dominance stays within [0, 1]
            gene.set_dominance(new_dominance)  # Set the new dominance value for the gene

    # Decrease dominance values for the bottom 50%
    for i in range(total_individuals - top_50_percent, total_individuals):
        chromosome = population[i]  # Get the chromosome at index 'i' from the sorted population
        highest_dominance_genes = chromosome.get_highest_dominance_genes()  # Get genes with the highest dominance

        for gene in highest_dominance_genes:
            current_dominance = gene.get_dominance()  # Get the current dominance value of the gene
            new_dominance = current_dominance - (
                    current_dominance * dom_decrease_factor)  # Decrease dominance by a factor
            new_dominance = min(max(new_dominance, 0), 1)  # Ensure new dominance stays within [0, 1]
            gene.set_dominance(new_dominance)  # Set the new dominance value for the gene

    random_genrator.shuffle(population)


def modify_dominance_top_and_bottom_10_percent(random_genrator, population, dom_increase_factor, dom_decrease_factor,
                                               mut_increase_factor, mut_decrease_factor):
    """
    Modify the dominance values of the top and bottom 50% of individuals' expressed genes in the population.

    Args:
        population (Population): The population to modify.
        dom_increase_factor (float): The factor by which to increase the dominance values for the top 10%.
        dom_decrease_factor (float): The factor by which to decrease the dominance values for the bottom 10%.
        mut_increase_factor (float): The factor by which to increase the mutation values for the top 10%.
        mut_decrease_factor (float): The factor by which to decrease the mutation values for the bottom 10%.

    Returns:
        None
    """
    # Sort the population by fitness from highest to lowest
    population.sort(key=lambda x: x.get_fitness(), reverse=True)

    # Calculate the number of individuals in the top and bottom 50%
    total_individuals = len(population)
    top_10_percent = int(total_individuals * 0.1)

    # Increase dominance values for the top 10%
    for i in range(top_10_percent):
        chromosome = population[i]  # Get the chromosome at index 'i' from the sorted population
        highest_dominance_genes = chromosome.get_highest_dominance_genes()  # Get genes with the highest dominance

        for gene in highest_dominance_genes:
            # Handle dominance values
            current_dominance = gene.get_dominance()  # Get the current dominance value of the gene
            new_dominance = current_dominance + (
                    current_dominance * dom_increase_factor)  # Increase dominance by a factor
            new_dominance = min(max(new_dominance, 0), 1)  # Ensure new dominance stays within [0, 1]
            gene.set_dominance(new_dominance)  # Set the new dominance value for the gene
            # Handle mutation values
            current_mutation = gene.get_mutation()  # Get the current mutation value of the gene
            new_mutation = current_mutation - (
                    current_mutation * mut_decrease_factor)  # Decrease mutation by a factor
            new_mutation = min(max(new_mutation, 0.01), 1)  # Ensure new mutation stays within [0, 1]
            gene.set_mutation(new_mutation)  # Set the new mutation value for the gene

    # Decrease dominance values for the bottom 10%
    for i in range(total_individuals - top_10_percent, total_individuals):
        chromosome = population[i]  # Get the chromosome at index 'i' from the sorted population
        highest_dominance_genes = chromosome.get_highest_dominance_genes()  # Get genes with the highest dominance

        for gene in highest_dominance_genes:
            current_dominance = gene.get_dominance()  # Get the current dominance value of the gene
            new_dominance = current_dominance - (
                    current_dominance * dom_decrease_factor)  # Decrease dominance by a factor
            new_dominance = min(max(new_dominance, 0), 1)  # Ensure new dominance stays within [0, 1]
            gene.set_dominance(new_dominance)  # Set the new dominance value for the gene
            # Handle mutation values
            current_mutation = gene.get_mutation()  # Get the current mutation value of the gene
            new_mutation = current_mutation + (
                    current_mutation * mut_increase_factor)  # Increase mutation by a factor
            new_mutation = min(max(new_mutation, 0.01), 1)  # Ensure new mutation stays within [0, 1]
            gene.set_mutation(new_mutation)  # Set the new mutation value for the gene

    random_genrator.shuffle(population)

