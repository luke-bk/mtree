# def modify_dominance_top_and_bottom_50_percent(population, increase_factor, decrease_factor):
#     """
#     Modify the dominance values of the top and bottom 50% of individuals' expressed genes in the population.
#
#     Args:
#         population (Population): The population to modify.
#         increase_factor (float): The factor by which to increase the dominance values for the top 50%.
#         decrease_factor (float): The factor by which to decrease the dominance values for the bottom 50%.
#
#     Returns:
#         None
#     """
#     # Sort the population by fitness from highest to lowest
#     population.sort(key=lambda x: x.get_fitness(), reverse=True)
#
#     # Calculate the number of individuals in the top and bottom 50%
#     total_individuals = len(population)
#     top_50_percent = int(total_individuals * 0.5)
#
#     # Increase dominance values for the top 50%
#     for i in range(top_50_percent):
#         chromosome = population[i]  # Get the chromosome at index 'i' from the sorted population
#         expressed_genes = chromosome.express_highest_dominance()  # Get the expressed genes of the chromosome
#
#         for j in range(len(expressed_genes)):
#             gene_index = j  # Assuming gene_index corresponds to the gene in the expressed_genes list
#             gene = chromosome.part_chromosomes[0].genes[gene_index]  # Assuming it's the first part chromosome
#             current_dominance = gene.get_dominance()  # Get the current dominance value of the gene
#             new_dominance = current_dominance + (current_dominance * increase_factor)  # Increase dominance by a factor
#             new_dominance = min(max(new_dominance, 0), 1)  # Ensure new dominance stays within [0, 1]
#             gene.set_dominance(new_dominance)  # Set the new dominance value for the gene
#
#     # Decrease dominance values for the bottom 50%
#     for i in range(total_individuals - top_50_percent, total_individuals):
#         chromosome = population[i]  # Get the chromosome at index 'i' from the sorted population
#         expressed_genes = chromosome.express_highest_dominance()  # Get the expressed genes of the chromosome
#
#         for j in range(len(expressed_genes)):
#             gene_index = j  # Assuming gene_index corresponds to the gene in the expressed_genes list
#             gene = chromosome.part_chromosomes[0].genes[gene_index]  # Assuming it's the first part chromosome
#             current_dominance = gene.get_dominance()  # Get the current dominance value of the gene
#             new_dominance = current_dominance - (current_dominance * decrease_factor)  # Decrease dominance by a factor
#             new_dominance = min(max(new_dominance, 0), 1)  # Ensure new dominance stays within [0, 1]
#             gene.set_dominance(new_dominance)  # Set the new dominance value for the gene


def modify_dominance_top_and_bottom_50_percent(random_genrator, population, increase_factor, decrease_factor):
    """
    Modify the dominance values of the top and bottom 50% of individuals' expressed genes in the population.

    Args:
        population (Population): The population to modify.
        increase_factor (float): The factor by which to increase the dominance values for the top 50%.
        decrease_factor (float): The factor by which to decrease the dominance values for the bottom 50%.

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
                        current_dominance * increase_factor)  # Increase dominance by a factor
            new_dominance = min(max(new_dominance, 0), 1)  # Ensure new dominance stays within [0, 1]
            gene.set_dominance(new_dominance)  # Set the new dominance value for the gene

    # Decrease dominance values for the bottom 50%
    for i in range(total_individuals - top_50_percent, total_individuals):
        chromosome = population[i]  # Get the chromosome at index 'i' from the sorted population
        highest_dominance_genes = chromosome.get_highest_dominance_genes()  # Get genes with the highest dominance

        for gene in highest_dominance_genes:
            current_dominance = gene.get_dominance()  # Get the current dominance value of the gene
            new_dominance = current_dominance - (
                        current_dominance * decrease_factor)  # Decrease dominance by a factor
            new_dominance = min(max(new_dominance, 0), 1)  # Ensure new dominance stays within [0, 1]
            gene.set_dominance(new_dominance)  # Set the new dominance value for the gene

    random_genrator.shuffle(population)
