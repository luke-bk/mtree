from evolutionary_algorithm.Chromosome import Chromosome
from evolutionary_algorithm.MtreeGeneReal import MtreeGeneReal

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    realGene = MtreeGeneReal(0.0, 1.0, 0.0, 1.0, 0.0, 1.0)

    print(f"My value is {realGene.gene_value}")
