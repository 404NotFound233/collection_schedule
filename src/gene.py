'''
    定义染色体类
'''

class Gene:
    fitness = 0.0
    chromosome = ""
    def __init__(self, chromosome, fitness):
        self.fitness = fitness
        self.chromosome = chromosome
