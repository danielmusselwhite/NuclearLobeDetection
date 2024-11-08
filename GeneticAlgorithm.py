import random
import matplotlib.pyplot as plt
import seaborn as sns
from utils import Color  # Import Color class from Utils.py
from pprint import pprint

# Initialize the Color object from Utils.py
color_printer = Color()

# Define hyperparameters to tune and their ranges
# TODO - find more hyperparemeters to tune and optimal ranges
paramRanges = {
    "epochs": (int, (10, 100)),
    "batch": (int, (8, 32)),
    "lr0": (float, (1e-5, 1e-2)),
    "momentum": (float, (0.8, 0.99)),
    "weight_decay": (float, (0.0001, 0.01)),
    "imgsz": (int, (320, 1280))
}

# Generate a random set of hyperparameters within defined ranges
def generateRandomParams(baseParams):
    params = baseParams.copy()
    # Generate random values for each parameter within the specified range
    for param, (paramType, (low, high)) in paramRanges.items():
        if paramType == int:
            params[param] = random.randint(low, high)  # Random integer between low and high
        elif paramType == float:
            params[param] = random.uniform(low, high)  # Random float between low and high
    return params

# Genetic algorithm optimization function
def geneticAlgorithmOptimize(trainFunc, valFunc, baseParams, optimizationMetric='metrics/mAP50-95(B)', generations=3, populationSize=4):
    # TODO - add early termination if: 1. no improvement in n x generations, or 2. we found a good enough model (0.95 mAP?)

    # Generate initial population by randomly sampling hyperparameters
    population = [generateRandomParams(baseParams) for _ in range(populationSize)]
    allGenerations = []  # List to track all generations' (params, scores)
    bestModel = None  # To track the best model and its details (score, generation, index)
    bestScore = -float('inf')  # Initialize with very low score
    generationScores = []  # To track average fitness for each generation

    for generation in range(generations):
        color_printer.print(f"Generation {generation + 1}/{generations}", color="red", bold=True, underline=True)

        # Store current generation's (params, score) pairs
        thisGeneration = []

        # Evaluate population fitness (validation score) for each individual in the population
        for i, params in enumerate(population):
            color_printer.print(f"Training individual {i+1}/{populationSize} from Generation {generation + 1}/{generations}", color="red")
            model = trainFunc(params)
            metrics = valFunc(model)
            fitness = metrics[optimizationMetric]
            thisGeneration.append((fitness, params))
            color_printer.print(f"Fitness Score ({optimizationMetric}): {fitness}", color="green")

            # Track the best model
            if fitness > bestScore:
                bestScore = fitness
                bestModel = (fitness, params, generation, i)  # Store (score, params, generation, index)

        # Update allGenerations with this generation's results
        allGenerations.append(thisGeneration)

        # Calculate the average fitness of this generation for trend plotting
        avgFitness = sum(fitness[0] for fitness in thisGeneration) / len(thisGeneration)
        generationScores.append(avgFitness)

        # Select top performers (e.g., top 50%) to serve as parents for the next generation
        thisGeneration.sort(reverse=True, key=lambda x: x[0]) # Sort by fitness score in descending order
        topPerformers = thisGeneration[:populationSize // 2] # Select top 50% of performers

        # Generate new population with crossover and mutation
        newPopulation = []
        for _ in range(populationSize):
            parent1, parent2 = random.sample(topPerformers, 2) # Randomly select 2 parents from top performers
            childParams = crossover(parent1[1], parent2[1]) # Combine parameters of two parents
            childParams = mutate(childParams) # Slightly adjust parameters
            newPopulation.append(childParams) # Add child to new population

        # Replace old population with new population
        population = newPopulation

    # Return the best parameters from all generations
    color_printer.print(f"The best model was individual {bestModel[3] + 1} from generation {bestModel[2] + 1} with a fitness score of {bestModel[0]}", color="magenta")

    # Generate graphs
    plotGraphs(generationScores, allGenerations)

    return bestModel[1]  # Returning the best parameters

# Crossover function to combine parameters of two parents
# Crossover type = Uniform (each parent has equal chance for each parameter)
def crossover(params1, params2):
    childParams = {}
    for key in params1:
        if random.random() > 0.5:
            color_printer.print(f"Inheriting parameter from parent 1: {key}", color="cyan")
            childParams[key] = params1[key]
        else:
            color_printer.print(f"Inheriting parameter from parent 2: {key}", color="cyan")
            childParams[key] = params2[key]
    return childParams

# Mutation function to slightly adjust parameters
# Mutation Type = Gaussian (add random noise to each parameter)
# Do it based on max of n% change of the min and max for that parameter so it scales with the range (eg for a param with 0-1 then mutation can be at max 0.1% change, vs for range of 1-10000 then mutation can be at max 1000 change)
# Mutation Rate = (probability of mutating each parameter)
# Mutation Size = (max % change (in relation to the range, not current value) in parameter value)
def mutate(params, mutationRate=0.2, mutationSize=0.3):
    mutatedParams = {}
    for key, value in params.items():
        # if this parameter is not in the range dictionary, skip it as it is not tunable
        if key not in paramRanges:
            mutatedParams[key] = value
            continue

        # else it is tunable randomly decide whether to mutate this parameter
        if random.random() < mutationRate:
            low, high = paramRanges[key][1]
            mutationRange = (high - low) * mutationSize
            mutation = random.uniform(-mutationRange, mutationRange) # Random increase vs decrease percent
            mutatedValue = value + mutation
            mutatedValue = max(low, min(high, mutatedValue))  # Ensure value is within range
            
            # Ensure the mutated value is of the correct type
            if isinstance(value, int):
                mutatedValue = int(round(mutatedValue))  # Round to nearest integer if it was an integer
            
            color_printer.print(f"Mutating parameter {key} from {value} to {mutatedValue} ({mutation / (high - low) * 100}% (range {low} to {high}))", color="yellow")
            mutatedParams[key] = mutatedValue
        else:
            mutatedParams[key] = value
    
    return mutatedParams

# Function to plot graphs showing average fitness over generations
def plotGraphs(generationScores, allGenerations):
    # Plot average fitness per generation
    plt.figure(figsize=(10, 6))
    plt.plot(generationScores, marker='o', label='Average Fitness')
    plt.title('Average Fitness per Generation')
    plt.xlabel('Generation')
    plt.ylabel('Average Fitness')
    plt.grid(True)
    plt.legend()
    plt.show()
    plt.savefig('AverageFitnessPerGeneration.png')

    # Plot boxplot for fitness distribution per generation
    fitnessPerGeneration = [[fitness[0] for fitness in gen] for gen in allGenerations]
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=fitnessPerGeneration)
    plt.title('Fitness Distribution per Generation')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.grid(True)
    plt.show()
    plt.savefig('FitnessDistributionPerGeneration.png')

