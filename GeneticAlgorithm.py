import random
import matplotlib.pyplot as plt
import seaborn as sns
from utils import Color  # Import Color class from Utils.py
import pprint

# Initialize the Color object from Utils.py
color_printer = Color()

# Define hyperparameters to tune and their ranges
paramRanges = {
    "epochs": (10, 100),
    "batch": (8, 32),
    "lr0": (1e-5, 1e-2),
    "momentum": (0.8, 0.99),
    "weight_decay": (0.0001, 0.01),
    "imgsz": (320, 1280)
}

# Generate a random set of hyperparameters within defined ranges
def generateRandomParams(baseParams):
    params = baseParams.copy()
    # Generate random values for each parameter within the specified range
    for param, (low, high) in paramRanges.items():
        params[param] = random.uniform(low, high) if isinstance(low, float) else random.randint(low, high) # Random float or int between the low and high for each parameter
    return params

# Genetic algorithm optimization function
def geneticAlgorithmOptimize(trainFunc, valFunc, baseParams, optimizationMetric='metrics/mAP50-95(B)', generations=4, populationSize=4):
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
            color_printer.print(f"Training inidividual {i+1}/{populationSize}", color="red")
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
        thisGeneration.sort(reverse=True, key=lambda x: x[0])
        topPerformers = thisGeneration[:populationSize // 2]

        # Generate new population with crossover and mutation
        newPopulation = []
        for _ in range(populationSize):
            parent1, parent2 = random.sample(topPerformers, 2)
            childParams = crossover(parent1[1], parent2[1])
            childParams = mutate(childParams)
            newPopulation.append(childParams)

        # Replace old population with new population
        population = newPopulation

    # Return the best parameters from all generations
    color_printer.print(f"The best model was individual {bestModel[3] + 1} from generation {bestModel[2] + 1} with a fitness score of {bestModel[0]}", color="yellow")

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
# Mutation Size = (max % change in parameter value)
def mutate(params, mutationRate=0.15, mutationSize=0.2):
    mutatedParams = {}
    for key, value in params.items():
        # if this parameter is not in the range dictionary, skip it as it is not tunable
        if key not in paramRanges:
            mutatedParams[key] = value
            continue
        # else it is tunable randomly decide whether to mutate this parameter
        if random.random() < mutationRate:
            low, high = paramRanges[key]
            mutationRange = (high - low) * mutationSize
            mutation = random.uniform(-mutationRange, mutationRange)
            mutation *= random.choice([-1, 1]) # randomly pick if its a % increase or decrease
            mutatedValue = value + mutation
            mutatedValue = max(low, min(high, mutatedValue))  # Ensure value is within range
            color_printer.print(f"Mutating parameter {key} from {value} to {mutatedValue} (value changed by {mutation / params[key] * 100}%)", color="cyan")
            mutatedParams[key] = mutatedValue
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

    # Plot boxplot for fitness distribution per generation
    fitnessPerGeneration = [[fitness[0] for fitness in gen] for gen in allGenerations]
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=fitnessPerGeneration)
    plt.title('Fitness Distribution per Generation')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.grid(True)
    plt.show()
