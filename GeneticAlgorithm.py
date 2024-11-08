import random
from utils import Color  # Import Color class from Utils.py
import pprint

# Define hyperparameters to tune and their ranges
paramRanges = {
    "epochs": (10, 100),
    "batch": (8, 32),
    "lr0": (1e-5, 1e-2),
    "momentum": (0.8, 0.99),
    "weightDecay": (0.0001, 0.01),
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
def geneticAlgorithmOptimize(trainFunc, valFunc, baseParams, optimizationMetric='metrics/mAP50-95(B)', generations=2, populationSize=2):
    # Generate initial population by randomly sampling hyperparameters
    population = [generateRandomParams(baseParams) for _ in range(populationSize)]
    scores = []

    for generation in range(generations):
        print(f"{Color.RED}{Color.BOLD}{Color.UNDERLINE}Generation {generation + 1}/{generations}{Color.END}")

        # Evaluate population fitness (validation score) for each individual in the population
        fitnessScores = []
        for i, params in enumerate(population):
            print(f"{Color.BLUE}Training with parameters {i+1}/{populationSize}: {params}{Color.END}")
            model = trainFunc(params)
            metrics = valFunc(model)
            fitness = metrics[optimizationMetric]
            fitnessScores.append((fitness, params))
            print(f"{Color.GREEN}Fitness Score: {fitness}{Color.END}")

        # Select top performers (e.g., top 50%) to serve as parents for the next generation
        fitnessScores.sort(reverse=True, key=lambda x: x[0])
        topPerformers = fitnessScores[:populationSize // 2]
        scores.extend(topPerformers)

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
    bestParams = max(scores, key=lambda x: x[0])[1]

    return bestParams

# Crossover function to combine parameters of two parents
# Crossover type = Uniform (each parent has equal chance for each parameter)
def crossover(params1, params2):
    childParams = {}
    for key in params1:
        if (random.random() > 0.5):
            print(f"{Color.CYAN}Inheriting parameter from parent 1: {key}{Color.END}")
            childParams[key] = params1[key]
        else:
            print(f"{Color.CYAN}Inheriting parameter from parent 2: {key}{Color.END}")
            childParams[key] = params2[key]
    return childParams

# Mutation function to slightly adjust parameters
# Mutation Type = Gaussian (add random noise to each parameter)
# Do it based on max of 10% change of the min and max for that parameter so it scales with the range (eg for a param with 0-1 then mutation can be at max 0.1% change, vs for range of 1-10000 then mutation can be at max 1000 change)
def mutate(params, mutationRate=0.15, mutationSize=0.1):
    mutatedParams = {}
    for key, value in params.items():
        # randomly decide whether to mutate this parameter
        if random.random() < mutationRate:
            low, high = paramRanges[key]
            mutationRange = (high - low) * mutationSize
            mutation = random.uniform(-mutationRange, mutationRange)
            mutation *= random.choice([-1, 1]) # randomly pick if its a % increase or decrease
            mutatedValue = value + mutation
            mutatedValue = max(low, min(high, mutatedValue))  # Ensure value is within range
            print(f"{Color.DARKCYAN}Mutating parameter {key} from {value} to {mutatedValue} (value changed by {mutation/mutatedParams[key]*100}%) {Color.END}")
            mutatedParams[key] = mutatedValue
    return mutatedParams
