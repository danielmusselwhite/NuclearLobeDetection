import random
import matplotlib.pyplot as plt
import seaborn as sns
from utils import Color  # Import Color class from Utils.py
from pprint import pprint

# Initialize the Color object from Utils.py
color_printer = Color()

# Define hyperparameters to tune and their ranges - https://docs.ultralytics.com/modes/train/#resuming-interrupted-trainings
paramRanges = {
    # Train Settings
    "epochs": (int, (100, 600)),  # Number of training epochs
    "batch": (int, (8, 32)),  # Batch size
    "lr0": (float, (1e-5, 1e-2)),  # Initial learning rate
    "momentum": (float, (0.8, 0.99)),  # Momentum factor
    "weight_decay": (float, (0.0001, 0.01)),  # Weight decay for L2 regularization
    "imgsz": (int, (80, 2560)),  # Target image size
    "lrf": (float, (0.1, 1.0)),  # Final learning rate fraction, affects learning rate decay
    "patience": (int, (10, 100)),  # Early stopping patience to prevent overfitting
    "warmup_epochs": (float, (0.0, 10.0)),  # Epochs for learning rate warmup
    "warmup_momentum": (float, (0.0, 0.95)),  # Initial momentum during warmup
    "warmup_bias_lr": (float, (0.0, 0.2)),  # Bias learning rate during warmup phase
    "cos_lr": (bool, (False, True)),  # Toggle for cosine learning rate scheduler
    "optimizer": (str, ('SGD', 'Adam', 'AdamW', 'RMSProp')),  # Choice of optimizer
    "box": (float, (0.05, 10.0)),  # Box loss weight
    "cls": (float, (0.05, 5.0)),  # Classification loss weight
    "dfl": (float, (0.0, 5.0)),  # Distribution focal loss weight
    "label_smoothing": (float, (0.0, 0.2)),  # Label smoothing factor for class labels
    "dropout": (float, (0.0, 0.5)),  # Dropout rate for regularization
    "rect": (bool, (False, True)),  # Enables rectangular training for more efficient padding
    "single_cls": (bool, (False, True)),  # Binary classification mode toggle
    "close_mosaic": (int, (0, 20)),  # Number of epochs to disable mosaic augmentation before end
    "freeze": (int, (0, 10)),  # Number of layers to freeze for transfer learning

    # Augmentation Settings and Hyperparameters
    "hsv_h": (float, (0.0, 1.0)), # adjusts hue
    "hsv_s": (float, (0.0, 1.0)), # adjusts saturation
    "hsv_v": (float, (0.0, 1.0)), # adjusts value
    "degrees": (float, (-180.0, 180.0)), # Rotation angle
    "translate": (float, (0.0, 1.0)), # Translation factor
    "scale": (float, (0.0, 100.0)),  # Scale
    "shear": (float, (-180.0, 180.0)), # Shear angle
    "perspective": (float, (0.0, 0.001)), # Perspective distortion
    "flipud": (float, (0.0, 1.0)), # flip image upside down
    "fliplr": (float, (0.0, 1.0)), # flip image left to right
    "bgr": (float, (0.0, 1.0)), # flips image RGB to BGR
    "mosaic": (float, (0.0, 1.0)), # combines 4 images into one
    "mixup": (float, (0.0, 1.0)), # blends two images and their labels creating a composite image
    "copy_paste": (float, (0.0, 1.0)), # copy and paste a section of an image onto another image
    "copy_paste_mode": (str, ('flip', 'mixup')), # copy and paste mode
    "auto_augment": (str, ('randaugment', 'autoaugment', 'augmix')), # automatic applies a predefined augmentation policy
    "erasing": (float, (0.0, 0.9)), # random erasies a portion of the image
    "crop_fraction": (float, (0.1, 1.0)), # crops image to a fraction of its size
}

# Generate a random set of hyperparameters within defined ranges
def generateRandomParams(baseParams):
    params = baseParams.copy()
    # Generate random values for each parameter within the specified range
    for key, value in paramRanges.items():
        # if this is a string pick a random value from the list of possible values
        if value[0] == str:
            params[key] = random.choice(value[1])
        # if this is a boolean pick a random value from 0 or 1
        elif value[0] == bool:
            params[key] = random.choice([True, False])
        # if this is a int pick a random value from the range
        elif value[0] == int:
            params[key] = random.randint(value[1][0], value[1][1])
        # if this is a float pick a random value from the range
        elif value[0] == float:
            params[key] = random.uniform(value[1][0], value[1][1])
    return params

# Genetic algorithm optimization function
def geneticAlgorithmOptimize(trainFunc, valFunc, baseParams, optimizationMetric='metrics/mAP50-95(B)', generations=10, populationSize=24, noImprovementThreshold=3, goodEnoughThreshold=0.95):
    # Generate initial population by randomly sampling hyperparameters
    population = [generateRandomParams(baseParams) for _ in range(populationSize)]
    allGenerations = []  # List to track all generations' (params, scores)
    bestModel = None  # To track the best model and its details (score, generation, index)
    bestScore = -float('inf')  # Initialize with very low score
    generationScores = []  # To track average fitness for each generation
    
    # Early termination counters
    generationsWithoutImprovement = 0

    for generation in range(generations):
        color_printer.print(f"Generation {generation + 1}/{generations}", color="red", bold=True, underline=True)

        # Store current generation's (params, score) pairs
        thisGeneration = []

        #region "Evaluate population fitness (validation score) for each individual in the population"
        populationBestFitness = -float('inf')
        populationBestModel = None
        individualNumber = -1
        for i, params in enumerate(population):
            color_printer.print(f"Training individual {i+1}/{populationSize} from Generation {generation + 1}/{generations}", color="red")
            model = trainFunc(params)
            metrics = valFunc(model)
            fitness = metrics[optimizationMetric]
            thisGeneration.append((fitness, params))
            color_printer.print(f"Fitness Score ({optimizationMetric}): {fitness}", color="green")

            # Track the best model
            if fitness > populationBestFitness:
                populationBestFitness = fitness
                populationBestModel = model
                individualNumber = i
        
        # If this populations best model is better than the best model so far, update the best model
        if populationBestFitness > bestScore:
            bestScore = populationBestFitness
            bestModel = (bestScore, populationBestModel, generation, individualNumber)
            generationsWithoutImprovement = 0 # Reset counter for no improvement as we have found a better model
        # Else we haven't improved so increment the counter
        else:
            generationsWithoutImprovement += 1
        #endregion

        # Update allGenerations with this generation's results
        allGenerations.append(thisGeneration)

        # Calculate the average fitness of this generation for trend plotting
        avgFitness = sum(fitness[0] for fitness in thisGeneration) / len(thisGeneration)
        generationScores.append(avgFitness)

        
        #region "Early termination"
        # Check if the model is good enough or if we have no improvement for 'n' generations
        if bestScore >= goodEnoughThreshold:
            color_printer.print(f"Early stopping: Found a good enough model with a fitness score of {bestScore} (>= {goodEnoughThreshold})", color="blue", bold=True)
            break
        if generationsWithoutImprovement >= noImprovementThreshold:
            color_printer.print(f"Early stopping: No improvement in the last {noImprovementThreshold} generations", color="blue", bold=True)
            break
        #endregion


        #region "Generating new population using crossover and mutation"
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
        #endregion

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
        # If this parameter is not in the range dictionary, skip it as it is not tunable
        if key not in paramRanges:
            mutatedParams[key] = value
            continue

        # Get the expected type for this parameter from paramRanges
        paramType = paramRanges[key][0]
        
        # Randomly decide whether to mutate this parameter
        if random.random() < mutationRate:
            # If this parameter is a float or int, apply Gaussian noise
            if paramType in [int, float]:
                low, high = paramRanges[key][1]
                mutationRange = (high - low) * mutationSize
                mutation = random.uniform(-mutationRange, mutationRange) # Random increase vs decrease percent
                mutatedValue = value + mutation
                mutatedValue = max(low, min(high, mutatedValue))  # Ensure value is within range
                
                # Ensure the mutated value is of the correct type
                if paramType == int:
                    mutatedValue = int(round(mutatedValue))  # Round to nearest integer if it was an integer
                
                color_printer.print(f"Mutating parameter {key} from {value} to {mutatedValue} ({mutation / (high - low) * 100}% (range {low} to {high}))", color="yellow")
                mutatedParams[key] = mutatedValue

            # If this parameter is boolean, flip it
            elif paramType == bool:
                color_printer.print(f"Mutating boolean parameter {key} from {value} to {not value}", color="yellow")
                mutatedParams[key] = not value

            # If this parameter is a string, randomly select a new value
            elif paramType == str:
                possibleValues = list(paramRanges[key][1])
                possibleValues.remove(value)  # Remove current value to choose a different one
                mutatedValue = random.choice(possibleValues)
                color_printer.print(f"Mutating parameter {key} from {value} to {mutatedValue}", color="yellow")
                mutatedParams[key] = mutatedValue

        else:
            mutatedParams[key] = value # No mutation, keep the original value
    
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
    plt.savefig('AverageFitnessPerGeneration.png')
    plt.show()
    plt.close()

    # Plot boxplot for fitness distribution per generation
    fitnessPerGeneration = [[fitness[0] for fitness in gen] for gen in allGenerations]
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=fitnessPerGeneration)
    plt.title('Fitness Distribution per Generation')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.grid(True)
    plt.savefig('FitnessDistributionPerGeneration.png')
    plt.show()
    plt.close()

