from ultralytics import YOLO
import os
from GeneticAlgorithm import geneticAlgorithmOptimize
from utils import Color
from pprint import pprint

# Set environment variable to avoid OMP error
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Initialize the Color object from Utils.py
color_printer = Color()

#region "Training"
# Load a pre-trained YOLOv8 model

# Define a function to train the YOLO model
def trainYolo(params):
    model = YOLO(params.get('model', 'yolov8s.pt'))

    # Train the model on the custom dataset
    model.train(**params)
    return model
#endregion



#region "Validation/Testing"

# Define a function to validate the YOLO model
def validateYolo(model):
    # Evaluate the model on the validation dataset
    validationResults = model.val()

    # Retrieve and print accuracy-related metrics from validation results
    metrics = validationResults.results_dict
    print("Validation Accuracy Metrics:")
    print(f"Precision: {metrics['metrics/precision(B)']:.2f}")
    print(f"Recall: {metrics['metrics/recall(B)']:.2f}")
    print(f"mAP@0.5: {metrics['metrics/mAP50(B)']:.2f}")
    print(f"mAP@0.5:0.95: {metrics['metrics/mAP50-95(B)']:.2f}")
    print(f"fitness: {metrics['fitness']:.2f}")

    return metrics

#endregion



#region: "Running inference"

# Define a function to run inference on a single image
def runInference(model, fullFileName):
    # Run inference on a single image
    fileName = fullFileName.split('.')[0]
    fileType = fullFileName.split('.')[1]

    # Perform inference
    results = model('inferenceImages/' + fullFileName)
    results = results[0]

    # Print detection results in a readable format
    print("Detection Results:")
    for result in results:
        for detection in result.boxes:
            classId = int(detection.cls[0])     # Class index (ID) of detected object
            className = model.names[classId]    # Class name using model's class list
            confidence = detection.conf[0]      # Confidence score of detection
            bbox = detection.xyxy[0]            # Bounding box coordinates in [x1, y1, x2, y2] format
            print(f"Class: {className}, Confidence: {confidence:.2f}, BBox: {bbox.tolist()}")

    # Save the annotated image with bounding boxes
    annotatedImagePath = f"inferenceImages/{fileName}_annotated.{fileType}"
    results.save(annotatedImagePath)
#endregion

# Main function
def main():
    # Initial parameters
    baseParams = {
        "model": 'yolov8s.pt',      # Yolo version
        "data": 'config.yaml',      # Data configuration
        "epochs": 50,               # Number of epochs
        "batch": 16,                # Batch size (number of images per batch)
        "imgsz": 640,               # Image size (number of pixels (all images will be resized to this size during training))
        "project": 'nuclearLobes',  # Project name
        "name": 'exp1',             # Experiment name
        "pretrained": True,         # Use pre-trained model
    }

    # Run genetic algorithm optimization
    bestParams = geneticAlgorithmOptimize(trainFunc=trainYolo, valFunc=validateYolo, baseParams=baseParams)
    color_printer.print(f"Best Parameters: ", color="magenta", bold=True, underline=True)
    pprint(f"{bestParams}")

    # Train final model with optimized parameters
    color_printer.print(f"Retraining the best model config (done as we can't store each trained due to size complexity, need to retrain after we know what the best is'): ", color="red", bold=True, underline=True)
    model = trainYolo(bestParams)
    metrics = validateYolo(model)

    # Save the above to validationResults.txt
    with open("validationResults.txt", "w") as f:
        f.write("Validation Accuracy Metrics:\n")
        f.write(f"Precision: {metrics['metrics/precision(B)']:.2f}\n")
        f.write(f"Recall: {metrics['metrics/recall(B)']:.2f}\n")
        f.write(f"mAP@0.5: {metrics['metrics/mAP50(B)']:.2f}\n")
        f.write(f"mAP@0.5:0.95: {metrics['metrics/mAP50-95(B)']:.2f}\n")
        f.write(f"fitness: {metrics['fitness']:.2f}\n")

    # Run inference on a test image
    runInference(model, "ToBeReplaced.png")

if __name__ == "__main__":
    main()
