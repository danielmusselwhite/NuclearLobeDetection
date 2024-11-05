from ultralytics import YOLO
import os

# Set environment variable to avoid OMP error
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'



#region "Training"
# Load a pre-trained YOLOv8 model

# Define a function to train the YOLO model
def trainYolo(params):
    model = YOLO('yolov8s.pt')

    # Train the model on the custom dataset
    model.train(
        **params
    )
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

    # Save the above to validationResults.txt
    with open("validationResults.txt", "w") as f:
        f.write("Validation Accuracy Metrics:\n")
        f.write(f"Precision: {metrics['metrics/precision(B)']:.2f}\n")
        f.write(f"Recall: {metrics['metrics/recall(B)']:.2f}\n")
        f.write(f"mAP@0.5: {metrics['metrics/mAP50(B)']:.2f}\n")
        f.write(f"mAP@0.5:0.95: {metrics['metrics/mAP50-95(B)']:.2f}\n")
        f.write(f"fitness: {metrics['fitness']:.2f}\n")

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
            class_id = int(detection.cls[0])       # Class index (ID) of detected object
            class_name = model.names[class_id]     # Class name using model's class list
            confidence = detection.conf[0]         # Confidence score of detection
            bbox = detection.xyxy[0]               # Bounding box coordinates in [x1, y1, x2, y2] format
            print(f"Class: {class_name}, Confidence: {confidence:.2f}, BBox: {bbox.tolist()}")

    # Save the annotated image with bounding boxes
    annotated_image_path = f"inferenceImages/{fileName}_annotated.{fileType}"
    results.save(annotated_image_path)
#endregion

# Main function
def main():
    # TODO - replace with a genetic algorithm or other HPOA, possibly also warm-start

    # Define params as a dictionary
    params = {
        "data": 'config.yaml',  # Path to config.yaml
        "epochs": 50,          # Number of epochs for training
        "batch": 16,            # Batch size
        "imgsz": 640,           # Image size
        "project": 'nuclearLobes',  # Project directory to save results
        "name": 'exp1',         # Experiment name
        "pretrained": True      # Start with pre-trained weights
    }

    # Train the model with the specified parameters
    model = trainYolo(params=params)

    # Validate the trained model
    validateYolo(model)
    # TODO - end of what should be wrapped in a GA

    # Run inference on a single image
    runInference(model, "ToBeReplaced.png")

# Ensure main() runs when script is executed directly
if __name__ == "__main__":
    main()