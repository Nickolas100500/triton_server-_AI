# infer.py
from triton_client.preprocess import InferenceModule
from triton_client.postprocess import YOLOv8Postprocessor
from triton_client.config import CONFIDENCE_THRESHOLD, IMAGE_PATH
import cv2
import numpy as np
import random


def plot(image, detections, ratio, padding, line_thickness=3, font_size=0.5, colors=None):
    """
    Draw detections on image (Ultralytics-style plot function)
    
    Args:
        image: Original image (BGR format)
        detections: List of detection dictionaries
        ratio: Scaling ratio from letterbox preprocessing
        padding: Padding values (dw, dh) from letterbox
        line_thickness: Thickness of bounding box lines
        font_size: Font size for labels
        colors: List of colors for different classes (if None, generates random colors)
    
    Returns:
        Image with drawn detections
    """
    img_draw = image.copy()
    dw, dh = padding
    
    # Generate colors for classes if not provided
    if colors is None:
        colors = {}
        for det in detections:
            class_name = det['class']
            if class_name not in colors:
                # Generate random color for each class
                colors[class_name] = (
                    random.randint(0, 255),
                    random.randint(0, 255), 
                    random.randint(0, 255)
                )
    
    for det in detections:
        x1, y1, x2, y2 = det['box']
        score = det['score']
        class_name = det['class']
        
        # Convert coordinates back to original image space
        x1 = int((x1 - dw) / ratio[0])
        y1 = int((y1 - dh) / ratio[1])
        x2 = int((x2 - dw) / ratio[0])
        y2 = int((y2 - dh) / ratio[1])
        
        # Get color for this class - always green
        color = (0, 255, 0)  # Green color for all detections
        
        # Draw bounding box
        cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, line_thickness)
        
        # Prepare label
        label = f"{class_name} {score:.2f}"
        
        # Calculate label size
        (label_width, label_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_size, 1
        )
        
        # Draw label background
        label_y = y1 - 10 if y1 - 10 > label_height else y1 + label_height + 10
        cv2.rectangle(
            img_draw,
            (x1, label_y - label_height - 5),
            (x1 + label_width + 5, label_y + 5),
            color,
            -1
        )
        
        # Draw label text
        cv2.putText(
            img_draw,
            label,
            (x1 + 2, label_y - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_size,
            (255, 255, 255),  # White text
            1,
            cv2.LINE_AA
        )
    
    return img_draw


def draw_detections(image, detections, ratio, padding):
    """Legacy function - use plot() instead"""
    return plot(image, detections, ratio, padding)

if __name__ == "__main__":
    # Configuration
    
    
    try:
        # Initialize modules
        infer = InferenceModule("localhost:8001")
        
        # Run inference
        result = infer.infer_image(IMAGE_PATH, "yolov8")
        output0 = result["output0"]
        
        # Postprocess
        post = YOLOv8Postprocessor(conf_threshold=CONFIDENCE_THRESHOLD)
        detections = post.process(output0)
        
        # Print results
        print("\nFinal Detections:")
        for det in detections:
            print(f"Class: {det['class']}, Score: {det['score']:.3f}, Box: {det['box']}")
            
        # Draw and save result using Ultralytics-style plot function
        original_img = cv2.imread(IMAGE_PATH)
        result_img = plot(original_img, detections, 
        result["ratio"], result["padding"], 
        line_thickness=3, font_size=0.6)
        cv2.imwrite("result.jpg", result_img)
        print("Result saved as 'result.jpg'")
        
    except Exception as e:
        print(f"Error: {e}")