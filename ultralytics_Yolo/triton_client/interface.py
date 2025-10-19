"""
Interface utilities for YOLOv8 Triton client
Contains plotting functions and performance monitoring tools
"""

import cv2
import time


class FPSCounter:
    """FPS counter class for real-time performance monitoring"""
    
    def __init__(self):
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.last_fps_display = time.time()
        self.current_fps = 0.0
        
    def update_frame(self):
        """Update frame counter and calculate FPS"""
        self.fps_counter += 1
        
        current_time = time.time()
        
        # Calculate and display FPS every second
        if current_time - self.last_fps_display >= 1.0:
            elapsed_time = current_time - self.fps_start_time
            self.current_fps = self.fps_counter / elapsed_time if elapsed_time > 0 else 0
            
            # Reset counters
            self.fps_counter = 0
            self.fps_start_time = current_time
            self.last_fps_display = current_time
            
            return True  # Indicates FPS was updated
        
        return False  # FPS not updated this frame
    
    def get_current_fps(self):
        """Get current FPS for display"""
        current_time = time.time()
        elapsed = current_time - self.fps_start_time
        if elapsed > 0:
            return self.fps_counter / elapsed
        return 0.0
    
    def get_fps_text(self):
        """Get formatted FPS text for display"""
        return f"FPS: {self.get_current_fps():.1f}"


def plot(image, detections, ratio, padding, line_thickness=2, font_size=0.4):
    """
    Draw detections on image (Ultralytics-style plot function)
    
    Args:
        image: Input image (OpenCV format)
        detections: List of detections from YOLOv8Postprocessor
        ratio: Scaling ratios from preprocessing
        padding: Padding values from preprocessing
        line_thickness: Thickness of bounding box lines
        font_size: Size of label text
        
    Returns:
        Image with drawn detections
    """
    img_draw = image.copy()
    dw, dh = padding
    
    for det in detections:
        x1, y1, x2, y2 = det['box']
        score = det['score']
        class_name = det['class']
        
        # Convert coordinates back to original image space
        x1 = int((x1 - dw) / ratio[0])
        y1 = int((y1 - dh) / ratio[1])
        x2 = int((x2 - dw) / ratio[0])
        y2 = int((y2 - dh) / ratio[1])
        
        # Green color for all detections
        color = (0, 255, 0)
        
        # Draw bounding box
        cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, line_thickness)
        
        # Prepare label
        label = f"{class_name} {score:.2f}"
        
        # Calculate label size
        (label_width, label_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_size, 1
        )
        
        # Draw label background
        label_y = y1 - 8 if y1 - 8 > label_height else y1 + label_height + 8
        cv2.rectangle(
            img_draw,
            (x1, label_y - label_height - 3),
            (x1 + label_width + 4, label_y + 3),
            color,
            -1
        )
        
        # Draw label text
        cv2.putText(
            img_draw,
            label,
            (x1 + 2, label_y - 1),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_size,
            (255, 255, 255),  # White text
            1,  # Тонкий текст
            cv2.LINE_AA
        )
    
    return img_draw


def draw_performance_info(image, fps_text, inference_time=None, detection_count=None, position=(10, 30)):
    """
    Draw performance information on image
    
    Args:
        image: Input image
        fps_text: FPS text to display
        inference_time: Inference time in ms (optional)
        detection_count: Number of detections (optional)
        position: Starting position for text (x, y)
        
    Returns:
        Image with performance info
    """
    img_draw = image.copy()
    x, y = position
    line_height = 30
    
    # Draw FPS
    cv2.putText(img_draw, fps_text, (x, y), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Draw inference time if provided
    if inference_time is not None:
        y += line_height
        inference_text = f"Inference: {inference_time:.1f}ms"
        cv2.putText(img_draw, inference_text, (x, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Draw detection count if provided
    if detection_count is not None:
        y += line_height
        detection_text = f"Objects: {detection_count}"
        cv2.putText(img_draw, detection_text, (x, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    return img_draw


def draw_error_message(image, error_message, fps_text=None):
    """
    Draw error message on image
    
    Args:
        image: Input image
        error_message: Error message to display
        fps_text: Optional FPS text
        
    Returns:
        Image with error message
    """
    img_draw = image.copy()
    
    # Draw error message
    cv2.putText(img_draw, error_message, (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Draw FPS if provided
    if fps_text:
        cv2.putText(img_draw, fps_text, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    return img_draw
