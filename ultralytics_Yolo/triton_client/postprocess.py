# postprocess.py
import numpy as np
from .config import CONFIDENCE_THRESHOLD, NMS_IOU_THRESHOLD, BATCH_SIZE, CLASS_NAMES

class YOLOv8Postprocessor:
    def __init__(self, conf_threshold=CONFIDENCE_THRESHOLD, class_names=CLASS_NAMES, min_score=0.01):
        self.conf_threshold = conf_threshold
        self.class_names = class_names
        self.min_score = min_score

    def calculate_iou(self, box1, box2):
        """
        Calculate IoU (Intersection over Union) between two boxes
        box1, box2: [x1, y1, x2, y2]
        """
        x1_inter = max(box1[0], box2[0])
        y1_inter = max(box1[1], box2[1])
        x2_inter = min(box1[2], box2[2])
        y2_inter = min(box1[3], box2[3])
        
        # Intersection area
        inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
        
        # Box areas
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        # IoU = intersection / union
        union_area = box1_area + box2_area - inter_area
        return inter_area / union_area if union_area > 0 else 0

    def non_max_suppression(self, detections, iou_threshold=NMS_IOU_THRESHOLD):
        """
        Apply Non-Maximum (NMS) Suppression to remove duplicate detections
        detections: list of detections with 'box', 'score', 'class' fields
        iou_threshold: IoU threshold for removing overlapping boxes
        """
        if not detections:
            return detections
        
        # Sort by score (best first)
        detections = sorted(detections, key=lambda x: x['score'], reverse=True)
        
        filtered_detections = []
        
        for current_det in detections:
            # Check if overlaps with already selected boxes
            keep = True
            for selected_det in filtered_detections:
                # Apply NMS only within the same class
                if current_det['class'] == selected_det['class']:
                    iou = self.calculate_iou(current_det['box'], selected_det['box'])
                    if iou > iou_threshold:
                        keep = False
                        break
            
            if keep:
                filtered_detections.append(current_det)
        
        return filtered_detections

    def process(self, output, min_score=None):
        """
        Corrected postprocessing for YOLOv8 output
        output shape: (1, 84, 8400) for 640x640 input
        """
        print(f"postprocess: output shape={output.shape}, dtype={output.dtype}")
        
        # Remove batch dimension
        output = output[0]  # shape: (84, 8400)
        
        detections = []
        min_score = min_score if min_score is not None else self.min_score
        
        # Process all boxes
        for i in range(output.shape[1]):
            row = output[:, i]
            
            # Get box coordinates (cx, cy, w, h)
            x, y, w, h = row[0], row[1], row[2], row[3]
            
            # Get class scores (including objectness)
            class_scores = row[4:]
            class_id = int(np.argmax(class_scores))
            score = float(class_scores[class_id])
            
            if score >= self.conf_threshold:
                # Convert from center format to corner format
                x1 = float(x - w / 2)
                y1 = float(y - h / 2)
                x2 = float(x + w / 2)
                y2 = float(y + h / 2)
                
                # Get class name
                if class_id < len(self.class_names):
                    class_name = self.class_names[class_id]
                else:
                    class_name = f"class_{class_id}"
                
                det = {
                    "class": class_name,
                    "score": round(score, 3),
                    "box": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                    "class_id": class_id
                }
                
                if det["score"] >= min_score:
                    detections.append(det)

        # Apply NMS to remove duplicates
        detections = self.non_max_suppression(detections, iou_threshold=NMS_IOU_THRESHOLD)

        # Print results
        if not detections:
            print("postprocess: No detections found above threshold.")
            max_score = np.max(output[4:, :]) if output.shape[0] > 4 else 0
            print(f"Max score in output: {max_score:.4f}")
        else:
            print(f"\nDetections found: {len(detections)} (after NMS)")
            print(f"{'Class':<12}{'Score':<8}{'Box':<30}")
            print("-" * 50)
            for det in detections:
                print(f"{det['class']:<12}{det['score']:<8}{str(det['box']):<30}")
            print("-" * 50)
        
        return detections