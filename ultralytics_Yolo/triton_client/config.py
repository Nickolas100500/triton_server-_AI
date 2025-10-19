# config.py
# Image preprocessing settings

# Width, Height 
IMAGE_SIZE = (640, 640) # Width, Height

# Inference confidence threshold
CONFIDENCE_THRESHOLD = 0.25  

# IoU threshold for Non-Maximum Suppression
NMS_IOU_THRESHOLD = 0.5  

BATCH_SIZE = 1

# Minimum score for detection filtering
MIN_SCORE = 0.25  

# Path to the input image for inference
IMAGE_PATH = "D:/Coding Program/Triton invidia/test_images/6.JPG"

# COCO class names for YOLOv8
CLASS_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
]

