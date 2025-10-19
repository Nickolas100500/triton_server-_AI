# preprocess.py
import cv2
import numpy as np
from tritonclient.grpc import InferInput, InferRequestedOutput
import tritonclient.grpc as grpcclient

class InferenceModule:
    def __init__(self, triton_url="localhost:8001"): 
        self.triton_client = grpcclient.InferenceServerClient(url=triton_url)
        # Check connection to Triton server
        try:
            if not self.triton_client.is_server_live():
                raise ConnectionError(f"Triton server at {triton_url} is not live")
            print(f"Successfully connected to Triton server at {triton_url}")
        except Exception as e:
            raise ConnectionError(f"Could not connect to Triton server at {triton_url}: {e}")

    def letterbox(self, img, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True):
        """Improved letterbox function"""
        shape = img.shape[:2]  # current shape [height, width]
        
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)
        
        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better test mAP)
            r = min(r, 1.0)
        
        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, 32), np.mod(dh, 32)  # wh padding
        elif scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios
        
        dw /= 2  # divide padding into 2 sides
        dh /= 2
        
        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return img, ratio, (dw, dh)
        
    def infer_array_image(self, img_array: np.ndarray, model_name: str = "yolov8"):
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        
        # Apply letterbox
        img_processed, ratio, (dw, dh) = self.letterbox(img_rgb, 640)
        
        # Normalize and transpose
        img_processed = img_processed.astype(np.float32) / 255.0
        img_processed = np.transpose(img_processed, (2, 0, 1))  # HWC to CHW
        img_processed = np.expand_dims(img_processed, 0)  # Add batch dimension

        print(f"preprocess: img range [{img_processed.min():.3f}, {img_processed.max():.3f}], shape {img_processed.shape}, dtype {img_processed.dtype}")

        # Prepare inference
        input_tensor = InferInput("images", img_processed.shape, "FP32")
        input_tensor.set_data_from_numpy(img_processed)
        output_tensor = InferRequestedOutput("output0")

        # Run inference
        results = self.triton_client.infer(
            model_name=model_name,
            inputs=[input_tensor],
            outputs=[output_tensor]
        )

        output = results.as_numpy("output0")
        print(f"preprocess array: output shape={output.shape}, dtype={output.dtype}")
        return {"output0": output, "original_shape": img_array.shape, "ratio": ratio, "padding": (dw, dh)}

    def infer_image(self, img_path: str = "5.jpg", model_name: str = "yolov8"):
        # Read and preprocess image
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Apply letterbox
        img_processed, ratio, (dw, dh) = self.letterbox(img_rgb, 640)
        
        # Normalize and transpose
        img_processed = img_processed.astype(np.float32) / 255.0
        img_processed = np.transpose(img_processed, (2, 0, 1))  # HWC to CHW
        img_processed = np.expand_dims(img_processed, 0)  # Add batch dimension

        print(f"preprocess: img range [{img_processed.min():.3f}, {img_processed.max():.3f}], shape {img_processed.shape}, dtype {img_processed.dtype}")

        # Prepare inference
        input_tensor = InferInput("images", img_processed.shape, "FP32")
        input_tensor.set_data_from_numpy(img_processed)
        output_tensor = InferRequestedOutput("output0")

        # Run inference
        results = self.triton_client.infer(
            model_name=model_name,
            inputs=[input_tensor],
            outputs=[output_tensor]
        )

        output = results.as_numpy("output0")
        print(f"preprocess: output shape={output.shape}, dtype={output.dtype}")
        return {"output0": output, "original_shape": img.shape, "ratio": ratio, "padding": (dw, dh)}