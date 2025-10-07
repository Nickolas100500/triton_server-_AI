import os
import numpy as np
from PIL import Image
from torchvision import transforms
import grpc
import tritonclient.grpc.aio as grpcclient
from tritonclient.grpc import service_pb2, service_pb2_grpc
from tritonclient.utils import triton_to_np_dtype
from .image_utils import center_crop, decode_img


class InferenceModule:
    def __init__(self) -> None:
        """Initialize."""
        self.url = os.environ.get("TRITON_SERVER_URL", "127.0.0.1:8001")
        self.triton_client = grpcclient.InferenceServerClient(url=self.url)

    async def infer_image(
        self,
        img: str,
        model_name: str = "classifier_onnx",
    ) -> dict:
        """
        Perform inference on the input image.

        Args:
            img (str): Base64 encoded image string.
            model_name (str): Name of the deployed model.

        Returns:
            dict: class ID, and logit.
        """
        model_meta, model_config = self.parse_model_metadata(model_name)
        shape = model_meta.inputs[0].shape
        channels, height, width = shape[1:]
        dtype = model_meta.inputs[0].datatype

        # Preprocess the image
        img = self.preprocess_image_torchvision(img)  # или preprocess_image_pil

        # Create input tensor for Triton
        inputs = [grpcclient.InferInput(model_meta.inputs[0].name, [1, channels, height, width], dtype)]
        inputs[0].set_data_from_numpy(img.astype(triton_to_np_dtype(dtype)))

        # Define output tensor
        outputs = [grpcclient.InferRequestedOutput(model_meta.outputs[0].name)]

        # Perform inference
        results = await self.triton_client.infer(
            model_name=model_name,
            inputs=inputs,
            outputs=outputs,
        )

        # Process the output
        output = results.as_numpy(model_meta.outputs[0].name)[0]
        cls_idx = np.argmax(output)
        cls_logit = output[cls_idx]

        return {"class_id": int(cls_idx), "logit": float(cls_logit)}

    def parse_model_metadata(self, model_name: str) -> object:
        """
        Parse metadata and configuration of the model.

        Args:
            model_name (str): Name of the deployed model.

        Returns:
            tuple: Metadata and configuration of the model.
        """
        channel = grpc.insecure_channel(self.url)
        grpc_stub = service_pb2_grpc.GRPCInferenceServiceStub(channel)
        metadata_request = service_pb2.ModelMetadataRequest(name=model_name)
        metadata_response = grpc_stub.ModelMetadata(metadata_request)

        config_request = service_pb2.ModelConfigRequest(name=model_name)
        config_response = grpc_stub.ModelConfig(config_request)

        return metadata_response, config_response

    def preprocess_image_pil(
        self,
        img: str,
    ) -> np.ndarray:
        """
        Preprocess the input image for ResNet.

        Args:
            img (str): Base64 encoded image string.
            height (int): Target height for resizing.
            width (int): Target width for resizing.

        Returns:
            np.ndarray: Preprocessed image as a NumPy array.
        """
        # Decode base64 image to PIL Image
        pil_img = decode_img(img)

        # Resize and center crop
        resized_img = pil_img.resize((256, 256), Image.BILINEAR)
        cropped_img = center_crop(resized_img, 224, 224)  # Центрированная обрезка до 224x224

        np_img = np.array(cropped_img).astype(np.float32)

        # Normalize using ImageNet mean and std
        normalized_img = (np_img / 255.0 - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])

        # Transpose to match NCHW format
        ordered_img = np.transpose(normalized_img, (2, 0, 1))

        # Add batch dimension
        batched_img = np.expand_dims(ordered_img, axis=0)

        return batched_img

    def preprocess_image_torchvision(
        self,
        img: str,
    ) -> np.ndarray:
        """
        Preprocess the input image for ResNet using torchvision.transforms.

        Args:
            img (str): Base64 encoded image string.

        Returns:
            np.ndarray: Preprocessed image as a NumPy array.
        """
        # Decode base64 image to PIL Image
        pil_img = decode_img(img)

        # Define preprocessing pipeline using torchvision.transforms
        self.preprocess = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
        ])

        # Apply preprocessing pipeline
        pil_img = self.preprocess(pil_img)

        # Center crop 224x224
        pil_img = center_crop(pil_img, 224, 224)

        # Convert PyTorch tensor to NumPy array and add batch dimension
        np_img = np.array(pil_img).astype(np.float32)

        # Normalize using ImageNet mean and std
        normalized_img = (np_img / 255.0 - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])

        # Transpose to match NCHW format
        ordered_img = np.transpose(normalized_img, (2, 0, 1))

        # Add batch dimension
        batched_img = np.expand_dims(ordered_img, axis=0)

        return batched_img
