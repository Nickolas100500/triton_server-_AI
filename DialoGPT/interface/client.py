import numpy as np
from tritonclient.grpc import InferenceServerClient, InferInput, InferRequestedOutput

class TritonClient:
    def __init__(self, url="127.0.0.1:8001"):
        self.client = InferenceServerClient(url=url)

    def run_inference(self, model_name, input_ids: np.ndarray):
        infer_input = InferInput("input_ids", input_ids.shape, "INT32")
        infer_input.set_data_from_numpy(input_ids)
        outputs = [InferRequestedOutput("logits")]
        result = self.client.infer(model_name, inputs=[infer_input], outputs=outputs)
        return result.as_numpy("logits")