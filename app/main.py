from fastapi import FastAPI
import onnxruntime as ort
import numpy as np
from pydantic import BaseModel

app = FastAPI()

# Load ONNX model
session = ort.InferenceSession("model.onnx")

class InputData(BaseModel):
    features: list

@app.post("/predict")
def predict(data: InputData):
    input_array = np.array([data.features], dtype=np.float32)
    inputs = {session.get_inputs()[0].name: input_array}
    pred = session.run(None, inputs)
    return {"prediction": float(pred[0][0])}
