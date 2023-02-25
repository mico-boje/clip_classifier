from io import BytesIO

import numpy as np
from fastapi import FastAPI, Form, UploadFile
from PIL import Image
from pydantic import Field

from clip_classifier.model.clip_classifier import ClipClassifier

app = FastAPI()
classifier = ClipClassifier()


def load_image(data):
    return Image.open(BytesIO(data))


@app.post("/predict")
async def predict(image: UploadFile , labels: list[str] = Form(..., description="A list of labels to use for classification. Must not contain commas.")):
    labels = labels[0].split(",")
    print(labels)
    img = load_image(await image.read())
    pred = classifier(labels, img)
    return pred