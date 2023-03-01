from io import BytesIO

from fastapi import FastAPI, Form, UploadFile
from PIL import Image

from clip_models.models.clip_classifier import ClipClassifier

app = FastAPI()
classifier = ClipClassifier()

def load_image(data):
    return Image.open(BytesIO(data))

@app.post("/predict")
async def predict(image: UploadFile , labels: list[str] = Form(..., description="A list of labels to use for classification. Must not contain commas.")):
    labels = labels[0].split(",")
    img = load_image(await image.read())
    pred = classifier(labels, img)
    return pred
