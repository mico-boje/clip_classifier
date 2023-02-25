from PIL import Image
from transformers import CLIPModel, CLIPProcessor


class ClipClassifier:
    def __init__(self, model = "openai/clip-vit-base-patch16") -> None:
        self.model = CLIPModel.from_pretrained(model)
        self.processor = CLIPProcessor.from_pretrained(model)

    def __call__(self, labels: list, image: Image) -> list:
        inputs = self.processor(text=labels, images=image, return_tensors="pt", padding=True)
        predictions = self._predict(inputs)
        return self._format_prediction(labels, predictions[0])
    
    def _predict(self, inputs):
        outputs = self.model(**inputs)
        logits_per_image = outputs.logits_per_image # this is the image-text similarity score
        probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities
        return probs
    
    def _format_prediction(self, labels: list, probs: list) -> list:
        return [{"label": label, "prob": "{:.2f}%".format(prob.item())} for label, prob in zip(labels, probs)]