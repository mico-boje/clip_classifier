import annoy
import clip
import torch


class AnnoyIndex:
    def __init__(self, vector_len = 512, model = "ViT-B/32", device = "cuda" if torch.cuda.is_available() else "cpu") -> None:
        self.device = device
        self.model, self.preprocess = clip.load(model, device=device)
        self.index = annoy.AnnoyIndex(vector_len, "angular")
        self.index_is_built = False
        self.id = 0
        self.mapping_id_to_image = {}
        
          
    def get_nearest_images(self, text, n = 3):
        nns = self.index.get_nns_by_vector(self._process_text(text)[0], n, include_distances=True)
        print(nns)
        result = {}
        for img, distance in zip(nns[0], nns[1]):
            result[self.mapping_id_to_image[img]] = distance
        return result

    def add_image(self, image, image_name):
        image_features = self._process_image(image)
        self.index.add_item(self.id, image_features[0])
        self.mapping_id_to_image[self.id] = image_name
        self.id += 1
        
    def build(self, n_trees = 10):
        self.index.build(n_trees)
        self.index_is_built = True
    
    def _process_image(self, image):
        image = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(image)
        return image_features.detach().cpu().numpy()
    
    def _process_text(self, text):
        text = clip.tokenize([text]).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text)
        return text_features.detach().cpu().numpy()