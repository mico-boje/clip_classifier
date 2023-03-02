# cython: language_level=3

import annoy
import numpy as np
import torch

cimport cython
cimport numpy as np
from cython.view cimport array as cvarray

from clip import clip


cdef class AnnoyIndex:
    cdef int id
    cdef dict mapping_id_to_image
    cdef object index
    cdef object model
    cdef object preprocess
    cdef str device

    def __init__(self, vector_len = 512, model = "ViT-B/32", device = "cuda" if torch.cuda.is_available() else "cpu") -> None:
        self.device = device
        self.model, self.preprocess = clip.load(model, device=device)
        self.index = annoy.AnnoyIndex(vector_len, "angular")
        #self.index_is_built = False
        self.id = 0
        self.mapping_id_to_image = {}
        
          
    def get_nearest_images(self, str text, int n = 5):
        cdef tuple nns = self.index.get_nns_by_vector(self.process_text(text)[0], n, include_distances=True)
        cdef dict result = {}
        cdef int i
        for i in range(n):
            img = nns[0][i]
            distance = nns[1][i]
            result[self.mapping_id_to_image[img]] = distance
        return result

    def add_image(self, image, image_name):
        image_features = self._process_image(image)
        self.index.add_item(self.id, image_features[0])
        self.mapping_id_to_image[self.id] = image_name
        self.id += 1
        
    def build(self, n_trees = 10):
        self.index.build(n_trees)
        #self.index_is_built = True
    
    def _process_image(self, image):
        image = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(image)
        return image_features.detach().cpu().numpy()
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef np.ndarray[np.float32_t, ndim=2] _process_text(self, str text):
        text_tensor = clip.tokenize([text]).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text_tensor)
        return text_features.detach().cpu().numpy()

    def process_text(self, str text):
        cdef np.ndarray[np.float32_t, ndim=2] features
        features = self._process_text(text)
        return features