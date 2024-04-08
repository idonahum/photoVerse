from diffusers.image_processor import VaeImageProcessor

class PhotoVerseImageProcessor(VaeImageProcessor):
    def __init__(self):
        super().__init__()

    def preprocess(self, img):
        pass