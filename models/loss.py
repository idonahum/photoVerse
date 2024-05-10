import torch
from torch.nn import CosineEmbeddingLoss
from torch.nn import functional as F
from facenet_pytorch import InceptionResnetV1
from models.arcface_resnet import ArcFaceResNet18
import cv2


class FaceLoss(torch.nn.Module):
    def __init__(self, device, model_name='arcface'):
        super(FaceLoss, self).__init__()
        assert model_name in ['arcface', 'facenet'], f"Model {model_name} not supported."
        self.device = device
        self.model_name = model_name
        self.input_size = 128 if model_name == 'arcface' else 160
        self.model = self._load_model(model_name).eval().to(device)
        self.cosine_loss = CosineEmbeddingLoss()

    @staticmethod
    def _load_model(model_name):
        if model_name == 'arcface':
            return ArcFaceResNet18(pretrained=True)
        else:
            return InceptionResnetV1(pretrained='vggface2')

    def preprocess(self, image_tensor, normalize=True):
        """
        Preprocess the input image tensor.
        """
        if self.model_name == 'arcface' and image_tensor.size(1) == 3:
            image_tensor = self.rgb_to_grayscale(image_tensor)
        resized_image = F.interpolate(image_tensor, size=(self.input_size, self.input_size), mode='bilinear',
                                      align_corners=False)
        if normalize:
            resized_image = resized_image / 127.5 - 1
        return resized_image

    @staticmethod
    def rgb_to_grayscale(image_tensor):
        """
        Converts an RGB image tensor to grayscale.

        Parameters:
            image_tensor (torch.Tensor): Tensor of shape (C, H, W) or (B, C, H, W)
                                         where C is 3 for the RGB channels.

        Returns:
            torch.Tensor: Grayscale image tensor of shape (H, W) or (B, 1, H, W)
        """
        # Define the weights for converting RGB to grayscale
        weights = torch.tensor([0.2989, 0.5870, 0.1140], device=image_tensor.device)

        # Check if the input is a batch or a single image
        if image_tensor.dim() == 4:  # If shape is (B, C, H, W)
            grayscale_image = torch.tensordot(image_tensor, weights, dims=([1], [0]))
            grayscale_image = grayscale_image.unsqueeze(1)  # Add a channel dimension
        elif image_tensor.dim() == 3:  # If shape is (C, H, W)
            grayscale_image = torch.tensordot(image_tensor, weights, dims=([0], [0]))
        else:
            raise ValueError("Input tensor should have 3 or 4 dimensions")

        return grayscale_image

    def forward(self, x, x_gen, maximize=True, normalize=True):
        """
        Compute cosine similarity between embeddings.
        If target
        """
        target = torch.ones(x.size(0)).to(self.device)
        if not maximize:
            target = -1 * target

        x = self.preprocess(x, normalize)
        x_gen = self.preprocess(x_gen, normalize)
        embedding1 = self.model(x)
        embedding2 = self.model(x_gen)

        return self.cosine_loss(embedding1, embedding2, target)


if __name__ == "__main__":
    # Specify the device (GPU if available, otherwise CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize the FaceNetLoss module with the specified device
    face_net_loss = FaceLoss(device=device, model_name='arcface')

    # Load images and convert them to tensors
    image1 = cv2.imread('/Users/ido.nahum/dev/photoVerse/CelebaHQMaskDataset/test/images/0.jpg')
    image2 = cv2.imread('/Users/ido.nahum/dev/photoVerse/CelebaHQMaskDataset/test/images/0.jpg')
    image1 = torch.from_numpy(image1).permute(2, 0, 1).float()
    image2 = torch.from_numpy(image2).permute(2, 0, 1).float()

    # Expand dimensions to match the expected input shape of the model
    input_tensor = image1.unsqueeze(0).repeat(8, 1, 1, 1).to(device)
    output_tensor = image2.unsqueeze(0).repeat(8, 1, 1, 1).to(device)

    # Calculate similarity using the FaceNetLoss module
    similarity = face_net_loss(input_tensor, output_tensor, maximize=False)
    print("Cosine Similarity between embeddings:", similarity.item())

    # Add noise or apply augmentation to the images
    noise = torch.randn_like(output_tensor) * 0.8  # Adding Gaussian noise
    images_augmented = torch.clamp(output_tensor + noise, 0, 1)  # Clamp values between 0 and 1

    # Calculate similarity using the FaceNetLoss module with the augmented images
    similarity_augmented = face_net_loss(input_tensor, images_augmented, maximize=False)
    print("Cosine Similarity between augmented embeddings:", similarity_augmented.item())

    print(input_tensor.shape)