import torch
import torch.nn.functional as F
from facenet_pytorch import InceptionResnetV1
import cv2


class FaceNetLoss(torch.nn.Module):
    def __init__(self, device):
        super(FaceNetLoss, self).__init__()
        self.model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

    def forward(self, x, x_gen):
        """
        Compute cosine similarity between embeddings.
        """
        embedding1 = self.model(x)
        embedding2 = self.model(x_gen)
        
        similarity = F.cosine_similarity(embedding1, embedding2, dim=1).mean()
        return similarity


if __name__ == "__main__":
    # Specify the device (GPU if available, otherwise CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize the FaceNetLoss module with the specified device
    face_net_loss = FaceNetLoss(device=device)

    # Load images and convert them to tensors
    image1 = cv2.imread('CelebaHQMaskDataset/test/images/27.jpg')
    image2 = cv2.imread('CelebaHQMaskDataset/test/images/27.jpg')
    image1 = torch.from_numpy(image1).permute(2, 0, 1).float() / 255.0
    image2 = torch.from_numpy(image2).permute(2, 0, 1).float() / 255.0

    # Expand dimensions to match the expected input shape of the model
    input_tensor = image1.unsqueeze(0).repeat(8, 1, 1, 1).to(device)
    output_tensor = image2.unsqueeze(0).repeat(8, 1, 1, 1).to(device)

    # Calculate similarity using the FaceNetLoss module
    similarity = face_net_loss(input_tensor, output_tensor)
    print("Cosine Similarity between embeddings:", similarity.item())

    # Add noise or apply augmentation to the images
    noise = torch.randn_like(output_tensor) * 0.8  # Adding Gaussian noise
    images_augmented = torch.clamp(output_tensor + noise, 0, 1)  # Clamp values between 0 and 1

    # Calculate similarity using the FaceNetLoss module with the augmented images
    similarity_augmented = face_net_loss(input_tensor, images_augmented)
    print("Cosine Similarity between augmented embeddings:", similarity_augmented.item())

    print(input_tensor.shape)