import argparse
import os
from datasets.utils import download_celebhq_masks, create_celebahq_masks, split_celebhqmasks_train_test
# from utils.arcface_utils import download_face_detection_models

parser = argparse.ArgumentParser(description="Downloading, extracting and splitting CelebHQ dataset, in addition, downloading face detection models")
parser.add_argument(
    "--save_path",
    type=str,
    default='./CelebaHQMaskDataset',
    help="The path to download extract and split the CelebHQ dataset.",
)

parser.add_argument(
    "--gdrive_file_id",
    type=str,
    default='1RGiGeumP_xVDN4RBC0K2m7Vh43IKSUPn',
    help="Google Drive file ID for CelebaHQMasks dataset.",
)

parser.add_argument(
    "--dataset_src_img_folder",
    type=str,
    default='CelebA-HQ-img',
    help="The sub folder containing the CelebA-HQ images.",
)

parser.add_argument(
    "--dataset_src_masks_folder",
    type=str,
    default='CelebAMask-HQ-mask-anno',
    help="The sub folder containing the CelebA-HQ masks.",
)

parser.add_argument(
    "--dataset_src_folder",
    type=str,
    default='CelebAMask-HQ',
    help="The main folder containing the CelebA-HQ dataset. Will be used if can automaticly find the main folder after extraction",
)

parser.add_argument(
    "--train_ratio",
    type=float,
    default=0.9,
    help="The ratio of images to be used for training.",
)

parser.add_argument(
    "--force_download",
    action='store_true',
    help="Force download the dataset even if it already exists.",
)

parser.add_argument(
    "--force_extract",
    action='store_true',
    help="Force extract the dataset even if it already exists.",
)

parser.add_argument(
    "--force_mask_creation",
    action='store_true',
    help="Force create the masks even if they already exists.",
)

parser.add_argument(
    "--num_of_samples",
    type=int,
    default=30000,
    help="Number of samples to use for the dataset creation, max is 30000.",
    choices=range(100, 30001),
)

parser.add_argument(
    "--force_split",
    action='store_true',
    help="Force split the dataset even if it already exists.",
)

# parser.add_argument(
#     "--gdrive_file_id_face_detection",
#     type=str,
#     default='1HJ4MlkOOqUQwxaRCPaCVh22Wpdi_Uhwj',
#     help="Google Drive file ID for face detection models.",
# )

# parser.add_argument(
#     "--face_detection_models_folder",
#     type=str,
#     default='./model_repository',
#     help="The folder to download the face detection models.",
# )

if __name__ == "__main__":
    args = parser.parse_args()

    # download and extract the dataset into the save path
    dataset_src_folder = download_celebhq_masks(args.gdrive_file_id, args.save_path, args.force_download, args.force_extract)

    # if cant automaticly find the main folder after extraction use the provided one
    if dataset_src_folder is None:
        dataset_src_folder = args.dataset_src_folder

    # create the masks from all masks per label files
    src_masks_folder = os.path.join(dataset_src_folder, 'masks')
    create_celebahq_masks(os.path.join(dataset_src_folder, args.dataset_src_masks_folder), src_masks_folder, args.force_mask_creation, args.num_of_samples)

    # split the images and masks into train and test folders
    src_img_folder = os.path.join(dataset_src_folder, args.dataset_src_img_folder)
    train_folder, test_folder = split_celebhqmasks_train_test(src_img_folder, src_masks_folder, args.save_path, args.train_ratio, args.force_split)

    # download the face detection models - it must be in models directory
    # arcface_models_folder = os.path.join(args.face_detection_models_folder, 'models')
    # os.makedirs(arcface_models_folder, exist_ok=True)
    # download_face_detection_models(arcface_models_folder, args.gdrive_file_id_face_detection)

    print(f'Train folder: {train_folder}')
    print(f'Test folder: {test_folder}')
    # print(f'ArcFace models folder: {args.face_detection_models_folder}')