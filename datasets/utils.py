import os
import cv2
import numpy as np
import gdown
import shutil
import zipfile
from tqdm import tqdm

NUM_OF_IMAGES_IN_CELEBAHQ = 30000
MASKS_LABEL_LIST_CELEBAHQ = ['skin', 'nose', 'eye_g', 'l_eye', 'r_eye', 'l_brow', 'r_brow', 'l_ear', 'r_ear', 'mouth', 'u_lip', 'l_lip', 'hair', 'hat', 'ear_r', 'neck_l', 'neck', 'cloth']


def make_folder(path):
    if not os.path.exists(os.path.join(path)):
        os.makedirs(os.path.join(path))


def create_celebahq_masks(masks_path, save_path, force_create=False, num_of_images=NUM_OF_IMAGES_IN_CELEBAHQ):
    make_folder(save_path)
    # if the masks folder already exists, check if all the masks are created if so skip the creation
    if os.path.exists(save_path) and not force_create:
        num_files = len(os.listdir(save_path))
        if num_files >= num_of_images:
            print('CelebaHQ masks already created, skipping creation')
            return
    with tqdm(total=num_of_images, desc="Creating CelebaHQ masks", unit="image") as pbar:
        for k in range(num_of_images):
            folder_num = k // 2000
            im_base = np.zeros((512, 512))
            for idx, label in enumerate(MASKS_LABEL_LIST_CELEBAHQ):
                filename = os.path.join(masks_path, str(folder_num), str(k).rjust(5, '0') + '_' + label + '.png')
                if os.path.exists(filename):
                    im = cv2.imread(filename)
                    im = im[:, :, 0]
                    im_base[im != 0] = (idx + 1)

            filename_save = os.path.join(save_path, str(k) + '.png')
            cv2.imwrite(filename_save, im_base)
            pbar.update(1)


def download_celebhq_masks(gdrive_file_id, save_path, force_download=False, force_extract=False):
    make_folder(save_path)
    url = f'https://drive.google.com/uc?id={gdrive_file_id}&export=download'
    zip_file = os.path.join(save_path, 'CelebaHQMask.zip')
    if not os.path.exists(zip_file) or force_download:
        print('Downloading CelebaHQMasks dataset')
        gdown.download(url, zip_file, quiet=False)
    else:
        print('CelebaHQ masks already downloaded, skipping download')

    dataset_src_folder = None
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        file_list = zip_ref.namelist()
        folder_list = [file.split('/')[0] for file in file_list if '/' in file]
        unique_folders = list(set(folder_list))
        if len(unique_folders) == 1:
            dataset_src_folder = unique_folders[0]
        if dataset_src_folder:
            if os.path.exists(os.path.join(save_path, dataset_src_folder)) and not force_extract:
                print('CelebaHQ masks already extracted, skipping extraction')
                return os.path.join(save_path, dataset_src_folder)
        with tqdm(total=len(file_list), desc="Extracting CelebaHQMasks dataset", unit="files") as pbar:
            zip_ref.extractall(save_path, members=_progress_bar(zip_ref.infolist(), pbar))
    return os.path.join(save_path, dataset_src_folder) if dataset_src_folder else None


def _progress_bar(members, pbar):
    for member in members:
        yield member
        pbar.update(1)


def _create_spilliting_folders(dest_folder):
    make_folder(dest_folder)
    make_folder(os.path.join(dest_folder, 'train'))
    make_folder(os.path.join(dest_folder, 'test'))
    make_folder(os.path.join(dest_folder, 'train', 'images'))
    make_folder(os.path.join(dest_folder, 'train', 'masks'))
    make_folder(os.path.join(dest_folder, 'test', 'images'))
    make_folder(os.path.join(dest_folder, 'test', 'masks'))


def split_celebhqmasks_train_test(src_img_folder, src_masks_folder, dest_folder, train_ratio=0.9, force_split=False):
    # create image and mask folders for both train and test
    _create_spilliting_folders(dest_folder)

    # list and sort the src images
    src_images = sorted(os.listdir(src_img_folder), key=lambda x: int(x.split('.')[0]))
    src_masks = sorted(os.listdir(src_masks_folder), key=lambda x: int(x.split('.')[0]))
    min_size = min(len(src_images), len(src_masks))

    # zip the images and masks
    src_images = list(zip(src_images[:min_size], src_masks[:min_size]))

    # shuffle the images
    np.random.shuffle(src_images)

    # copy the images and masks to the train and test folders
    num_images = len(src_images)
    num_train_images = int(train_ratio * num_images)

    # check if num of images in train is equal or greater than num_train_images, if yes skip the split
    exist_num_train_images = len(os.listdir(os.path.join(dest_folder, 'train', 'images')))
    exist_num_test_images = len(os.listdir(os.path.join(dest_folder, 'test', 'images')))
    if exist_num_train_images + exist_num_test_images >= num_images and not force_split:
        print('CelebaHQ images already split, skipping split')
        return os.path.join(dest_folder, 'train'), os.path.join(dest_folder, 'test')

    else:
        shutil.rmtree(os.path.join(dest_folder, 'train'))
        shutil.rmtree(os.path.join(dest_folder, 'test'))
        _create_spilliting_folders(dest_folder)

    with tqdm(total=num_images, desc="Splitting images and masks", unit="image") as pbar:
        for i, (img, mask) in enumerate(src_images):
            if i < num_train_images:
                shutil.copy(os.path.join(src_img_folder, img), os.path.join(dest_folder, 'train', 'images', img))
                shutil.copy(os.path.join(src_masks_folder, mask), os.path.join(dest_folder, 'train', 'masks', mask))
            else:
                shutil.copy(os.path.join(src_img_folder, img), os.path.join(dest_folder, 'test', 'images', img))
                shutil.copy(os.path.join(src_masks_folder, mask), os.path.join(dest_folder, 'test', 'masks', mask))
            pbar.update(1)

    return os.path.join(dest_folder, 'train'), os.path.join(dest_folder, 'test')

