import os
import cv2
import numpy as np


NUM_OF_IMAGES_IN_CELEBAHQ = 30000
MASKS_LABEL_LIST_CELEBAHQ = ['skin', 'nose', 'eye_g', 'l_eye', 'r_eye', 'l_brow', 'r_brow', 'l_ear', 'r_ear', 'mouth', 'u_lip', 'l_lip', 'hair', 'hat', 'ear_r', 'neck_l', 'neck', 'cloth']


def make_folder(path):
    if not os.path.exists(os.path.join(path)):
        os.makedirs(os.path.join(path))


def create_celebahq_masks(masks_path, save_path):
    make_folder(save_path)
    for k in range(NUM_OF_IMAGES_IN_CELEBAHQ):
        folder_num = k // 2000
        im_base = np.zeros((512, 512))
        for idx, label in enumerate(MASKS_LABEL_LIST_CELEBAHQ):
            filename = os.path.join(masks_path, str(folder_num), str(k).rjust(5, '0') + '_' + label + '.png')
            if os.path.exists(filename):
                im = cv2.imread(filename)
                im = im[:, :, 0]
                im_base[im != 0] = (idx + 1)

        filename_save = os.path.join(save_path, str(k) + '.png')
        print(filename_save)
        cv2.imwrite(filename_save, im_base)


if __name__ == "__main__":
    folder_base = '/Users/ido.nahum/Downloads/CelebAMask-HQ/CelebAMask-HQ-mask-anno'
    folder_save = '/Users/ido.nahum/dev/photoVerse/CelebaHQMask/masks'
    create_celebahq_masks(folder_base, folder_save)
