"""
Script to generate an augmented version of the 10k cats dataset.
Usage:
    1. Get 10k cats dataset at https://web.archive.org/web/20150520175645/http://137.189.35.203/WebUI/CatDatabase/catData.html
    2. Extract it somewhere, e.g. at /foo/bar/10k_cats
       That directory should then have the direct subdirectories CAT_00 to CAT_06.
    3. Start the script with:
           python generate_dataset.py --path="/foo/bar/10k_cats"
       replace "/foo/cats" with your path
"""
from __future__ import print_function, division
import os
import random
import numpy as np
import argparse
from dataset import Dataset
from scipy import misc

random.seed(42)
np.random.seed(42)

PADDING = 30
AUGMENTATIONS = 9
SCALE = 64

WRITE_UNAUG = True
WRITE_AUG = True
WRITE_UNAUG_TO = "out_unaug_%dx%d" % (SCALE, SCALE)
WRITE_AUG_TO = "out_aug_%dx%d" % (SCALE, SCALE)

def main():
    """Main function. Normalizes and augments the 10k cats dataset."""

    parser = argparse.ArgumentParser(description="Normalize and augment the 10k cats dataset.")
    parser.add_argument("--path", required=True, help="Path to your dataset directory, " \
                                                      "should contain CAT_* folders")
    args = parser.parse_args()

    subdir_names = ["CAT_00", "CAT_01", "CAT_02", "CAT_03", "CAT_04", "CAT_05", "CAT_06"]
    subdirs = [os.path.join(args.path, subdir) for subdir in subdir_names]

    dataset = Dataset(subdirs)

    for img_idx, image in enumerate(dataset.get_images()):
        print("Image %d" % (img_idx,))

        # debug code that shows the original image with face rectangles and keypoints
        #img = image.copy()
        #img.draw_keypoints()
        #img.draw_face_rectangles()
        #img.show()

        image.remove_rotation()

        # debug code that shows the cat image with face rectangles and keypoints after
        # rotation was removed (so that eyeline is parallel to x-axis)
        #img = image.copy()
        #img.draw_face_rectangles()
        #img.draw_keypoints()
        #img.show()

        # get the face with some pixels of padding around it (padding is useful for the
        # augmentation)
        face_orig = image.extract_face(pad=PADDING)
        faces = [face_orig]

        # augment
        faces.extend(face_orig.augment(AUGMENTATIONS,
                                       hflip=True, vflip=False,
                                       scale_to_percent=(0.93, 1.08), scale_axis_equally=True,
                                       rotation_deg=8, shear_deg=0,
                                       translation_x_px=4, translation_y_px=4,
                                       brightness_change=0.15, noise_mean=0.0, noise_std=0.02))

        # save unaugmented face + augmentations
        for aug_idx, face in enumerate(faces):
            # remove the padding around the face, it was only useful for the augmentation process
            face.unpad(pad=PADDING)

            # save images
            # one folder containing only unaugmented versions
            # one folder containing augmented versions
            filename = "{:0>6}_{:0>3}.jpg".format(img_idx, aug_idx)
            if WRITE_UNAUG and aug_idx == 0:
                face_cp = face.copy()
                face_cp.resize(new_height=SCALE, new_width=SCALE)
                misc.imsave(os.path.join(WRITE_UNAUG_TO, filename), face_cp.image_arr)
            if WRITE_AUG:
                face_cp = face.copy()
                face_cp.resize(new_height=SCALE, new_width=SCALE)
                misc.imsave(os.path.join(WRITE_AUG_TO, filename), face_cp.image_arr)

if __name__ == "__main__":
    main()
