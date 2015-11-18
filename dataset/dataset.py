"""
Helper classes to handle the process of normalizing and augmenting the 10k cats dataset.

Classes:
    Dataset                 Handles loading of cat images with keypoints (e.g. eyes, ears)
    ImageWithKeypoints      Container for one example images with its keypoints,
                            supports e.g. resizing the image, showing it (in a window), drawing
                            points/rectangles on it or augmenting it.
    Keypoints               Helper class to handle the keypoints of one image,
                            supports e.g. shifting/translating them by N pixels, warping them
                            via an affine transformation matrix, flipping them or calculating
                            a face rectangle from them.
    Point2D                 A class to encapsulate a (y, x) coordinate.
    PointsList              A list of Point2D
    Rectangle               A rectangle in an image, used for the face rectangles.

Note that coordinates are usually provided as (y, x) not (x, y).
"""
from __future__ import print_function, division
import os
import re
import math
import random
from scipy import misc
import numpy as np
from ImageAugmenter import create_aug_matrices
from skimage import transform as tf
from skimage import color

WARP_KEYPOINTS_MODE = "constant"
WARP_KEYPOINTS_CVAL = 0.0
WARP_KEYPOINTS_INTERPOLATION_ORDER = 1

class Dataset(object):
    """Helper class to load images with facial keypoints."""
    def __init__(self, dirs):
        """Initialize the class.
        Args:
            dirs    A list of directories (filepaths) to load from."""
        self.dirs = dirs
        self.fps = self.get_image_filepaths()

    def get_images(self, start_at=None, count=None):
        """Load images with keypoints.
        Args:
            start_at    Index of first image to load.
            count       Maximum number of images to load.
        Returns:
            List of ImageWithKeypoints (generator)"""
        start_at = 0 if start_at is None else start_at
        end_at = len(self.fps) if count is None else start_at+count
        for fp in self.fps[start_at:end_at]:
            image = misc.imread(fp)
            keypoints = Keypoints(self.get_keypoints(fp, image.shape[0], image.shape[1]))
            yield ImageWithKeypoints(image, keypoints)

    def get_image_filepaths(self):
        """Loads filepaths of example images.
        Returns:
            List of strings (filepaths)"""
        result_img = []
        for fp_dir in self.dirs:
            fps = [f for f in os.listdir(fp_dir) if os.path.isfile(os.path.join(fp_dir, f))]
            fps = [os.path.join(fp_dir, f) for f in fps]
            fps_img = [fp for fp in fps if re.match(r".*\.jpg$", fp)]
            fps_img = [fp for fp in fps if os.path.isfile("%s.cat" % (fp,))]
            result_img.extend(fps_img)

        return result_img

    def get_keypoints(self, image_filepath, image_height, image_width):
        """Loads the keypoints of one image.
        Args:
            image_filepath  Filepath of the image for which to load keypoints.
            image_height    Height of the image.
            image_width     Width of the image.
        Returns:
            Numpy array of shape (19,)"""
        fp_keypoints = "%s.cat" % (image_filepath,)
        if not os.path.isfile(fp_keypoints):
            raise Exception("Could not find keypoint coordinates for image '%s'." \
                            % (image_filepath,))
        else:
            coords_raw = open(fp_keypoints, "r").readlines()[0].strip().split(" ")
            coords_raw = [abs(int(coord)) for coord in coords_raw]
            keypoints_arr = np.zeros((9*2,), dtype=np.uint16)
            for i in range(1, len(coords_raw), 2): # first element is the number of coords
                y = clip(0, coords_raw[i+1], image_height-1)
                x = clip(0, coords_raw[i], image_width-1)
                keypoints_arr[(i-1)] = y
                keypoints_arr[(i-1) + 1] = x
            return keypoints_arr

class ImageWithKeypoints(object):
    """Container for an example image and its keypoints."""
    def __init__(self, image_arr, keypoints):
        """Instantiate an object.
        Args:
            image_arr   Numpy array of the image, shape (height, width, channels)
            keypoints   Keypoints object"""
        assert len(image_arr.shape) == 3
        assert image_arr.shape[2] == 3
        self.image_arr = image_arr
        self.keypoints = keypoints

    def copy(self):
        """Copy the object.
        Returns: ImageWithKeypoints"""
        return ImageWithKeypoints(np.copy(self.image_arr), self.keypoints.copy())

    def get_height(self):
        """Get the image height.
        Returns: height, integer"""
        return self.image_arr.shape[0]

    def get_width(self):
        """Get the image width.
        Returns: width, integer"""
        return self.image_arr.shape[1]

    def get_center(self):
        """Get the center of the image.
        Returns: Point2D"""
        y, x = self.get_height()/2, self.get_width()/2
        return Point2D(y=int(y), x=int(x))

    def resize(self, new_height, new_width):
        """Resize the image to given height and width.
        Args:
            new_height  Height to resize to.
            new_width   Width to resize to."""
        self.keypoints.normalize(self)
        # unclear in scipy doc if (new_height, new_width) or (new_width, new_height) is correct
        #print(self.image_arr.shape)
        #self.image_arr = misc.imresize(np.rollaxis(self.image_arr, 2, 0), (new_height, new_width))
        self.image_arr = misc.imresize(self.image_arr, (new_height, new_width))
        #self.image_arr = np.rollaxis(self.image_arr, 0, 3)
        #print(self.image_arr.shape)
        self.keypoints.unnormalize(self)

    def grayscale(self):
        """Converts the image to grayscale."""
        self.image_arr = color.rgb2gray(self.image_arr)

    def unpad(self, pad):
        """Removes padding around the image. Updates keypoints accordingly.
        Args: pad: Number of pixels of padding to remove"""
        self.image_arr = self.image_arr[pad:self.get_height()-pad, pad:self.get_width()-pad, ...]
        self.keypoints.shift_y(-pad, self)
        self.keypoints.shift_x(-pad, self)

    def remove_rotation(self):
        """Removes the image's rotation by aligning its eyeline parallel to the x axis."""
        angle = math.radians(self.keypoints.get_angle_between_eyes(normalize=False))

        # move eyes center to top left of image
        eyes_center = self.keypoints.get_eyes_center()
        img_center = self.get_center()
        matrix_to_topleft = tf.SimilarityTransform(translation=[-eyes_center.x, -eyes_center.y])

        # rotate the image around the top left corner by -$angle degrees
        matrix_transforms = tf.AffineTransform(rotation=-angle)

        # move the face to the center of the image
        # this protects against parts of the face leaving the image (because of the rotation)
        matrix_to_center = tf.SimilarityTransform(translation=[img_center.x, img_center.y])

        # combine to one affine transformation
        matrix = matrix_to_topleft + matrix_transforms + matrix_to_center
        matrix = matrix.inverse

        # apply transformations
        new_image = tf.warp(self.image_arr, matrix, mode="nearest")
        new_image = np.array(new_image * 255, dtype=np.uint8)
        self.image_arr = new_image

        # create new image with N channels for N coordinates
        # mark each coordinate's pixel in the respective channel
        # rotate
        # read out new coordinates (after rotation)
        self.keypoints.warp(self, matrix)

        if self.keypoints.mouth().y < self.keypoints.left_eye().y:
            print("Warning: mouth is above left eye")
            # unclear where this problem comes from, fix it with flipping for now
            #self.image_arr = np.flipud(self.image_arr)
            #self.keypoints.flipud(self)
        if self.keypoints.right_eye().x < self.keypoints.left_eye().x:
            print("Warning: right eye is left, left eye is right")

    def extract_rectangle(self, rect, pad):
        """Extracts a rectangle within the image as a new ImageWithKeypoints.
        Args:
            rect    Rectangle object
            pad     Padding in pixels around the rectangle
        Returns:
            ImageWithKeypoints"""
        pad_black_top = 0
        pad_black_right = 0
        pad_black_bottom = 0
        pad_black_left = 0

        if rect.tl_y - pad < 0:
            pad_black_top = abs(rect.tl_y - pad)
        if rect.tl_x - pad < 0:
            pad_black_left = abs(rect.tl_x - pad)
        if rect.br_y + pad > (self.get_height() - 1):
            pad_black_bottom = (rect.br_y + pad) - (self.get_height() - 1)
        if rect.br_x + pad > (self.get_width() - 1):
            pad_black_right = (rect.br_x + pad) - (self.get_width() - 1)

        tl_y = clip(0, rect.tl_y - pad, self.get_height()-1)
        tl_x = clip(0, rect.tl_x - pad, self.get_width()-1)
        br_y = clip(0, rect.br_y + pad, self.get_height()-1)
        br_x = clip(0, rect.br_x + pad, self.get_width()-1)

        img_rect = self.image_arr[tl_y:br_y+1, tl_x:br_x+1, ...]
        keypoints = self.keypoints.copy()
        img = ImageWithKeypoints(img_rect, keypoints)
        keypoints.shift_y(-tl_y, img)
        keypoints.shift_x(-tl_x, img)

        img.image_arr = np.pad(img.image_arr, ((pad_black_top, pad_black_bottom), \
                                               (pad_black_left, pad_black_right), \
                                               (0, 0)), \
                                               mode="median")
        keypoints.shift_y(pad_black_top, img)
        keypoints.shift_x(pad_black_left, img)

        return img

    def extract_face(self, pad):
        """Extracts the cat face within the image.
        Args:
            pad     Padding in pixels around the face.
        Returns:
            ImageWithKeypoints"""
        face_rect = self.keypoints.get_rectangle(self)
        return self.extract_rectangle(face_rect, pad)

    def augment(self, n, hflip=False, vflip=False, scale_to_percent=1.0, scale_axis_equally=True,
                rotation_deg=0, shear_deg=0, translation_x_px=0, translation_y_px=0,
                brightness_change=0.0, noise_mean=0.0, noise_std=0.0):
        """Generates randomly augmented versions of the image.
        Also augments the keypoints accordingly.

        Args:
            n                   Number of augmentations to generate.
            hflip               Allow horizontal flipping (yes/no).
            vflip               Allow vertical flipping (yes/no)
            scale_to_percent    How much scaling/zooming to allow. Values are around 1.0.
                                E.g. 1.1 is -10% to +10%
                                E.g. (0.7, 1.05) is -30% to 5%.
            scale_axis_equally  Whether to enforce equal scaling of x and y axis.
            rotation_deg        How much rotation to allow. E.g. 5 is -5 degrees to +5 degrees.
            shear_deg           How much shearing to allow.
            translation_x_px    How many pixels of translation along the x axis to allow.
            translation_y_px    How many pixels of translation along the y axis to allow.
            brightness_change   How much change in brightness to allow. Values are around 0.0.
                                E.g. 0.2 is -20% to +20%.
            noise_mean          Mean value of gaussian noise to add.
            noise_std           Standard deviation of gaussian noise to add.
        Returns:
            List of ImageWithKeypoints
        """
        assert n >= 0
        result = []
        if n == 0:
            return result

        matrices = create_aug_matrices(n,
                                       img_width_px=self.get_width(),
                                       img_height_px=self.get_height(),
                                       scale_to_percent=scale_to_percent,
                                       scale_axis_equally=scale_axis_equally,
                                       rotation_deg=rotation_deg,
                                       shear_deg=shear_deg,
                                       translation_x_px=translation_x_px,
                                       translation_y_px=translation_y_px)
        for i in range(n):
            img = self.copy()
            matrix = matrices[i]

            # random horizontal / vertical flip
            if hflip and random.random() > 0.5:
                img.image_arr = np.fliplr(img.image_arr)
                img.keypoints.fliplr(img)
            if vflip and random.random() > 0.5:
                img.image_arr = np.flipud(img.image_arr)
                img.keypoints.flipud(img)

            # random brightness adjustment
            by_percent = random.uniform(1.0 - brightness_change, 1.0 + brightness_change)
            img.image_arr = img.image_arr * by_percent

            # gaussian noise
            # numpy requires a std above 0
            if noise_std > 0:
                img.image_arr = img.image_arr \
                                + (255 * np.random.normal(noise_mean, noise_std,
                                                          (img.image_arr.shape)))

            # clip to 0-255
            img.image_arr = np.clip(img.image_arr, 0, 255).astype(np.uint8)

            arr = tf.warp(img.image_arr, matrix, mode="nearest") # projects to float 0-1
            img.image_arr = np.array(arr * 255, dtype=np.uint8)
            img.keypoints.warp(img, matrix)
            result.append(img)

        return result

    def draw_rectangle(self, rect, color_tuple=None):
        """Draw a rectangle with given color onto the image.
        Args:
            rect            The rectangle object
            color_tuple     Color of the rectangle, e.g. (255, 0, 0) for red."""
        self.draw_rectangles([rect], color_tuple=color_tuple)

    def draw_rectangles(self, rects, color_tuple=None):
        """Draw several rectangles onto the image."""
        if color_tuple is None:
            color_tuple = (255, 0, 0)

        for rect in rects:
            for x in range(rect.tl_x, rect.br_x+1):
                self.image_arr[rect.tl_y, x, ...] = color_tuple
                self.image_arr[rect.br_y, x, ...] = color_tuple
            for y in range(rect.tl_y, rect.br_y+1):
                self.image_arr[y, rect.tl_x, ...] = color_tuple
                self.image_arr[y, rect.br_x, ...] = color_tuple

    def draw_face_rectangles(self):
        """Draw all face rectangles onto the image according to the 5 existing methods.
        Colors:
            Green = Method 0
            Blue = Method 1
            Red = Method 2
            Yellow = Method 3
            Cyan = Method 4
        """
        self.draw_rectangle(self.keypoints.get_rectangle(self, method=0), color_tuple=(0, 255, 0))
        self.draw_rectangle(self.keypoints.get_rectangle(self, method=1), color_tuple=(0, 0, 255))
        self.draw_rectangle(self.keypoints.get_rectangle(self, method=2), color_tuple=(255, 0, 0))
        self.draw_rectangle(self.keypoints.get_rectangle(self, method=3), color_tuple=(255, 255, 0))
        self.draw_rectangle(self.keypoints.get_rectangle(self, method=4), color_tuple=(0, 255, 255))

    def draw_point(self, pnt, color_tuple=None):
        """Draw a point onto the image."""
        self.draw_point([pnt], color_tuple=color_tuple)

    def draw_points(self, pnts, color_tuple=None):
        """Draw several points onto the image."""
        if color_tuple is None:
            color_tuple = (255, 0, 0)

        height = self.get_height()
        width = self.get_width()

        for pnt in pnts:
            self.image_arr[pnt.y, clip(0, pnt.x-1, width-1) \
                           :clip(0, pnt.x+2, width-1), ...] = (255, 0, 0)
            self.image_arr[clip(0, pnt.y-1, height-1) \
                           :clip(0, pnt.y+2, height-1), pnt.x, ...] = (255, 0, 0)

    def draw_keypoints(self, color_tuple=None):
        """Draw all image's keypoints as crosses."""
        self.draw_points(self.keypoints.get_points(), color_tuple=color_tuple)

    def show(self):
        """Show the image in a window."""
        misc.imshow(self.image_arr)

    def to_array(self):
        """Return the image content's numpy array.
        Returns: numpy array of shape (height, width, channels)"""
        return self.image_arr

class Keypoints(object):
    """Helper class to encapsulate the facial keypoints.

    Existing keypoints:
        point number | meaning
        1 = left eye
        2 = right eye
        3 = mouth
        4 = left ear 1 (left side start)
        5 = left ear 2 (tip)
        6 = left ear 3 (right side start)
        7 = right ear 1 (left side start)
        8 = right ear 2 (tip)
        9 = right ear 3 (right side start)
    (left/right when looking at cat (not from the perspective of the cat))

    Rough outline on image (frontal perspective on cat):

          5             8
             6      7
        4               9

             1     2

                3

    """
    def __init__(self, keypoints_arr, is_normalized=False):
        """Instantiate a new keypoints object.
        Args:
            keypoints_arr   Numpy array of the keypoints of shape (18,)
            is_normalized   Whether the keypoints are in the range 0-1 (true) or have integer
                            pixel values.
        """
        assert len(keypoints_arr.shape) == 1
        assert len(keypoints_arr) == 9*2
        if is_normalized:
            assert keypoints_arr.dtype == np.float32 and all([0 <= v <= 1.0 for v in keypoints_arr])
        else:
            assert keypoints_arr.dtype == np.uint16 and all([v >= 0 for v in keypoints_arr])
        self.keypoints_arr = keypoints_arr
        self.is_normalized = is_normalized

    def copy(self):
        """Creates a copy of the keypoints object.
        Returns: Keypoints"""
        return Keypoints(np.copy(self.keypoints_arr))

    def normalize(self, image):
        """Normalizes the keypoint value to 0-1 floats with respect to the given image's dimensions.
        Args:
            image   ImageWithKeypoints"""
        assert not self.is_normalized
        height = image.get_height()
        width = image.get_width()
        for i in range(0, len(self.keypoints_arr), 2):
            self.keypoints_arr[i] = self.keypoints_arr[i] / height
            self.keypoints_arr[i+1] = self.keypoints_arr[i+1] / width
        self.is_normalized = True

    def unnormalize(self, image):
        """Converts back from 0-1 floats to integer pixel values with respect to the given
        image's dimensions.
        Args:
            image   ImageWithKeypoints"""
        assert self.is_normalized
        height = image.get_height()
        width = image.get_width()
        for i in range(0, len(self.keypoints_arr), 2):
            self.keypoints_arr[i] = self.keypoints_arr[i] * height
            self.keypoints_arr[i+1] = self.keypoints_arr[i+1] * width
        self.is_normalized = False

    def left_eye(self):
        """Returns the coordinates of the left eye as Point2D."""
        return self.get_nth_keypoint(0)

    def right_eye(self):
        """Returns the coordinates of the right eye as Point2D."""
        return self.get_nth_keypoint(1)

    def mouth(self):
        """Returns the coordinates of the mouth eye as Point2D."""
        return self.get_nth_keypoint(2)

    def get_nth_keypoint(self, nth):
        """Returns the coordinates of the n-th (starting with 0) keypoint as Point2D."""
        y = self.keypoints_arr[nth*2]
        x = self.keypoints_arr[nth*2 + 1]
        if self.is_normalized:
            y = float(y)
            x = float(x)
        else:
            y = int(y)
            x = int(x)
        return Point2D(y=y, x=x)

    def get_face_center(self):
        """Returns the coordinates of the face center as Point2D."""
        face_center_x = (self.left_eye().x + self.right_eye().x + self.mouth().x) / 3
        face_center_y = (self.left_eye().y + self.right_eye().y + self.mouth().y) / 3
        face_center = Point2D(y=int(face_center_y), x=int(face_center_x))
        return face_center

    def get_eyes_center(self):
        """Returns the coordinates of center between the eyes as Point2D."""
        x = (self.left_eye().x + self.right_eye().x) / 2
        y = (self.left_eye().y + self.right_eye().y) / 2
        return Point2D(y=int(y), x=int(x))

    def get_angle_between_eyes(self, normalize):
        """Returns with angle of the eyeline with respect to the x axis in degrees.
        E.g. a value of -5 indicates that the face is rotated by 5 degrees counter clock wise.
        Args:
            normalize   Whether to normalize the value to the range of -1 (-180) to +1 (+180).
        Returns:
            Angle in degrees relative to x axis"""
        left_eye = self.left_eye().to_array()
        right_eye = self.right_eye().to_array()
        # conversion to int is here necessary, otherwise eyes_vector cant have negative values
        eyes_vector = right_eye.astype(np.int) - left_eye.astype(np.int)
        x_axis_vector = np.array([0, 1])
        angle = angle_between(x_axis_vector, eyes_vector)
        angle_deg = math.degrees(angle)

        assert -180 <= angle_deg <= 180, angle_deg
        if normalize:
            return angle_deg / 180
        else:
            return angle_deg

    def get_points(self):
        """Returns all facial keypoints as Point2D-s.
        Returns: List of Point2D."""
        result = []
        for i in range(0, len(self.keypoints_arr)//2):
            result.append(self.get_nth_keypoint(i))
        return result

    def get_min_x(self):
        """Returns the minimum x value among all facial keypoints."""
        return min([point.x for point in self.get_points()])

    def get_min_y(self):
        """Returns the minimum y value among all facial keypoints."""
        return min([point.y for point in self.get_points()])

    def get_max_x(self):
        """Returns the maximum x value among all facial keypoints."""
        return max([point.x for point in self.get_points()])

    def get_max_y(self):
        """Returns the maximum y value among all facial keypoints."""
        return max([point.y for point in self.get_points()])

    def shift_x(self, n_pixels, image):
        """Shifts all keypoints by N pixels on the x axis.
        Args:
            n_pixels    Shift by that number of pixels
            image       Image with maximum dimensions, i.e. dont shift further than image.width"""
        for i in range(0, len(self.keypoints_arr), 2):
            new_val = int(self.keypoints_arr[i+1]) + n_pixels
            new_val = clip(0, new_val, image.get_width()-1)
            self.keypoints_arr[i+1] = new_val

    def shift_y(self, n_pixels, image):
        """Shifts all keypoints by N pixels on the y axis.
        Args:
            n_pixels    Shift by that number of pixels
            image       Image with maximum dimensions, i.e. dont shift further than image.height"""
        for i in range(0, len(self.keypoints_arr), 2):
            new_val = int(self.keypoints_arr[i]) + n_pixels
            new_val = clip(0, new_val, image.get_height()-1)
            self.keypoints_arr[i] = new_val

    def warp(self, image, matrix):
        """Warp all keypoints according to an affine transformation matrix.
        Args:
            image   Image with maximum dimensions
            matrix  Affine transformation matrix from scikit-image."""
        points = self.get_points()
        for i, pnt in enumerate(points):
            pnt.warp(image, matrix)
            self.keypoints_arr[i*2:(i*2)+2] = [pnt.y, pnt.x]

    def fliplr(self, image):
        """Flip all keypoints horizontally.
        Args:
            image   Image with maximum dimensions."""
        for i in range(0, len(self.keypoints_arr), 2):
            self.keypoints_arr[i+1] = (image.get_width()-1) - self.keypoints_arr[i+1]
        # switch points
        # 9 with 4 (right ear 3, left ear 1)
        self._switch_points(9-1, 4-1)
        # 8 with 5 (right ear 2, left ear 2)
        self._switch_points(8-1, 5-1)
        # 7 with 6 (right ear 1, left ear 3)
        self._switch_points(7-1, 6-1)
        # 2 with 1 (right eye, left eye)
        self._switch_points(2-1, 1-1)

    def flipud(self, image):
        """Flip all keypoints vertically.
        Args:
            image   Image with maximum dimensions."""
        for i in range(0, len(self.keypoints_arr), 2):
            self.keypoints_arr[i] = (image.get_height()-1) - self.keypoints_arr[i]

    def _switch_points(self, index1, index2):
        """Switch the coordinates of two keypoints.
        Args:
            index1      Index of the first keypoint
            index1      Index of the second keypoint
        """
        y1 = self.keypoints_arr[index1*2]
        x1 = self.keypoints_arr[index1*2+1]
        y2 = self.keypoints_arr[index2*2]
        x2 = self.keypoints_arr[index2*2+1]
        self.keypoints_arr[index1*2] = y2
        self.keypoints_arr[index1*2+1] = x2
        self.keypoints_arr[index2*2] = y1
        self.keypoints_arr[index2*2+1] = x1

    def get_rectangle(self, image, method=4):
        """Generate face rectangles based on various methods.

        Face rectangles are rectangles around the facial keypoints that contain various parts
        of the face.
        Methods:
            - 0: Bounding box around all keypoints
            - 1: Rectangle 0, translated to the center of the face
            - 2: Rectangle 0, translated half-way to the center of the face
            - 3: Bounding box around the corners of Rectangle 0 and 2
            - 4: Rectangle 3, squared (this is the main rectangle used)

        Args:
            image   Image with maximum dimensions
            method  Index of the method
        Returns:
            Rectangle object
        """

        image_width = image.get_width()
        image_height = image.get_height()

        face_center = self.get_face_center()

        if method == 0:
            # rectangle 0: bounding box around provided keypoints
            return Rectangle(self.get_min_y(), self.get_min_x(), self.get_max_y(), self.get_max_x())
        elif method == 1:
            # rectangle 1: the same rectangle as rect 0, but translated to the center of the face
            rect = self.get_rectangle(image, method=0)
            rect_center = rect.get_center()
            diff_y = face_center.y - rect_center.y
            diff_x = face_center.x - rect_center.x

            min_x_fcenter = max(0, rect.tl_x + diff_x)
            min_y_fcenter = max(0, rect.tl_y + diff_y)
            max_x_fcenter = min(image_width-1, rect.br_x + diff_x)
            max_y_fcenter = min(image_height-1, rect.br_y + diff_y)

            return Rectangle(min_y_fcenter, min_x_fcenter, max_y_fcenter, max_x_fcenter)
        elif method == 2:
            # rectangle 2: the same rectangle as rect 0, but translated _half-way_ towards the
            # center of the face
            rect = self.get_rectangle(image, method=0)
            rect_center = rect.get_center()
            diff_y = face_center.y - rect_center.y
            diff_x = face_center.x - rect_center.x

            min_x_half = int(max(0, rect.tl_x + (diff_x/2)))
            min_y_half = int(max(0, rect.tl_y + (diff_y/2)))
            max_x_half = int(min(image_width-1, rect.br_x + (diff_x/2)))
            max_y_half = int(min(image_height-1, rect.br_y + (diff_y/2)))

            return Rectangle(min_y_half, min_x_half, max_y_half, max_x_half)
        elif method == 3:
            # rectangle 3: a merge between rect 0 and 2 rectangle, essentially a bounding box around
            # the corners of both rectangles

            rect0 = self.get_rectangle(image, method=0)
            rect2 = self.get_rectangle(image, method=2)

            min_x_merge = max(0, min(rect0.tl_x, rect2.tl_x))
            min_y_merge = max(0, min(rect0.tl_y, rect2.tl_y))
            max_x_merge = min(image_width-1, max(rect0.br_x, rect2.br_x))
            max_y_merge = min(image_height-1, max(rect0.br_y, rect2.br_y))

            return Rectangle(min_y_merge, min_x_merge, max_y_merge, max_x_merge)
        elif method == 4:
            # rectangle 4: like 3, but squared with Rectangle.square()

            rect3 = self.get_rectangle(image, method=3)
            rect3.square(image)
            return rect3
        else:
            raise Exception("Unknown rectangle generation method %d chosen." % (method,))

    def get_rectangles(self, image):
        """Returns all facial rectangles.
        Args: image: Image with maximum dimensions
        Returns: List of Rectangle"""
        return [self.get_rectangle(image, method=i) for i in range(0, 5)]

    def to_array(self):
        """Returns the keypoints as array of shape (18,)."""
        return self.keypoints_arr

    def __str__(self):
        """Converts object to string representation."""
        return str(self.keypoints_arr)

class PointsList(object):
    """A helper class encapsulating multiple Point2D."""

    def __init__(self, points):
        """Instantiates a new points list.
        Args:
            points  List of Point2D."""
        self.points = points

    def normalize(self, image):
        """Normalizes each point to 0-1 with respect to an image's dimensions."""
        for point in self.points:
            point.normalize(image)

    def unnormalize(self, image):
        """Unnormalizes each point from 0-1 to integer pixel values with respect to an
        image's dimensions."""
        for point in self.points:
            point.unnormalize(image)

    def any_normalized(self):
        """Returns whether any point in the list has normalized coordinates."""
        return any([point.is_normalized for point in self.points])

    def all_normalized(self):
        """Returns whether all points in the list have normalized coordinates."""
        return all([point.is_normalized for point in self.points])

    def to_array(self):
        """Returns the list of points as a numpy array of shape (nb_points*2)."""
        result = np.zeros((len(self.points)*2,), dtype=np.float32)
        for i, point in enumerate(self.points):
            result[i*2] = point.y
            result[i*2 + 1] = point.x
        return result

    def __str__(self):
        """Returns a string representation of this point list."""
        return str([str(pnt) for pnt in self.points])

class Point2D(object):
    """A helper class encapsulating a (y, x) coordinate."""

    def __init__(self, y, x, is_normalized=False):
        """Instantiate a new Point2D object.
        Args:
            y               Y-coordinate of point
            x               X-coordinate of point
            is_normalized   Whether the coordinates are normalized to 0-1 instead of integer
                            pixel values"""
        if is_normalized:
            assert isinstance(y, float), type(y)
            assert isinstance(x, float), type(x)
        else:
            assert isinstance(y, int), type(y)
            assert isinstance(x, int), type(x)
        self.y = y
        self.x = x
        self.is_normalized = is_normalized

    def normalize(self, image):
        """Normalize the integer pixel values to 0-1 with respect to an image's dimensions.
        Args: image: The image which's dimensions to use."""
        assert not self.is_normalized
        self.y /= image.shape[0]
        self.x /= image.shape[1]
        self.is_normalized = True

    def unnormalize(self, image):
        """Unnormalize the 0-1 coordinate value to integer pixel values with respect to an
        image's dimensions.
        Args: image: The image which's dimensions to use."""
        assert self.is_normalized
        self.y *= image.shape[0]
        self.x *= image.shape[1]
        self.is_normalized = False

    def warp(self, image, matrix):
        """Warp the point's coordinates according to an affine transformation matrix.
        Args:
            image   The image which's dimensions to use.
            matrix  The affine transformation matrix (from scikit-image)
        """
        assert not self.is_normalized

        # This method draws the point as a white pixel on a black image,
        # then warps that image according to the matrix
        # then reads out the new position of the pixel
        # (if its not found / outside of the image then the coordinates will be unchanged).
        # This is a very wasteful process as many pixels have to be warped instead of just one.
        # There is probably a better method for that, but I don't know it.
        image_pnt = np.zeros((image.get_height(), image.get_width()), dtype=np.uint8)
        image_pnt[self.y, self.x] = 255
        image_pnt_warped = tf.warp(image_pnt, matrix, mode=WARP_KEYPOINTS_MODE,
                                   cval=WARP_KEYPOINTS_CVAL,
                                   order=WARP_KEYPOINTS_INTERPOLATION_ORDER)
        maxindex = np.argmax(image_pnt_warped)
        if maxindex == 0 and image_pnt_warped[0, 0] < 0.5:
            # dont change coordinates
            #print("Note: Coordinate (%d, %d) not changed" % (self.y, self.x))
            pass
        else:
            (y, x) = np.unravel_index(maxindex, image_pnt_warped.shape)
            self.y = y
            self.x = x

    def to_array(self):
        """Returns the coordinate as a numpy array."""
        if self.is_normalized:
            return np.array([self.y, self.x], dtype=np.float32)
        else:
            return np.array([self.y, self.x], dtype=np.uint16)

    def __str__(self):
        """Returns a string representation of the coordinate."""
        if self.is_normalized:
            return "PN(%.4f, %.4f)" % (self.y, self.x)
        else:
            return "P(%d, %d)" % (self.y, self.x)

class Rectangle(object):
    """Class representing a rectangle in an image."""
    def __init__(self, tl_y, tl_x, br_y, br_x, is_normalized=False):
        """Instantiate a new rectangle.
        Args:
            tl_y            y-coordinate of top left corner
            tl_x            x-coordinate of top left corner
            br_y            y-coordinate of bottom right corner
            br_x            x-coordinate of bottom right corner
            is_normalized   Whether the coordinates are normalized to 0-1 instead of integer
                            pixel values"""
        assert tl_y >= 0 and tl_x >= 0 and br_y >= 0 and br_x >= 0
        assert tl_y < br_y and tl_x < br_x
        if is_normalized:
            assert all(isinstance(v, float) for v in [tl_y, tl_x, br_y, br_x])
        else:
            assert all(isinstance(v, int) for v in [tl_y, tl_x, br_y, br_x])

        self.tl_y = tl_y
        self.tl_x = tl_x
        self.br_y = br_y
        self.br_x = br_x
        self.is_normalized = is_normalized

    def get_width(self):
        """Returns the width of the rectangle."""
        return self.br_x - self.tl_x

    def get_height(self):
        """Returns the height of the rectangle."""
        return self.br_y - self.tl_y

    def get_center(self):
        """Returns the center of the rectangle as a Point2D."""
        y = self.tl_y + (self.get_height() / 2)
        x = self.tl_x + (self.get_width() / 2)
        if self.is_normalized:
            return Point2D(y=float(y), x=float(x), is_normalized=True)
        else:
            return Point2D(y=int(y), x=int(x), is_normalized=False)

    def square(self, image):
        """Squares the rectangle.
        It first adds columns/rows until the image's borders are reached.
        Then deletes columns/rows until the rectangle is squared.
        Args:
            image   Image which's dimensions to use, i.e. rectangle won't be increased in size
                    beyond that image's height/width.
        """
        assert not self.is_normalized

        img_height = image.get_height()
        img_width = image.get_width()
        height = self.get_height()
        width = self.get_width()

        # extend by adding cols / rows until borders of image are reached
        # removed, because only removing cols/rows was really tested.
        # Fixme: test with adding cols/rows
        # Todo: change method so that it adds and removes cols/rows at the same time
        """
        i = 0
        while width < height and self.br_x < img_width and self.tl_x > 0:
            if i % 2 == 0:
                self.tl_x -= 1
            else:
                self.br_x += 1
            width += 1
            i += 1

        while height < width and self.br_y < img_height and self.tl_y > 0:
            if i % 2 == 0:
                self.tl_y -= 1
            else:
                self.br_y += 1
            height += 1
            i += 1
        """

        # remove cols / rows until rectangle is squared
        # this part was written at a different time, which is why the removal works differently,
        # it does however the exactle same thing (move yx coordinates of topleft/bottemright
        # corners)
        if height > width:
            diff = height - width
            remove_top = math.floor(diff / 2)
            remove_bottom = math.floor(diff / 2)
            if diff % 2 != 0:
                remove_top += 1
            self.tl_y += int(remove_top)
            self.br_y -= int(remove_bottom)
        elif width > height:
            diff = width - height
            remove_left = math.floor(diff / 2)
            remove_right = math.floor(diff / 2)
            if diff % 2 != 0:
                remove_left += 1
            self.tl_x += int(remove_left)
            self.br_x -= int(remove_right)

    def normalize(self, image):
        """Normalize integer pixel values to 0-1 with respect to an image.
        Args: image: Image which's dimensions to use."""
        assert not self.is_normalized
        self.tl_y /= image.shape[0]
        self.tl_x /= image.shape[1]
        self.br_y /= image.shape[0]
        self.br_x /= image.shape[1]
        self.is_normalized = True

    def unnormalize(self, image):
        """Normalize from 0-1 to integer pixel values with respect to an image.
        Args: image: Image which's dimensions to use."""
        assert self.is_normalized
        self.tl_y *= image.shape[0]
        self.tl_x *= image.shape[1]
        self.br_y *= image.shape[0]
        self.br_x *= image.shape[1]
        self.is_normalized = False

    def __str__(self):
        """Returns a string representation of the rectangle."""
        if self.is_normalized:
            return "RN(%.4f, %.4f)x(%.4f, %.4f)" % (self.tl_y, self.tl_x, self.br_y, self.br_x)
        else:
            return "R(%d, %d)x(%d, %d)" % (self.tl_y, self.tl_x, self.br_y, self.br_x)

def unit_vector(vector):
    """Returns the unit vector of the vector."""
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    angle = np.arccos(np.dot(v1_u, v2_u))
    if np.isnan(angle):
        if (v1_u == v2_u).all():
            v = 0.0
        else:
            v = np.pi
    else:
        v = angle

    if v2_u[0] < 0:
        return -v
    else:
        return v

def clip(minval, val, maxval):
    """Clips a value between min and max (both including)."""
    if val < minval:
        return minval
    elif val > maxval:
        return maxval
    else:
        return val
