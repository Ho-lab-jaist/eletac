"""Processing routine of the Camera Images"""

from abc import ABC, abstractmethod
import numpy as np
import torch

import cv2
# from protac.util import image_processing_tools as it
import image_processing_tools as it


def process_input_image(
    image: np.ndarray,
    threshold: int = 42,
    filter_size: int = 3,
    cropped_size: tuple = (400, 400),
    resized_size: tuple = (256, 256),
    apply_mask: bool = False,
    mask_radius: int = -1,
    apply_block_mask: bool = False,
    block_mask_radius: int = -1,
    block_mask_center: tuple = (128, 128)
) -> np.ndarray:
    """Wrapper function for processing the image"""

    processor = BinaryTacImageProcessor(
                                threshold,
                                filter_size,
                                cropped_size,
                                resized_size,
                                apply_mask,
                                mask_radius,
                                apply_block_mask,
                                block_mask_radius,
                                block_mask_center)

    return processor.process(image)

def process_input_image_rgb(
    image: np.ndarray,
    cropped_size: tuple = (400, 400),
    resized_size: tuple = (256, 256),
) -> np.ndarray:
    """Wrapper function for processing the image"""
    processor = TacImageProcessorRGB(cropped_size,
                                     resized_size)
    return processor.process(image)

def warp_input_image(
        image: np.ndarray,
        homography: np.ndarray,
        threshold: int = 42,
        filter_size: int = 3,
        cropped_size: tuple = (400, 400),
        resized_size: tuple = (256, 256),
        apply_mask = False,
        mask_radius: int = 180,
        block_mask_radius: int = 27,
        block_mask_center: tuple = (131, 120)):

    processor = WrapedBinaryTacImageProcessor(
                        homography = homography,
                        threshold = threshold,
                        filter_size = filter_size,
                        cropped_size = cropped_size,
                        resized_size = resized_size,
                        apply_mask = apply_mask,
                        mask_radius = mask_radius,
                        block_mask_radius = block_mask_radius,
                        block_mask_center = block_mask_center)

    return processor.process(image)

class BaseTacImageProcessor(ABC):
    """This class is an abstract base class (ABC)
    for tactile image processors
    To create a subclass, you need to impement the following functions:
        -- <__init__>:  initialize the class; first call BaseTacImageProcessor.__init__(self, args)
        -- <process>:   tactile image processing function
    """

    def __init__(
        self,
        cropped_size: tuple = (400, 400),
        resized_size: tuple = (256, 256)
    ):
        assert isinstance(cropped_size, tuple)
        assert isinstance(resized_size, tuple)

        self.cropped_size = cropped_size
        self.resized_size = resized_size

    def __call__(self, sample: np.ndarray):
        image = self.process(sample)
        return self.__img2tensor(image).unsqueeze(0)

    @abstractmethod
    def process(self, image: np.ndarray) -> np.ndarray:
        """Process tactile images"""
        pass

    @staticmethod
    def __img2tensor(img: np.ndarray) -> torch.Tensor:
        """Method for translating image into tensor"""
        img = (img / 255.0).astype(np.float32)
        return torch.from_numpy(img)

class TacImageProcessorRGB(BaseTacImageProcessor):
    """Class for handling the image transforms"""

    def __init__(
        self,
        cropped_size: tuple = (400, 400),
        resized_size: tuple = (256, 256),
        apply_mask: bool = False,
        mask_radius: int = 120
    ):
        assert isinstance(cropped_size, (int, tuple))
        assert isinstance(resized_size, (int, tuple))

        BaseTacImageProcessor.__init__(self, cropped_size, resized_size)
        self.apply_mask = apply_mask
        self.mask_radius = mask_radius

    def __call__(self, sample: np.ndarray):
        image = self.process(sample)
        return self.__img2tensor(image)

    def process(self, image: np.ndarray):
        image = it.crop_image_at_center(image, self.cropped_size)
        image = it.resize_image(image, self.resized_size)
        if self.apply_mask:
            image = it.apply_mask_to_image(image, self.mask_radius)

        return image

    @staticmethod
    def __img2tensor(img: np.ndarray) -> torch.Tensor:
        """Method for translating image into tensor"""
        img = (img / 255.0).astype(np.float32)
        img = np.transpose(img, (2, 0, 1))
        return torch.from_numpy(np.ascontiguousarray(img))

class BinaryTacImageProcessor(BaseTacImageProcessor):
    """Class for handling the image transforms"""

    def __init__(
        self, backbone_path,
        threshold: int = 42,
        filter_size: int = 3,
        cropped_size: tuple = (400, 400),
        resized_size: tuple = (256, 256),
        apply_mask: bool = False,
        mask_radius: int = 120,
        apply_block_mask: bool = False,
        block_mask_radius: int = 20,
        block_mask_center: tuple = (128, 128)
    ):
        assert isinstance(threshold, int)
        assert isinstance(filter_size, int)
        assert isinstance(cropped_size, (int, tuple))
        assert isinstance(resized_size, (int, tuple))

        BaseTacImageProcessor.__init__(
                            self,
                            cropped_size,
                            resized_size)

        self.threshold = threshold
        self.filter_size = filter_size
        self.apply_mask = apply_mask
        self.mask_radius = mask_radius
        self.apply_block_mask = apply_block_mask
        self.block_mask_radius = block_mask_radius
        self.block_mask_center = block_mask_center
        self.backbone_path = backbone_path

    def process(self, image: np.ndarray):
        image = it.rbg_to_grayscale(image)
        image = it.blur_image(image, self.filter_size)
        image = it.create_binary_image(image, self.threshold)

        img_bb = cv2.imread(str(self.backbone_path), 2)
        ret_bb, bw_img_bb = cv2.threshold(img_bb, 50, 255, cv2.THRESH_BINARY)
        image = image + bw_img_bb

        image = it.crop_image_at_center(image, self.cropped_size)
        image = it.resize_image(image, self.resized_size)
        if self.apply_mask:
            image = it.apply_mask_to_image(image, self.mask_radius)
        if self.apply_block_mask:
            image = it.apply_invert_mask_to_image(image,
                                                radius = self.block_mask_radius,
                                                center = self.block_mask_center)
        return image


class BinaryTacImageProcessorInference(BaseTacImageProcessor):
    """Class for handling the image transforms"""

    def __init__(
        self, backbone_path,
        threshold: int = 42,
        filter_size: int = 3,
        cropped_size: tuple = (400, 400),
        resized_size: tuple = (256, 256),
        apply_mask: bool = False,
        mask_radius: int = 120,
        apply_block_mask: bool = False,
        block_mask_radius: int = 20,
        block_mask_center: tuple = (128, 128)
    ):
        assert isinstance(threshold, int)
        assert isinstance(filter_size, int)
        assert isinstance(cropped_size, (int, tuple))
        assert isinstance(resized_size, (int, tuple))

        BaseTacImageProcessor.__init__(
                            self,
                            cropped_size,
                            resized_size)

        self.threshold = threshold
        self.filter_size = filter_size
        self.apply_mask = apply_mask
        self.mask_radius = mask_radius
        self.apply_block_mask = apply_block_mask
        self.block_mask_radius = block_mask_radius
        self.block_mask_center = block_mask_center
        self.backbone_path = backbone_path

    def process(self, image: np.ndarray):
        image = it.rbg_to_grayscale(image)
        image = it.blur_image(image, self.filter_size)
        image = it.create_binary_image(image, self.threshold)

        img_bb = cv2.imread(str(self.backbone_path), 2)
        ret_bb, bw_img_bb = cv2.threshold(img_bb, 50, 255, cv2.THRESH_BINARY)
        image = image

        image = it.crop_image_at_center(image, self.cropped_size)
        image = it.resize_image(image, self.resized_size)
        if self.apply_mask:
            image = it.apply_mask_to_image(image, self.mask_radius)
        if self.apply_block_mask:
            image = it.apply_invert_mask_to_image(image,
                                                radius = self.block_mask_radius,
                                                center = self.block_mask_center)
        return image

# class WrapedBinaryTacImageProcessor(BaseTacImageProcessor):
#     """Class for handling the image transforms"""

#     def __init__(
#         self,
#         homography: np.ndarray,
#         threshold: int = 42,
#         filter_size: int = 3,
#         cropped_size: tuple = (400, 400),
#         resized_size: tuple = (256, 256),
#         apply_mask = False,
#         mask_radius: int = 180,
#         block_mask_radius: int = 27,
#         block_mask_center: tuple = (131, 120)
#     ):
#         assert isinstance(threshold, int)
#         assert isinstance(filter_size, int)
#         assert isinstance(cropped_size, (int, tuple))
#         assert isinstance(resized_size, (int, tuple))

#         BaseTacImageProcessor.__init__(
#                             self,
#                             cropped_size,
#                             resized_size)

#         self.homography = homography
#         self.threshold = threshold
#         self.filter_size = filter_size
#         self.apply_mask = apply_mask
#         self.mask_radius = mask_radius
#         self.block_mask_radius = block_mask_radius
#         self.block_mask_center = block_mask_center

#     # comment this function if you want to use
#     # the one of abstract ABS class
#     def __call__(self, sample: np.ndarray):
#         image = self.process(sample)
#         return image


#     def process(self, image: np.ndarray):
#         image = it.rbg_to_grayscale(image)
#         image = it.blur_image(image, self.filter_size)
#         image = it.create_binary_image(image, self.threshold)
#         image = it.crop_image_at_center(image, self.cropped_size)
#         image = it.resize_image(image, self.resized_size)
#         image = cv2.warpPerspective(image,
#                                     self.homography,
#                                     image.shape)
#         # image = cv2.warpPerspective(image,
#         #                     np.array([[0.9983223359,0.0055955921,-0.7200676756],
#         #                              [-0.0025290397,1.0036331255,-0.2131794838],
#         #                              [-0.0000163044,0.0000112643,1]]),
#         #                     (256, 256))

#         if self.apply_mask:
#             image = it.apply_mask_to_image(image, self.mask_radius)

#         image = it.apply_invert_mask_to_image(image,
#                                               radius = self.block_mask_radius,
#                                               center = self.block_mask_center)

#         # image = it.apply_invert_mask_to_image(image,
#         #                                       radius = 27,
#         #                                       center = (123, 133))

#         return image

class WrapedBinaryTacImageProcessor(BaseTacImageProcessor):
    """Class for handling the image transforms"""

    def __init__(
        self,
        homography: np.ndarray,
        threshold: int = 42,
        filter_size: int = 3,
        cropped_size: tuple = (400, 400),
        resized_size: tuple = (256, 256),
        apply_mask = False,
        mask_radius: int = 180,
        apply_block_mask = True,
        block_mask_radius: int = 27,
        block_mask_center: tuple = (131, 120)
    ):
        assert isinstance(threshold, int)
        assert isinstance(filter_size, int)
        assert isinstance(cropped_size, (int, tuple))
        assert isinstance(resized_size, (int, tuple))

        BaseTacImageProcessor.__init__(
                            self,
                            cropped_size,
                            resized_size)

        self.homography = homography
        self.threshold = threshold
        self.filter_size = filter_size
        self.apply_mask = apply_mask
        self.mask_radius = mask_radius
        self.apply_block_mask = apply_block_mask
        self.block_mask_radius = block_mask_radius
        self.block_mask_center = block_mask_center

    # comment this function if you want to use
    # the one of abstract ABS class
    # def __call__(self, sample: np.ndarray):
    #     image = self.process(sample)
    #     return image

    def process(self, image: np.ndarray):
        image = it.rbg_to_grayscale(image)
        image = it.blur_image(image, self.filter_size)
        image = it.create_binary_image(image, self.threshold)
        image = it.crop_image_at_center(image, self.cropped_size)
        image = it.resize_image(image, self.resized_size)
        if self.apply_mask:
            image = it.apply_mask_to_image(image, self.mask_radius)
        if self.apply_block_mask:
            image = it.apply_invert_mask_to_image(image,
                                                radius = self.block_mask_radius,
                                                center = self.block_mask_center)
        image = cv2.warpPerspective(
                    image,
                    self.homography,
                    image.shape)
        return image

class WrapedBinaryTacImageProcessor2(BaseTacImageProcessor):
    """Class for handling the image transforms"""

    def __init__(
        self,
        homography: np.ndarray,
        threshold: int = 42,
        filter_size: int = 3,
        cropped_size: tuple = (400, 400),
        resized_size: tuple = (256, 256),
        apply_mask = False,
        mask_radius: int = 180,
        apply_block_mask = True,
        block_mask_radius: int = 27,
        block_mask_center: tuple = (131, 120)
    ):
        assert isinstance(threshold, int)
        assert isinstance(filter_size, int)
        assert isinstance(cropped_size, (int, tuple))
        assert isinstance(resized_size, (int, tuple))

        BaseTacImageProcessor.__init__(
                            self,
                            cropped_size,
                            resized_size)

        self.homography = homography
        self.threshold = threshold
        self.filter_size = filter_size
        self.apply_mask = apply_mask
        self.mask_radius = mask_radius
        self.apply_block_mask = apply_block_mask
        self.block_mask_radius = block_mask_radius
        self.block_mask_center = block_mask_center

    # comment this function if you want to use
    # the one of abstract ABS class
    # def __call__(self, sample: np.ndarray):
    #     image = self.process(sample)
    #     return image

    def process(self, rgb_image: np.ndarray):
        std_image = np.std(rgb_image, axis=2, ddof=0).astype(np.uint8)
        std_image = it.create_binary_image(std_image, 30)
        image = it.rbg_to_grayscale(rgb_image)
        image = it.blur_image(image, self.filter_size)
        image = it.create_binary_image(image, self.threshold) # (480, 640) uint8 0 255
        image = np.logical_xor(image, std_image)
        image = image.astype(np.uint8) * 255
        image = it.crop_image_at_center(image, self.cropped_size)
        image = cv2.warpPerspective(image,
                                    self.homography,
                                    image.shape)
        image = it.resize_image(image, self.resized_size)
        if self.apply_mask:
            image = it.apply_mask_to_image(image, self.mask_radius)
        if self.apply_block_mask:
            image = it.apply_invert_mask_to_image(image,
                                                radius = self.block_mask_radius,
                                                center = self.block_mask_center)
        return image


class WrapedSimBinaryTacImageProcessor(BaseTacImageProcessor):
    """Class for handling the image transforms"""

    def __init__(
        self,
        homography: np.ndarray,
        threshold: int = 42,
        filter_size: int = 3,
        cropped_size: tuple = (400, 400),
        resized_size: tuple = (256, 256),
        apply_mask = False,
        mask_radius: int = 180,
        block_mask_radius: int = 27,
        block_mask_center: tuple = (131, 120)
    ):
        assert isinstance(threshold, int)
        assert isinstance(filter_size, int)
        assert isinstance(cropped_size, (int, tuple))
        assert isinstance(resized_size, (int, tuple))

        BaseTacImageProcessor.__init__(
                            self,
                            cropped_size,
                            resized_size)

        self.homography = homography
        self.threshold = threshold
        self.filter_size = filter_size
        self.apply_mask = apply_mask
        self.mask_radius = mask_radius
        self.block_mask_radius = block_mask_radius
        self.block_mask_center = block_mask_center
        self.fine_homography = np.array([[0.9735614975, -0.0608098947, -0.0219200518],
                                         [-0.0265503514, 0.9370677405, -5.3430283852],
                                         [-0.0001828783, -0.0004730493, 1]])

    def process(self, image: np.ndarray):
        image = it.rbg_to_grayscale(image)
        image = it.blur_image(image, self.filter_size)
        image = it.create_binary_image(image, self.threshold)
        image = it.crop_image_at_center(image, self.cropped_size)
        if self.apply_mask:
            image = it.apply_mask_to_image(image, self.mask_radius)
        image = it.resize_image(image, self.resized_size)
        image = cv2.warpPerspective(image,
                                    self.homography,
                                    (256, 256))
        image = it.apply_invert_mask_to_image(image,
                                              radius = self.block_mask_radius,
                                              center = self.block_mask_center)
        image = cv2.warpPerspective(image,
                                    self.fine_homography,
                                    (256, 256))
        image = it.apply_mask_to_image(image, 113)

        return image