import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

import random
from PIL import Image, ImageEnhance, ImageOps, ImageChops
import numpy as np

class Operations:
    def __init__(self, rotation_range=(0, 30), noise_mean=0, noise_std=0.1):
        pass        
    
    # def apply_transforms(self, img, tgt):
    #     img = self.random_rotate(img)  # Применяем случайный поворот
    #     img = self.random_noise(img)    # Применяем шум
    #     return img
    
    def toImage(self,image):
        if image.is_cuda:
            image = image.cpu()
        image = image.numpy()
        if image.shape[0] == 1:  # assuming shape (1, 28, 28)
            image = image.squeeze(0)  # squeeze out the channel dimension
        return Image.fromarray((image * 255).astype(np.uint8), 'L')
    
    def toTensor(self, image):
        transform = transforms.Compose([
            transforms.Resize((28, 28)),  # Resize image to 28x28 if not already
            transforms.ToTensor(),        # Convert to tensor
        ])
        tensor = transform(image)
        return tensor
    
    def tensors_to_images(self,tensor):
        images = []
        for i in range(tensor.size(0)):
            # Convert the tensor to PIL Image
            image = TF.to_pil_image(tensor[i])
            images.append(image)
        return images

    def images_to_tensors(self,images):
        tensor_list = []
        for image in images:
            # Convert the PIL Image to tensor
            tensor = TF.to_tensor(image)
            tensor_list.append(tensor)
        return torch.stack(tensor_list)  # Stack all tensor images into a single tensor

    def apply_augmentation(self,images, augmentation):
        augmented_images = []
        for image in images:
            # Example of an augmentation: rotate by 25 degrees
            augmented_image = augmentation(image)
            augmented_images.append(augmented_image)
        return augmented_images


    def shear_x(self,image, magnitude):
        """ Shear the image along the horizontal axis """
        return image.transform(image.size, Image.AFFINE, (1, magnitude, 0, 0, 1, 0))

    def shear_y(self,image, magnitude):
        """ Shear the image along the vertical axis """
        return image.transform(image.size, Image.AFFINE, (1, 0, 0, magnitude, 1, 0))

    def translate_x(self,image, pixels):
        """ Translate the image in the horizontal direction """
        return image.transform(image.size, Image.AFFINE, (1, 0, pixels, 0, 1, 0))

    def translate_y(self,image, pixels):
        """ Translate the image in the vertical direction """
        return image.transform(image.size, Image.AFFINE, (1, 0, 0, 0, 1, pixels))

    def rotate(self,image, degrees):
        """ Rotate the image """
        return image.rotate(degrees)

    def auto_contrast(self, image):
        """ Maximize image contrast """
        return ImageOps.autocontrast(image)

    def invert(self, image):
        """ Invert the image colors """
        return ImageOps.invert(image)

    def equalize(self, image):
        """ Equalize the image histogram """
        return ImageOps.equalize(image)

    def solarize(self, image, threshold):
        """ Solarize the image """
        return ImageOps.solarize(image, threshold)

    def posterize(self, image, bits):
        """ Reduce the number of bits for each color channel """
        return ImageOps.posterize(image, bits)

    def adjust_contrast(self, image, factor):
        """ Adjust the image contrast """
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(factor)

    def adjust_color(self, image, factor):
        """ Adjust the color balance """
        enhancer = ImageEnhance.Color(image)
        return enhancer.enhance(factor)

    def adjust_brightness(self, image, factor):
        """ Adjust the brightness of the image """
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(factor)

    def adjust_sharpness(self, image, factor):
        """ Adjust the sharpness of the image """
        enhancer = ImageEnhance.Sharpness(image)
        return enhancer.enhance(factor)

    def cutout(self, image, size):
        """ Apply a cutout of a square patch of size 'size' """
        x = np.random.randint(0, image.width - size)
        y = np.random.randint(0, image.height - size)
        mask = Image.new('L', (size, size), 0)
        image.paste(mask, (x, y))
        return image

    def sample_pairing(self, images, weight):
        """ Perform sample pairing """
        img1, img2 = images[0], images[np.random.randint(1, len(images))]
        return Image.blend(img1, img2, weight)

