import torch
import torchvision.transforms as transforms
import random

class RandomTransform():
    def __init__(self, rotation_range=(0, 30), noise_mean=0, noise_std=0.1):
        self.rotation_range = rotation_range
        self.noise_mean = noise_mean
        self.noise_std = noise_std
    def init_params(self,rotation_range=(0, 30), noise_mean=0, noise_std=0.1):
        self.rotation_range = rotation_range
        self.noise_mean = noise_mean
        self.noise_std = noise_std

    def random_rotate(self, img):
        # Генерация случайного угла для поворота
        angle = random.uniform(self.rotation_range[0], self.rotation_range[1])
        rotate_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomRotation(angle),
            transforms.ToTensor()
        ])
        
        # Обработка каждого изображения в батче
        return torch.stack([rotate_transform(img[i]) for i in range(img.size(0))])

    def random_noise(self, img):
        # Генерация случайного шума
        noise = torch.randn_like(img) * self.noise_std + self.noise_mean
        noisy_img = img + noise
        return torch.clamp(noisy_img, 0, 1)  # Ограничиваем значения в пределах [0, 1]

    def apply_transforms(self, img, tgt):
        img = self.random_rotate(img)  # Применяем случайный поворот
        img = self.random_noise(img)    # Применяем шум
        return img
    
