from .FGSM import FGSM
from .noising import RandomTransform
from .operations import Operations

def initialize_all_attacks(model):
    eps = 0.1
    mul = 1.5
    attack_model1 = FGSM(model = model)
    attack_model2 = RandomTransform(rotation_range=(0, 60*mul), noise_mean=0, noise_std=0.2 * mul)
    operations = Operations()

    attacks = [
        lambda x, y, eps=eps: attack_model1.fgsm_augment(x, y, eps) ,
        lambda x,y, attack_model=attack_model2: attack_model.apply_transforms(x,y) , 
    ]
    attacks += [
        lambda x, y: operations.images_to_tensors( operations.apply_augmentation( operations.tensors_to_images(x) , augmentation= lambda z: operations.shear_x( z, 0.4 ))) ,
        lambda x, y: operations.images_to_tensors( operations.apply_augmentation( operations.tensors_to_images(x) , augmentation= lambda z: operations.shear_y( z, 0.4 ))) ,
        lambda x, y: operations.images_to_tensors( operations.apply_augmentation( operations.tensors_to_images(x) , augmentation= lambda z: operations.translate_x( z, 5 ))) ,
        lambda x, y: operations.images_to_tensors( operations.apply_augmentation( operations.tensors_to_images(x) , augmentation= lambda z: operations.translate_y( z, 5 ))) ,
        lambda x, y: operations.images_to_tensors( operations.apply_augmentation( operations.tensors_to_images(x) , augmentation= lambda z: operations.invert( z ))) ,
        lambda x, y: operations.images_to_tensors( operations.apply_augmentation( operations.tensors_to_images(x) , augmentation= lambda z: operations.equalize( z))) ,
        lambda x, y: operations.images_to_tensors( operations.apply_augmentation( operations.tensors_to_images(x) , augmentation= lambda z: operations.solarize( z , 0.9))) ,
        lambda x, y: operations.images_to_tensors( operations.apply_augmentation( operations.tensors_to_images(x) , augmentation= lambda z: operations.adjust_contrast( z, 8 ))) ,
        lambda x, y: operations.images_to_tensors( operations.apply_augmentation( operations.tensors_to_images(x) , augmentation= lambda z: operations.adjust_sharpness( z, 8 ))) ,
        lambda x, y: operations.images_to_tensors( operations.apply_augmentation( operations.tensors_to_images(x) , augmentation= lambda z: operations.adjust_brightness( z, 8 ))) 
    ]
    return attacks