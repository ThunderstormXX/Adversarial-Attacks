import torch 
import torch.nn.functional as F

class FGSM():
    def __init__(self, model = None):
        self.model = model

    def init_params(self, model):
        self.model = model

    def fgsm_attack(self, img, eps, data_grad):
        perturbed_img = img + eps*data_grad.sign()
        perturbed_img = torch.clamp(perturbed_img, 0, 1)
        return perturbed_img

    def fgsm_augment(self, data, target, eps):
        data.requires_grad = True
        output = self.model(data)
        loss = F.nll_loss(output, target)
        
        gradients = torch.autograd.grad(loss, data)[0]
        
        perturbed_data = self.fgsm_attack(data, eps, gradients)
        data.requires_grad = False
        return perturbed_data