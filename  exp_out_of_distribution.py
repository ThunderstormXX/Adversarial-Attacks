import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt
from src.utils import MNIST, MNIST_v2, train, train_epoch_adv, test_epoch
from src.models import CNN
from src.advattack.attacks import test, plot_examples
from src.utils import train_epoch_adv_v2,train_epoch_adv_v3,train_epoch_adv_v4,  test_epoch_adv, train_epoch_default
from src.advattack.FGSM import FGSM
from src.advattack.noising import RandomTransform
from src.advattack.operations import Operations
from src.advattack.initialize_all_attacks import initialize_all_attacks , initialize_all_aggressive_attacks
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# device = torch.device("mps" if torch.mps.is_available() else "cpu")
device = torch.device("cpu")
train_dataset, test_dataset, train_loader, test_loader = MNIST_v2()
print(len(train_dataset))


def CombineImagesHorizontally(*images):
    if not images:
        raise ValueError("No images provided")

    total_width = sum(img.width for img in images)
    max_height = max(img.height for img in images)

    combined_image = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for img in images:
        combined_image.paste(img, (x_offset, 0))
        x_offset += img.width

    return combined_image
def display_images(images, transpose = True):
    M = len(images)
    N = len(images[0])
    if transpose:
        M, N = N, M
    
    fig, axes = plt.subplots(M, N, figsize=(N * 2, M * 2))
    
    for i in range(M):
        for j in range(N):
            ax = axes[i, j]
            if transpose:
                img = images[j][i] 
            else:
                img = images[i][j]
            ax.imshow(img, cmap='gray')
            ax.axis('off')
            
    plt.tight_layout()
    plt.show()


n_epochs = 3

seeds = np.arange(1)

for seed in seeds:
    print('Adversarial training')
    print('SEED: ', seed)
    model = CNN().to(device)
    model.load_state_dict(torch.load('./checkpoints/simple_cnn_mnist.pth'))
    criterion = torch.nn.functional.cross_entropy
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    print(next(model.parameters()).device)
    
    pi = torch.ones(len(train_dataset))/len(train_dataset)
    pi.to(device)
    all_losses = torch.zeros_like(pi)

    pi_history = [pi]
    adv_test_history = []
    adv_full_loss_history = []

    for i in range(n_epochs):
        print('EPOSH: ', i)
        logs = train_epoch_adv_v4(model, pi, optimizer, train_loader, criterion, device, tau = 1, gamma= 1e-1, init_losses = all_losses, seed = seed)
        pi = logs['pi']

        test_acc = test_epoch(model, test_loader, device)
        
        adv_test_history.append(test_acc)
        
        adv_full_loss_history += [ logs['all_losses']]
        pi_history+=[pi]

    adv_full_loss_history = np.array(adv_full_loss_history)

    pi_history = np.array([pi.detach().numpy() for pi in pi_history])
    adv_test_history = np.array(adv_test_history)
    adv_full_loss_history = np.array(adv_full_loss_history)
    
    ## SAVING
    torch.save(model.state_dict(), f'./checkpoints/exp_2/simple_cnn_mnist_adversarial_seed_{seed}.pth')
    # np.save(f'./checkpoints/adv_loss_history_seed_{seed}.npy', adv_loss_history)
    np.save(f'./checkpoints/exp_2/pi_history_seed_{seed}.npy', pi_history)
    np.save(f'./checkpoints/exp_2/adv_test_history_seed_{seed}.npy', adv_test_history)
    np.save(f'./checkpoints/exp_2/adv_full_loss_history_seed_{seed}.npy', adv_full_loss_history)

    
    print('Default training')
    print('SEED: ', seed)
    model = CNN().to(device)
    model.load_state_dict(torch.load('./checkpoints/simple_cnn_mnist.pth'))
    criterion = torch.nn.functional.cross_entropy
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    print(next(model.parameters()).device)

    all_losses = torch.zeros_like(pi)
    # default_loss_history = []
    default_test_history = []
    default_full_loss_history = []
    for i in range(n_epochs):
        print('EPOSH: ', i)
        logs = train_epoch_adv_v4(model, pi, optimizer, train_loader, criterion, device, tau = 1, gamma= 1e-1, init_losses = all_losses, default = True, seed = seed)
        pi = logs['pi']
        test_acc = test_epoch(model, test_loader, device)
        
        default_test_history.append(test_acc)
        default_full_loss_history  += [ logs['all_losses']]
        
    default_full_loss_history = np.array(default_full_loss_history)

    default_test_history = np.array(default_test_history)

    ## SAVING
    torch.save(model.state_dict(), f'./checkpoints/exp_2/simple_cnn_mnist_default_seed_{seed}.pth')
    np.save(f'./checkpoints/exp_2/default_test_history_seed_{seed}.npy', default_test_history)
    np.save(f'./checkpoints/exp_2/default_full_loss_history_seed_{seed}.npy', default_full_loss_history)