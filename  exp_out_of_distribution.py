import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from matplotlib import pyplot as plt
from src.utils import MNIST, MNIST_v2,MNIST_v3_fake, train, train_epoch_adv, test_epoch
from src.models import CNN
from src.advattack.attacks import test, plot_examples
from src.utils import train_epoch_adv_v2,train_epoch_adv_v3,train_epoch_adv_v4,  test_epoch_adv, train_epoch_default
from src.advattack.FGSM import FGSM
from src.advattack.noising import RandomTransform
from src.advattack.operations import Operations
from src.advattack.initialize_all_attacks import initialize_all_attacks , initialize_all_aggressive_attacks
from PIL import Image
import warnings
from copy import deepcopy
warnings.filterwarnings('ignore')


device = torch.device("cpu")
train_dataset, test_dataset, train_loader, test_loader = MNIST_v3_fake(p = 0.1)
print(len(train_dataset))
fake_indices = np.array(list(train_dataset.get_fake_indices()))
np.save(f'./checkpoints/exp_2/fake_indices.npy', fake_indices)

n_epochs = 8
n_cold_epochs = 2

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

    pi_history_cold = [pi]
    adv_test_history_cold = []
    adv_full_loss_history_cold = []

    for i in range(n_cold_epochs):
        print('COLD EPOCH: ', i)
        logs = train_epoch_adv_v4(model, pi, optimizer, train_loader, criterion, device, tau = 1, gamma= 1e-7, init_losses = all_losses, seed = seed)
        pi = logs['pi']
        test_acc = test_epoch(model, test_loader, device)
        
        adv_test_history_cold.append(test_acc)
        
        adv_full_loss_history_cold += logs['mean_losses']
        pi_history_cold+=[pi]

    pi_history_cold = np.array([pi.detach().numpy() for pi in pi_history_cold])
    adv_test_history_cold = np.array(adv_test_history_cold)
    adv_full_loss_history_cold = np.array(adv_full_loss_history_cold)

    np.save(f'./checkpoints/exp_2/pi_history_cold_seed_{seed}.npy', pi_history_cold)
    np.save(f'./checkpoints/exp_2/adv_test_history_cold_seed_{seed}.npy', adv_test_history_cold)
    np.save(f'./checkpoints/exp_2/adv_full_loss_history_cold_seed_{seed}.npy', adv_full_loss_history_cold)

    ## UPDATE DATALOADER

    all_indices = np.arange(len(train_dataset))
    sorted_indices = np.argsort(pi_history_cold[-1])
    indices_to_remove = sorted_indices[len(sorted_indices)//2:]
    new_train_dataset = deepcopy(train_dataset)
    new_train_dataset.filter_indices(indices_to_remove)
    new_train_loader = DataLoader(new_train_dataset, batch_size=64, shuffle=True)
    
    print(len(list(set(indices_to_remove).intersection(set(fake_indices)))) / len(fake_indices) )
    
    pi = torch.ones(len(new_train_dataset))/len(new_train_dataset)
    pi.to(device)
    all_losses = torch.zeros_like(pi)

    adv_test_history = []
    adv_full_loss_history = []
    for i in range(n_epochs):
        print('EPOCH: ', i)
        logs = train_epoch_adv_v4(model, pi, optimizer, new_train_loader, criterion, device, tau = 1, gamma= 1e-1, init_losses = all_losses, default = True, seed = seed)
        pi = logs['pi']
        test_acc = test_epoch(model, test_loader, device)
        
        adv_test_history.append(test_acc)
        adv_full_loss_history  += logs['mean_losses']

    adv_full_loss_history = np.array(adv_full_loss_history)

    adv_test_history = np.array(adv_test_history)
    adv_full_loss_history = np.array(adv_full_loss_history)
    
    ## SAVING
    torch.save(model.state_dict(), f'./checkpoints/exp_2/simple_cnn_mnist_adversarial_seed_{seed}.pth')
    np.save(f'./checkpoints/exp_2/adv_test_history_seed_{seed}.npy', adv_test_history)
    np.save(f'./checkpoints/exp_2/adv_full_loss_history_seed_{seed}.npy', adv_full_loss_history)


    ####################################################### DEFAULT #######################################################################
    
    print('Default training')
    print('SEED: ', seed)
    model = CNN().to(device)
    model.load_state_dict(torch.load('./checkpoints/simple_cnn_mnist.pth'))
    criterion = torch.nn.functional.cross_entropy
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    print(next(model.parameters()).device)

    

    all_losses = torch.zeros(len(train_dataset))
    pi = torch.ones(len(train_dataset))/len(train_dataset)
    pi.to(device)
    
    default_test_history_cold = []
    default_full_loss_history_cold = []
    for i in range(n_cold_epochs):
        print('EPOCH: ', i)
        logs = train_epoch_adv_v4(model, pi, optimizer, train_loader, criterion, device, tau = 1, gamma= 1e-1, init_losses = all_losses, default = True, seed = seed)
        pi = logs['pi']
        test_acc = test_epoch(model, test_loader, device)
        
        default_test_history_cold.append(test_acc)
        default_full_loss_history_cold += logs['mean_losses']

    default_full_loss_history_cold = np.array(default_full_loss_history_cold)
    default_test_history_cold = np.array(default_test_history_cold)

    np.save(f'./checkpoints/exp_2/default_test_history_cold_seed_{seed}.npy', default_test_history_cold)
    np.save(f'./checkpoints/exp_2/default_full_loss_history_cold_seed_{seed}.npy', default_full_loss_history_cold)

    ## UPDATE DATALOADER
    all_indices = np.arange(len(train_dataset))
    sorted_indices = np.argsort( logs['all_losses'].detach().numpy() )
    indices_to_remove = sorted_indices[len(sorted_indices)//2:]
    new_train_dataset = deepcopy(train_dataset)
    new_train_dataset.filter_indices(indices_to_remove)
    new_train_loader = DataLoader(new_train_dataset, batch_size=64, shuffle=True)
    print(len(list(set(indices_to_remove).intersection(set(fake_indices)))) / len(fake_indices) )
    
    
    pi = torch.ones(len(new_train_dataset))/len(new_train_dataset)
    pi.to(device)
    all_losses = torch.zeros_like(pi)

    default_test_history = []
    default_full_loss_history = []

    for i in range(n_epochs):
        print('EPOCH: ', i)
        logs = train_epoch_adv_v4(model, pi, optimizer, new_train_loader, criterion, device, tau = 1, gamma= 1e-1, init_losses = all_losses, default = True, seed = seed)
        pi = logs['pi']
        test_acc = test_epoch(model, test_loader, device)
        
        default_test_history.append(test_acc)
        default_full_loss_history  += logs['mean_losses']
        

    default_full_loss_history = np.array(default_full_loss_history)
    default_test_history = np.array(default_test_history)

    ## SAVING
    torch.save(model.state_dict(), f'./checkpoints/exp_2/simple_cnn_mnist_default_seed_{seed}.pth')
    np.save(f'./checkpoints/exp_2/default_test_history_seed_{seed}.npy', default_test_history)
    np.save(f'./checkpoints/exp_2/default_full_loss_history_seed_{seed}.npy', default_full_loss_history)