import torch
from torchvision import transforms,datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

import torch.nn.functional as F

def MNIST():
    # Загрузка и подготовка данных
    transform = transforms.Compose([
        # transforms.Resize(32),  # Увеличиваем размер изображений до 32x32
        # transforms.Grayscale(3),  # Преобразуем в 3 канала
        transforms.ToTensor(),
        # transforms.Normalize((0.5,), (0.5,))  # Нормализация для одного канала, дублируется для всех трех
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    return train_dataset, test_dataset, train_loader, test_loader

class MNISTWithIndices(datasets.MNIST):
    def __getitem__(self, index):
        img, target = super(MNISTWithIndices, self).__getitem__(index)
        return img, target, index

def MNIST_v2():
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    train_dataset = MNISTWithIndices(root='./data', train=True, download=True, transform=transform)
    test_dataset = MNISTWithIndices(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    return train_dataset, test_dataset, train_loader, test_loader


# Обучение модели
def train_model(model, train_loader, criterion, optimizer, num_epochs=10, device='cpu'):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in tqdm(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}')

def train_epoch(model, optimizer, train_loader, criterion, device, prepare = None):
    model.train()
    for it, traindata in enumerate(train_loader):
        train_inputs, train_labels = traindata
        train_inputs = train_inputs.to(device) 
        train_labels = train_labels.to(device)
        train_labels = torch.squeeze(train_labels)

        model.zero_grad()        
        if prepare is not None:
            train_inputs = prepare(train_inputs, train_labels)

        output = model(train_inputs)
        loss = criterion(output, train_labels.long())
        loss.backward()
        optimizer.step()
def evaluate_loss_acc(loader, model, criterion, device):
    model.eval()
    total_acc = 0.0
    total_loss = 0.0
    total = 0.0
    for it, data in enumerate(loader):
        inputs, labels = data
        inputs = inputs.to(device) 
        labels = labels.to(device)
        labels = torch.squeeze(labels)

        output = model(inputs) # pay attention here!
        loss = criterion(output, labels.long())# + torch.norm(WW^T - I)
        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct = pred == labels.byte()
        total_acc += torch.sum(correct).item() / len(correct)

    total = it + 1
    return total_loss / total, total_acc / total

def train(model, opt, train_loader, test_loader, criterion, n_epochs, \
          device, verbose=True, prepare = None):
    train_log, train_acc_log = [], []
    val_log, val_acc_log = [], []

    for epoch in range(n_epochs):
        train_epoch(model, opt, train_loader, criterion, device, prepare = prepare)
        train_loss, train_acc = evaluate_loss_acc(train_loader,
                                                  model, criterion,
                                                  device)
        val_loss, val_acc = evaluate_loss_acc(test_loader, model,
                                              criterion, device)

        train_log.append(train_loss)
        train_acc_log.append(train_acc)

        val_log.append(val_loss)
        val_acc_log.append(val_acc)

        if verbose:
             print (('Epoch [%d/%d], Loss (train/test): %.4f/%.4f,'+\
               ' Acc (train/test): %.4f/%.4f' )
                   %(epoch+1, n_epochs, \
                     train_loss, val_loss, train_acc, val_acc))

    return train_log, train_acc_log, val_log, val_acc_log


def train_epoch_adv(model, pi, attacks, optimizer, train_loader, criterion, device, tau=1):
    model.train()
    pi_array = []
    for traindata in tqdm(train_loader):
        train_inputs, train_labels = traindata
        train_labels = torch.squeeze(train_labels)

        model.zero_grad()        
        all_inputs = []
        for attack in attacks:
            # print(train_inputs.device , train_labels.device)
            all_inputs.append(attack(train_inputs, train_labels))

        all_outputs = []
        for input in all_inputs:
            input = input.to(device)
            all_outputs.append(model(input))

        train_labels = train_labels.to(device)
        all_losses = []
        for output in all_outputs:
            all_losses.append(criterion(output, train_labels.long()))

        full_loss = 0
        for i in range(len(all_losses)):
            full_loss += all_losses[i] * pi[i]

        full_loss.backward()
        optimizer.step()

        pi = F.softmax( torch.tensor([loss*tau for loss in all_losses]))
        pi_array.append(pi)
    return pi, pi_array

def train_epoch_adv_v2(model, pi, attacks, optimizer, train_loader, criterion, device, tau=1, init_losses = None):
    model.train()
    pi_array = []
    if init_losses is None:
        all_losses = torch.zeros_like(pi)
    else:
        all_losses = init_losses
    for traindata in tqdm(train_loader):
        train_inputs, train_labels = traindata
        train_labels = torch.squeeze(train_labels)

        model.zero_grad()        

        probs = pi.clone().detach().numpy()
        ind_attack = np.random.choice(range(len(attacks)) , p=probs)
        attack = attacks[ind_attack]
        input = attack(train_inputs, train_labels)
        input = input.to(device)

        output = model(input)

        train_labels = train_labels.to(device)
        loss = criterion(output, train_labels.long())
        loss.backward()
        optimizer.step()

        all_losses[ind_attack] = loss
        if torch.all(all_losses != 0):
            # print(all_losses)
            # print(pi)
            # raise Exception('TEST')
            pi = F.softmax( torch.log(pi) * all_losses * tau )
            pi_array.append(pi)
    return pi, pi_array, all_losses

def train_epoch_adv_v3(model, pi, attacks, optimizer, train_loader, criterion, device, tau=1,gamma = 1, init_losses = None, default = False, seed = None):
    if seed is not None:
        np.random.seed(seed)  # Установка seed для NumPy
        torch.manual_seed(seed)  # Установка seed для PyTorch
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)  # Установка seed для всех GPU, если использу
    model.train()
    pi_array = []
    losses_array = []
    if init_losses is None:
        all_losses = torch.zeros_like(pi)
    else:
        all_losses = init_losses

    for traindata in tqdm(train_loader):
        train_inputs, train_labels = traindata
        train_labels = torch.squeeze(train_labels)

        model.zero_grad()        
        if default is True:
            probs = np.ones(len(pi)) / len(pi)
        else:
            probs = pi.clone().detach().numpy()

        ind_attack = np.random.choice(range(len(attacks)) , p=probs)
        attack = attacks[ind_attack]
        input = attack(train_inputs, train_labels)
        input = input.to(device)

        output = model(input)

        train_labels = train_labels.to(device)
        loss = criterion(output, train_labels.long())
        loss.backward()
        optimizer.step()

        # print(ind_attack , loss)
        # print(all_losses)
        all_losses[ind_attack] = loss.clone()
        # print(all_losses)
        if torch.all(all_losses != 0):
            pi = F.softmax( (torch.log(pi) + gamma * all_losses) / (1 + gamma * tau) - 1 )
            pi_array.append(pi)
            losses_array.append(all_losses.clone())
    
    logs = dict(
        pi_array = pi_array, 
        loss_array = losses_array, 
    )
    return pi, logs

def adversarial_loss_fn(x,y,weights, loss_fn):
    return loss_fn(x,y , reduction='none').mul(weights).sum()

def train_epoch_adv_v4(model, pi, optimizer, train_loader, criterion, device, tau=1,gamma = 1, init_losses = None, default = False, seed = None):
    if seed is not None:
        np.random.seed(seed)  # Установка seed для NumPy
        torch.manual_seed(seed)  # Установка seed для PyTorch
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)  # Установка seed для всех GPU, если использу
    model.train()

    # losses_array = []
    if init_losses is None:
        all_losses = torch.zeros_like(pi)
    else:
        all_losses = init_losses

    for traindata in tqdm(train_loader):
        train_inputs, train_labels, train_indices = traindata
        train_labels = torch.squeeze(train_labels)

        model.zero_grad()        
        input = train_inputs.to(device)
        train_labels = train_labels.to(device)
        
        output = model(input)
        
        weights = pi[train_indices]
        loss_default = criterion( output.float(),train_labels.long(), reduction='none')
        if default is True:
            loss = loss_default.sum()
        else:
            loss = loss_default.mul(weights).sum()
        loss.backward()
        optimizer.step()

        all_losses[train_indices] = loss_default.detach().clone().sum()

        pi = F.softmax( (torch.log(pi) + gamma * all_losses) / (1 + gamma * tau) - 1 )
        # losses_array.append(all_losses.clone().detach().numpy())
    
    logs = dict(
        all_losses = all_losses,
        pi = pi  
    )
    return logs

def train_epoch_default(model, attacks, optimizer, train_loader, criterion, device):
    model.train()
 
    for traindata in tqdm(train_loader):
        train_inputs, train_labels = traindata
        train_labels = torch.squeeze(train_labels)

        model.zero_grad()        

        ind_attack = np.random.choice(range(len(attacks)) )
        attack = attacks[ind_attack]
        input = attack(train_inputs, train_labels)
        input = input.to(device)

        output = model(input)

        train_labels = train_labels.to(device)
        loss = criterion(output, train_labels.long())
        loss.backward()
        optimizer.step()
    return loss

def test_epoch_adv(model, attacks, train_loader, device):
    model.eval()
    total_acc = np.zeros(len(attacks))
    total = 0.0
    
    for traindata in tqdm(train_loader):
        train_inputs, train_labels = traindata
        train_labels = torch.squeeze(train_labels)

        model.zero_grad()        
        all_inputs = []
        for attack in attacks:
            all_inputs.append(attack(train_inputs, train_labels))

        all_outputs = []
        for input in all_inputs:
            input = input.to(device)
            all_outputs.append(model(input))

        train_labels = train_labels.to(device)
        
        for e, output in enumerate(all_outputs):
            pred = output.argmax(dim=1)
            correct = pred == train_labels.byte()
            total_acc[e] += torch.sum(correct).item() / len(correct)

        total = total + 1
    return total_acc / total 


def test_epoch(model,train_loader, device):
    model.eval()
    total_acc = 0.0
    total = 0.0
    
    for traindata in tqdm(train_loader):
        train_inputs, train_labels, indices = traindata
        train_labels = torch.squeeze(train_labels)

        train_inputs = train_inputs.to(device)
        train_labels = train_labels.to(device)

        model.zero_grad()        
    
        outputs = model(train_inputs)
        
        pred = outputs.argmax(dim=1)
        correct = pred == train_labels.byte()
        total_acc += torch.sum(correct).item() / len(correct)

        total = total + 1
    return total_acc / total 