import torch
from tqdm import tqdm 
import matplotlib.pyplot as plt

def test( model, test_loader, attack ):
    real_acc = 0
    fake_acc = 0
    adv_examples = []

    for data, target in tqdm(test_loader):
        output = model(data)  
        init_pred = output.max(1, keepdim=True)[1].squeeze(-1)
        
        mask = init_pred == target
        real_acc += torch.sum(mask).item() / len(mask)

        perturbed_data = attack(data, target)
        
        output = model(perturbed_data)
        final_pred = output.max(1, keepdim=True)[1].squeeze(-1)
        
        correct = final_pred == target
        fake_acc += torch.sum(correct * mask ).item() / torch.sum(mask).item()
        
        adv_examples.append(( init_pred[0].item(), final_pred[0].item(), perturbed_data[0].squeeze().detach().numpy()))


    real_acc /= len(test_loader)
    fake_acc /= len(test_loader)

    print("Real Accuracy = {}, Fake Accuracy on the real true predictions = {}".format(real_acc, fake_acc))

    return fake_acc , real_acc , adv_examples

def plot_examples( vvalues, examples, cnt_ex = 5 ):
    cnt = 0
    plt.figure(figsize=(10,10))
    for i in range(len(vvalues)):
        for j in range(min(len(examples[i]), cnt_ex)):
            cnt += 1
            plt.subplot(len(vvalues),min(len(examples[0]), cnt_ex),cnt)
            plt.xticks([], [])
            plt.yticks([], [])
            if j == 0:
                plt.ylabel("par.: {}".format(vvalues[i]), fontsize=14)
            orig,adv,ex = examples[i][j]
            plt.title("{} -> {}".format(orig, adv))
            plt.imshow(ex, cmap="gray")
    plt.tight_layout()
    plt.show()