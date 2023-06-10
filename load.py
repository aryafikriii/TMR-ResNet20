import dill  # in order to save Lambda Layer
import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from resnet import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
'''
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 4

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
'''
def main():
    # get some random training images
    #dataiter = iter(trainloader)
    #images, labels = next(dataiter)
    #images, labels = images.cuda(), labels.cuda()

    # the network architecture cor6esponding to the checkpoint
    model = resnet20()

    # remember to set map_location
    check_point = torch.load('D:/Documents/Kuliah/S1/GARUDA ACE/Coding/1-20-23_resnet_pytorch/pretrained_models/resnet20-12fca82f.th', map_location={'cuda:1': 'cuda:0'})

    # cause the model are saved from Parallel, we need to wrap it
    model = torch.nn.DataParallel(model)
    model.load_state_dict(check_point['state_dict'])

    # pay attention to .module! without this, if you load the model, it will be attached with [Parallel.module]
    # that will lead to some trouble!
    torch.save(model.module, 'D:/Documents/Kuliah/S1/GARUDA ACE/Coding/1-20-23_resnet_pytorch/save_temp/resnet20_check_point.pth', pickle_module=dill)

    # load the converted pretrained model
    net = torch.load('D:/Documents/Kuliah/S1/GARUDA ACE/Coding/1-20-23_resnet_pytorch/save_temp/resnet20_check_point.pth', map_location={'cuda:1': 'cuda:0'})
    net.eval()
    net.cuda()
    '''
    outputs = net(images)

    _, predicted = torch.max(outputs, 1)

    print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}'
                                for j in range(4)))
    '''
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
                                     
    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=128, shuffle=False,
        num_workers=4, pin_memory=True)    

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.cuda(), labels.cuda()
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print('Accuracy of the network on the test images: %d %%' % (100 * accuracy))        

    for name, layer in net.named_modules():
        if isinstance(layer, nn.Conv2d):
            # Hitung skor dari setiap neuron dalam lapisan
            score = neuron_score(layer)
            # Urutkan neuron berdasarkan skor
            _, indices = torch.sort(score, descending=True)
            # Pilih neuron dengan skor tertinggi
            net.state_dict()[name + '.weight'][indices]   
            print(name , indices)   #+ '.weight'     

    for name, param in net.named_parameters():
        if 'weight' in name:
            # Modify the weight
            param.data = param.data * 2
            
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in val_loader:
            images, labels = data
            images, labels = images.cuda(), labels.cuda()
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')
    '''
    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    # again no gradients needed
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.cuda(), labels.cuda()
            outputs = net(images)
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1


    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]    
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')
    '''

# Fungsi untuk menghitung skor neuron menggunakan L2-norm
def neuron_score(layer):
    weight = layer.weight.data
    score = weight.norm(p=2, dim=0)
    return score

if __name__ == '__main__':
    main()  # execute this only when run directly, not when imported!
