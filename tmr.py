import dill  # in order to save Lambda Layer
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from resnet import *

def main():
    # Define the ResNet-20 model
    model = resnet20()
    # remember to set map_location
    check_point = torch.load('D:/Documents/Kuliah/S1/GARUDA ACE/Coding/2-24-23_TMR/pretrained_models/resnet20-12fca82f.th', map_location={'cuda:1': 'cuda:0'})

    model = torch.nn.DataParallel(model)
    model.load_state_dict(check_point['state_dict'])

    torch.save(model.module, 'D:/Documents/Kuliah/S1/GARUDA ACE/Coding/2-24-23_TMR/save_temp/resnet20_check_point.pth', pickle_module=dill)

    # load the converted pretrained model
    resnet = torch.load('D:/Documents/Kuliah/S1/GARUDA ACE/Coding/2-24-23_TMR/save_temp/resnet20_check_point.pth', map_location={'cuda:1': 'cuda:0'})
    #resnet = torch.hub.load('pytorch/vision:v0.9.0', 'resnet20', pretrained=True)

    # Define the TMR_ResNet class that implements TMR for ResNet-20
    class TMR_ResNet(nn.Module):
        def __init__(self, resnet):
            super(TMR_ResNet, self).__init__()
            self.model1 = resnet
            self.model2 = resnet
            self.model3 = resnet
        
        def forward(self, x):
            out1 = self.model1(x)
            out2 = self.model2(x)
            out3 = self.model3(x)
            return out1, out2, out3

    # Define the hook function to capture the output from each layer
    def get_layer_output(name):
        def hook(model, input, output):
            layer_outputs[name] = output[0].cpu().detach().numpy()
        return hook

    # Load the CIFAR-10 dataset
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

    # Create an instance of TMR_ResNet
    tmr_resnet = TMR_ResNet(resnet)
    tmr_resnet = tmr_resnet.cuda()

    # Register hooks for all layers in the model
    layer_outputs = {}
    for name, layer in tmr_resnet.named_modules():
        layer.register_forward_hook(get_layer_output(name))

    # Get the outputs from each redundant model
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.cuda(), labels.cuda()
            # calculate outputs by running images through the network
            output1, output2, output3 = tmr_resnet(images)
    '''
    dataiter = iter(testloader)
    images, labels = dataiter.next()
    images = images.cuda()

    output1, output2, output3 = tmr_resnet(images)
    '''
    # Print the outputs from each layer for each model
    for i in range(1, 4):
        print(f"Output from Model {i}:")
        for name, output in layer_outputs.items():
            print(f"{name}: {output[i-1]}")

if __name__ == '__main__':
    main()  # execute this only when run directly, not when imported!