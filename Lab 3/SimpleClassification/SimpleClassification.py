import sys
from torchvision import transforms
import torch
import Utils
import torch.optim as optim
from NetworkLinear import NetworkLinear
import numpy

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir_train = "C:/Users/subek/OneDrive/Documents/Visual Studio Code/CSC_592_ML_KM/simpleclassification/dataset_planescarsships/train"
    data_dir_test = "C:/Users/subek/OneDrive/Documents/Visual Studio Code/CSC_592_ML_KM/simpleclassification/dataset_planescarsships/test"
    image_transforms = {
            "train": transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5],
                                    [0.5, 0.5, 0.5])
            ]),
            
            "test": transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5],
                                    [0.5, 0.5, 0.5])
            ])
        }
    
        # ToTensor converts a PIL Image or numpy.ndarry (HxWxC) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
    batch_size = 16
    
    num_epochs = 25
    
    train_loader = Utils.get_train_loader(data_dir_train, batch_size, transform=image_transforms["train"])
    test_loader = Utils.get_test_loader(data_dir_test, batch_size, transform=image_transforms["test"])
    train_iter = iter(train_loader)
    images, labels = next(train_iter) # get a batch of data e.g., 16x3x224x224
    print(images[0].shape)
    
    Utils.plot_images(images, labels)  # plot images
    
    net = NetworkLinear()   # create the simple linear network
    #net = NetworkCNN()     # create the CNN network
    
    #----------------------- Trainable parameter ---------------------------------------------------------
    
    model_parameters = filter(lambda p: p.requires_grad, net.parameters())
    params = sum([numpy.prod(p.size()) for p in model_parameters])
    print("Number of trainable parameters:", params)

    #------------------------------------------------------------------------------------
    
    loss_criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    
    running_loss = 0 
    print_freq = 100
    for epoch in range(num_epochs):
        for i, data in enumerate(train_loader):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs) # forward pass 
            loss = loss_criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            #if i % print_freq == print_freq-1: 
        print('epoch:',epoch, i+1, running_loss/print_freq)
        running_loss = 0 

    #-----------compute accuracy on trained model-------------
    total = 0 # keeps track of how many images we have processed 
    correct = 0 # keeps track of how many correct images our net predicts
    with torch.no_grad():
        for i, data in enumerate(test_loader): 
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size()[0]
            correct += (predicted == labels).sum().item()
 
    print("Accuracy: ", correct/total)


if __name__ == "__main__":
    sys.exit(int(main() or 0))


        
        
        
        