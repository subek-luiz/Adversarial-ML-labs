import torch
import DataManagerPytorchL as DMP
import AttackWrappersWhiteBox
import ResNet 

def main():
    #Replace the next line with the file path of where you saved the ResNet model
    modelDir = "/work/pi_csc592_uri_edu/subek_uri/FGSMLab/FGSMLab.py"
    #Define the GPU device we are using 
    device = torch.device("cuda")
    #Parameters for the dataset
    batchSize = 64 
    numClasses = 10
    inputImageSize = [1, 3, 32, 32] #Batch size, color channel, height, width 
    #Create the ResNet model (note this does not include pre-trained weights)
    model = ResNet.resnet56(inputImageSize, numClasses).to(device)
    #Next load in the trained weights of the model 
    checkpoint = torch.load(modelDir)
    model.load_state_dict(checkpoint['state_dict'])
    #Switch the model into eval model for testing
    model = model.eval()

if __name__ == "__main__":
    main()