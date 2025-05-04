import torch
import DataManagerPytorchL as DMP
import AttackWrappersWhiteBox
import ResNet


def main():
    #Replace the next line with the file path of where you saved the ResNet models
    modelDir = "/work/pi_csc592_uri_edu/subek_uri/BlackBoxAttack/ModelResNet56-Run0.th"
    modelDirSynthetic = "/work/pi_csc592_uri_edu/subek_uri/BlackBoxAttack/ModelResNet164-Run0.th"
    #Define the GPU device we are using
    device = torch.device('cuda')
    #Parameters for the dataset
    batchSize = 64 
    numClasses = 10
    inputImageSize = [1, 3, 32, 32] #Batch size, color channel, height, width
    #Create the ResNet model
    model = ResNet.resnet56(inputImageSize, numClasses).to(device)
    #Next load in the trained weights of the model 
    checkpoint = torch.load(modelDir)
    model.load_state_dict(checkpoint['state_dict'])
    #Switch the model into eval model for testing
    model = model.eval()
    #Load the synthetic model for the pure black-box attack
    syntheticModel = ResNet.resnet164(inputImageSize, numClasses).to(device)
    checkpointSynthetic = torch.load(modelDirSynthetic)
    syntheticModel.load_state_dict(checkpointSynthetic['state_dict'])
    syntheticModel = syntheticModel.eval()
    #Load in the dataset
    valLoader = DMP.GetCIFAR10Validation(inputImageSize[2], batchSize)
    #Check the clean accuracy of the model
    cleanAcc = DMP.validateD(valLoader, model, device)
    print("CIFAR-10 Clean Val Loader Acc:", cleanAcc)
    #Get correctly classified, classwise balanced samples to do the attack
    totalSamplesRequired = 100
    correctLoader = DMP.GetCorrectlyIdentifiedSamplesBalanced(model, totalSamplesRequired, valLoader, numClasses)
    #Check to make sure the accuracy is 100% on the correct loader
    correctAcc = DMP.validateD(correctLoader, model, device)
    print("CIFAR-10 Clean Correct Loader Acc:", correctAcc)
    #Do the attacks
    epsilonMax = 0.031 #Maximum perturbation
    clipMin = 0.0 #Minimum value a pixel can take
    clipMax = 1.0 #Maximum value a pixel can take     
    numSteps = 10
    epsilonStep = epsilonMax/numSteps
    #Run the attacks
    advLoader = AttackWrappersWhiteBox.PGDNativePytorch(device, correctLoader, syntheticModel, epsilonMax, epsilonStep, numSteps, clipMin, clipMax)
    advAcc = DMP.validateD(advLoader, model, device)
    print("Black-box Robustness:", advAcc)
    
    
    
if __name__ == "__main__":
    main()
    
    
    
