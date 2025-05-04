import torch 
import DataManagerPytorchL as DMP
import torchvision
import foolbox 

#This operation can all be done in one line but for readability later
#the projection operation is done in multiple steps for l-inf norm
def ProjectionOperation(xAdv, xClean, epsilonMax):
    #First make sure that xAdv does not exceed the acceptable range in the positive direction
    xAdv = torch.min(xAdv, xClean + epsilonMax) 
    #Second make sure that xAdv does not exceed the acceptable range in the negative direction
    xAdv = torch.max(xAdv, xClean - epsilonMax)
    return xAdv

#Native (no attack library) implementation of the FGSM attack in Pytorch 
def FGSMNativePytorch(device, dataLoader, model, epsilonMax, clipMin, clipMax):
    model.eval() #Change model to evaluation mode for the attack 
    #Generate variables for storing the adversarial examples 
    numSamples = len(dataLoader.dataset) #Get the total number of samples to attack
    xShape = DMP.GetOutputShape(dataLoader) #Get the shape of the input (there may be easier way to do this)
    xAdv = torch.zeros(numSamples, xShape[0], xShape[1], xShape[2])
    yClean = torch.zeros(numSamples)
    advSampleIndex = 0 
    batchSize = 0 #just do dummy initalization, will be filled in later
    #Go through each sample 
    tracker = 0
    for xData, yData in dataLoader:
        batchSize = xData.shape[0] #Get the batch size so we know indexing for saving later
        tracker = tracker + batchSize
        print("Processing up to sample=", tracker)
        #Put the data from the batch onto the device 
        #xDataTemp = torch.from_numpy(xData.cpu().detach().numpy()).to(device)
        xData = xData.to(device)
        yData = yData.type(torch.LongTensor).to(device)
        # Set requires_grad attribute of tensor. Important for attack. (Pytorch comment, not mine) 
        xData.requires_grad = True
        # Forward pass the data through the model
        output = model(xData)
        # Calculate the loss
        loss = torch.nn.CrossEntropyLoss()
        # Zero all existing gradients
        model.zero_grad()
        # Calculate gradients of model in backward pass
        cost = loss(output, yData).to(device)
        cost.backward()
        #Compute the perturbed image
        perturbedImage = xData + epsilonMax*xData.grad.data.sign().detach()
        # Adding clipping to maintain the range for pixels
        perturbedImage = torch.clamp(perturbedImage, clipMin, clipMax)
        #Save the adversarial images from the batch 
        for j in range(0, batchSize):
            xAdv[advSampleIndex] = perturbedImage[j]
            yClean[advSampleIndex] = yData[j]
            advSampleIndex = advSampleIndex+1 #increment the sample index
    #All samples processed, now time to save in a dataloader and return 
    advLoader = DMP.TensorToDataLoader(xAdv, yClean, transforms= None, batchSize= dataLoader.batch_size, randomizer=None) #use the same batch size as the original loader
    return advLoader

#Native (no attack library) implementation of the PGD attack in Pytorch 
def PGDNativePytorch(device, dataLoader, model, epsilonMax, epsilonStep, numSteps, clipMin, clipMax):
    model.eval()  #Change model to evaluation mode for the attack
    #Generate variables for storing the adversarial examples
    numSamples = len(dataLoader.dataset)  #Get the total number of samples to attack
    xShape = DMP.GetOutputShape(dataLoader)  #Get the shape of the input (there may be easier way to do this)
    xAdv = torch.zeros(numSamples, xShape[0], xShape[1], xShape[2])
    yClean = torch.zeros(numSamples)
    advSampleIndex = 0
    batchSize = 0  #just do dummy initalization, will be filled in later
    loss = torch.nn.CrossEntropyLoss()
    tracker = 0
    #Go through each sample
    for xData, yData in dataLoader:
        batchSize = xData.shape[0]  # Get the batch size so we know indexing for saving later
        tracker = tracker + batchSize
        #Put the data from the batch onto the device
        xAdvCurrent = xData.to(device)
        yCurrent = yData.type(torch.LongTensor).to(device)
        for attackStep in range(0, numSteps): 
            xAdvCurrent.requires_grad = True
            # Forward pass the data through the model
            output = model(xAdvCurrent)
            # Calculate the loss
            # Zero all existing gradients
            model.zero_grad()
            # Calculate gradients of model in backward pass
            cost = loss(output, yCurrent)
            cost.backward()
            advTemp = xAdvCurrent + (epsilonStep * xAdvCurrent.grad.data.sign()).to(device)
            advTemp = ProjectionOperation(advTemp, xData.to(device), epsilonMax)
            #Adding clipping to maintain the range
            xAdvCurrent = torch.clamp(advTemp, min=clipMin, max=clipMax).detach_()
            # Save the adversarial images from the batch
        for j in range(0, batchSize):
            xAdv[advSampleIndex] = xAdvCurrent[j]
            yClean[advSampleIndex] = yData[j]
            advSampleIndex = advSampleIndex + 1  # increment the sample index
    #All samples processed, now time to save in a dataloader and return
    advLoader = DMP.TensorToDataLoader(xAdv, yClean, transforms=None, batchSize=dataLoader.batch_size, randomizer=None)  # use the same batch size as the original loader
    return advLoader

#CW attack untargeted method using Foolbox
#Returns a dataloader with the adversarial samples and the clean labels  
def CWAttackUntargetedFoolBox(device, dataLoader, model, clipMin, clipMax):
    model.eval() #Change model to evaluation mode for the attack 
    #Wrap the model using Foolbox's Pytorch wrapper 
    fmodel = foolbox.PyTorchModel(model, bounds=(clipMin, clipMax))
    #Create attack variable 
    #attack = foolbox.attacks.EADAttack(binary_search_steps=9, steps=1000, initial_stepsize=0.01, confidence=0.0, initial_const=0.001, regularization=0.01, decision_rule='EN', abort_early=True)
    attack = foolbox.attacks.L2CarliniWagnerAttack(binary_search_steps=9, steps=1000, stepsize=0.01, confidence=0, initial_const=0.001, abort_early=True)
    #Generate variables for storing the adversarial examples 
    numSamples = len(dataLoader.dataset) #Get the total number of samples to attack
    xShape = DMP.GetOutputShape(dataLoader) #Get the shape of the input (there may be easier way to do this)
    xAdv = torch.zeros(numSamples, xShape[0], xShape[1], xShape[2])
    yClean = torch.zeros(numSamples)
    advSampleIndex = 0 
    batchSize = 0 #just do dummy initalization, will be filled in later
    #Go through and generate the adversarial samples in batches 
    for i, (xCurrent, yCurrent) in enumerate(dataLoader):
        print("Attack batch=", i)
        batchSize = xCurrent.shape[0] #Get the batch size so we know indexing for saving later
        xCurrentCuda = xCurrent.to(device) #Load the data into the GPU
        yCurrentCuda = yCurrent.type(torch.LongTensor).to(device)
        criterion = foolbox.criteria.Misclassification(yCurrentCuda)
        #Next line actually runs the attack 
        _, advs, success = attack(fmodel, xCurrentCuda, epsilons=None, criterion=criterion)
        #Save the adversarial samples 
        for j in range(0, batchSize):
            xAdv[advSampleIndex] = advs[j]
            yClean[advSampleIndex] = yCurrent[j]
            advSampleIndex = advSampleIndex+1 #increment the sample index 
    #All samples processed, now time to save in a dataloader and return 
    advLoader = DMP.TensorToDataLoader(xAdv, yClean, transforms= None, batchSize= dataLoader.batch_size, randomizer=None) #use the same batch size as the original loader
    return advLoader
