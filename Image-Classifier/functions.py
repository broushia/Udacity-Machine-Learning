#collection of functions to be used in both programs

import numpy as np
import torch
from torch import nn
from torch import optim

from torchvision import datasets, transforms, models
from collections import OrderedDict

from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sb
import json

import argparse

def train_input_args():
    print("   Command Line Arguments")
    print("-"*40)
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str, default = 'aipnd-project',
                      help='data location')
    parser.add_argument('--save_dir', type=str, default = 'checkpoint.pth',
                      help='save location of model checkpoint')
    parser.add_argument('--arch', type=str, default ='vgg16',
                      help='type of pretrained model used - vgg16, vgg19, or alexnet')
    parser.add_argument('--learning_rate', type=float, default ='0.001',
                      help='model learning rate')
    parser.add_argument('--hidden_units', type=int, default ='5000',
                      help='number of nodes in middle layer of classifier')
    parser.add_argument('--epochs', type=int, default ='5',
                      help='number of training epochs')
    parser.add_argument('--device', type=str, default ='cuda',
                      help='device used to run model - cpu or cuda')    
    return parser.parse_args()

def predict_input_args():
    print("   Command Line Arguments")
    print("-"*40)
    parser = argparse.ArgumentParser()
    parser.add_argument('image_dir', type=str, default ='aipnd-project/flowers/test/1/image_06743.jpg', #I know the default doesn't do anything but its nice to have here
                       help="path to image used for prediction")
    parser.add_argument('load_dir', type=str, default ='aipnd-project/checkpoint', #I know the default doesn't do anything but its nice to have here
                      help="checkpoint used for prediction") 
    parser.add_argument('--top_k', type=int, default ='5',
                      help="returns 'K' most likely results") 
    parser.add_argument('--device', type=str, default ='cpu',
                      help="device used to run model - cpu or cuda") 
    parser.add_argument('--json', type=str, default ='aipnd-project/cat_to_name.json',
                      help='location of class names')
    return parser.parse_args()

def load_images():
    
    data_dir = 'aipnd-project/flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    dirs = {'train' : train_dir,
            'valid' : valid_dir,
            'test' : test_dir}
    
    data_transforms = {'train' : transforms.Compose([transforms.RandomRotation(30),
                                                     transforms.RandomResizedCrop(224),
                                                     transforms.RandomHorizontalFlip(),
                                                     transforms.ToTensor(),
                                                     transforms.Normalize([0.485,0.456,0.406],
                                                                          [0.229,0.224,0.225])]),
                       'valid' : transforms.Compose([transforms.Resize(224),
                                                     transforms.CenterCrop(224),
                                                     transforms.ToTensor(),
                                                     transforms.Normalize([0.485,0.456,0.406],
                                                                          [0.229,0.224,0.225])]),
                               
                       'test' : transforms.Compose([transforms.Resize(224),
                                                    transforms.CenterCrop(224),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize([0.485,0.456,0.406],
                                                                         [0.229,0.224,0.225])])}
    # Load the datasets with ImageFolder
    image_data = {'train': datasets.ImageFolder(dirs['train'], transform = data_transforms['train']),
                  'valid': datasets.ImageFolder(dirs['valid'], transform = data_transforms['valid']),
                  'test': datasets.ImageFolder(dirs['test'], transform = data_transforms['test'])}

    # Using the image datasets and the transforms, define the dataloaders
    data_loader = {'train' : torch.utils.data.DataLoader(image_data['train'], batch_size=32, shuffle=True),
                   'valid' : torch.utils.data.DataLoader(image_data['valid'], batch_size=32, shuffle=False),
                   'test' : torch.utils.data.DataLoader(image_data['test'], batch_size=32, shuffle=False)}
    
    return image_data, data_loader

def build_model(arch , hidden_units, learning_rate):
    
    #Define new network
    classifier_toplayer = 0
    if arch == "alexnet":
        model = models.alexnet(pretrained = True)
        classifier_toplayer = 9216
    elif arch ==  "vgg19":
        model = models.vgg19(pretrained = True)
        classifier_toplayer = 25088
    else:
        model = models.vgg16(pretrained = True)
        classifier_toplayer = 25088
        if arch != 'vgg16':
            print("\n   Invalid architecture selected. Default VGG16 selected")

    for param in model.parameters():
        param.requires_grad = False   

    classifier = nn.Sequential(OrderedDict([
                              ('layer0', nn.Linear(classifier_toplayer,hidden_units)),
                              ('relu0', nn.ReLU()),
                              ('layer1', nn.Linear(hidden_units,102)),
                              ('output', nn.LogSoftmax(dim = 1))]))
    
    classifier.dropout = nn.Dropout( p =0.1)
    model.classifier = classifier

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate) #remember to adjust learning rate as needed
    
    return model, optimizer, criterion

def train_model(model, data_loader, epochs, criterion, optimizer, device = 'cpu'):
    print("\n   ---Model is training---")
    
    model.to(device)
    loss_data = {}   
    
    for e in range(epochs):        
        for mode in ['train','valid']:
            if mode == 'train':
                model.train()
            else:
                model.eval()
            
            running_loss = 0
            correct = 0
            
            for images,labels in data_loader[mode]:
                images, labels = images.to(device), labels.to(device)
                
                optimizer.zero_grad()
            
                outputs = model.forward(images)
                _,predicted = torch.max(outputs.data, 1)            
                loss = criterion(outputs, labels)
                
                if mode == 'train':
                    loss.backward()
                    optimizer.step()
            
                running_loss += loss.item()*images.size(0)
                correct += (predicted == labels).sum().item()
                
            loss_data[mode] = (running_loss / len(data_loader[mode].dataset))                       
        
        accuracy = (correct / len(data_loader[mode].dataset)) * 100       
              
        print('\n   ---Epoch {}/{}---'.format(e+1,epochs))
        print('-'*40)
        print("Training Loss: {:.4f}  Validation Loss: {:.4f}  Accuracy: {:.0f}%\n".format(loss_data['train'],
                                                                                           loss_data['valid'], accuracy))
    print("-"*40)    
    print('   Training Complete - Fingers Crossed')

    return model

def save_model(model, optimizer, arch, data_dir, save_dir, image_data):

    model.to('cpu')
    model.class_to_idx = image_data['train'].class_to_idx
    
    checkpoint = {'arch':arch,
                  'hidden_layer': model.classifier.layer1.in_features,
                  'optimizer_dict': optimizer.state_dict(),
                  'model_dict': model.state_dict(),
                  'class_to_idx':model.class_to_idx}

    torch.save(checkpoint, data_dir + '/' + save_dir)
    
    print("\n   Checkpoint Save - Complete")
    
    
def load_model(file_path,device):

    checkpoint = torch.load(file_path)
    
    if checkpoint["arch"] == "alexnet":
        model = models.alexnet(pretrained = True)
        classifier_toplayer = 9216
    elif checkpoint["arch"] ==  "vgg19":
        model = models.vgg19(pretrained = True)
        classifier_toplayer = 25088
    else:
        model = models.vgg16(pretrained = True)
        classifier_toplayer = 25088
    
    for param in model.parameters():
        param.requires_grad = False
    
    model.class_to_idx = checkpoint['class_to_idx']    

    classifier = nn.Sequential(OrderedDict([
                          ('layer0', nn.Linear(classifier_toplayer,checkpoint['hidden_layer'])), #add toplayer to checkpoint
                          ('relu0', nn.ReLU()),
                          ('layer1', nn.Linear(checkpoint['hidden_layer'],102)),
                          ('output', nn.LogSoftmax(dim = 1))]))
     
    model.classifier = classifier    
    model.load_state_dict(checkpoint['model_dict'])   
    model.to(device)
    
    optimizer = optim.Adam(model.classifier.parameters())
    optimizer.load_state_dict(checkpoint['optimizer_dict'])
    
    print("\n   Model Load - Complete")

    return model, optimizer 

def process_image(image_path):
    image = Image.open(image_path)    
    
    #determine size of thumbnail so smallest side is 256 
    width,height = image.size
    if width < height:
        thumbh = 256 * height /width
        thumbw = 256
    else:
        thumbw = 256 * width / height
        thumbh = 256

    thumb_size = thumbw , thumbh
    image.thumbnail(thumb_size, Image.ANTIALIAS)
    width,height = image.size
    
    #find limits of cropped image
    left = (width - 224) / 2
    top = (height - 224) / 2
    right = (width + 224) / 2 
    bottom = (height + 224) /2
    
    image = image.crop((left,top,right,bottom))     
    
    #convert to numpy array
    np_image = np.array(image)
    np_image = np_image / 255
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    np_image = (np_image - mean) / std
    np_image = np_image.transpose((2,0,1))
    
    return np_image


def predict(image_path, model, j_son, topk=5, device = 'cpu'):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    with open(j_son, 'r') as f:
        cat_to_name = json.load(f)
    
    model.to(device)

    image_tensor = torch.from_numpy(process_image(image_path)).type(torch.FloatTensor).to(device)
    #adds a batch size
    image_tensor.unsqueeze_(0)
    
    output = model(image_tensor)

    probs, classes = output.topk(topk)
    #there has got to be a better way but at least it works
    probs = torch.exp(probs)
    probs = probs.data.cpu().tolist()[0] #.numpy()[0]
    classes = classes.data.cpu().tolist()[0] #.numpy()[0]
    
    labels = []
    
    for i in range(len(classes)):
        classes[i]=classes[i]+1
        labels.append(cat_to_name[str(classes[i])])
        #IT WORKS   
    return probs, classes, labels
