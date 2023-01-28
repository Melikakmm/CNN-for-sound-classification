#!/usr/bin/env python
# coding: utf-8

# In[16]:


import os
# from tqdm import tqdm
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F  
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import copy
from torchsummary import summary


# In[17]:


from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder, LabelBinarizer, StandardScaler


# In[18]:


import utils


# In[19]:


# define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[20]:


DATA_DIR = '/Users/melikakeshavarz/desktop/new/data/fma_small'

tracks = utils.load('/Users/melikakeshavarz/desktop/new/data/fma_metadata/tracks.csv')
features = utils.load('/Users/melikakeshavarz/desktop/new/data/fma_metadata/features.csv')#annotation files
echonest = utils.load('/Users/melikakeshavarz/desktop/new/data/fma_metadata/echonest.csv')

subset = tracks.index[tracks['set', 'subset'] <= 'small']

assert subset.isin(tracks.index).all()
assert subset.isin(features.index).all()

features_all = features.join(echonest, how='inner').sort_index(axis=1)
print('Not enough Echonest features: {}'.format(features_all.shape))

tracks = tracks.loc[subset]
features_all = features.loc[subset]

tracks.shape, features_all.shape

train = tracks.index[tracks['set', 'split'] == 'training'] #bunch of indexes (not ids) for training val and test
val = tracks.index[tracks['set', 'split'] == 'validation']
test = tracks.index[tracks['set', 'split'] == 'test']


# In[6]:


#pause
labels_onehot = LabelBinarizer().fit_transform(tracks['track', 'genre_top'])
labels_onehot = pd.DataFrame(labels_onehot, index=tracks.index)


# In[21]:


#Meeeeeeeeeee

#custome dataset class



from torch.utils.data import Dataset, DataLoader



class FMA(Dataset):
    def __init__(self, data_dir, track_ids,
                 target_sample_rate, transformation, num_samples, device , twoD = False, paper_cut = True):
        self.data_dir = data_dir
        self.track_ids = track_ids
        self.filenames = os.listdir(data_dir)
        self.target_sample_rate = target_sample_rate
        self.device = device
        self.transformation = transformation.to(self.device)
        self.twoD = twoD
        self.num_samples = num_samples
        self.paper_cut = paper_cut
        
        

    def __getitem__(self, index):
        tid = self.track_ids[index]
        filepath = utils.get_audio_path(self.data_dir, tid)
        label = torch.from_numpy(labels_onehot.loc[tid].values).float()
        waveform, sr = torchaudio.load(filepath)#be careful all of the sample rates aren't the same(resample)
        #waveform --> (2, 10000) #(number of channels, number of samples)
        waveform = waveform.to(self.device)
        waveform = self._resample_if_necessary(waveform, sr)
        waveform = self._mix_down_if_necessary(waveform)
        #we have to adjust the length of the audio waveforms before the transformation
        waveform = self._cut_if_necessary(waveform)
        waveform = self._right_pad_if_necessary(waveform)
        if self.twoD == True:
            waveform = self.transformation(waveform)
        else:
            pass
        
        
        if self.paper_cut == True:
            waveform = waveform[:, :128, :513]
        else:
            pass
            
        return waveform, label
    
    
    def _cut_if_necessary(self, waveform):
        #this method happens before the transformation
        if waveform.shape[1] > self.num_samples:
            waveform = waveform[:, :self.num_samples]
            return waveform
        
        
    def _right_pad_if_necessary(self, waveform):
        if waveform.shape[1] < self.num_samples:
            num_missing_samples = self.num_samples - waveform.shape[1]
            last_dim_padding = (0,num_missing_samples) # (1, 2) -> (left, right)   
            #(1, 2, 0, 1) -> (left, right, padnumleft, padnumright)
            # what happens is : [1, 1, 1] --> [0, 1, 1, 1, 0, 0]
            waveform = torch.nn.functional.pad(waveform, last_dim_padding)
            waveform = waveform.T
        return waveform
    
    
        
    def _resample_if_necessary(self, waveform , sr):
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            waveform = resampler(waveform)
        return waveform
    
    
    #from (2, 10000) to (1, 0000) taking the average between two waveforms
    def _mix_down_if_necessary(self, waveform):
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform , dim = 0, keepdim = True)
        return waveform
        

        
        return waveform.T, label 
    
    def __len__(self):#just gives us the number of samples in our datasets.
        return len(self.filenames) 

        

        


# In[28]:


#trying the class:

if __name__ == "__main__":
    

    SAMPLE_RATE=44100
    
    #maxlength
    NUM_SAMPLES = 1320000
    #working on GPU
    if torch.cuda.is_available():
        Device = "cuda"
    else:
        Device = "cpu"
        
    print(f"we are using {Device}.")  
    
    #50% hop_length is the best for accuracy
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate = SAMPLE_RATE, n_fft = 1024, hop_length = 256,
                                                        n_mels = 64) 
    
    
    n_fft = 1024    # FFT window size
    hop_length = 256    # number of samples between successive frames
    win_length = n_fft 
    
    spectrogram = torchaudio.transforms.Spectrogram(n_fft=1024, hop_length = 256, win_length = win_length )
    
    
    
    FL = FMA(DATA_DIR, train, SAMPLE_RATE, spectrogram, NUM_SAMPLES, Device, twoD =True)
    #print(f"there are {len(FL)} samples in the dataset" )
    waveform, label = FL[0] #track number 2
    a = 1
    
    


# In[29]:


# If melspectrogram is applied

waveform.shape


# In[24]:


####### very bad 2D architecture :))))))) just a test
from torch import nn

class CNN2D(nn.Module):
    
    
    def __init__(self):
        #vgg
        super().__init__()
        #4 conv blocks / flatten / linear / softmax
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels =1, out_channels =16, kernel_size =3, stride =1, padding =2),
                                  nn.ReLU(),
                                  nn.MaxPool2d(kernel_size = 2))
        
        
        
        
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels =16, out_channels =32, kernel_size =3, stride =1, padding =2),
                                  nn.ReLU(),
                                  nn.MaxPool2d(kernel_size = 2))
        
        
        
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels =32, out_channels =64, kernel_size =3, stride =1, padding =2),
                                  nn.ReLU(),
                                  nn.MaxPool2d(kernel_size = 2))
        
        
        
        self.conv4 = nn.Sequential(nn.Conv2d(in_channels =64, out_channels =128, kernel_size =3, stride =1, padding =2),
                                  nn.ReLU(),
                                  nn.MaxPool2d(kernel_size = 2))
        
        

        
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(38016, 8)
        self.softmax =  nn.Softmax(dim = 1)
        
        
     #in this method we tell pytorch how to pass data from layer to another layer   
    def forward(self, input_data):
        x = self.conv1(input_data)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.flatten(x)
        logits = self.linear(x)
        predictions = self.softmax(logits)
        return predictions
        
        
            





# In[25]:


if __name__ == "__main__":
    if torch.cuda.is_available():
        Device = "cuda"
    else:
        Device = "cpu"
    print(f"Using {Device}")
    cnn = CNN2D()
    summary(cnn.to(Device), (1, 128, 513) ) #summary(model, size of the spectogram)
    
    #warning: the input is on gpu that's why we have to have the model on the smae device


# In[27]:


BATCH = 64

# create a training dataset and dataloader
FL = FMA(DATA_DIR, train, SAMPLE_RATE,spectrogram, NUM_SAMPLES, Device, twoD =True)
val_dataset = FMA(DATA_DIR, val, SAMPLE_RATE, spectrogram, NUM_SAMPLES, Device, twoD =True)


# create a validation dataset and dataloader
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH, shuffle=True)
dataloader = torch.utils.data.DataLoader(FL, batch_size=BATCH, shuffle=True)

    
# create the CNN model
model = CNN2D().to(Device) # HERE YOU PUT UR NETWORK
model.to(device)

# define the loss function and the optimizer
loss_fn = nn.CrossEntropyLoss()

# Adam optimizer
# optimizer = torch.optim.Adam(model.parameters())


# Allamy 2021
# Define the optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=True)

# Define the scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.2, patience=5)



num_epochs = 10
i = 0
running_loss = 0.0


   
# train the model
for epoch in range(num_epochs):
    # evaluate the model on the training dataset
    train_correct = 0
    train_total = 0
    for waveform, label in dataloader:
        label = label.to(device)
        train_label = torch.argmax(label, dim=1)

        # clear the gradients
        optimizer.zero_grad()

        # forward pass
        waveform = waveform.squeeze(0)

        
        waveform = waveform.to(device)
        output = model(waveform)
            
        loss = loss_fn(output, label)

        # backward pass
        loss.backward()
        optimizer.step()  
        
        # Update the learning rate
        scheduler.step(loss)
            
        _, train_predicted = torch.max(output.data, 1)
        train_total += train_label.size(0)
        train_correct += (train_predicted == train_label).sum().item()
        # print statistics
        i += 1
        running_loss += loss.item()
            
           
    print('[%d, %5d subsamples] Training loss: %.3f' % (epoch + 1, i*BATCH, running_loss / len(dataloader)))
    running_loss = 0            
    # evaluate the model on the validation dataset
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for val_waveform, val_label in val_dataloader:
            val_label = val_label.to(device)
            val_label = torch.argmax(val_label, dim=1)
            val_waveform = val_waveform.squeeze(0)
            
            val_waveform = val_waveform.to(device)
            val_output = model(val_waveform)
            val_loss += loss_fn(val_output, val_label).item()
            _, val_predicted = torch.max(val_output.data, 1)
            val_total += val_label.size(0)
            val_correct += (val_predicted == val_label).sum().item()


    print('Validation Loss: {:.4f} | Validation Accuracy: {:.4f} | Training Accuracy: {:.4f}'.format(val_loss / len(val_dataloader), val_correct / val_total, train_correct / train_total))
print('Finished Training')



# In[30]:


###the first architecture on the paper 




class nnet1(nn.Module):
    
    
    def __init__(self):

        super().__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(in_channels =1, out_channels =128, kernel_size =(4, 513), stride =1, padding =2, bias=True),
                                  nn.ReLU(),
                                  nn.MaxPool2d(kernel_size = (2, 1), stride = (2, 1)))
        
        
        
        
        self.conv2 = nn.Sequential(nn.Conv2d( in_channels =128, out_channels =128,kernel_size =(4, 1), stride =1, padding =2, bias=True),
                                  nn.ReLU(),
                                  nn.MaxPool2d(kernel_size = (2, 1), stride = (2, 1)))
        
        
        
        self.conv3 = nn.Sequential(nn.Conv2d( in_channels =128, out_channels =256, kernel_size =(4, 1), stride =1, padding =2, bias=True),
                                  nn.ReLU())
        
        
        
        self.Max = nn.MaxPool2d(kernel_size = (26, 1))
        self.Avg = nn.AvgPool2d(kernel_size = (26, 1))
        
        

        
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(6656, 300)
        self.linear2 = nn.Linear(300, 150 )
        self.linear3 = nn.Linear(150, 8 )
        self.softmax =  nn.Softmax(dim = 1)
        
        
     #in this method we tell pytorch how to pass data from layer to another layer   
    def forward(self, input_data):
        x = self.conv1(input_data)

        x = self.conv2(x)

        x = self.conv3(x)

        y = self.Max(x)

        z = self.Avg(x)

        x = torch.cat((y, z), dim=1)

        x = self.flatten(x)

        x = self.linear1(x)

        x = self.linear2(x)

        logits = self.linear3(x)

        
        predictions = self.softmax(logits)
        return predictions
        
        
            



# In[32]:


if __name__ == "__main__":
    if torch.cuda.is_available():
        Device = "cuda"
    else:
        Device = "cpu"
    print(f"Using {Device}")
    cnn =nnet1()
    summary(cnn.to(Device), (1, 128, 513) ) #summary(model, size of the spectogram)
    
    #warning: the input is on gpu that's why we have to have the model on the smae device


# In[34]:


BATCH = 64

# create a training dataset and dataloader
FL = FMA(DATA_DIR, train, SAMPLE_RATE,spectrogram, NUM_SAMPLES, Device, twoD =True)
val_dataset = FMA(DATA_DIR, val, SAMPLE_RATE, spectrogram, NUM_SAMPLES, Device, twoD =True)


# create a validation dataset and dataloader
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH, shuffle=True)
dataloader = torch.utils.data.DataLoader(FL, batch_size=BATCH, shuffle=True)

    
# create the CNN model
model = nnet1().to(Device) # HERE YOU PUT UR NETWORK
model.to(device)

# define the loss function and the optimizer
loss_fn = nn.CrossEntropyLoss()

# Adam optimizer
# optimizer = torch.optim.Adam(model.parameters())


# Allamy 2021
# Define the optimizer
optimizer = torch.optim.Adadelta(model.parameters(), lr=1)

# Define the scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.2, patience=5)



num_epochs = 10
i = 0
running_loss = 0.0


    
# train the model

for epoch in range(num_epochs):
    # evaluate the model on the training dataset
    train_correct = 0
    train_total = 0
    for waveform, label in dataloader:
        label = label.to(device)
        train_label = torch.argmax(label, dim=1)

        # clear the gradients
        optimizer.zero_grad()

        # forward pass
        waveform = waveform.squeeze(0)

        
        waveform = waveform.to(device)
        output = model(waveform)
            
        loss = loss_fn(output, label)

        # backward pass
        loss.backward()
        optimizer.step()  
        
        # Update the learning rate
        scheduler.step(loss)
            
        _, train_predicted = torch.max(output.data, 1)
        train_total += train_label.size(0)
        train_correct += (train_predicted == train_label).sum().item()
        # print statistics
        i += 1
        running_loss += loss.item()
            
           
    print('[%d, %5d subsamples] Training loss: %.3f' % (epoch + 1, i*BATCH, running_loss / len(dataloader)))
    running_loss = 0            
    # evaluate the model on the validation dataset
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for val_waveform, val_label in val_dataloader:
            val_label = val_label.to(device)
            val_label = torch.argmax(val_label, dim=1)
            val_waveform = val_waveform.squeeze(0)
            
            val_waveform = val_waveform.to(device)
            val_output = model(val_waveform)
            val_loss += loss_fn(val_output, val_label).item()
            _, val_predicted = torch.max(val_output.data, 1)
            val_total += val_label.size(0)
            val_correct += (val_predicted == val_label).sum().item()


    print('Validation Loss: {:.4f} | Validation Accuracy: {:.4f} | Training Accuracy: {:.4f}'.format(val_loss / len(val_dataloader), val_correct / val_total, train_correct / train_total))
print('Finished Training')


# In[35]:



class ResidualBlock(nn.Module):
    def __init__(self):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channels = 1, out_channels = 128, kernel_size = (4, 513), stride = 1, padding = 2, bias = False),
                        nn.BatchNorm2d(128),
                        nn.ReLU(inplace = True))
        self.conv2 = nn.Sequential(
                        nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = (4, 1), stride = 1, padding = 2, bias = False),
                        nn.BatchNorm2d(128))
        self.conv3 = nn.Sequential(
                        nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = (4, 1), stride = 1, padding = 2, bias = False),
                        nn.BatchNorm2d(256))
        self.relu = nn.ReLU()
        
        
        self.downsample =nn.Conv2d(in_channels = 1, out_channels = 256, kernel_size = (4, 505), padding = (3, 2), stride = 1, bias= False )

        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        
        #print(residual.size(), out.size())
        if self.downsample:
            residual = self.downsample(x)
        #print(residual.size(), out.size())
        out += residual
        out = self.relu(out)
        return out       




class nnet2(nn.Module):
    def __init__(self):
        super(nnet2, self).__init__()
        
        self.block = ResidualBlock()
        
        self.Max = nn.MaxPool2d(kernel_size = (26, 1))
        self.Avg = nn.AvgPool2d(kernel_size = (26, 1))
        
        

        
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(33280, 300)
        self.linear2 = nn.Linear(300, 150 )
        self.linear3 = nn.Linear(150, 8 )
        self.softmax =  nn.Softmax(dim = 1)
        
        
        
    def forward(self, x):
        
        x = self.block(x)

        
        y = self.Max(x)
        z = self.Avg(x)
        x = torch.cat((y, z), dim=1)
        
        x = self.flatten(x)

        x = self.linear1(x)

        x = self.linear2(x)

        logits = self.linear3(x)

        
        #predictions = self.softmax(logits)
        return logits
        
        
        
        
    
    
    
    
    
    
    
    


# In[36]:


if __name__ == "__main__":
    if torch.cuda.is_available():
        Device = "cuda"
    else:
        Device = "cpu"
    print(f"Using {Device}")
    
    
   
    
    cnn =nnet2()
    summary(cnn.to(Device), (1, 128, 513) ) #summary(model, size of the spectogram)
    
    #warning: the input is on gpu that's why we have to have the model on the smae device


# In[38]:


BATCH = 64

# create a training dataset and dataloader
FL = FMA(DATA_DIR, train, SAMPLE_RATE,spectrogram, NUM_SAMPLES, Device, twoD =True)
val_dataset = FMA(DATA_DIR, val, SAMPLE_RATE, spectrogram, NUM_SAMPLES, Device, twoD =True)


# create a validation dataset and dataloader
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH, shuffle=True)
dataloader = torch.utils.data.DataLoader(FL, batch_size=BATCH, shuffle=True)

    
# create the CNN model
model = nnet2().to(Device) # HERE YOU PUT UR NETWORK
model.to(device)

# define the loss function and the optimizer
loss_fn = nn.CrossEntropyLoss()

# Adam optimizer
# optimizer = torch.optim.Adam(model.parameters())


# Allamy 2021
# Define the optimizer
optimizer = torch.optim.Adadelta(model.parameters(), lr=1)

# Define the scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.2, patience=5)



num_epochs = 10
i = 0
running_loss = 0.0


    
# train the model

for epoch in range(num_epochs):
    # evaluate the model on the training dataset
    train_correct = 0
    train_total = 0
    for waveform, label in dataloader:
        label = label.to(device)
        train_label = torch.argmax(label, dim=1)

        # clear the gradients
        optimizer.zero_grad()

        # forward pass
        waveform = waveform.squeeze(0)

        
        waveform = waveform.to(device)
        output = model(waveform)
            
        loss = loss_fn(output, label)

        # backward pass
        loss.backward()
        optimizer.step()  
        
        # Update the learning rate
        scheduler.step(loss)
            
        _, train_predicted = torch.max(output.data, 1)
        train_total += train_label.size(0)
        train_correct += (train_predicted == train_label).sum().item()
        # print statistics
        i += 1
        running_loss += loss.item()
            
           
    print('[%d, %5d subsamples] Training loss: %.3f' % (epoch + 1, i*BATCH, running_loss / len(dataloader)))
    running_loss = 0            
    # evaluate the model on the validation dataset
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for val_waveform, val_label in val_dataloader:
            val_label = val_label.to(device)
            val_label = torch.argmax(val_label, dim=1)
            val_waveform = val_waveform.squeeze(0)
            
            val_waveform = val_waveform.to(device)
            val_output = model(val_waveform)
            val_loss += loss_fn(val_output, val_label).item()
            _, val_predicted = torch.max(val_output.data, 1)
            val_total += val_label.size(0)
            val_correct += (val_predicted == val_label).sum().item()


    print('Validation Loss: {:.4f} | Validation Accuracy: {:.4f} | Training Accuracy: {:.4f}'.format(val_loss / len(val_dataloader), val_correct / val_total, train_correct / train_total))
print('Finished Training')

