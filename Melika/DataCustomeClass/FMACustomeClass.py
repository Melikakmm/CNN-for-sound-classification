#!/usr/bin/env python
# coding: utf-8

# In[1]:



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
#Confusion matrix:
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import itertools


# In[2]:


from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder, LabelBinarizer, StandardScaler


# In[3]:


import utils


# In[4]:


# define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("mps" if torch.backends.mps.is_available() else "cpu") !!!!! MACOS M1 !!!!


# In[5]:


DATA_DIR = '/Users/melikakeshavarz/desktop/fma/data/fma_small'

tracks = utils.load('/Users/melikakeshavarz/desktop/fma/data/fma_metadata/tracks.csv')
features = utils.load('/Users/melikakeshavarz/desktop/fma/data/fma_metadata/features.csv')#annotation files
echonest = utils.load('/Users/melikakeshavarz/desktop/fma/data/fma_metadata/echonest.csv')

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


len(train)


# In[7]:


tracks_index = tracks.index
tracks_index


# In[8]:


#From data to one hot labels
labels_onehot = LabelBinarizer().fit_transform(tracks['track', 'genre_top'])
labels_onehot_Ten = torch.tensor(labels_onehot)
labels_onehot = pd.DataFrame(labels_onehot, index=tracks.index)




#from onehot labels to encoded targets.
targets = torch.argmax(labels_onehot_Ten, dim=1)


# In[9]:


#creating an annotation dataframe from checksums in the FMA data folder. We are interested in the number of folders
#containing the songs.


df = pd.read_csv('/Users/melikakeshavarz/Desktop/fma/data/fma_small/checksums.txt', sep='  |/', header = None,
                 names = ['id', 'fold', 'songs'], converters={'fold': str})
df.index = tracks_index
df.loc[5][1]


# In[10]:








from torch.utils.data import Dataset, DataLoader


#custome dataset class
class FMA(Dataset):
    def __init__(self, data_dir, track_ids, annotation,
                 target_sample_rate, transformation, num_samples, device , twoD = False, paper_cut = False):
        self.annotation = annotation
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
        filepath = self._get_audio_sample_path(tid)
        label = torch.from_numpy(labels_onehot.loc[tid].values).float()
        
        try:
            waveform, sr = torchaudio.load(filepath)
        except:
            print(filepath)
        #be careful all of the sample rates aren't the same(resample)
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
    
    
    def _get_audio_sample_path(self, dex):
        fold = self.annotation.loc[dex][1]
        path = os.path.join(self.data_dir, fold, self.annotation.loc[dex][2])
        return path
        

            
    
    
    
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
     
    

        
        return waveform, label 
    
    def __len__(self):#just gives us the number of samples in our datasets.
        return len(self.track_ids) 

        


# In[11]:


#trying the class:

if __name__ == "__main__":
    

    SAMPLE_RATE=44100
    
    #maxlength
    NUM_SAMPLES = 44100
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
    
    
    
    FL = FMA(DATA_DIR, train, df, SAMPLE_RATE, mel_spectrogram, NUM_SAMPLES, Device, twoD =True)
    print(f"there are {len(FL)} samples in the dataset" )
    waveform, label = FL[0] #track number 2
    a = 1
    


# In[ ]:


#Here are corruted songs!
#it's your choice how to deal with the :)

Dex = tracks_index

def _get_audio_sample_path(data_dir, dex):
        fold = df.loc[dex][1]
        path = os.path.join(data_dir, fold, df.loc[dex][2])
        return path
    
for i in Dex:
    p = _get_audio_sample_path(DATA_DIR, i)
    try:
            w, sr = torchaudio.load(p)
    except:
            print(p)
    


# In[ ]:




