import pandas as pd
import h5py
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import gc
import random
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import scipy
import os
from scipy import signal as sp
import torch.nn.functional as F

import utils_pre_post_es46_online as u

print(torch.cuda.is_available())
if torch.cuda.is_available():  
    dev = "cuda" 
    map_location=None
else:  
    dev = "cpu"  
    map_location='cpu'
device = torch.device(dev)


dataToProcess = "NRCA" #sys.argv[1]
num_classes = 4 #int(sys.argv[2])
seed = 123 #int(sys.argv[3])

print("num_classes",num_classes, ". If you want to change this variable, please change this in the code.")
path=os.getcwd()+'/'
print("path: ",path)


    
preprocessing=False # this is True only the firse time, then it is no more needed
plot_model_hidden=False # Set True to visualize hidden states plots, during test
train=True
trValTest_split_rnd=False
plot_model_hidden=False #this forces plot_model_hidden to be false in the case of CNN, because this is not implemented

u.seed_everything(seed)
traces_in_test="NO" # traces to be forced in test set: "NO" or "similar_events" or "NRCA_2000To2022" or "jan_aug_2017" or "pre2000" or "first_last_part" or "visso_test"
force_traces_in_test=[]#['NRCA.IV.100758332_EV','NRCA.IV.100401915_EV'] #
                
df_empty = pd.DataFrame(columns = ['E_channel', 'N_channel', 'Z_channel', 'trace_name', 'label',
    'trace_start_time', 'network_code', 'receiver_name', 'receiver_type',
    'receiver_elevation_m', 'receiver_latitude', 'receiver_longitude',
    'source_id', 'source_depth_km', 'source_latitude', 'source_longitude',
    'source_magnitude_type', 'source_magnitude', 'source_origin_time', 'p_travel_sec'])
df_pre = df_empty.copy() 
df_visso = df_empty.copy() # if num_classes!=9 this df will remain empty
df_post = df_empty.copy()

df_pre = pd.read_pickle(path+'dataframe_pre_'+dataToProcess+'.csv')
df_post = pd.read_pickle(path+'dataframe_post_'+dataToProcess+'.csv')
if num_classes==9:
    df_visso = pd.read_pickle(path+'dataframe_visso_'+dataToProcess+'.csv')
df_pre, df_visso, df_post=u.pre_post_equal_length(df_pre, df_visso, df_post,force_traces_in_test, num_classes)
    
for i in force_traces_in_test:
    if (i not in df_pre['trace_name'].values) and (i not in df_visso['trace_name'].values) and (i not in df_post['trace_name'].values):
        print("WARNING: ", i," not in df_pre and df_post. This will cause an error.")

df_pre['trace_start_time'] = df_pre['trace_start_time'].apply(lambda x: datetime.strptime(x, '%Y-%m-%dT%H:%M:%S.%fZ'))
df_visso['trace_start_time'] = df_visso['trace_start_time'].apply(lambda x: datetime.strptime(x, '%Y-%m-%dT%H:%M:%S.%fZ'))
df_post['trace_start_time'] = df_post['trace_start_time'].apply(lambda x: datetime.strptime(x, '%Y-%m-%dT%H:%M:%S.%fZ'))

if num_classes==2:
    df=pd.concat([df_pre, df_post], ignore_index=True)
else:
    frames_pre = u.frames_N_classes(df_pre,num_classes, pre_or_post="pre")
    frames_post = u.frames_N_classes(df_post,num_classes, pre_or_post="post")
    if num_classes==9:
        frames_visso = u.frames_N_classes(df_visso,num_classes, pre_or_post="visso")
        df=pd.concat([pd.concat(frames_pre),pd.concat(frames_visso),pd.concat(frames_post)], ignore_index=True)
    else:
        df=pd.concat([pd.concat(frames_pre),pd.concat(frames_post)], ignore_index=True)

df['source_origin_time'] = df['source_origin_time'].apply(lambda x: datetime.strptime(x, '%Y-%m-%dT%H:%M:%S'))
df['TTF'] = df.apply (lambda row: u.add_TTF_in_sec(row), axis=1)

df, X_train, y_train, index_train, X_val, y_val, index_val, X_test, y_test, index_test=u.train_val_test_split(df, train_percentage=0.70, val_percentage=0.10, test_percentage=0.20,force_in_test=force_traces_in_test, split_random=trValTest_split_rnd)
batch_size = 32
tr_dl = u.create_dataloader(X=X_train, y=y_train, index=index_train,target_dataset="train_dataset", batch_size=batch_size)
val_dl = u.create_dataloader(X=X_val, y=y_val, index=index_val,target_dataset="val_dataset", batch_size=batch_size)
test_dl = u.create_dataloader(X=X_test, y=y_test, index=index_test,target_dataset="test_dataset", batch_size=batch_size)
            
loss_function = nn.CrossEntropyLoss()
min_loss = np.Inf
num_epochs = 50 # 100
learning_rate = 0.001#0.00001
init_lr = learning_rate
lr_decay = 0.99

inp_size=X_train.shape[1]
out_size=y_train.shape[1]
print('inp_size',inp_size, 'out_size', out_size)
s_max=False #if True it add a softmax layer before the output
model=u.CNN(num_feature=inp_size, num_class=out_size).to(device)
    
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.001)
best_model=type(model)(num_feature=inp_size, num_class=out_size).to(device)

for epoch in range(num_epochs):
    model.train()
    sum_loss_tr = 0
    sum_acc_tr = 0
    for idx, batch in tqdm(enumerate(tr_dl), total=len(tr_dl)):#enumerate(tr_dl):
        model.zero_grad() 
        inp = batch[0].to(device)
        label = batch[1].to(device)
        output = model(inp.float(), batch_size=batch_size, steps_in=inp_size, softmax=s_max)[0]
        
        current_loss = F.cross_entropy(output, label)#loss_function(output, label)
        current_loss.backward()
        optimizer.step()
        sum_loss_tr += current_loss.item()*batch_size
        confusion_matrix = torch.zeros(out_size, out_size)
        _, preds = torch.max(output, 1)
        _, classes = torch.max(label, 1)
        for t, p in zip(classes.view(-1), preds.view(-1)):
            confusion_matrix[t.long(), p.long()] += 1
        curr_avg_accuracy=(torch.mean(confusion_matrix.diag()/confusion_matrix.sum(1)).item())*100
        sum_acc_tr+=curr_avg_accuracy

    curr_tr_loss = sum_loss_tr/len(tr_dl)
    curr_tr_acc = sum_acc_tr/len(tr_dl)
                    
    model.eval()
    with torch.no_grad():#.inference_mode():#
        sum_loss_val = 0
        sum_acc_val = 0

        for idx, batch in tqdm(enumerate(val_dl), total=len(val_dl)):#enumerate(val_dl):
            inp = batch[0].to(device)
            label = batch[1].to(device)
            output = model(inp.float(), batch_size=batch_size, steps_in=inp_size, softmax=s_max)[0]

            sum_loss_val += F.cross_entropy(output, label).item()*batch_size#loss_function(output, label)
            confusion_matrix = torch.zeros(out_size, out_size)
            _, preds = torch.max(output, 1)
            _, classes = torch.max(label, 1)
            for t, p in zip(classes.view(-1), preds.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
            curr_avg_accuracy=(torch.mean(confusion_matrix.diag()/confusion_matrix.sum(1)).item())*100
            sum_acc_val+=curr_avg_accuracy

        curr_val_loss = sum_loss_val/len(val_dl)
        curr_val_acc = sum_acc_val/len(val_dl)

        if curr_val_loss < min_loss:
            min_loss = curr_val_loss
            del best_model
            best_model = type(model)(num_feature=inp_size, num_class=out_size)
            best_model.load_state_dict(model.state_dict())
            print("Best Epoch:", epoch+1)
    
    model.eval()
    with torch.no_grad():#.inference_mode():#

        sum_loss_te = 0 
        sum_acc_te = 0 
        for idx, batch in tqdm(enumerate(test_dl), total=len(test_dl)):#enumerate(test_dl):
            inp = batch[0].to(device)
            label = batch[1].to(device)
            output = model(inp.float(), batch_size=batch_size, steps_in=inp_size, softmax=s_max)[0]
            sum_loss_te += F.cross_entropy(output, label).item()*batch_size#loss_function(output, label)
            confusion_matrix = torch.zeros(out_size, out_size)
            _, preds = torch.max(output, 1)
            _, classes = torch.max(label, 1)
            for t, p in zip(classes.view(-1), preds.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
            curr_avg_accuracy=(torch.mean(confusion_matrix.diag()/confusion_matrix.sum(1)).item())*100
            sum_acc_te+=curr_avg_accuracy

        curr_te_loss = sum_loss_te/len(test_dl)
        curr_te_acc = sum_acc_te/len(test_dl)
        
    print("Epoch", epoch+1, "\tTrain Loss:", curr_tr_loss, "\tValid Loss:", curr_val_loss, "\tTest Loss:", curr_te_loss)
    print("Train Accuracy:", curr_tr_acc, "\tValid Accuracy:", curr_val_acc, "\tTest Accuracy:", curr_te_acc)
    
best_model.to(dev)

inputte=[]
labeltte=[]
outputte=[]
indexte=[]
confusion_matrix = torch.zeros(out_size, out_size)

best_model.eval()
with torch.no_grad():#.inference_mode():#
    sum_loss_te = 0 
    for idx, batch in tqdm(enumerate(test_dl), total=len(test_dl)):
        inp = batch[0].to(dev)
        inputte.append(inp)
        label = batch[1].to(dev)
        labeltte.append(label)
        indexte.append(batch[2].to(dev))
        output = best_model(inp.float(), batch_size=batch_size, steps_in=inp_size, softmax=s_max)[0]
        _, preds = torch.max(output, 1)
        _, classes = torch.max(label, 1)
        for t, p in zip(classes.view(-1), preds.view(-1)):
            confusion_matrix[t.long(), p.long()] += 1               
        outputte.append(output)
        sum_loss_te += loss_function(output, label).item()*batch_size
    curr_te_loss = sum_loss_te/len(test_dl)
    print("curr_te_loss: ",curr_te_loss)
print("confusion_matrix",confusion_matrix)
# print("per-class accuracy: ",confusion_matrix.diag()/confusion_matrix.sum(1))
# print("Average accuracy: ",torch.mean(confusion_matrix.diag()/confusion_matrix.sum(1)))
print("per-class accuracy: ",[float("{:.2f}".format((t.item())*100)) for t in confusion_matrix.diag()/confusion_matrix.sum(1)]) # actually this is recall
avg_accuracy=(torch.mean(confusion_matrix.diag()/confusion_matrix.sum(1)).item())*100
print("Average accuracy: ","{:.2f}".format(avg_accuracy), "%")
# now we normalize the confusion matrix
confusion_matrix = (confusion_matrix/confusion_matrix.sum(axis=1))*100
                