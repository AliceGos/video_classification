#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt 
import seaborn as sns
import random
from modules_dataprep import create_newidx
from sklearn.preprocessing import StandardScaler
path = '/home/I0259079/workdir/surf_dataset/'
from sklearn.utils import class_weight
 
#%% Dataset
class getdataset(Dataset):
    def __init__(self, df):
        self.df = df
        self.variables = list(self.df.columns[:6])	
        self.length = len(self.variables)

    def __len__(self):
        return len(self.df['video_newidx'].unique())

    def __getitem__(self, idx):
        data  = self.df[self.df['video_newidx']==idx]
        label = data['label'].unique()
        data  = data.loc[:,data.columns.isin(self.variables)]
        data = data.T
        return np.array(data).astype(np.float32), np.array(label)

#%% Model
class conv1dmodel(nn.Module):
    """Model for video classification."""
    def __init__(self, input_size, num_classes):
        super(conv1dmodel, self).__init__()
        self.conv1 = nn.Conv1d(input_size, 6, 10)
        self.conv2 = nn.Conv1d(6, 6, 5)

        self.drp = nn.Dropout(0.3)
    
        self.fc = nn.Linear(6, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.drp(x)
        x = F.relu(self.conv2(x))
        x = self.drp(x)
        x = x.mean(2)
        x = x.view(x.shape[0], 6)        
        x = self.fc(x)

        return x

#%% Train, validate and test model using cross validation
path = '/home/I0259079/workdir/surf_dataset/'
option = ''              # can be switched to '_all' if explore results on surf+longboard
train_formodel  = pd.read_csv(path+f'train{option}.csv') 
test_formodel   = pd.read_csv(path+f'test{option}.csv')  
y_test_formodel = pd.read_csv(path+f'y_test{option}.csv')
kfold = 4
# Keep accuracies accross K-fold cross validation
list_trainacc = []
list_valacc = []
list_testacc = []
# Keep predictions accross K-fold cross validation
list_trainpreds = []
list_valpreds = []
list_testpreds = []
# Keep loss accross K-fold cross validation
list_trainloss_allk = []
list_valloss_allk = []
# Keep true labels accross K-fold cross validation
list_ytrain = []
list_yval = []


listyes = list(train_formodel.loc[train_formodel['label'] == 1, 'video_id'].unique())
listno  = list(train_formodel.loc[train_formodel['label'] == 0, 'video_id'].unique())

for k in range(kfold):
    
    listval = random.sample(listyes,2)+random.sample(listno,1)

    train_set = train_formodel[~train_formodel['video_id'].isin(listval)]
    val_set   = train_formodel[train_formodel['video_id'].isin(listval)]

    df = train_set
    group = df.groupby('video_id')
    df2 = group.apply(lambda x: x['label'].unique())
    l1 = [list(x) for x in df2]
    y_train_formodel = [item for sublist in l1 for item in sublist]
    df = val_set
    group = df.groupby('video_id')
    df2 = group.apply(lambda x: x['label'].unique())
    l1 = [list(x) for x in df2]
    y_val_formodel = [item for sublist in l1 for item in sublist]

    create_newidx(train_set)
    create_newidx(val_set)

    train_dset      = getdataset(train_set)
    validation_dset = getdataset(val_set)

    train_loader      = DataLoader(train_dset, batch_size=1)
    validation_loader = DataLoader(validation_dset, batch_size=1)

    crossentropy_weights=torch.tensor([5,1],dtype=torch.float)
    #crossentropy_weights=class_weight.compute_class_weight(class_weight ='balanced',classes =[0,1],y=np.array(y_train_formodel))
    #crossentropy_weights=torch.tensor(crossentropy_weights,dtype=torch.float)
    criterion = torch.nn.CrossEntropyLoss(weight=crossentropy_weights, reduction='none')

    nepochs = 300

    model = conv1dmodel(input_size=6, num_classes=2)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)

    # # Print model's state_dict
    # print("Model's state_dict:")
    # for param_tensor in model.state_dict():
    # print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    # # Print optimizer's state_dict
    # print("Optimizer's state_dict:")
    # for var_name in optimizer.state_dict():
    # print(var_name, "\t", optimizer.state_dict()[var_name])

    trainloss  = []
    valloss    = []
    accval = []
    acctrain = []
    best_loss  = 1e8
    for epoch in range(nepochs):
        list_trainloss = []
        list_valloss = []
        list_trainacc_n = []
        list_valacc_n = []

        # training part
        model.train()
        for it, data in enumerate(train_loader):
            X,Y  = data
            Y    = Y.squeeze(1)
            out  = model(X) # shape: Batch_size, 2
            loss = criterion(out, Y)
            #print('y',Y,'out',out.squeeze(),loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            list_trainloss.append(loss.item())
            acc = np.where(torch.argmax(out,1)==Y,1,0)
            list_trainacc_n.append(acc)
        acctrain.append(np.mean(list_trainacc_n))
        trainloss.append(np.mean(list_trainloss))

        # validation part
        model.eval()
        for it, data in enumerate(validation_loader):
            X,Y = data
            Y   = Y.squeeze(1)
            out = model(X)
            loss = criterion(out, Y)
            #print('y',Y,'out',out.squeeze(),loss.item())
            list_valloss.append(loss.item())
            acc = np.where(torch.argmax(out,1)==Y,1,0)
            list_valacc_n.append(acc)
        accval.append(np.mean(list_valacc_n))
        mean_loss = np.mean(list_valloss)
        if mean_loss < best_loss:
            best_loss = mean_loss
            torch.save(model.state_dict(), path+f'best_model{option}_{k}.ckpt')
            
        valloss.append(np.mean(list_valloss))

        if epoch%10==0:
            print('train loss', np.mean(list_trainloss))
            print('val loss', np.mean(list_valloss))

    list_trainloss_allk.append(trainloss)
    list_valloss_allk.append(valloss)

    # Predict on train set        
    preds = []
    #model = conv1dmodel(input_size=6,num_classes=2)
    #model.load_state_dict(torch.load(path+f'best_model{option}_{k}.ckpt'))
    model.eval()

    for it, data in enumerate(train_loader):
        X,Y  = data
        pred = model(X)
        pred = torch.argmax(pred,1)    
        preds.append(pred)
    tp    = [x.tolist() for x in preds]
    preds = [item for sublist in tp for item in sublist]
    acc   = accuracy_score(y_train_formodel,preds)
    list_ytrain.append(y_train_formodel)
    list_trainacc.append(acc)
    list_trainpreds.append(preds)

    # Predict on validation set        
    preds = []
    for it, data in enumerate(validation_loader):
        X,Y  = data
        pred = model(X)
        pred = torch.argmax(pred,1)    
        preds.append(pred)
    tp    = [x.tolist() for x in preds]
    preds = [item for sublist in tp for item in sublist]
    acc   = accuracy_score(y_val_formodel,preds)
    list_yval.append(y_val_formodel)
    list_valacc.append(acc)
    list_valpreds.append(preds)

    # Predict on test set        
    test_dset   = getdataset(test_formodel)
    test_loader = DataLoader(test_dset, batch_size=1)
    preds = []
    for it, data in enumerate(test_loader):
        X,Y  = data
        pred = model(X)
        pred = torch.argmax(pred,1)    
        preds.append(pred)
    tp    = [x.tolist() for x in preds]
    preds = [item for sublist in tp for item in sublist]
    acc   = accuracy_score(y_test_formodel,preds)
    list_testacc.append(acc)
    list_testpreds.append(preds)

#%% Plot Losses and Confusion Matrices
for k in range(kfold):
    plt.figure()
    plt.plot(list_trainloss_allk[k],color='red', label='Train')
    plt.plot(list_valloss_allk[k],color='green', label='Validation')
    plt.legend()
    plt.xlabel('Epoch')
    plt.title(f'Train and Validation Loss - Fold {k}')
    plt.show()

    conf_matrix = confusion_matrix(list_ytrain[k],list_trainpreds[k])
    plt.figure()
    sns.heatmap(conf_matrix,annot=True,cmap='Blues',cbar=False)
    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title(f'Train Confusion Matrix- Fold {k}', fontsize=18)
    plt.show()

    conf_matrix = confusion_matrix(list_yval[k],list_valpreds[k])
    plt.figure()
    sns.heatmap(conf_matrix,annot=True,cmap='Blues',cbar=False)
    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title(f'Validation Confusion Matrix- Fold {k}', fontsize=18)
    plt.show()

    conf_matrix = confusion_matrix(y_test_formodel,list_testpreds[k])
    plt.figure()
    sns.heatmap(conf_matrix,annot=True,cmap='Blues',cbar=False)
    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title(f'Test Confusion Matrix- Fold {k}', fontsize=18)
    plt.show()

#%% Print accuracies
print(list_testacc)
