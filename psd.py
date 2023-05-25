import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mne 
from mne.datasets import misc
import pyxdf
import os, re
import matplotlib.pyplot as plt

import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load XDF file
# xdf_data, xdf_header = pyxdf.load_xdf("C:/Users/eugen/sandbox/bio_feedback_app/firstfocused.xdf")

def files_to_mne_eeg(dir_list_zip):
    path = dir_list_zip[0]
    dir_list = dir_list_zip[1]
    files = []
    psd_li = np.array([])
    count = 0
    for file in dir_list:
        if re.search(f"relaxed\d*.xdf$", file):
            files.append((file, "relaxed")) # add a tuple of filepath and boolean of event
        elif re.search(f"focused\d*.xdf$", file):
            files.append((file, "focused")) # add a tuple of filepath and boolean of event

    for file, event_id in files: # event_id is used later to identify stimulus type. Here, it is relaxed:0, focus:1
        filepath = path + "/" + file
        fname = (misc.data_path() / 'xdf' /
            filepath)
        streams, header = pyxdf.load_xdf(fname)
        data = streams[0]["time_series"].T
        assert data.shape[0] == 5
        data[:6] *= (1e-6 / 50 / 2)  # uV -> V and preamp gain
        sfreq = int(float(streams[0]["info"]["nominal_srate"][0]))
        info = mne.create_info(5, sfreq, ["eeg", "eeg", "eeg", "eeg", "eeg"])
        raw = mne.io.RawArray(data, info)
        # raw.filter(0, 50)

        # Create epochs        
        if event_id == "focused":
            # events = np.array([[0, 0, 1]])
            events = [[x * sfreq, 0, 1] for x in range(raw.n_times // int(sfreq))]
            event_dict = {'focused': 1}
        else:
            events = [[x * sfreq, 0, 2] for x in range(raw.n_times // int(sfreq))]
            event_dict = {'relaxed': 2}
        # if event_id == "focused":
        #     # events = np.array([[0, 0, 1]])
        #     events = [[2 * x, 0, 1] for x in range(raw.n_times // 2)]
        #     event_dict = {'focused': 1}
        # else:
        #     events = [[2 * x, 0, 2] for x in range(raw.n_times // 2)]
        #     event_dict = {'relaxed': 2}

        tmin = 0
        tmax = 2 # 5*sfreq #the whole duration of the data?
        epoch_dur = 10 # 5*sfreq
        # tmin = -0.5
        # tmax = 5 * sfreq
        # epoch_dur = 5 * sfreq
        # events = mne.make_fixed_length_events(raw, duration=epoch_dur, overlap=epoch_dur*0.5)
        # np.hstack((events, np.zeros(len(events)))) 
        # event_dict = {'focused': 1}
        epochs = mne.Epochs(
            raw, events, event_dict, tmin, tmax, baseline=None #, reject=None #preload=True
        )
        # Compute power spectral density using Welch's method
        fmin, fmax = 0, 50
        n_fft = 2048
        spectrum = epochs.compute_psd(
            "welch",
            n_fft=n_fft,
            n_overlap=2,
            n_per_seg = int(tmax * sfreq),
            tmin=tmin,
            tmax=tmax,
            fmin=fmin,
            fmax=fmax,
            window="boxcar",
        )   
        psd, freqs = spectrum.get_data(return_freqs=True)
        psd = psd.reshape(len(psd), -1)
        psd = psd * 1e15 #reshape so you can compute
        
        # scaler = StandardScaler()
        # X_scaled = scaler.fit_transform(psd)
        # pca = PCA(n_components=59)
        # psd = pca.fit_transform(psd)

        add_label = np.full(len(psd), event_dict[event_id]-1).reshape(-1, 1)
        psd = np.hstack((psd, add_label)) # has 59, 2005 shape 2006 with labels
        if len(psd_li) == 0:
            psd_li = psd
            # freqs_li = freqs
        else:
            psd_li = np.vstack((psd_li, psd))
    return psd_li

train_path = "C:/Users/eugen/sandbox/bio_feedback_app" # add your own file path
test_path = "C:/Users/eugen/sandbox/bio_feedback_app/Test files"

dir_list_train = (train_path, os.listdir(train_path))
dir_list_test = (test_path, os.listdir(test_path))
files_to_mne_eeg(dir_list_train)
from sklearn.model_selection import StratifiedKFold
X = files_to_mne_eeg(dir_list_train)
# try one nearest neighbor?
# final_model_eval = files_to_mne_eeg(dir_list_test) # 
X_train, X_test, y_train, y_test = train_test_split(X[:, :-1], X[:, -1], test_size=0.25, random_state=30)
ii8
print("Testing final Eval")
print("----------------!")

eval_ = files_to_mne_eeg(dir_list_test)
X_test = eval_[:, :-1]
y_test = eval_[:, -1]


# X[:, :-1]

# pd.Series(X_train[0]).value_counts()
### startified kfold to implement later
# for train_idx, test_idx in stratified_kfold.split(X, y):
#     X_train, X_test = X[train_index], X[test_index]
#     y_train, y_test = y[train_index], y[test_index]
# np.std(X_train, axis=0).max()
### SVM
# from sklearn.svm import SVC

# svm = SVC(kernel='rbf')
# svm.fit(X_train, y_train)
# y_pred = svm.predict(X_test)
# train_acc = accuracy_score(y_train, svm.predict(X_train))
# accuracy = accuracy_score(y_test, y_pred)
# print("Train_acc", train_acc)
# print("Test_acc:", accuracy)

# X_train.shape
# Define the network architecture
# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         # Define the layers
#         self.fc1 = nn.Linear(2005, 328)  # input layer
#         self.fc2 = nn.Linear(328, 64)  # hidden layer
#         self.fc3 = nn.Linear(64, 2)  # output layer

#     def forward(self, x):
#         # Pass the input tensor through each of our operations
#         x = self.fc1(x)
#         x = torch.relu(x)
#         x = self.fc2(x)
#         x = torch.relu(x)
#         x = self.fc3(x)
#         # x = torch.log_softmax(x, dim=1)
#         return x

# # Instantiate the network
# model = Net()
# print(model)

# # X_train = torch.from_numpy(X_train).float()
# # y_train = torch.from_numpy(y_train).long()
# # X_test = torch.from_numpy(X_test).float()
# # y_test = torch.from_numpy(y_test).long()

# # Create data loaders - they help with shuffling and batching the data
# train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=1, shuffle=True)
# test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=1)

# # Instantiate the model
# model = Net()

# # Define loss function and optimizer
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.01)

# # Training loop
# for epoch in range(10):  # loop over the dataset multiple times
#     for inputs, labels in train_loader:
#         # zero the parameter gradients
#         optimizer.zero_grad()

#         # forward + backward + optimize
#         outputs = model(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()

# # Testing loop
# correct = 0
# total = 0
# with torch.no_grad():
#     for inputs, labels in test_loader:
#         outputs = model(inputs)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()

# print('Accuracy of the network on the test data: %d %%' % (100 * correct / total))

# print(y_train.unique())
# print(y_test.unique())

# from collections import Counter
# Counter(y_train)
### Random Forest (dear, Leo Breiman)
# from sklearn.ensemble import RandomForestClassifier

# X_train.shape
# np.unique(X_train).size
# X_train.size
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
train_acc = accuracy_score(y_train, rf.predict(X_train))
accuracy = accuracy_score(y_test, y_pred)
print("Train_acc", train_acc)
print("Test_acc:", accuracy)


### GradientBoosting Classifier
from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier()
gb.fit(X_train, y_train)
y_pred = gb.predict(X_test)
train_acc = accuracy_score(y_train, gb.predict(X_train))
accuracy = accuracy_score(y_test, y_pred)
print("Train_acc", train_acc)
print("Test_acc:", accuracy)

### Logistic Regression
# from sklearn.linear_model import LogisticRegression

# lr = LogisticRegression()
# lr.fit(X_train, y_train)
# train_acc = accuracy_score(y_train, lr.predict(X_train))
# accuracy = accuracy_score(y_test, lr.predict(X_test))
# print("Train_acc", train_acc)
# print("Test_acc:", accuracy)
# lr.predict(X_test)

import xgboost as xgb
# xgb_model = xgb.XGBClassifier(objective="reg:linear", random_state=42)
# xgb_model = xgb.XGBClassifier(objective="reg:squaredlogerror", random_state=42)
# xgb_model = xgb.XGBClassifier(objective="reg:squarederror", random_state=42)
# xgb_model = xgb.XGBClassifier(objective="reg:logistic", random_state=42) # best!
# xgb_model = xgb.XGBClassifier(objective="binary:logistic", early_stopping_rounds=10, reg_alpha=1, random_state=42) # best! does same as reg:logistic
# xgb_model = xgb.XGBClassifier(objective="binary:hinge", random_state=42)
# xgb_model = xgb.XGBClassifier(objective="reg:tweedie", random_state=42)
# xgb_model = xgb.XGBClassifier(objective="binary:logitraw", random_state=42)
# xgb_model = xgb.XGBClassifier(objective="binary:logitraw", random_state=42)

# xgb_model = xgb.XGBClassifier(objective="binary:logistic", max_depth=2, random_state=42) # best! does same as reg:logistic
xgb_model = xgb.XGBClassifier(objective="binary:logistic", random_state=42) # best! does same as reg:logistic

xgb_model.fit(X_train, y_train)
train_pred = xgb_model.predict(X_train)
train_acc = (y_train == train_pred).mean()
y_pred = xgb_model.predict(X_test)
test_acc = (y_pred == y_test).mean()
print("Train_acc", train_acc)
print("Test_acc:", test_acc)

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
roc_auc = roc_auc_score(y_test, y_pred)
print("ROC AUC: %.2f" % roc_auc)


