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

def psd_vis(dir_list_zip):
    path = dir_list_zip[0]
    dir_list = dir_list_zip[1]
    files = []
    for file in dir_list:
        if re.search(f".xdf$", file):
            files.append(file) # add a tuple of filepath and boolean of event
    for file in files:
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
        raw.plot_psd()

def eeg_vis(dir_list_zip):
    path = dir_list_zip[0]
    dir_list = dir_list_zip[1]
    files = []
    for file in dir_list:
        if re.search(f".xdf$", file):
            files.append(file) # add a tuple of filepath and boolean of event
    for file in files:
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
        raw.plot()

def extract_data_eeg_xdf(dir_list_zip):
    path = dir_list_zip[0]
    dir_list = dir_list_zip[1]
    files = []
    psd_li = np.array([])
    for file in dir_list:
        if re.search(f".xdf$", file):
            files.append(file) # add a tuple of filepath and boolean of event
    
    for file in files:
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
        # Create epochs        
        events = [[x * sfreq, 0, 0] for x in range(raw.n_times // int(sfreq))]
        event_dict = {'unknown': 0}
        tmin = 0
        tmax = 0.5 
        epochs = mne.Epochs(
            raw, events, event_dict, tmin, tmax, baseline=None
        )
        # Compute power spectral density using Welch's method
        fmin, fmax = 0, 50
        n_fft = 2048
        spectrum = epochs.compute_psd(
            "welch",
            n_fft=n_fft,
            n_overlap=int(tmax * sfreq)-1,
            n_per_seg = int(tmax * sfreq),
            tmin=tmin,
            tmax=tmax,
            fmin=fmin,
            fmax=fmax,
            window="boxcar",
        )   
        psd, freqs = spectrum.get_data(return_freqs=True)
        psd = psd.reshape(len(psd), -1)
        psd = psd * 1e16 #reshape so you can compute
        if len(psd_li) == 0:
            psd_li = psd
        else:
            psd_li = np.vstack((psd_li, psd))
    return psd_li

### This one is to build the model. Not used for real data with unknown labels
def files_to_mne_eeg(dir_list_zip):
    path = dir_list_zip[0]
    dir_list = dir_list_zip[1]
    files = []
    psd_li = np.array([])
    for file in dir_list:
        if re.search(f"relaxed\d*.xdf$", file):
            files.append((file, "relaxed")) # add a tuple of filepath and boolean of event
        elif re.search(f"focused\d*.xdf$", file):
            files.append((file, "focused")) # add a tuple of filepath and boolean of event

    for file, event_id in files: # event_id is used later to identify stimulus type. Here, it is focused:0, relaxed:1
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

        # Create epochs        
        if event_id == "focused":
            # events = np.array([[0, 0, 1]])
            events = [[x * sfreq, 0, 1] for x in range(raw.n_times // int(sfreq))]
            event_dict = {'focused': 1}
        else:
            events = [[x * sfreq, 0, 2] for x in range(raw.n_times // int(sfreq))]
            event_dict = {'relaxed': 2}
        tmin = 0
        tmax = 0.5 
        epochs = mne.Epochs(
            raw, events, event_dict, tmin, tmax, baseline=None
        )
        # Compute power spectral density using Welch's method
        fmin, fmax = 0, 50
        n_fft = 2048
        spectrum = epochs.compute_psd(
            "welch",
            n_fft=n_fft,
            n_overlap=int(tmax * sfreq)-1,
            n_per_seg = int(tmax * sfreq),
            tmin=tmin,
            tmax=tmax,
            fmin=fmin,
            fmax=fmax,
            window="boxcar",
        )   
        psd, freqs = spectrum.get_data(return_freqs=True)
        psd = psd.reshape(len(psd), -1)
        psd = psd * 1e16 #reshape so you can compute
        
        add_label = np.full(len(psd), event_dict[event_id]-1).reshape(-1, 1)
        psd = np.hstack((psd, add_label)) # has ~59, 2005 shape. 2006 with labels
        if len(psd_li) == 0:
            psd_li = psd
        else:
            psd_li = np.vstack((psd_li, psd))
    return psd_li
if __name__ == "__main__":
    train_path = "C:/Users/eugen/sandbox/bio_feedback_app" # add your own file path
    test_path = "C:/Users/eugen/sandbox/bio_feedback_app/Test files"

    dir_list_train = (train_path, os.listdir(train_path))
    dir_list_test = (test_path, os.listdir(test_path))

    from sklearn.model_selection import StratifiedKFold, cross_val_score
    data = files_to_mne_eeg(dir_list_train)
    X, y = data[:, :-1], data[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=30)

    ##### Model Experiments #####
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=25)

    ### SVM
    from sklearn.svm import SVC
    svm = SVC(kernel='rbf')
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    train_acc = accuracy_score(y_train, svm.predict(X_train))
    accuracy = accuracy_score(y_test, y_pred)
    print("Train_acc", train_acc)
    print("Test_acc:", accuracy)
    svm_score = cross_val_score(svm, X, y, cv=cv).mean()

    ### Random Forest
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    y_train_pred = rf.predict(X_train)
    y_pred = rf.predict(X_test)
    train_acc = accuracy_score(y_train, y_train_pred)
    accuracy = accuracy_score(y_test, y_pred)
    print("Train_acc", train_acc)
    print("Test_acc:", accuracy)
    rf_score = cross_val_score(rf, X, y, cv=cv).mean()

    ### GradientBoosting Classifier
    # from sklearn.ensemble import GradientBoostingClassifier
    # gb = GradientBoostingClassifier()
    # gb.fit(X_train, y_train)
    # y_pred = gb.predict(X_test)
    # train_acc = accuracy_score(y_train, gb.predict(X_train))
    # accuracy = accuracy_score(y_test, y_pred)
    # print("Train_acc", train_acc)
    # print("Test_acc:", accuracy)
    # gb_score = cross_val_score(gb, X, y, cv=cv).mean()

    ### Logistic Regression
    # from sklearn.linear_model import LogisticRegression
    # lr = LogisticRegression()
    # lr.fit(X_train, y_train)
    # train_acc = accuracy_score(y_train, lr.predict(X_train))
    # accuracy = accuracy_score(y_test, lr.predict(X_test))
    # print("Train_acc", train_acc)
    # print("Test_acc:", accuracy)
    # lr_score = cross_val_score(lr, X, y, cv=cv).mean() # Doesn't work because it's not scaled?

    ### XGB model
    import xgboost as xgb
    xgb_ = xgb.XGBClassifier(objective="binary:logistic", random_state=25) # best! does same as reg:logistic
    xgb_.fit(X_train, y_train)
    train_pred = xgb_.predict(X_train)
    y_pred = xgb_.predict(X_test)
    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, y_pred)
    print("Train_acc", train_acc)
    print("Test_acc:", test_acc)
    xgb_score = cross_val_score(xgb_, X, y, cv=cv).mean()

    ##### Model Selection & Final Evaluation #####
    print("cv scores of each model")
    print(f"xgb score: {xgb_score}")
    print(f"rf score: {rf_score}")
    print(f"svm_score: {svm_score}")

    ## Chose final model: Random Forest. Train the whole model again
    ## Here, RF is chosen over xgb (close competitor) because it is more stable (higher k-fold cv accuracy). 
    ## One thing to note, when checking final model evaluation, xgb however does a slightly better job. 

    eval_ = files_to_mne_eeg(dir_list_test)
    X_eval, y_eval = eval_[:, :-1], eval_[:, -1]

    # xgb_.fit(X,y)
    # xgb_eval_pred = xgb_.predict(X_eval)
    # print(accuracy_score(y_eval, xgb_eval_pred))
    rf.fit(X, y)
    y_eval_pred = rf.predict(X_eval)
    print(accuracy_score(y_eval, y_eval_pred))

    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import classification_report
    from sklearn.metrics import roc_auc_score

    print(confusion_matrix(y_eval, y_eval_pred))
    print(classification_report(y_eval, y_eval_pred))
    roc_auc = roc_auc_score(y_eval, y_eval_pred)
    print("ROC AUC: %.2f" % roc_auc)


    ##### saving our final model #####
    ## Only uncomment when you want to over-write a new model.
    # import pickle
    # with open('model.pkl', 'wb') as f:
    #     pickle.dump(rf, f)