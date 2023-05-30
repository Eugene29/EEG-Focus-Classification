import pickle, os
from TNT_eeg_func import extract_data_eeg_xdf, eeg_vis, psd_vis

with open('model.pkl', 'rb') as f:
    clf_loaded = pickle.load(f)

test_path = "C:/Users/eugen/sandbox/bio_feedback_app/Test files" # default unseen data. 
# demo_path = "C:/Users/eugen/sandbox/bio_feedback_app/Test files/Demo data" # new data you want to experiment

dir_list_zip = (test_path, os.listdir(test_path))
data = extract_data_eeg_xdf(dir_list_zip) # This will gather all of files named "relaxed\d*.xdf$". Vice versa with Focused.
pred = clf_loaded.predict(data) # event_dic: {0:focused, 1:relaxed} Each value represents focused/relaxed at 1 second.
pred_w_time = {k:v for k, v in enumerate(pred)} 

# eeg_vis(dir_list_zip) # to visualize eeg data. ** it is not scaled. So some channels are hard to see. 
# psd_vis(dir_list_zip) # to visualize psd data. Think of it as feature extraction. 
# print(pred_w_time) # if you want time indicator using dictionary
print(pred) 

# This combines the files and show that the predictions are valid. Here, the test data we use are completely unseen.
# So, when demoing, use only file in per pathway. 