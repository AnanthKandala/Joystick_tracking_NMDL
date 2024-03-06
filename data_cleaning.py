import scipy.io
import numpy as np
import os
import pandas as pd


def generate_clean_data(data_file):
    """data cleaning_steps:
            - calculate average cursor position using a rolling window and normalize it
            - Normalize the electrode activities w.r.t each electrode and divide it into windows
            - generate testing and training set using these snipped images 
    """ 
    data = scipy.io.loadmat(data_file)
    t0 = 5*1000
    frame = pd.DataFrame({key.replace('Pos', ''): np.squeeze(value)[t0:] for key, value in data.items() if 'Pos' in key})  #cursor postions
    window_size = 70; step = 5
    avg_frame = frame.rolling(window=window_size, step=step).mean().iloc[1 + window_size//step:,:]
    def normalize_array(column):
        return (column - column.min())/(column.max() - column.min())
    avg_frame = avg_frame.apply(normalize_array) #final cursor postions used for training/testing
    
    electrode_signal = data['data'].astype(np.float32)[t0:,:]
    def normalize_signal(electrode_signal):
        min_signal = np.min(electrode_signal, axis=0)
        max_signal = np.max(electrode_signal, axis=0)
        return -1 + 2*(electrode_signal - min_signal)/ (max_signal - min_signal)
    electrode_signal = normalize_signal(electrode_signal) #final electrode signal used for training/testing
    
    cleaned_data = []
    indices = avg_frame.index.to_list()
    for i in range(1, len(indices)):
        print(i, len(indices))
        index = indices[i]
        row = avg_frame.loc[index]
        current_position = row[['CursorX', 'CursorY']].to_numpy()
        previous_position = avg_frame.iloc[indices[i+1]][['CursorX', 'CursorY']].to_numpy()
        position = np.concatenate((current_position, future_position))
        activity = electrode_signal[index-window_size:index,:] #activity of electrodes in the window
        cleaned_data.append((activity, position))
    return cleaned_data
    
    
    
    
    
    