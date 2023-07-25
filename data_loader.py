import scipy.io
import numpy as np



def create_images(data_file, im_len, stride_len):
    '''takes the matlab file for a patient and creates images of the ecog data for training/testing. We have a 2D array made up of 
    electrode voltage vs time. This single, long signal needs to chopped up into smaller peices to create the desired dataset of 'images'
    args:
        data_file: matlab file for a patient
        i_len (int): length of each image (in ms)
        stride_len (int): length of stride (in ms)
    returns: 
        image_data (list): list of (image, label) pairs. image (2D array) -->ecog voltages vs time, label (2D array) --> joystick coordinates vs time '''
    data = scipy.io.loadmat(data_file)
    array = data['data']
    indices = [(i, i+im_len) for i in list(range(0, array.shape[0] - im_len +1, stride_len))]
    cut_signals = [array[i[0]:i[1], :] for i in indices]
    assert len(data['CursorPosX']) == len(data['CursorPosY'])
    t_sample = [i+im_len-1 for i in list(range(0, len(data['CursorPosX']) - im_len +1, stride_len))]
    trajectory = [(data['CursorPosX'][i,0], data['CursorPosY'][i,0]) for i in t_sample]
    return [(cut_signals[i], trajectory[i]) for i in range(len(cut_signals))]
