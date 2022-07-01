"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%  PRE-PROCESSING PIPELINE 2 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% This pipeline is a basic pre-processing pipeline for ERPs 
% as used in the introductory EEG analysis course for master students (hence
% the excessive comments) 
% You can try it out with data from an oddball task, which is also provided. 

"""

###############################################################################
""" housekeeping"""
###############################################################################

#To be able to manipulate the figures
#Change tools/preferences/ipython console/graphics/backend to qt5
#If plots still dont appear as independent windows, restart Python

#Import the necessary Python modules:
import numpy as np
import mne
import matplotlib.pyplot as plt
import os

#%%############################################################################
"""import data"""
###############################################################################

# define your working directory:
working_dir = "C:/Users/raque/Dropbox/EEG_Course/EEGCourse 2021/EEGCourse/" #change this to match yours
P3_data_dir = working_dir + 'DataOddball/' 

# load the channel information for BioSemi files (.fif)
montage = mne.channels.make_standard_montage('biosemi64')

# Loads your BioSemi data file
raw = mne.io.read_raw_fif(P3_data_dir + 'BioSemi_P3_Sub14_raw.fif',preload=True)

#%%############################################################################
"""filter the data"""
###############################################################################

# filter the data
filt_raw = raw.copy() #create a copy of the raw data and name it "filt_raw"
filt_raw.load_data().filter(l_freq=0.1, h_freq=40) #apply the bandpass filter

# notice that now we have two datasets, the raw set and the filtered set.
# this allows us to go back to the raw data if we want to change anything in the analysis.
# it also helps us keep track of all the transformations that have been applied to the data.

# Check if the filters have been correctly applied to the data by
# visual inspection and by checking the output in the console
filt_raw.plot_psd() #frequency domain
filt_raw.plot() #time domain

#%%############################################################################
"""remove and interpolate bad channels"""
###############################################################################

# Bad channels are marked by adding them to the list of 'bads' within the info object.
# scroll through your data (plot filtered data in the time domain) and identify whether there
# are any channels that look VERY bad. 
filt_raw.plot()

# To mark a channel as bad, simply click on it, and you will see it turn grey.
# Once you are finished, close the figure and check the list of bads:
print(filt_raw.info['bads'])

# Interpolate the bad channels
interp_filt_raw = filt_raw.copy()
interp_filt_raw.load_data().interpolate_bads(reset_bads=False)

#%%############################################################################
"""epoch data"""
###############################################################################

# find events from the RAW dataset '3' marks the onset of the standard and '4' 
# marks the onset of an oddball
events = mne.find_events(raw)

# The event_id is a dictionary in which you can name your conditions. tmin/tmax
# defines the length of the epoch relative to the event in seconds
# no baseline correction is done at this point to avoid problems for ICA
epochs = mne.Epochs(interp_filt_raw, events, event_id = {'Standard':3, 'Oddball':4}, tmin=-0.2, tmax=1,
                     proj=False, baseline=(None),
                     preload=True, reject=None)

# Check how many epochs of each type were created
np.count_nonzero(epochs.events[:,2] == 3)
np.count_nonzero(epochs.events[:,2] == 4)

#%%############################################################################
"""re-reference to the average"""
###############################################################################

# rereference to average (average reference makes all the ICA scalp
# topographies have zero total potential (i.e. red and blue always balances))
epochs.set_eeg_reference().apply_proj().average()

#%%############################################################################
"""visual artifact rejection"""
###############################################################################

# Plot the epochs to do visual trial rejection.
epochs.plot(n_epochs = 5, n_channels = 64)
# - Click on an epoch to reject it from the dataset
# - Use keyboard shortcuts to adapt the window size (HELP to see keyboard controls)
# - Close the figure for the selected epochs to be rejected

epochs.drop #this shows you the number of remaining epochs in each condition
   
    
#%%############################################################################
"""Independent Components Analysis"""
###############################################################################

# To run the ICA we need to know the rank of the data (how many unique channels)
# The rank of the data defines the number of components we can get from the ICA.
# Interpolated channels do not have any unique information and we also need to 
# subtract 1 channel for the average reference:
ncomp = len(epochs.pick_types(eeg = True).ch_names) - len(epochs.info['bads']) - 1

# create ICA object with desired parameters
ica = mne.preprocessing.ICA(n_components = ncomp)

# do ICA decomposition
ica.fit(epochs) 

# Plot the components
# clicking on a component will mark it as bad 
ica.plot_components()

# Plot the properties of a single component (e.g. to check its frequency profile)
ica.plot_properties(epochs, picks=11)

#Look at the timecourse of the component
ica.plot_sources(raw)

# Decide which component(s) to "project out of the data"
# Click on their name; this will turn them grey

clean_data = epochs.copy() #create a copy of the raw data and name it ica_raw
ica.apply(clean_data) #Apply the weights of the ICA to the copy of the raw data
clean_data.plot() #check your data 

# Save the clean data
clean_data.save(P3_data_dir + 'clean_data-epo.fif')



