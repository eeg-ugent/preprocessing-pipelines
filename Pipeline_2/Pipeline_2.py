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

#%%############################################################################
"""import data"""
###############################################################################

# define your working directory:
working_dir = "C:/Users/raque/Dropbox/EEG_Course/EEGCourse 2021/EEGCourse/" #change this to match yours
P3_data_dir = working_dir + 'DataOddball/' 

# load the channel information for BioSemi files (.fif)
montage = mne.channels.make_standard_montage('biosemi64')

# Loads your BioSemi data file
raw = mne.io.read_raw_bdf(P3_data_dir + 'p2_oddball.bdf',preload=True)

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

# Event times stamped to the acquisition software can be extracted using mne.find_events():

# find events from the RAW dataset
events = mne.find_events(raw)

# The event_id is a dictionary in which you can name your conditions. tmin/tmax
# defines the length of the epoch relative to the event in seconds
epochs = mne.Epochs(interp_filt_raw, events, event_id = {'Standard':3, 'Oddball':4}, tmin=-0.2, tmax=1,
                     proj=False, baseline=(None, 0),
                     preload=True, reject=None)

# Check how many epochs of each type were created
np.count_nonzero(epochs.events[:,2] == 3)
np.count_nonzero(epochs.events[:,2] == 4)

#%%############################################################################
"""re-reference to the average"""
###############################################################################

# Plot ERPs (over all trials) for each electrode before re-referencing and make a screenshot
evoked_pre_average = epochs.average()
evoked_pre_average.plot(window_title = 'original reference')

# rereference to average (average reference makes all the ICA scalp
# topography have zero total potential (i.e. red and blue always balances))
epochs.set_eeg_reference().apply_proj().average()

# ------>>>>>>  ASSIGNMENT Nr 11:
# Write 2 lines of code to plot epochs after re-referencing and make a screenshot
evoked_reref_to_average = epochs.average()
evoked_reref_to_average.plot(window_title = 'rereferenced')
# Compare the two plots, what do you notice?

#%%############################################################################
"""visual artifact rejection"""
###############################################################################

# ------>>>>>>  ASSIGNMENT Nr 12:
# Plot the epochs to do visual trial rejection.
epochs.plot(n_epochs = 5, n_channels = 64)
# - Click on an epoch to reject it from the dataset
# - Use keyboard shortcuts to adapt the window size (HELP to see keyboard controls)
# - First scroll through the entire data set to get a feel for it
# - Adjust the scale to a comfortable level 
# - Then start at the beginning and click on epochs you think should be rejected
# - Don't reject trials with eye blinks or eye movements; we will get these with ICA
# - Close the figure for the selected epochs to be rejected


# ------>>>>>>  ASSIGNMENT Nr 13:
# Use the variable explorer to write one line of code to answer the following question:
# How many epochs of each condition do you have left? 
epochs.drop
   
    
#%%############################################################################
"""Independent Components Analysis"""
###############################################################################

# Now we prepare to run ICA
# We need to know how many ICA components we want
# This is the same amount as the number of UNIQUE channels
# We have to take into account that an interpolated channel does not have any unique
# information, since it is made up of information from the surrounding channels
# We also need to subtract 1 channel, because we are using an average reference
# because this leads to all the channels having in common an amount of
# information equal to 1/number of channels
# So, the correct number of components to get out of the ICA =
# Original number of channels - number of interpolated channels - 1

# ------>>>>>>  ASSIGNMENT Nr 14:
# write a line of code to find the correct number of components
# to get from the ICA
ncomp = len(epochs.pick_types(eeg = True).ch_names) - len(epochs.info['bads']) - 1
print(ncomp)

# create ICA object with desired parameters
ica = mne.preprocessing.ICA(n_components = ncomp)

# do ICA decomposition
ica.fit(epochs) 

# Plot the components
# clicking on a component will mark it as bad 
ica.plot_components()

# Plot the properties of a single component (e.g. to check its frequency profile)
ica.plot_properties(epochs, picks=7)

#Look at the timecourse of the component
ica.plot_sources(raw)

# Decide which component(s) to "project out of the data"
# Click on their name; this will turn them grey
# Remember; the component will not ONLY represent artifactual activity
# There will also be some brain activity mixed into it; ICA is not perfect
# This is why you want to be very conservative and not project just any
# component out of your data. Once you selected the component(s), move on to 
# the next assignment:

# ------>>>>>>  ASSIGNMENT Nr 15:
# Make a screenshot of the following plot and describe what it represents
ica.plot_overlay(evoked_reref_to_average)

#%%############################################################################
"""Apply ICA weights to the data"""
###############################################################################

# Now we take the ICA weights and apply them to the data
clean_data = epochs.copy() #create a copy of the raw data and name it ica_raw
ica.apply(clean_data) #Apply the weights of the ICA to the copy of the raw data

# Save the clean data
clean_data.save(P3_data_dir + 'clean_data-epo.fif')



