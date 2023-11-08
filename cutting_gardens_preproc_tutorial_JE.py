# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 13:36:04 2023

@author: jeayrs

Preprocessing with MNE python tutorial

Ghent Cutting gardens, 16/10/2023

"""
###############################################################################
""" housekeeping"""
###############################################################################
# In order to be able to manipulate the figures
# Change tools/preferences/ipython console/graphics/backend to qt5
# If plots still don't appear as independent windows, restart Python

# Import some libraries
import os
import numpy as np
import mne

#%%############################################################################
"""import data"""
###############################################################################

# define the file directory where you saved the data
working_dir = "D:/cutting_gardens/data/" 
file = "practice_dataset.bdf" # the file we're going to use
fpath = os.path.join(working_dir, file)

# Load the 'raw' dataset (i.e. fresh from the lab) and load it into memory
# (necessary in order to manipulate the data but not to look at basic info)
raw = mne.io.read_raw_bdf(fpath, preload = True)

#%%############################################################################
"""explore data"""
###############################################################################
# the 'raw' object has various attributes which we can already use to learn
# about the data. for example, raw.info is a dictionary containing some of the
# basic information about the data.
isinstance(raw.info, dict)

print(raw)
print(raw.info)

# The fields in .info are Python dictionary keys, indexed using square brackets
# and strings to access the contents. For example, to access the channel names:
print(raw.info['ch_names'])
print('there are {} channels'.format(len(raw.info['ch_names']))) # of channels
# the sampling frequency
print(raw.info['sfreq'])
# a list of channels labelled as 'bad' (should be empty for now)
print(raw.info['bads'])

# to speed things up, we can downsample the data
raw.resample(400)
print(raw.info['sfreq']) # check

# We can see from raw.info that there are 73 channels, one for triggers
# ('Stimulus') and 72 are apparently 'EEG' - but we used a 64 channel cap...
# We can see from raw.info['ch_names'] that most channels have sensible names
# (Fp1, AF7...) but the last few are still labelled EXG1, EXG2 etc. - if you're 
# familiar with biosemi this should make sense - these are the extra electrodes
# for EOG and references etc, we'll rename them to something more intuitive
fix_chans = {'EXG1':'Mastoid1','EXG2':'Mastoid2',
             'EXG3':'eye_right','EXG4':'eye_left',
             'EXG5':'eye_above','EXG6':'eye_below'}
raw.rename_channels(fix_chans)

# we still have two exg channels which weren't actually recorded though (EXG7
# and EXG8) these are empty, so we'll drop them
raw.drop_channels(['EXG7', 'EXG8'])
print(raw.info['ch_names'])

# we'll also reset the channel types, so MNE knows what is 'brain' data
raw.set_channel_types({#'Mastoid1':'eeg', 'Mastoid2':'eeg',
                       'eye_above':'eog', 'eye_below':'eog',
                       'eye_left':'eog', 'eye_right': 'eog'})
print(raw.info)

# The raw object also has various methods, which we can use for example to 
# visually explore the data in a few ways:
# Plot the data in the time domain. Assuming you have enabled interactive
# plotting (top of script) then you should be able to scroll around through
# time etc. If you click on a channel then it will be marked as 'bad' and turn 
# grey.
raw.plot()

# take a moment to explore the data and the plotting tool. You can click the 
# HELP icon to open a list of keyboard shortcuts. It's useful to adjust the 
# scale so that the data doesn't overlap itself too much and to adjust the 
# number of channels so that you can see them all on one screen (or maybe half
# depending on the size of your screen and quailty of your eyesight).

# Plot the data in the frequency domain
raw.plot_psd()

# if we want, we can extract the data as an np.array (maybe to use in some
# other software)
raw_data = raw.get_data()
# inspect it a bit
raw_data.view()
raw_data.shape

# or, a bit neater, in a dataframe (if you have the pandas library)
#raw_df = raw.to_data_frame()
#raw_df.head()

#%%############################################################################
"""re-reference the data"""
###############################################################################
# our data still has it's original default referencing scheme. We can easily
# change that with the set_eeg_reference method

# rereference to average of the mastoids
raw.set_eeg_reference(ref_channels = ['Mastoid1', 'Mastoid2'])
raw.plot(n_channels = 64)

# rereference to average of all electrodes (default)
raw.set_eeg_reference(ref_channels = "average")

# take another look
raw.plot(n_channels = 64)

## I'll drop the mastoids because I'm not using them
raw.drop_channels(['Mastoid1', 'Mastoid2'])

#%%############################################################################
"""Make annotations for data rejection"""
###############################################################################
# we can annotate bad spans of raw data in the interactive plot by pressing the
# 'a' key, useful if we know there is a period we aren't interested in such
# as a mis-start of the task etc.
raw_annot = raw.copy()
raw_annot.plot(n_channels = 64)

# we can also identify eog events algorithmically via "find_eog_events" this
# produces a list of 'events' around each blink (hopefully). This applies a 
# filter and then identifies peaks in the eog to find likely blinks. We can 
# adjust the threshold, via thresh. but default should be okay for now.
eog_events = mne.preprocessing.find_eog_events(raw_annot)
# we'll say that the blinks start a tiny bit earlier than 
onsets = eog_events[:, 0] / raw_annot.info["sfreq"] - 0.25
# we'll assume they're all half a second long
dur = [0.5] * len(eog_events)
descriptions = ["blink"] * len(eog_events)
blink_annot = mne.Annotations(onsets,
                              dur,
                              descriptions,
                              orig_time = raw_annot.info["meas_date"])
raw_annot.set_annotations(blink_annot)

# let's take a look at what has been detected
raw_annot.plot(n_channels = 64)

#%%############################################################################
"""filter the data"""
###############################################################################
# To remove slow drifts, we'll apply a filter to the raw data. To do this we'll
# first make a copy of 'raw' called 'filt_raw' and apply a bandpass filter to
# that with the .filter method. This applies a finite impulse response function
# (FIR) filter with the lower and upper frequency bands that we set with l_freq
# and h_freq respectively.
filt_raw = raw.copy() 
filt_raw.load_data().filter(l_freq = 1, h_freq = 40)

# Check what effect the filters have had on the data
filt_raw.plot_psd() # frequency domain
filt_raw.plot(n_channels = 64) # time domain

#%%############################################################################
"""remove and interpolate bad channels"""
###############################################################################
# Earlier, we saw that raw had an attribute 'bads' which was an empty list
# 'bads' lists the channels which have been marked as 'bad' either by manual 
# choice (during plotting) or by some algorithm. You can check this by plotting
# the data, clicking on a channel to mark it as 'bad' and printing 'bads' again
filt_raw.plot(n_channels = 64)
print(filt_raw.info['bads'])

# a butterfly plot overlays all of the channels, here we highlight the 'bad'
# channels by presenting them in a different colour. This can be helpful to 
# see how much that channel truly stands out.
fig = filt_raw.plot(butterfly = True,
                    color = '#00000022',
                    bad_color = 'r',
                    title = 'bads_marked')


# in order to interpolate data we need to include the channel locations in a 
# montage. Our data was recorded on a biosemi 64-channel system. MNE has built- 
# in electrode montages for this and various other systems, we'll load it here
montage = mne.channels.make_standard_montage('biosemi64')
# Check the montage in 3d
montage.plot(kind = '3d')

# apply the montage to the data
filt_raw.set_montage(montage)
# plot sensor locations in 2d view
# 'bad' channels will be red, click individual dots to see the label
filt_raw.plot_sensors()

# Now we'll make another copy of the (filtered) data and interpolate the bad 
# channels using the 'interpolate_bads' method. We'll specify 'reset_bads = 
# False' so that we keep a record of the number of interpolated channels
# (useful for ICA later)
interp_filt_raw = filt_raw.copy()
interp_filt_raw.load_data().interpolate_bads(reset_bads = False)
print(interp_filt_raw.info['bads']) #check that the 'bads' are still listed


#%%############################################################################
"""epoch data"""
###############################################################################
# previously, we made a list of 'events' which were based around the 
# algorithmically detected blinks. We used those events to annotate the data
# where there was a blink so that it could be removed from the data. Here,
# we'll create a similar list of 'events' but instead of basing this around
# the blink detection algorithm, we'll use the 'status' channel to create
# events associated with our experimental events (i.e. stimulus onset triggers)


# These data were collected from a simple oddball task, where we used triggers
# '3' for standard stimuli and '4' for oddballs. We'll now identify when the 
# diferent events happened in our raw data with .find_events(raw)
events = mne.find_events(interp_filt_raw)

# a dictionary to identify which trigger corresponds to which stimulus type
event_id = {'standard':3, 'oddball':4}

# we can visualise the paradigm (timecourse of the events), to confirm nothing
# weird has happened
fig = mne.viz.plot_events(events, 
                          sfreq = interp_filt_raw.info['sfreq'],
                          event_id = event_id)

# revisit the data plot with events included
interp_filt_raw.plot(n_channels = 64,
         event_id = event_id,
         events = events)

# We'll use the event codes to create an 'epochs' object. This is another 
# departure from the raw data, resulting in a time * channel * event matrix
# of epochs instead of the continuous time * channel data we started with

# mne.Epochs has several arguments:
# - event_id allows us to include event names as above (in place of numbers)
# - tmin and tmax allow us to define the start and end time of our epochs
# - proj = False means we want to actually change the data rather than simply 
# 'projecting' the change onto the data (which would allow us to switch the
# projection on or off, which can be useful but unnecessary here)
# - baseline allows us to define what time period we want to use for baseline 
# correction (None = start of epoch, 0 = onset of event code)
epochs = mne.Epochs(interp_filt_raw,
                    events,
                    event_id = event_id,
                    tmin = -0.2, tmax = 1,
                    proj = False, baseline = (None, 0),
                    preload = True, reject = None)

# The epochs object we just created now has various attributes, just as 'raw'
# did. For example, the list of event codes we used to create it.
epochs.events

# As before, we can extract the data:
epoched_data = epochs.get_data()
epoched_data.view()
epoched_data.shape

# How many epochs of each type were created?
np.count_nonzero(epochs.events[:,2] == 3)
np.count_nonzero(epochs.events[:,2] == 4)

#%%############################################################################
"""artifact rejection"""
###############################################################################
# Plot the epochs to do visual trial rejection.
epochs.plot(n_epochs = 5, n_channels = 64)
# - Click on an epoch to reject it from the dataset
# - Use keyboard shortcuts to adapt window size (HELP to see keyboard controls)
# - First scroll through the entire data set to get a feel for it
# - Adjust the scale to a comfortable level 
# - Start at the beginning and click on epochs you think should be rejected
# - When the figure is closed, the list of epochs to be rejected will be saved

# alternatively we can drop epochs via list of indices
#epochs.drop([1,2,3...])

# If you don't like visual data inspection, then you can instead automatically
# detect bad spans by setting a peak-to-peak threshold. This means that any
# epoch with a maximum-to-minimum difference exceeding the threshold value
# in any channel (of a given type) will be automatically rejected
reject_dictionary = dict(eog = 250e-6)#, eeg = 40e-6)

rej_epochs = mne.Epochs(interp_filt_raw,
                    events,
                    event_id = event_id,
                    tmin = -0.2, tmax = 1,
                    proj = False, baseline = (None, 0),
                    preload = True, 
                    reject = reject_dictionary) # this is the only change

# ^this probably led to an awful lot of epochs being rejected... and that's
# only based on eog thresholds - it would be even more if we also include eeg.
# Of course this could be refined in various ways. For example, for more 
# selective detection and correction you could look into the autoreject
# project: https://autoreject.github.io/stable/index.html which includes
# implementations of various algorithmic trial rejection methods (NOTE: you
# would need to also install the AutoReject library from the link above). A
# basic implementation would be:    

#from autoreject import AutoReject
#epochs = AutoReject.fit_transform(epochs) 

# for the RANSAC method:
#from autoreject import Ransac
#epochs = Ransac.fit_transform(epochs)  

    
#%%############################################################################
"""Independent Components Analysis"""
###############################################################################
# To apply an ICA to our data is pretty simple in MNE python, but first we need
# some basic info: how many components to compute (i.e. determined by the 
# number of unique - uninterpolated - data channels). We applied an average 
# reference, so we'll subtract 1 from the total number.

# use pick_types to find only the eeg channels (ignoring face electrodes) and
# get the list of channel names via ch_names. The length (len) of this list 
# will tell us how many channels there are in total (i.e. 64) 
nchan = len(epochs.pick_types(eeg = True).ch_names)

# find out how many channels we marked as 'bad' and interpolated earlier by
# taking the length (len) of the list of 'bads' from epochs.info
nbad = len(epochs.info['bads'])

# subtract nbad from nchan and subtract 1 more to find the number of components
# we want to compute
ncomp = (nchan - nbad) -1
print(ncomp)

# computing the ICA is as simple as calling the 'ICA' function from the 
# mne.preprocessing library and then 'fitting' it to the our 'epochs' data
ica = mne.preprocessing.ICA(n_components = ncomp)

# do the work (this might take a while, mne is comparatively fast though)
ica.fit(epochs) 

### Visually inspect the components
# Plot the properties of a single component to inspect it
ica.plot_properties(epochs, picks = [0])

# plot a few more...
ica.plot_properties(epochs, picks=[1,2,3,4,5])

# Look at the timecourse of the component
ica.plot_sources(interp_filt_raw)


# =============================================================================
# ### For soemthing more replicable, we could correlate the ICs with the eog 
# # channel to see which components (if any) are strongly correlated
# from mne.preprocessing import create_eog_epochs
# 
# # we automatically detect eye events from the eog channels
# eog_epochs = create_eog_epochs(interp_filt_raw)
# # identify bad components via their correlation with the eog epochs
# eog_inds, scores = ica.find_bads_eog(eog_epochs)
# 
# # plot the correlation values
# ica.plot_scores(scores, exclude=eog_inds)
# 
# # plot the properties of the components specifically around eog events
# ica.plot_properties(eog_epochs, picks = eog_inds)
# 
# =============================================================================


# Decide which component(s) to "project out of the data"
# Click on their name; this will turn them grey or list them with ica.exclude()
# Remember; the component will not ONLY represent artifactual activity
# There will also be some brain activity mixed into it; ICA is not perfect
ica.exclude = [0]

# As with the EEG data itself, we can access the actual ICA components in a 
# similar way as a np.array via get_components().
components = ica.get_components()
components.view()

# we can save the ica solution for use later if needed, by convention the file
# should end with -ica.fif
#fname = 'name_for_ICA_result-ica.fif'
#ica.save(fname = fname, overwrite = True)


#%%############################################################################
"""Apply ICA weights to the data"""
###############################################################################

# read in our ICA solution (this isn't necessary if you haven't closed it)
#fname = "name_for_ICA_result-ica.fif"
#ica = mne.preprocessing.read_ica(fname)

# use our original 'raw' data, apply a less aggressive filter and then apply 
# the ICA solution to it
raw.load_data().filter(l_freq = 0.01, h_freq = 40)
raw.set_montage(montage) # apply the montage again
ica.apply(raw)

# re-epoch the data (the same as above, the only difference is the filter)
epochs = mne.Epochs(raw,
                    events,
                    event_id = event_id,
                    tmin = -0.2, tmax = 1,
                    proj = False, baseline = (None, 0),
                    preload = True, reject = None)

# plot the data, select any remaining bad bits to reject/ interpolate
epochs.plot(n_channels = 64)

# apply the changes
epochs.interpolate_bads()
#epochs.drop([1,2,3...])

# save the final version of the data, by convention epochs files should end
# with -epo.fif
epo_fName = "clean_data-epo.fif"
epochs.save(epo_fName, overwrite = True)

#%%############################################################################
'''make ERPs'''
###############################################################################
# after the pre-processing, computing an ERP (i.e. 'evoked' object) for a given
# condition is the easy part:
    
# average all of the trials for a grand-average (collapsed across conditions)
evoked = epochs.average()
evoked.plot()
evoked.plot_joint()

# make separate ERPs for each condition
evoked_standard = epochs['standard'].average()
evoked_oddball = epochs['oddball'].average()

# select a specific electrode to plot
evoked_standard.plot('Pz')

# we can visualise both ERPs on one plot
mne.viz.plot_compare_evokeds([evoked_standard, evoked_oddball], picks = 'Fp1')

# let's take the mean of a more sensible set of electrodes
mne.viz.plot_compare_evokeds([evoked_standard, evoked_oddball],
                             picks = ['Pz', 'P1', 'P2'], 
                             combine = 'mean')

# If we're tired of MNE we can just get the data as an np.array as before
standard_data = evoked_standard.get_data()
standard_data.view()
standard_data.shape

# or as a dataframe (if you have pandas):
#standard_df = evoked_standard.to_data_frame()


