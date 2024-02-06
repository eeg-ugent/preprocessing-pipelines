
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22

@author: raquel
""" 

#%%############################################################################    
""" housekeeping"""
###############################################################################

#   1   #
#In order to be able to manipulate the figures
#Change tools/preferences/ipython console/graphics/backend to qt5
#If plots still dont appear as independent windows, restart Python

#   2   #
#Import the necessary Python modules:
import mne
import os

#%%############################################################################
"""import, rereference"""
###############################################################################

# define your read and write directories:
read_dir = "D:/Anna_Jessen_Minimizer/Raw/" 
write_dir = "D:/Anna_Jessen_Minimizer/Better_Pipeline/Raw_Rref/"

for sub in range(len(os.listdir(read_dir))):  # Loop over participants

    raw = mne.io.read_raw_bdf(read_dir + os.listdir(read_dir)[sub],preload=True) #Load raw data
  
    fix_chans = {'EXG1':'Mastoid1','EXG2':'Mastoid2',
                 'EXG3':'eye_right','EXG4':'eye_left',
                 'EXG5':'eye_above','EXG6':'eye_below'}
    raw.rename_channels(fix_chans) #rename externals
    
    # we still have two exg channels which weren't actually recorded though (EXG7
    # and EXG8) these are empty, so we'll drop them
    raw.drop_channels(['EXG7', 'EXG8'])
    
    # rereference to average of the mastoids
    raw.set_eeg_reference(ref_channels = ['Mastoid1', 'Mastoid2'])
    
    # we'll also reset the channel types, so MNE knows what is 'brain' data
    raw.set_channel_types({'eye_above':'eog', 'eye_below':'eog',
                           'eye_left':'eog', 'eye_right': 'eog'})
    
    raw.drop_channels(['Mastoid1', 'Mastoid2']) # we dont need these anymore
    
    montage = mne.channels.make_standard_montage('biosemi64')
    raw.set_montage(montage) #apply the montage
    
    raw.save(write_dir + os.listdir(read_dir)[sub][0:4] + '_raw_rref.fif', overwrite=True)
    
#%%############################################################################
"""Filter and Epoch"""
###############################################################################   
    
# define your read and write directories:
read_dir = "D:/Anna_Jessen_Minimizer/Better_Pipeline/Raw_Rref/"
write_dir = "D:/Anna_Jessen_Minimizer/Better_Pipeline/Raw_Rref_BPF_Epo/"

for sub in range(len(os.listdir(read_dir))):  # Loop over participants

    raw = mne.io.read_raw_fif(read_dir + os.listdir(read_dir)[sub],preload=True) #Load raw data

    # filter the data harshly for ICA
    raw.filter(l_freq=1, h_freq=40) #apply the bandpass filter 1-40 Hz
    
    # epoch data
    events = mne.find_events(raw, stim_channel = ['Status'], shortest_event = 1) 
    raw_epo = mne.Epochs(raw, events, event_id = {'101': 101, 
                                                 '102': 102, 
                                                 '103': 103,
                                                 '104': 104}, 
                        tmin=-0.2, tmax=1.2,
                         proj=False, baseline= (None,0),
                         preload=True, reject=None)
    
    raw_epo.save(write_dir + os.listdir(read_dir)[sub][0:4] + '_raw_rref_bpf_epo.fif', overwrite=True)
    
#%%############################################################################
"""Artefact & Channel rejection"""
###############################################################################

# define your read and write directories:
read_dir = "D:/Anna_Jessen_Minimizer/Better_Pipeline/Raw_Rref_BPF_Epo/"
write_dir = "D:/Anna_Jessen_Minimizer/Better_Pipeline/Raw_Rref_BPF_Clean_Epo/"

sub = 0

raw_clean_epo = mne.read_epochs(read_dir + os.listdir(read_dir)[sub],preload=True) #Load epochs

# first reref to average
raw_clean_epo.set_eeg_reference().apply_proj().average()

# mark bad epochs & channels
raw_clean_epo.plot_psd() #frequency domain
raw_clean_epo.plot(n_epochs = 10, n_channels = 64)

# Close the figure for the selected epochs to be rejected

# Interpolate the bad channels
raw_clean_epo.load_data().interpolate_bads()

# save file as .fif
raw_clean_epo.save(write_dir + os.listdir(read_dir)[sub][0:4] + '_raw_rref_bpf_clean_epo.fif', overwrite= True)
    
#%%############################################################################
"""ICA"""
###############################################################################
read_dir = "D:/Anna_Jessen_Minimizer/Better_Pipeline/Raw_Rref_BPF_Clean_Epo/" 
write_dir = "D:/Anna_Jessen_Minimizer/Better_Pipeline/Raw_Rref_BPF_Clean_Epo_ICA/"

sub = 0 # input subnumber manually

raw_clean_epo_ica = mne.read_epochs(read_dir + os.listdir(read_dir)[sub],preload=True)

# create ICA object with desired parameters
ica = mne.preprocessing.ICA(n_components = 40)

# do ICA decomposition
ica.fit(raw_clean_epo_ica) 

raw_clean_epo_ica.plot(n_epochs = 10, n_channels = 64)

# Plot the components
# clicking on a component will mark it as bad 
ica.plot_components()

# Plot the properties of a single component (e.g. to check its frequency profile)
ica.plot_properties(raw_clean_epo_ica, picks=[0,1,2,3,4,5,6,7,8,9,10])

#Look at the timecourse of the component
ica.plot_sources(raw_clean_epo_ica)

# save file as .fif
ica.save(write_dir + os.listdir(read_dir)[sub][0:4] + '_raw_rref_bpf_clean_epo_ica.fif', overwrite= True)

#%%############################################################################
"""Apply ICA weights to the data"""
###############################################################################
sub = 0 # input subnumber manually

# read in our ICA solution 
# define your read directory:
read_dir = "D:/Anna_Jessen_Minimizer/Better_Pipeline/Raw_Rref_BPF_Clean_Epo_ICA/"
ica = mne.preprocessing.read_ica(read_dir + os.listdir(read_dir)[sub]) # load ICA weights

# define your read and write directories:
read_dir = "D:/Anna_Jessen_Minimizer/Better_Pipeline/Raw_Rref/"
write_dir = "D:/Anna_Jessen_Minimizer/Better_Pipeline/Clean_Data/"

# Load raw unprocessed data again
raw = mne.io.read_raw_fif(read_dir + os.listdir(read_dir)[sub],preload=True) # load raw, rereferenced data
# reref to average
raw.set_eeg_reference('average')

# Apply a less aggressive filter and then apply the ICA solution to it
raw.filter(l_freq = 0.2, h_freq = 30)
ica.apply(raw) #apply the ICA weights to the data

# re-epoch the data (the same as above, the only difference is the filter)
events = mne.find_events(raw, stim_channel = ['Status'], shortest_event = 1)
epochs = mne.Epochs(raw, events, event_id = {'101': 101, 
                                             '102': 102, 
                                             '103': 103,
                                             '104': 104}, 
                    tmin=-0.2, tmax=1.2,
                     proj=False, baseline= (None,0),
                     preload=True, reject=None)

# plot the data, select any channels to interpolate
epochs.plot(n_epochs = 10, n_channels = 64)

# apply the changes
epochs.interpolate_bads()

# plot the data, select any remaining bad bits to reject
epochs.plot(n_epochs = 10, picks = ['C3','Cz','C4','CPz','P3','Pz','P4'])

# if you selected additional channels, interpolate again
epochs.interpolate_bads()

# save the final version of the data, by convention epochs files should end
# with -epo.fif
epochs.save(write_dir + os.listdir(read_dir)[sub][0:4] + '_clean_data-epo.fif', overwrite = True)
   
#%%############################################################################
"""Create ERP's"""
###############################################################################

# define your read and write directories:
read_dir = "D:/Anna_Jessen_Minimizer/Better_Pipeline/Clean_Data/" 
write_dir = "D:/Anna_Jessen_Minimizer/Better_Pipeline/Results/"

all_evokeds_unproductive = []
all_evokeds_productive = []
all_evokeds_highfreq = []
all_evokeds_lowfreq = []

for sub in range(len(os.listdir(read_dir))):  # Loop over participants

    clean_data = mne.read_epochs(read_dir + os.listdir(read_dir)[sub],preload=True)
    
    clean_data.apply_baseline(baseline=(-0.2, 0))
    
    evoked_unproductive = clean_data['101'].average()
    evoked_productive = clean_data['102'].average()
    evoked_highfreq = clean_data['103'].average()
    evoked_lowfreq = clean_data['104'].average()
    
    chans = ['C3','Cz','C4','CPz','P3','Pz','P4']
    
    mne.viz.plot_compare_evokeds(dict(unproductive=evoked_unproductive, 
                                      productive=evoked_productive),
                                 chans)
        
    # Now, we take these individual evokeds and put them in our list
    # together with the other individuals
    all_evokeds_unproductive.append(evoked_unproductive)
    all_evokeds_productive.append(evoked_productive)
    all_evokeds_highfreq.append(evoked_highfreq)
    all_evokeds_lowfreq.append(evoked_lowfreq)
          
GA_unproductive = mne.grand_average(all_evokeds_unproductive)
GA_productive = mne.grand_average(all_evokeds_productive)
GA_highfreq = mne.grand_average(all_evokeds_highfreq)
GA_lowfreq = mne.grand_average(all_evokeds_lowfreq)

mne.viz.plot_compare_evokeds(dict(unproductive=all_evokeds_unproductive, 
                                  productive=all_evokeds_productive),
                              chans)

mne.viz.plot_compare_evokeds(dict(highfreq=all_evokeds_highfreq, 
                                  lowfreq=all_evokeds_lowfreq),
                              chans)
