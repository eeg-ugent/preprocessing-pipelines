%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%  PRE-PROCESSING PIPELINE  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%  EMILIE CASPAR RWANDA DATA        %%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%  FIRST CREATED BY RAQUEL LONDON 09-11-21  %%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% Pre-clean taking out square waves (manually) and downsample to 512

clear all; close all; clc;

readfolder = 'D:\Emilie_Caspar\Raw_Data\';
writefolder = 'D:\Emilie_Caspar\Processed_Data\sr512_PreClean\';
cd(readfolder);
filelist = dir('*.bdf');

eeglab

%manually input subject number:
s = 1;
    
% 1.- Import with average of mastoid
EEG = pop_biosig([readfolder 'P' filelist(s).name(2:4) '_PAIN.bdf'],'channels',1:66,'ref',65:66); %import data with average of mastoids
[ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, 0,'gui','off'); 
EEG = eeg_checkset( EEG );

% 2.- Downsample to 512
EEG = pop_resample( EEG, 512); 
EEG = eeg_checkset( EEG );
   
% 3.- Visual or automatic artifact rejection on the continuous data (only to throw out square waves)
%if spectrum shows ripple, then look for square waves and cut them out
%(you can set the time to display high so you can go through the data
%very fast). Make sure to remove DC offset to visualize so you dont
%miss any channels. If the spectrum was OK and you dont suspect any
%square waves you can just continue and ignore the next plots, and then
%continue again to save the downsampled dataset.

% check for square waves (the spectrum will look weird with ripple and/or other strangeness)
figure
pop_spectopo(EEG, 1, [0  EEG.pnts / EEG.srate], 'EEG' , 'freqrange',[0 100],'electrodes','off');

% if spectrum looks weird; plot data and mark square waves for rejection
% (click "reject" after going through the whole dataset)
pop_eegplot( EEG, 1, 1, 1);

% recheck spectrum
figure
pop_spectopo(EEG, 1, [0  EEG.pnts / EEG.srate], 'EEG' , 'freqrange',[0 100],'electrodes','off');

% If everything is OK
% save dataset   
EEG = pop_saveset( EEG, 'filename',[writefolder 'P' filelist(s).name(2:4) '_PAIN_sr512_PreClean.set']);
clear EEG


 
%% Highpass > 1Hz (for better ICA), zapline to eliminate line noise, lowpass < 40 to remove residual linenoise

clear all; close all; clc;

readfolder = 'D:\Emilie_Caspar\Processed_Data\sr512_PreClean\';
writefolder = 'D:\Emilie_Caspar\Processed_Data\sr512_PreClean_HP1_ZapP_LP40\';
cd(readfolder);
filelist = dir('*.set');

eeglab

for s = 1:length(filelist)
    
    %open dataset
    EEG = pop_loadset('filename', [readfolder 'P' filelist(s).name(2:4) '_PAIN_sr512_PreClean.set']);
    EEG = eeg_checkset( EEG );
    
    % 4a.- Highpass at 1 Hz for ICA 
    EEG = pop_eegfiltnew(EEG, [], 1, [], true, [], 1); 
    [ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, 1,'gui','off'); 
    EEG = eeg_checkset( EEG ); 

    % 5a.- Apply zapline plus (https://github.com/MariusKlug/zapline-plus)
    EEG.data = clean_data_with_zapline_plus(EEG.data, EEG.srate);
    EEG = eeg_checkset( EEG );
     
    % 6a.- Lowpass at 40 Hz 
    EEG = pop_eegfiltnew(EEG,  40, [], [], true, [], 1); 
    [ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, 1,'gui','off'); 
    EEG = eeg_checkset( EEG ); 
    
    figure
    pop_spectopo(EEG, 1, [0      4849685.5574], 'EEG' , 'percent', 10, 'freqrange',[2 55],'electrodes','off');

    EEG = pop_saveset( EEG, 'filename',[writefolder 'P' filelist(s).name(2:4) '_PAIN_sr512_PreClean_HP1_ZapP_LP40.set']);
      
end


%% Remove bad channels and interpolate 

clear all; close all; clc;

readfolder = 'D:\Emilie_Caspar\Processed_Data\sr512_PreClean_HP1_ZapP_LP40\';
writefolder = 'D:\Emilie_Caspar\Processed_Data\sr512_PreClean_HP1_ZapP_LP40_IP\';
cd(readfolder);
filelist = dir('*.set'); 

eeglab

%manually input subject number:
s = 1;
    
%open dataset
EEG = pop_loadset('filename', [readfolder 'P' filelist(s).name(2:4) '_PAIN_sr512_PreClean_HP1_ZapP_LP40.set']);
EEG = eeg_checkset( EEG );

%read channel information from file
EEG = pop_editset(EEG, 'chanlocs', 'D:\Emilie_Caspar\66channs.locs');
[ALLEEG EEG] = eeg_store(ALLEEG, EEG, CURRENTSET);
EEG = eeg_checkset( EEG );

%plot time and frequency domain data to visually identify bad channels
figure
pop_eegplot( EEG, 1, 1, 1); %(in settings of the plot you can choose to display channel label or number)
figure 
pop_spectopo(EEG, 1, [0      4849685.5574], 'EEG' , 'freq', [6 10 22], 'freqrange',[2 60],'electrodes','off');

%put the bad channels in an array manually and then continue from the
%breakpoint
badchans = [4 8]; %for example: [4 8]
save (['D:\Emilie_Caspar\Processed_Data\BadChans\PP_' filelist(s).name(2:4) '_badchans.mat'], 'badchans','-v7.3') %save for later 
save([writefolder filelist(s).name(1:4) '_JND.mat'], 'JND','-v7.3');

% 7a.- remove bad channels       
EEG = pop_select( EEG, 'nochannel',badchans); 
[ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, 1,'gui','off'); 
EEG = eeg_checkset( EEG );
% 8a.- interpolate using the electrode locations from the original dataset
EEG = pop_interp(EEG, ALLEEG(1).chanlocs, 'spherical');
[ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, 2,'gui','off'); 

clear badchans

%save dataset
EEG = pop_saveset( EEG, 'filename',[writefolder 'P' filelist(s).name(2:4) '_PAIN_sr512_PreClean_HP1_ZapP_LP40_IP.set']);

%% Average to reference and epoch
clear all; close all; clc;

readfolder = 'D:\Emilie_Caspar\Processed_Data\sr512_PreClean_HP1_ZapP_LP40_IP\';
writefolder = 'D:\Emilie_Caspar\Processed_Data\sr512_PreClean_HP1_ZapP_LP40_IP_AvRef\';
cd(readfolder);
filelist = dir('*.set'); 

eeglab

for s = 1:length(filelist)

    %open dataset
    EEG = pop_loadset('filename', [readfolder 'P' filelist(s).name(2:4) '_PAIN_sr512_PreClean_HP1_ZapP_LP40_IP.set']);
    EEG = eeg_checkset( EEG );

    % 9a.- Exclude mastoids from the dataset and then rereference to average of remaining head electrodes 
    EEG = pop_reref( EEG, [],'exclude',[EEG.nbchan-1 EEG.nbchan] ); %this only works if the mastoids are indeed the last two channels 
    [ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, 1,'gui','off'); 
    EEG = pop_select( EEG, 'nochannel',{'M1','M2'});
    [ALLEEG, EEG, CURRENTSET] = eeg_store( ALLEEG, EEG, 0 );

    % 10a.- create SHORT epochs timelocked to stimulus 
    % The length of the epoch is for ICA, assuming the period of interest (including baseline) is -700 to 1300 ms and freqrange is 3 - 30 Hz
    EEG = pop_epoch( EEG, {  '11' '12' '13' '14' '15' '16' '17' '18' '21' '22' '23' '24' '25' '26' '27' '28' '31' '32' '33' '34' '35' '36' '37' '38' '41' '42' '43' '44' '45' '46' '47' '48' '51' '52' '53' '54' '55' '56' '57' '58' '61' '62' '63' '64' '65' '66' '67' '68'   }, [.7  1.3], 'newname', 'BDF file resampled epochs', 'epochinfo', 'yes'); 
    EEG = eeg_checkset( EEG );

    %save dataset
    EEG = pop_saveset( EEG, 'filename',[writefolder 'P' filelist(s).name(2:4) '_PAIN_sr512_PreClean_HP1_ZapP_LP40_IP_AvRef.set']);

end
    


%% Visual artifact rejection, very aggressively throw out any unique one-off artifacts
clear all; close all; clc;

readfolder = 'D:\Emilie_Caspar\Processed_Data\sr512_PreClean_HP1_ZapP_LP40_IP_AvRef\';
writefolder = 'D:\Emilie_Caspar\Processed_Data\sr512_PreClean_HP1_ZapP_LP40_IP_AvRef_Art\';
cd(readfolder);
filelist = dir('*.set'); 

eeglab

%manually input subject number:
s = 1;
    
%open dataset
EEG = pop_loadset('filename', [readfolder 'P' filelist(s).name(2:4) '_PAIN_sr512_PreClean_HP1_ZapP_LP40_IP_AvRef.set']);
EEG = eeg_checkset( EEG );

%plot data to scroll
pop_eegplot( EEG, 1, 1, 1);
%click on any trial that shows a unique and large artefact
%click on reject at the end

% 11a.- 
%once the artifacts are marked and rejected continue to save 

%save dataset
EEG = pop_saveset( EEG, 'filename',[writefolder 'P' filelist(s).name(2:4) '_PAIN_sr512_PreClean_HP1_ZapP_LP40_IP_AvRef_Art.set']);



%% ICA (project out blinks and eye-movements, save weights)
clear all; close all; clc;

readfolder = 'D:\Emilie_Caspar\Processed_Data\sr512_PreClean_HP1_ZapP_LP40_IP_AvRef_Art\';
writefolder = 'D:\Emilie_Caspar\Processed_Data\sr512_PreClean_HP1_ZapP_LP40_IP_AvRef_Art_ICA\';
cd(readfolder);
filelist = dir('*.set'); 

eeglab

for s = 1:length(filelist)
    
    % open dataset
    EEG = pop_loadset('filename', [readfolder 'P' filelist(s).name(2:4) '_PAIN_sr512_PreClean_HP1_ZapP_LP40_IP_AvRef_Art.set']);
    EEG = eeg_checkset( EEG );

    %load previously identified bad channels, this to count how many
    %channels were interpolated and calculate the rank of the data (the
    %number of unique sources of information)
    load (['D:\Emilie_Caspar\Processed_Data\BadChans\PP_' filelist(s).name(2:4) '_badchans.mat'])
    
    %12a.- Run ICA and make sure to adjust rank: -1 for average reference and -
    % n(badchans) to account for interpolated channels
    EEG = pop_runica(EEG , 'extended',1,'interupt','on','pca',EEG.nbchan - length(badchans) -1);
    [ALLEEG, EEG] = eeg_store(ALLEEG, EEG, CURRENTSET);
    EEG = eeg_checkset( EEG );
    
    % Check if the composition was succesful, but don't project out yet.
    
    % save dataset
    EEG = pop_saveset( EEG, 'filename',[writefolder 'P' filelist(s).name(2:4) '_PAIN_sr512_PreClean_HP1_ZapP_LP40_IP_AvRef_Art_ICA.set']);

end

%% Go back to data after step 3
% 
%  ROUND 2 BEGINS 
%% Highpass > 0.1Hz, zapline to eliminate line noise, lowpass < 40 to remove residual linenoise

clear all; close all; clc;

readfolder = 'D:\Emilie_Caspar\Processed_Data\sr512_PreClean\';
writefolder = 'D:\Emilie_Caspar\Processed_Data\round2\sr512_PreClean_HP01_ZapP_LP40\';
cd(readfolder);
filelist = dir('*.set');

eeglab

for s = 1:length(filelist)
    
    %open dataset
    EEG = pop_loadset('filename', [readfolder 'P' filelist(s).name(2:4) '_PAIN_sr512_PreClean.set']);
    EEG = eeg_checkset( EEG );
    
    % 4b.- Highpass at 0.1 Hz 
    EEG = pop_eegfiltnew(EEG, [], 0.1, [], true, [], 1); 
    [ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, 1,'gui','off'); 
    EEG = eeg_checkset( EEG ); 

    % 5b.- Apply zapline plus (https://github.com/MariusKlug/zapline-plus)
    EEG.data = clean_data_with_zapline_plus(EEG.data, EEG.srate);
    EEG = eeg_checkset( EEG );
     
    % 6b.- Lowpass at 40 Hz 
    EEG = pop_eegfiltnew(EEG,  40, [], [], true, [], 1); 
    [ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, 1,'gui','off'); 
    EEG = eeg_checkset( EEG ); 
    
    figure
    pop_spectopo(EEG, 1, [0      4849685.5574], 'EEG' , 'percent', 10, 'freqrange',[2 55],'electrodes','off');

    EEG = pop_saveset( EEG, 'filename',[writefolder 'P' filelist(s).name(2:4) '_PAIN_sr512_PreClean_HP01_ZapP_LP40.set']);
      
end


%% Remove bad channels and interpolate 

clear all; close all; clc;

readfolder = 'D:\Emilie_Caspar\Processed_Data\round2\sr512_PreClean_HP01_ZapP_LP40\';
writefolder = 'D:\Emilie_Caspar\Processed_Data\round2\sr512_PreClean_HP01_ZapP_LP40_IP\';
cd(readfolder);
filelist = dir('*.set'); 

eeglab

for s = 1:length(filelist)
    
    %open dataset
    EEG = pop_loadset('filename', [readfolder 'P' filelist(s).name(2:4) '_PAIN_sr512_PreClean_HP01_ZapP_LP40.set']);
    EEG = eeg_checkset( EEG );
    
    %read channel information from file
    EEG = pop_editset(EEG, 'chanlocs', 'D:\Emilie_Caspar\66channs.locs');
    [ALLEEG EEG] = eeg_store(ALLEEG, EEG, CURRENTSET);
    EEG = eeg_checkset( EEG );
    
    %load previously identified bad channels
    load (['D:\Emilie_Caspar\Processed_Data\BadChans\PP_' filelist(s).name(2:4) '_badchans.mat'])
    
    % 7b.- remove bad channels       
    EEG = pop_select( EEG, 'nochannel',badchans); 
    [ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, 1,'gui','off'); 
    EEG = eeg_checkset( EEG );
    % 8b.- interpolate using the electrode locations from the original dataset
    EEG = pop_interp(EEG, ALLEEG(1).chanlocs, 'spherical');
    [ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, 2,'gui','off'); 
    
    clear badchans
    
    %save dataset
    EEG = pop_saveset( EEG, 'filename',[writefolder 'P' filelist(s).name(2:4) '_PAIN_sr512_PreClean_HP01_ZapP_LP40_IP.set']);

end

%% Average to reference and epoch
clear all; close all; clc;

readfolder = 'D:\Emilie_Caspar\Processed_Data\round2\sr512_PreClean_HP01_ZapP_LP40_IP\';
writefolder = 'D:\Emilie_Caspar\Processed_Data\round2\sr512_PreClean_HP01_ZapP_LP40_IP_AvRef\';
cd(readfolder);
filelist = dir('*.set'); 

eeglab

for s = 1:length(filelist)

    %open dataset
    EEG = pop_loadset('filename', [readfolder 'P' filelist(s).name(2:4) '_PAIN_sr512_PreClean_HP01_ZapP_LP40_IP.set']);
    EEG = eeg_checkset( EEG );

    % 9b.- Exclude mastoids from the dataset and then rereference to average of remaining head electrodes 
    EEG = pop_reref( EEG, [],'exclude',[EEG.nbchan-1 EEG.nbchan] ); %this only works if the mastoids are indeed the last two channels 
    [ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, 1,'gui','off'); 
    EEG = pop_select( EEG, 'nochannel',{'M1','M2'});
    [ALLEEG, EEG, CURRENTSET] = eeg_store( ALLEEG, EEG, 0 );

    % 10b.- create LONG epochs with bufferzone for TF analysis timelocked to stimulus 
    % The length of the epoch is for ICA, assuming the period of interest (including baseline) is -700 to 1300 ms and freqrange is 3 - 30 Hz
    EEG = pop_epoch( EEG, {  '11' '12' '13' '14' '15' '16' '17' '18' '21' '22' '23' '24' '25' '26' '27' '28' '31' '32' '33' '34' '35' '36' '37' '38' '41' '42' '43' '44' '45' '46' '47' '48' '51' '52' '53' '54' '55' '56' '57' '58' '61' '62' '63' '64' '65' '66' '67' '68'   }, [-1.7  2.3], 'newname', 'BDF file resampled epochs', 'epochinfo', 'yes'); 
    EEG = eeg_checkset( EEG );

    %save dataset
    EEG = pop_saveset( EEG, 'filename',[writefolder 'P' filelist(s).name(2:4) '_PAIN_sr512_PreClean_HP01_ZapP_LP40_IP_AvRef.set']);

end
    
%% Apply previously calculated ICA weights
clear all; close all; clc;

readfolder = 'D:\Emilie_Caspar\Processed_Data\round2\sr512_PreClean_HP01_ZapP_LP40_IP_AvRef\';
ICA_readfolder = 'D:\Emilie_Caspar\Processed_Data\sr512_PreClean_HP1_ZapP_LP40_IP_AvRef_Art_ICA\';
writefolder = 'D:\Emilie_Caspar\Processed_Data\round2\sr512_PreClean_HP01_ZapP_LP40_IP_AvRef_ICA\';
cd(readfolder);
filelist = dir('*.set'); 

eeglab

%manually input subject number:
s = 1;

% load dataset with ICA weights
[ALLEEG EEG CURRENTSET ALLCOM] = eeglab;
EEG = pop_loadset('filename', [ICA_readfolder 'P' filelist(s).name(2:4) '_PAIN_sr512_PreClean_HP1_ZapP_LP40_IP_AvRef_Art_ICA.set']);
[ALLEEG, EEG, CURRENTSET] = eeg_store( ALLEEG, EEG, 0 );

% open current dataset
EEG = pop_loadset('filename', [readfolder 'P' filelist(s).name(2:4) '_PAIN_sr512_PreClean_HP01_ZapP_LP40_IP_AvRef.set']);
[ALLEEG, EEG, CURRENTSET] = eeg_store( ALLEEG, EEG, 0 );

% 11b.- apply ICA weights to current dataset
EEG = eeg_checkset( EEG );
EEG = pop_editset(EEG, 'run', [], 'icaweights', 'ALLEEG(1).icaweights', 'icasphere', 'ALLEEG(1).icasphere', 'icachansind', 'ALLEEG(1).icachansind');
[ALLEEG EEG] = eeg_store(ALLEEG, EEG, CURRENTSET);

% visualize top ICA components maps
pop_selectcomps(EEG, [1:24] );
% plot component activations
pop_eegplot( EEG, 0, 1, 1);

% project out the selected components
EEG = eeg_checkset( EEG );
EEG = pop_subcomp( EEG, [1  2  3], 1); %check both plot options (single trials and ERP's) to check result before clicking OK
[ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, 2,'gui','off'); 

%save dataset
EEG = pop_saveset( EEG, 'filename',[writefolder 'P' filelist(s).name(2:4) '_PAIN_sr512_PreClean_HP01_ZapP_LP40_IP_AvRef_ICA.set']);


%% Baselining and artefact rejection

clear all; close all; clc;

readfolder = 'D:\Emilie_Caspar\Processed_Data\round2\sr512_PreClean_HP01_ZapP_LP40_IP_AvRef_ICA\';
writefolder = 'D:\Emilie_Caspar\Processed_Data\round2\sr512_PreClean_HP01_ZapP_LP40_IP_AvRef_ICA_Art\';
cd(readfolder);
filelist = dir('*.set'); 

eeglab

%manually input subject number:
s = 1;
    
%open dataset
EEG = pop_loadset('filename', [readfolder 'P' filelist(s).name(2:4) '_PAIN_sr512_PreClean_HP01_ZapP_LP40_IP_AvRef_ICA.set']);
EEG = eeg_checkset( EEG );

%  Make sure to check which baseline to use. I've chosen epoch mean here, but
%  this might not be optimal for your type of analysis

% 12b.- remove epoch mean (linear baseline subtraction)
EEG = pop_rmbase( EEG, []);
[ALLEEG EEG] = eeg_store(ALLEEG, EEG, CURRENTSET);
EEG = eeg_checkset( EEG );

%plot data to scroll
pop_eegplot( EEG, 1, 1, 1);
%click on any artefactual trial 
%click on reject at the end

%save dataset
EEG = pop_saveset( EEG, 'filename',[writefolder 'P' filelist(s).name(2:4) '_PAIN_sr512_PreClean_HP01_ZapP_LP40_IP_AvRef_ICA_BL_Art.set']);


%% Laplacian transform
clear all; close all; clc;

readfolder = 'D:\Emilie_Caspar\Processed_Data\round2\sr512_PreClean_HP01_ZapP_LP40_IP_AvRef_ICA_BL_Art\';
writefolder = 'D:\Emilie_Caspar\Processed_Data\round2\sr512_PreClean_HP01_ZapP_LP40_IP_AvRef_ICA_BL_Art_Lap\';
cd(readfolder);
filelist = dir('*.set'); 

eeglab

for s = 1:length(filelist)

    %open dataset
    EEG = pop_loadset('filename', [readfolder 'P' filelist(s).name(2:4) '_PAIN_sr512_PreClean_HP01_ZapP_LP40_IP_AvRef_ICA_BL_Art.set']);
    EEG = eeg_checkset( EEG );

    % 13b.- perform Laplacian
    interelectrodedist=zeros(EEG.nbchan);
    for chani=1:EEG.nbchan
        for chanj=chani+1:EEG.nbchan
            interelectrodedist(chani,chanj) = sqrt( (EEG.chanlocs(chani).X-EEG.chanlocs(chanj).X)^2 + (EEG.chanlocs(chani).Y-EEG.chanlocs(chanj).Y)^2 + (EEG.chanlocs(chani).Z-EEG.chanlocs(chanj).Z)^2);
        end
    end

    valid_gridpoints = find(interelectrodedist);

    % extract XYZ coordinates from EEG structure
    X = [EEG.chanlocs.X];
    Y = [EEG.chanlocs.Y];
    Z = [EEG.chanlocs.Z];

    % create G and H matrices
    [ldata,G,H] = laplacian_perrinX(EEG.data(:,:,:),X,Y,Z,10,1e-6);
    
    % replace non transformed data with Laplacian transformed data
    EEG.data = ldata;
    
    %save dataset
    EEG = pop_saveset( EEG, 'filename',[writefolder 'P' filelist(s).name(2:4) '_PAIN_sr512_PreClean_HP01_ZapP_LP40_IP_AvRef_ICA_BL_Art_Lap.set']);

end

    
   
%% Convert to fieldtrip format
% =========================================================================
%            Convert .set files to FieldTrip format
% =========================================================================
clear all; close all; clc;

readfolder = 'D:\Emilie_Caspar\Processed_Data\round2\sr512_PreClean_HP01_ZapP_LP40_IP_AvRef_ICA_BL_Art_Lap\';
writefolder = 'D:\Emilie_Caspar\Processed_Data\round2\sr512_PreClean_HP01_ZapP_LP40_IP_AvRef_ICA_BL_Art_Lap_FT\';
cd(readfolder);
filelist = dir('*.set'); 

eeglab

for s = 1:length(filelist)
    
    EEG = pop_loadset('filename', filelist(s).name, 'filepath', readfolder);
    FtEEG = eeglab2fieldtrip( EEG, 'preprocessing', 'none' );          
    % attach event and epcoh fields to fieldtrip structure
    FtEEG.epoch = EEG.epoch;
    FtEEG.event = EEG.event;                                      
    % Specify name you wish dataset to be saved with
    cd(writefolder);
    fieldname = ['P' filelist(s).name(2:4) '_PAIN_sr512_PreClean_HP01_ZapP_LP40_IP_AvRef_ICA_BL_Art_Lap_FT'];
    save (fieldname, 'FtEEG');                                             
    
end
