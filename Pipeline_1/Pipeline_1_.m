%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%  PRE-PROCESSING PIPELINE #1 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% This pipeline was created specifically for data that Emilie Caspar and
% Guillaume Pech collected in Rwanda. The main issue was that, since the
% electrical system was very contaminated and unstable, there were large
% electrical artifacts in the data. The huge square waves had to be taken
% out manually.

% To run each section, you need to create a write folder according to your 
% preferred naming convention. Here, I put the applied pre-processing steps 
% in the file name and the folder name. 

% You can merge several sections and save fewer intermediate files.

% Yor data is not collected with a Biosemi system with a 64 electrode cap?
% Make sure to adapt the script accordingly.

%% housekeeping                                         MANUAL INPUT NEEDED

clear all %start with a clean workspace

% set main folders for raw and processed data
raw = 'D:\Emilie_Caspar\Raw_Data'; 
processed = 'D:\Processed_Data';

% define triggers (event codes that you want to use as t=0 for your epochs)
triggers = {  '11' '12' '13' '14' '15' '16' '17' '18' '21' '22' '23' '24' '25' '26' '27' '28' '31' '32' '33' '34' '35' '36' '37' '38' '41' '42' '43' '44' '45' '46' '47' '48' '51' '52' '53' '54' '55' '56' '57' '58' '61' '62' '63' '64' '65' '66' '67' '68'   }; %for example: {'11' '12' '13' '14'}

% This script assumes that your datafiles have a participant-number which
% goes from 01 (or 001, 0001 etc) to nparticipants. It also assumes there 
% is some text before and after the number.
pretext = 'P'; %input text before the number
posttext = '_PAIN_'; %input text after the number
numstart = 2; %input the position of the first number in the filename
numend = 4;  %input the position of the last number in the filename

eeglab %start eeglab

%% Remove square waves & downsample 512                 MANUAL INPUT NEEDED

readfolder = raw;
writename = 'sr512_PreClean';
writefolder = [processed '\' writename]; %make sure this folder exists
cd(readfolder);
filelist = dir('*.bdf');

% Manually input subject number:
% Do this section for all PP's first, then you can run the following
% section on everyone at the same time
s = 1;        % <<< ------------------------------ MANUAL INPUT NEEDED HERE
    
% 1.- Import with average of mastoids
EEG = pop_biosig(filelist(s).name, 'channels',1:66,'ref',65:66); 

% 2.- Downsample to 512
EEG = pop_resample( EEG, 512); 
   
% 3.- Visual or automatic artifact rejection on the continuous data (only 
% to throw out square waves)
% If spectrum shows ripple, then look for square waves or other strong edges 
% and cut them out (you can set the time to display high so you can go 
% through the data very fast). Make sure to remove DC offset to visualize 
% so you don't miss any channels. If the spectrum was OK and you don't 
% suspect any square waves you can just continue and ignore the next plots, 
% and then continue again to save the downsampled dataset.

% check for square waves (the spectrum will look weird with ripple and/or 
% other strangeness)
figure
pop_spectopo(EEG, 1, [0  EEG.pnts / EEG.srate], 'EEG' , 'freqrange',...
    [0 100],'electrodes','off');

% if spectrum looks weird; plot data and mark square waves for rejection
% (click "reject" after going through the whole dataset)
pop_eegplot( EEG, 1, 1, 1);

% recheck spectrum
figure
pop_spectopo(EEG, 1, [0  EEG.pnts / EEG.srate], 'EEG' , 'freqrange',...
    [0 100],'electrodes','off');

% If everything is OK save dataset   
EEG = pop_saveset( EEG, 'filename',[writefolder '\' ...
    pretext filelist(s).name(numstart:numend) posttext writename '.set']);

clear EEG
 
%% Highpass > 1Hz (for better ICA), zapline to eliminate line noise, 
% lowpass < 40 to remove residual line noise

readname = 'sr512_PreClean'; %previous writename. You can softcode this, 
% but it makes it more confusing when you are still going back and forth 
% between sections while experimenting
readfolder = [processed '\' readname];
writename = [readname '_HP1_ZapP_LP40'];
writefolder = [readfolder '_HP1_ZapP_LP40'];%make sure this folder exists

cd(readfolder);
filelist = dir('*.set');

for s = 1:length(filelist)
    
    %open dataset
    EEG = pop_loadset('filename', filelist(s).name);
    
    % 4a.- Highpass at 1 Hz for ICA 
    EEG = pop_eegfiltnew(EEG, [], 1, [], true, [], 1); 

    % 5a.- Apply zapline plus (https://github.com/MariusKlug/zapline-plus)
    EEG.data = clean_data_with_zapline_plus(EEG.data, EEG.srate);
     
    % 6a.- Lowpass at 40 Hz 
    EEG = pop_eegfiltnew(EEG,  40, [], [], true, [], 1); 

    % Uncomment to check the data:
    
    figure
    pop_spectopo(EEG, 1, [0  EEG.pnts / EEG.srate], 'EEG' , 'freqrange',...
    [0 100],'electrodes','off');

    EEG = pop_saveset( EEG, 'filename',[writefolder '\' ...
    pretext filelist(s).name(numstart:numend) posttext writename '.set']);

    clear EEG

end


%% Remove bad channels and interpolate                  MANUAL INPUT NEEDED

readname = 'sr512_PreClean_HP1_ZapP_LP40';
readfolder = [processed '\' readname];
writename = [readname '_IP'];
writefolder = [readfolder '_IP'];%make sure this folder exists

cd(readfolder);
filelist = dir('*.set'); 

% manually input subject number:
% Do this section for all PP's first, then you can run the following
% section on everyone at the same time
s = 1;        % <<< ------------------------------ MANUAL INPUT NEEDED HERE
    
%open dataset
EEG = pop_loadset('filename', filelist(s).name);

% read channel information from file
% edit to the name of your channel location file 
EEG = pop_editset(EEG, 'chanlocs', [processed '\66channs.locs']);

[ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, 1,'gui','off'); 
EEG = eeg_checkset( EEG ); %store dataset in the ALLEEG struct

%plot time and frequency domain data to visually identify bad channels
figure
pop_eegplot( EEG, 1, 1, 1); %(in plot settings choose to display channel
% label or number)
figure
pop_spectopo(EEG, 1, [0  EEG.pnts / EEG.srate], 'EEG' , 'freqrange',...
[0 100],'electrodes','off');

% if there are bad channels replace nan with their numbers, for 
% example: bads = [4 8], if not, leave as nan
bads = [nan]; % <<< ------------------------------ MANUAL INPUT NEEDED HERE
badchans = nan(2,length(bads));  
badchans(1,:) = bads; %stores the numbers of the bad channels
badchans(2,1) = nnz(~isnan(badchans(1,:))); %stores amount of bad channels
save ([processed '\BadChans\PP_' filelist(s).name(2:4) '_badchans.mat'], ...
    'badchans','-v7.3') %save for later 

% run this loop to remove channels, interpolate and/or save
if ~isnan(bads) %if there are bad channels

    % 7a.- remove bad channels       
    EEG = pop_select( EEG, 'nochannel',bads); %remove the channels 
    [ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, 2,'gui','off'); 
    EEG = eeg_checkset( EEG );
    
    % 8a.- interpolate using the electrode locations from the original dataset
    EEG = pop_interp(EEG, ALLEEG(1).chanlocs, 'spherical');
    [ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, 3,'gui','off'); 
    
    %save dataset
    EEG = pop_saveset( EEG, 'filename',[writefolder '\' ...
        pretext filelist(s).name(numstart:numend) posttext writename '.set']);

else %if there are no bad channels
    
    %save dataset
    EEG = pop_saveset( EEG, 'filename',[writefolder '\' ...
        pretext filelist(s).name(numstart:numend) posttext writename '.set']);
end

clear badchans EEG

%% Average reference and short epochs

readname = 'sr512_PreClean_HP1_ZapP_LP40_IP';
readfolder = [processed '\' readname];
writename = [readname '_Ref_sEp'];
writefolder = [readfolder '_Ref_sEp'];%make sure this folder exists
cd(readfolder);
filelist = dir('*.set'); 

for s = 1:length(filelist)

    %open dataset
    EEG = pop_loadset('filename', filelist(s).name);
    
    % 9a.- Exclude mastoids from the dataset and then rereference to 
    % average of remaining head electrodes 
    EEG = pop_reref( EEG, [],'exclude',[EEG.nbchan-1 EEG.nbchan] ); %this 
    % only works if the mastoids are indeed the last two channels 
    EEG = pop_select( EEG, 'nochannel',{'M1','M2'});

    % 10a.- create SHORT epochs timelocked to stimulus 
    % The length of the epoch is for ICA, assuming the period of interest
    % (including baseline) is -700 to 1300 ms and freqrange is 3 - 30 Hz
    EEG = pop_epoch( EEG, triggers, [.7  1.3], 'newname', ...
        'BDF file resampled epochs', 'epochinfo', 'yes'); 

    %save dataset
    EEG = pop_saveset( EEG, 'filename',[writefolder '\' ...
    pretext filelist(s).name(numstart:numend) posttext writename '.set']);

    clear EEG 

end
    
%% Visual artifact rejection                            MANUAL INPUT NEEDED
% very aggressively throw out any unique one-off artifacts

readname = 'sr512_PreClean_HP1_ZapP_LP40_IP_Ref_sEp';
readfolder = [processed '\' readname];
writename = [readname '_Art'];
writefolder = [readfolder '_Art'];
cd(readfolder);
filelist = dir('*.set'); 

% manually input subject number:
% Do this section for all PP's first, then you can run the following
% section on everyone at the same time
s = 1;        % <<< ------------------------------ MANUAL INPUT NEEDED HERE
    
%open dataset
EEG = pop_loadset('filename', filelist(s).name);

%plot data to scroll
pop_eegplot( EEG, 1, 1, 1);
% click on any trial that shows a unique and large artefact
% DONT't click on reject at the end and don't close the % window!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

% save log of marked trials
% now click on REJECT

% 11a.- 
%once the artifacts are marked and rejected continue to save 

%save dataset
EEG = pop_saveset( EEG, 'filename',[writefolder '\' ...
    pretext filelist(s).name(numstart:numend) posttext writename '.set']);

clear EEG

%% ICA (save weights)

readname = 'sr512_PreClean_HP1_ZapP_LP40_IP_Ref_sEp_Art';
readfolder = [processed '\' readname];
writename = [readname '_ICA'];
writefolder = [readfolder '_ICA'];
cd(readfolder);
filelist = dir('*.set'); 

for s = 1:length(filelist)
    
    % open dataset
    EEG = pop_loadset('filename', [readfolder '\' pretext ...
        filelist(s).name(numstart:numend) posttext readname  '.set']);

    %load previously identified bad channels, this to count how many
    %channels were interpolated and calculate the rank of the data (the
    %number of unique sources of information)
    load ([processed '\BadChans\PP_' ...
        filelist(s).name(2:4) '_badchans.mat'])
    
    %12a.- Run ICA and make sure to adjust rank: -1 for average reference 
    % and - n(badchans) to account for interpolated channels
    EEG = pop_runica(EEG , 'extended',1,'interupt','on','pca',EEG.nbchan...
        - badchans(2,1) -1); %If using a different reference dont subtract 
    % the last "-1"
     
    % save dataset
    EEG = pop_saveset( EEG, 'filename',[writefolder '\' ...
    pretext filelist(s).name(numstart:numend) posttext writename '.set']);

    clear EEG badchans

end

%% Go back to data after step 3
% 
%  ROUND 2 BEGINS 
%
%% R2-Highpass > 0.1Hz, zapline to eliminate line noise, 
% lowpass < 40 to remove residual linenoise

readname = 'sr512_PreClean';
readfolder = [processed '\' readname];
writename = [readname '_HP01_ZapP_LP40'];
writefolder = [processed '\round2\' writename];% make sure this folder exists

cd(readfolder);
filelist = dir('*.set');

eeglab

for s = 1:length(filelist)
    
    %open dataset
    EEG = pop_loadset('filename', [readfolder '\' pretext ...
        filelist(s).name(numstart:numend) posttext readname '.set']);
    
    % 4b.- Highpass at 0.1 Hz 
    EEG = pop_eegfiltnew(EEG, [], 0.1, [], true, [], 1); 

    % 5b.- Apply zapline plus (https://github.com/MariusKlug/zapline-plus)
    EEG.data = clean_data_with_zapline_plus(EEG.data, EEG.srate);
     
    % 6b.- Lowpass at 40 Hz 
    EEG = pop_eegfiltnew(EEG,  40, [], [], true, [], 1); 
    
    % Uncomment to check the data:
    
    %figure
    %pop_spectopo(EEG, 1, [0  EEG.pnts / EEG.srate], 'EEG' , 'freqrange',...
    %[0 100],'electrodes','off');

    EEG = pop_saveset( EEG, 'filename',[writefolder '\' ...
    pretext filelist(s).name(numstart:numend) posttext writename '.set']);
     
    clear EEG

end


%% R2-Remove bad channels and interpolate 

readname = 'sr512_PreClean_HP01_ZapP_LP40';
readfolder = [processed '\round2\' readname];
writename = [readname '_IP'];
writefolder = [processed '\round2\' writename];% make sure this folder exists
cd(readfolder);
filelist = dir('*.set'); 

for s = 1:length(filelist)
    
    %open dataset
    EEG = pop_loadset('filename', [readfolder '\' pretext ...
        filelist(s).name(numstart:numend) posttext readname '.set']);
    
    %read channel information from file
    EEG = pop_editset(EEG, 'chanlocs', [processed '\66channs.locs']);
  
    %load previously identified bad channels
    load ([processed '\BadChans\PP_' ...
        filelist(s).name(2:4) '_badchans.mat'])
  
    % run this loop to remove channels, interpolate and/or save
    if ~isnan(bads) %if there are bad channels

        % 7a.- remove bad channels       
        EEG = pop_select( EEG, 'nochannel',bads); %remove the channels 
        [ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, 2,'gui','off'); 
        EEG = eeg_checkset( EEG );
        
        % 8a.- interpolate using the electrode locations from the original dataset
        EEG = pop_interp(EEG, ALLEEG(1).chanlocs, 'spherical');
        [ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, 3,'gui','off'); 
        
        %save dataset
        EEG = pop_saveset( EEG, 'filename',[writefolder '\' ...
            pretext filelist(s).name(numstart:numend) posttext writename '.set']);
    
    else %if there are no bad channels
        
        %save dataset
        EEG = pop_saveset( EEG, 'filename',[writefolder '\' ...
            pretext filelist(s).name(numstart:numend) posttext writename '.set']);

    end
    
    clear EEG badchans

end

%% R2-Average to reference and epoch

readname = 'sr512_PreClean_HP01_ZapP_LP40_IP';
readfolder = [processed '\round2\' readname];
writename = [readname '_Ref_lEp'];
writefolder = [processed '\round2\' writename];% make sure this folder exists
cd(readfolder);
filelist = dir('*.set'); 

for s = 1:length(filelist)

    %open dataset
    EEG = pop_loadset('filename', [readfolder '\' pretext ...
        filelist(s).name(numstart:numend) posttext readname '.set']);

    % 9b.- Exclude mastoids from the dataset and then rereference to 
    % average of remaining head electrodes 
    EEG = pop_reref( EEG, [],'exclude',[EEG.nbchan-1 EEG.nbchan] ); %this 
    % only works if the mastoids are indeed the last two channels 
    EEG = pop_select( EEG, 'nochannel',{'M1','M2'});

    % 10b.- LONG epochs with bufferzone for TF analysis timelocked to stimulus 
    % The length of the epoch is for ICA, assuming the period of interest 
    % (including baseline) is -700 to 1300 ms and freqrange is 3 - 30 Hz
    % replace '11' '12' '13' '14' with your own triggers
    EEG = pop_epoch( EEG, triggers, [-1.7  2.3], 'newname', ...
        'BDF file resampled epochs', 'epochinfo', 'yes'); 

    %save dataset
    EEG = pop_saveset( EEG, 'filename',[writefolder '\' ...
        pretext filelist(s).name(numstart:numend) posttext writename '.set']);

    clear EEG

end
    
%% R2-Apply ICA                                   MANUAL INPUT NEEDED HERE

readname = 'sr512_PreClean_HP01_ZapP_LP40_IP_Ref_lEp';
readfolder = [processed '\round2\' readname];
ICA_readfolder = [processed '\sr512_PreClean_HP1_ZapP_LP40_IP_Ref_sEp_Art_ICA'];
writename = [readname '_ICA'];
writefolder = [processed '\round2\' writename];% make sure this folder exists

cd(readfolder);
filelist = dir('*.set'); 

% manually input subject number:
% Do this section for all PP's first, then you can run the following
% section on everyone at the same time
s = 1;%        <<< ------------------------------ MANUAL INPUT NEEDED HERE

% load dataset with ICA weights
[ALLEEG EEG CURRENTSET ALLCOM] = eeglab;
EEG = pop_loadset('filename', [ICA_readfolder '\' pretext ...
    filelist(s).name(numstart:numend) posttext readname '_ICA.set']);

% open current dataset
EEG = pop_loadset('filename', [readfolder 'P' filelist(s).name(2:4)...
    '_PAIN_sr512_PreClean_HP01_ZapP_LP40_IP_AvRef.set']);
[ALLEEG, EEG, CURRENTSET] = eeg_store( ALLEEG, EEG, 0 );

% 11b.- apply ICA weights to current dataset
EEG = pop_editset(EEG, 'run', [], 'icaweights', 'ALLEEG(1).icaweights',...
    'icasphere', 'ALLEEG(1).icasphere', 'icachansind', 'ALLEEG(1).icachansind');
[ALLEEG EEG] = eeg_store(ALLEEG, EEG, CURRENTSET);

% visualize top 24 ICA components maps
pop_selectcomps(EEG, [1:24] );
% plot component activations
pop_eegplot( EEG, 0, 1, 1);

% project out the selected components
EEG = pop_subcomp( EEG, [1  2  3], 1); %check both plot options (single 
% trials and ERP's) to check result before clicking OK
[ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, 2,'gui','off'); 

%save dataset
EEG = pop_saveset( EEG, 'filename',[writefolder 'P' filelist(s).name(2:4)...
    '_PAIN_sr512_PreClean_HP01_ZapP_LP40_IP_AvRef_ICA.set']);

clear EEG CURRENTSET ALLEEG


%% R2-Baselining and artefact rejection

readfolder = writefolder;
writefolder = [readfolder '_Art\'];%make sure this folder exists
cd(readfolder);
filelist = dir('*.set'); 

% manually input subject number:
% Do this section for all PP's first, then you can run the following
% section on everyone at the same time
s = 1;
    
%open dataset
EEG = pop_loadset('filename', [readfolder '\' pretext ...
    filelist(s).name(numstart:numend) posttext readname '.set']);

%  Make sure to check which baseline to use. I've chosen epoch mean here, but
%  this might not be optimal for your type of analysis

% 12b.- remove epoch mean (linear baseline subtraction)
EEG = pop_rmbase( EEG, []);

%plot data to scroll
pop_eegplot( EEG, 1, 1, 1);
%click on any artefactual trial 
%click on reject at the end

%save dataset
EEG = pop_saveset( EEG, 'filename',[writefolder 'P' filelist(s).name(2:4) ...
    '_PAIN_sr512_PreClean_HP01_ZapP_LP40_IP_AvRef_ICA_BL_Art.set']);

clear EEG

%% R2-Laplacian transform

readfolder = writefolder;
writefolder = [readfolder '_Lap\'];%make sure this folder exists
cd(readfolder);
filelist = dir('*.set'); 

for s = 1:length(filelist)

    %open dataset
    EEG = pop_loadset('filename', [readfolder '\' pretext ...
        filelist(s).name(numstart:numend) posttext readname '.set']);

    % 13b.- perform Laplacian
    interelectrodedist=zeros(EEG.nbchan);
    for chani=1:EEG.nbchan
        for chanj=chani+1:EEG.nbchan
            interelectrodedist(chani,chanj) = sqrt( (EEG.chanlocs(chani).X-...
                EEG.chanlocs(chanj).X)^2 + (EEG.chanlocs(chani).Y-...
                EEG.chanlocs(chanj).Y)^2 + (EEG.chanlocs(chani).Z-...
                EEG.chanlocs(chanj).Z)^2);
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
    EEG = pop_saveset( EEG, 'filename',[writefolder 'P' filelist(s).name(2:4)...
        '_PAIN_sr512_PreClean_HP01_ZapP_LP40_IP_AvRef_ICA_BL_Art_Lap.set']);

    clear EEG

end
   
   
%% R2-Convert .set files to FieldTrip format
% This is only useful if you want to conduct your analyses in Fieldtrip

readfolder = writefolder;
writefolder = [readfolder '_FT\'];%make sure this folder exists
cd(readfolder);
filelist = dir('*.set'); 

for s = 1:length(filelist)
    
    EEG = pop_loadset('filename', filelist(s).name, 'filepath', readfolder);
    FtEEG = eeglab2fieldtrip( EEG, 'preprocessing', 'none' );          
    % attach event and epcoh fields to fieldtrip structure
    FtEEG.epoch = EEG.epoch;
    FtEEG.event = EEG.event;                                      
    % Specify name you wish dataset to be saved with
    cd(writefolder);
    fieldname = ['AP' filelist(s).name(2:4) ...
        '_PAIN_sr512_PreClean_HP01_ZapP_LP40_IP_AvRef_ICA_BL_Art_Lap_FT'];
    save (fieldname, 'FtEEG');                                             
    
    clear EEG FtEEG
end
