# Pipeline_1
Preprocessing pipeline for very noisy data. Works for TF and ERP analyses. Originally written for Biosemi 64 data. This is a very "extra" pipeline, with a backprojection of ICA weights. In many cases, you might want to save time and use a more basic pipeline.

Steps:
- Manual pre-clean cutting out huge artifacts (e.g. square waves or other artifacts so huge that you can't filter the data without causing other problems) 
- Downsample
- Highpass (1 Hz, for better working ZaplinePlus ánd better ICA decomposition)
- ZaplinePlus (clean line noise)
- Lowpass
- Interpolate bad channels 
- Average reference and cut SHORT epochs (only period of interest, no buffer zones for TF)
- Visual artefact rejection
- ICA (save weights, don't project out any components)
- Go back to the downsampled data and start over
- Highpass 
- ZaplinePlus
- Lowpass
- Interpolate bad channels
- Average to reference and epoch (include buffer zones for TF)
- Apply previously calculated ICA weights
- Project out components
- Baseline
- Visual artefact rejection
- Laplacian transform (optional)
- Convert to Fieldtrip format (optional)

## Dependencies
NoiseTools http://audition.ens.fr/adc/NoiseTools/ <br /> 
EEGlab https://sccn.ucsd.edu/eeglab/download.php <br /> 
ZaplinePlus https://github.com/MariusKlug/zapline-plus <br /> 

## Comments

## Log
Raquel - created script <br /> 
Raquel - tested on 3 participants - seems to work well <br /> 

