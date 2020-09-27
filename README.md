# MINT_Frequency_Spectrogram_Model
 Frequency spectrogram analysis of EEG data

## Directory structure
- `train.py` is the engine. The model is configured there, the optimizer, loss, and data iterators are declared there, and tensorboard is set up there. Training and validation are done afterwards. The model is saved there every set number of epochs. This file should remain *mostly* static if you introduce new models - you can play around with the learning rate and such, but the process doesn't really need to be changed.
- `myargs.py` determines some input arguments for training. It's pretty self-explanatory.
- in `utils/` you will find data processing files (like `spectrogram_generate.py`) that are required for certain models. If you need other data processing, you should write it in a script here and process the data to a location that you determine. Moreover, `model.py` contains the training models architectures. Each model should be implemented as a class and will be imported to `train.py`.
- data are located at the Google Drive: `MINT/Experiment Data/Machine Learning Data/data`. Simply download the entire folder to your project repository; it's already set up in the right format. In the folder you will see `raw` for the raw data from the EEG and `train` for data that's already processed in the spectrogram form. You'll have to re-process the raw data if you do not want to use spectrograms.
- `trained_models/` contains saved models.
- `runs/` contains outputs from tensorboard.
- the other stuff is miscellaneous.   

## Tensorboard
Make sure tensorboard is installed.
While in project directory, start with
`python -m tensorboard.main --logdir=runs`
View data at
`http://localhost:6006/`
