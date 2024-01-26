 Implementation of attentiveFP to train a model to predict UVspectra using smilestrings.

 Currently implemented using Cosine Similarity and predicitng 170 points between 80, 420 nm
 No Datasets shipped if you would like to demo models make datasets hat are pkl files with first element being smile string and rest of row is the discretized spectra.

 Download all packages as they've been downloaded in packetlist.

 then run setup.py before training.

To train a model after giving it data to train on run command python train.py --model-name AttentiveFPModel
