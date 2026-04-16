# AI_Infrasound_Phase_Identification

The initial notebooks in this repository outline how to read in, preprocess, and visualize the data from McAAP. Following notebooks show how to construct and pre-train InfraCoder. The last notebooks fine-tune and interpret the model's cluster assignments. These approaches are described extensively in the manuscript, "Unsupervised Deep Representation Learning for Infrasound Phase Identification" (in review).

## Cardinal

The Cardinal software package must be installed prior to environment set up. Information on installing Cardinal can be found here: https://github.com/sjarrowsmith/cardinal.git

## Install and Activate

Navigate to directory and type:

 - conda env create -f deep_learning_env.yml

 - source activate deep_learning

 - pip install tensorflow==2.18.0 keras==3.8.0

 - pip install graphviz pydot dask cartopy networkx future pisces
   
## Notebooks

Navigate to directory and type: 

 - jupyter nbclassic&

## Data

The data used in this study (and additional data details) can be found at 10.5281/zenodo.18318050.

## Scalers

The Scalers/ archive contains the preprocessing scalers used during training:

 - cwt_scaler.pkl - scaler for scalogram

 - env_scaler.pkl - scaler for waveform envelope

 - phase_cos_scaler.pkl - scaler for cosine-transformed wavelet phase

 - phase_sin_scaler.pkl - scaler for sine-transformed wavelet phase

## Model Weights

The Model_Weights/ archive contains trained model checkpoints:

 - InfraCoder100.keras - pretrained InfraCoder autoencoder (100 epochs)

 - InfraCoder100_Attention.keras - attention extraction model

 - DEC_InfraCoder_10.weights.h5 - InfraCoder model after fine-tuning
