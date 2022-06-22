"""

This script shows how we trained SynthSeg.
Importantly, it reuses numerous parameters seen in the previous tutorial about image generation
(i.e., 2-generation_explained.py), which we strongly recommend reading before this one.



If you use this code, please cite one of the SynthSeg papers:
https://github.com/BBillot/SynthSeg/blob/master/bibtex.bib

Copyright 2020 Benjamin Billot

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in
compliance with the License. You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software distributed under the License is
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
implied. See the License for the specific language governing permissions and limitations under the
License.
"""

# project imports
from SynthSeg.training import training
import pandas as pd
import numpy as np


def create_label_mappings(lut_synth_file, base="/autofs/vast/lzgroup/Projects/FastInfantSurfer/hdf5_sets/",
                          cols=["AllLabels", "SegmentationLabels", "GenerationClasses"]):
    separator = {"tsv": "\t", "csv": ",", "txt": " "}
    lut = pd.read_csv(lut_synth_file, sep=separator[lut_synth_file[-3:]])
    for col in cols:
        np.save(base + col + ".npy", lut[col].values)


if __name__ == "__main__":
    # path training label maps
    path_training_label_maps = '/autofs/vast/lzgroup/Users/LeonieHenschel/FastInfantSurfer/data/dataset_split_training.csv'
    path_model_dir = '/autofs/vast/lzgroup/Projects/FastInfantSurfer/experiments/SynthSeg_orig'
    batchsize = 1

    # architecture parameters
    n_levels = 5           # number of resolution levels
    nb_conv_per_level = 2  # number of convolution per level
    conv_size = 3          # size of the convolution kernel (e.g. 3x3x3)
    unet_feat_count = 24   # number of feature maps after the first convolution
    activation = 'elu'     # activation for all convolution layers except the last, which will use sofmax regardless
    feat_multiplier = 2    # if feat_multiplier is set to 1, we will keep the number of feature maps constant throughout the
    #                        network; 2 will double them(resp. half) after each max-pooling (resp. upsampling);
    #                        3 will triple them, etc.

    # training parameters
    lr = 1e-4               # learning rate
    lr_decay = 0            # learning rate decay (knowing that Adam already has its own internal decay)
    wl2_epochs = 1          # number of pre-training epochs with wl2 metric w.r.t. the layer before the softmax
    dice_epochs = 1 #100       # number of training epochs; for FastSurferCNN = 400k steps, adopt here to same number
    steps_per_epoch = 4000  # number of iteration per epoch


    # ---------- Generation parameters ----------
    # these parameters are from the previous tutorial, and thus we do not explain them again here

    # ge    neration and segmentation labels
    # List of labels (first sided, second non-sided)
    path_generation_labels = '/autofs/vast/lzgroup/Projects/FastInfantSurfer/hdf5_sets/AllLabels.npy'
    n_neutral_labels = 18 # Non-sided labels
    # Labels we want to predict (equivalent to generation_labels here)
    path_segmentation_labels = '/autofs/vast/lzgroup/Projects/FastInfantSurfer/hdf5_sets/SegmentationLabels.npy'

    # shape and resolution of the outputs
    target_res = None  # set to something else (not None) if training samples at different resolutions are required
    output_shape = 96  # crop the training examples to this size (fit on GPU)
    n_channels = 1 # one for uni-modal, two for multi-modal

    # GMM sampling
    prior_distributions = 'uniform'
    # Regroup labels with similar tissue types into K "classes", so that intensities of similar regions are sampled
    # from the same Gaussian distribution. Provide a list indicating the class of each label.
    path_generation_classes = '/autofs/vast/lzgroup/Projects/FastInfantSurfer/hdf5_sets/GenerationClasses.npy'

    # spatial deformation parameters
    flipping = True  # enable right/left flipping
    scaling_bounds = 0.15  # the scaling coefficients will be sampled from U(1-scaling_bounds; 1+scaling_bounds)
    rotation_bounds = 15  # the rotation angles will be sampled from U(-rotation_bounds; rotation_bounds)
    shearing_bounds = 0.012  # the shearing coefficients will be sampled from U(-shearing_bounds; shearing_bounds)
    translation_bounds = False  # no translation is performed, as this is already modelled by the random cropping
    nonlin_std = 3.  # this controls the maximum elastic deformation (higher = more deformation)
    bias_field_std = 0.5  # his controls the maximum bias field corruption (higher = more bias)

    # acquisition resolution parameters
    randomise_res = False # Do not randomise the resolution
    blur_range = 1.0

    # ------------------------------------------------------ Training ------------------------------------------------------

    training(path_training_label_maps,
            path_model_dir,
            generation_labels=path_generation_labels,
            segmentation_labels=path_segmentation_labels,
            n_neutral_labels=n_neutral_labels,
            batchsize=batchsize,
            n_channels=n_channels,
            target_res=target_res,
            output_shape=output_shape,
            prior_distributions=prior_distributions,
            generation_classes=path_generation_classes,
            flipping=flipping,
            scaling_bounds=scaling_bounds,
            rotation_bounds=rotation_bounds,
            shearing_bounds=shearing_bounds,
            translation_bounds=translation_bounds,
            nonlin_std=nonlin_std,
            randomise_res=randomise_res,
            blur_range=blur_range,
            bias_field_std=bias_field_std,
            n_levels=n_levels,
            nb_conv_per_level=nb_conv_per_level,
            conv_size=conv_size,
            unet_feat_count=unet_feat_count,
            feat_multiplier=feat_multiplier,
            activation=activation,
            lr=lr,
            lr_decay=lr_decay,
            wl2_epochs=wl2_epochs,
            dice_epochs=dice_epochs,
            steps_per_epoch=steps_per_epoch)
