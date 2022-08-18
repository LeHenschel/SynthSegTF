"""

Very simple script to generate an example of the synthetic data used to train SynthSeg.
This is for visualisation purposes, since it uses all the default parameters.



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
import sys
sys.path.append("/autofs/vast/lzgroup/Users/LeonieHenschel/SynthSegTF")
from ext.lab2im import utils
from SynthSeg.brain_generator import BrainGenerator

# generate an image from the label map.
# path training label maps
path_training_label_maps = '/autofs/vast/lzgroup/Users/LeonieHenschel/FastInfantSurfer/data/dataset_split_training.csv'
label_name = "/dhcp_mapped23_conformed_nn.mgz"
path_model_dir = '/autofs/vast/lzgroup/Projects/FastInfantSurfer/experiments/SynthSeg_net1/run_dhcp'
resume_ckpt = None  # path_model_dir + '/dice_071.h5'
batchsize = 1
n_levels = 5           # number of resolution levels
# ---------- Generation parameters ----------
# these parameters are from the previous tutorial, and thus we do not explain them again here

# generation and segmentation labels
# List of labels (first sided, second non-sided)
path_generation_labels = '/autofs/vast/lzgroup/Projects/FastInfantSurfer/hdf5_sets/dhcp_AllLabels.npy'
n_neutral_labels = 4  # Non-sided labels (this was wrongly set to 18?)
# Labels we want to predict (equivalent to generation_labels here)
path_segmentation_labels = '/autofs/vast/lzgroup/Projects/FastInfantSurfer/hdf5_sets/dhcp_SegmentationLabels.npy'

# shape and resolution of the outputs
target_res = None  # set to something else (not None) if training samples at different resolutions are required
output_shape = None  # crop the training examples to this size (fit on GPU)
n_channels = 1  # one for uni-modal, two for multi-modal

# GMM sampling
prior_distributions = 'uniform'
# Regroup labels with similar tissue types into K "classes", so that intensities of similar regions are sampled
# from the same Gaussian distribution. Provide a list indicating the class of each label.
path_generation_classes = '/autofs/vast/lzgroup/Projects/FastInfantSurfer/hdf5_sets/dhcp_GenerationClasses.npy'

# spatial deformation parameters
flipping = True  # enable right/left flipping
scaling_bounds = 0.15  # the scaling coefficients will be sampled from U(1-scaling_bounds; 1+scaling_bounds)
rotation_bounds = 15  # the rotation angles will be sampled from U(-rotation_bounds; rotation_bounds)
shearing_bounds = 0.012  # the shearing coefficients will be sampled from U(-shearing_bounds; shearing_bounds)
translation_bounds = False  # no translation is performed, as this is already modelled by the random cropping
nonlin_std = 3.  # this controls the maximum elastic deformation (higher = more deformation)
bias_field_std = 0.5  # his controls the maximum bias field corruption (higher = more bias)
nonlin_shape_factor = .04

# acquisition resolution parameters
randomise_res = False  # Do not randomise the resolution
blur_range = 1.0

# get label lists
generation_labels, _ = utils.get_list_labels(label_list=path_generation_labels, labels_dir=path_training_label_maps)
if path_segmentation_labels is not None:
    segmentation_labels, _ = utils.get_list_labels(label_list=path_segmentation_labels)
else:
    segmentation_labels = generation_labels

# instantiate BrainGenerator object
brain_generator = BrainGenerator(labels_dir=path_training_label_maps,
                                 generation_labels=generation_labels,
                                 n_neutral_labels=n_neutral_labels,
                                 output_labels=segmentation_labels,
                                 subjects_prob=None,
                                 patch_dir=None,
                                 batchsize=batchsize,
                                 n_channels=n_channels,
                                 target_res=target_res,
                                 output_shape=output_shape,
                                 output_div_by_n=2 ** n_levels,  #
                                 generation_classes=path_generation_classes,
                                 prior_distributions=prior_distributions,
                                 prior_means=None,  #
                                 prior_stds=None,  #
                                 use_specific_stats_for_channel=False,  #
                                 mix_prior_and_random=False,  #
                                 flipping=flipping,
                                 scaling_bounds=scaling_bounds,
                                 rotation_bounds=rotation_bounds,
                                 shearing_bounds=shearing_bounds,
                                 translation_bounds=translation_bounds,
                                 nonlin_std=nonlin_std,
                                 nonlin_shape_factor=nonlin_shape_factor,  #
                                 randomise_res=randomise_res,
                                 blur_range=blur_range,  #
                                 bias_field_std=bias_field_std,
                                 label_img_name=label_name)


result_dir = "./generated_examples/"
n_examples = 1

import os

for n in range(n_examples):

    # generate new image and corresponding labels
    im, lab = brain_generator.generate_brain()

    # save output image and label map
    utils.save_volume(im, brain_generator.aff, brain_generator.header,
                      os.path.join(result_dir, f'image_{n+1}.nii.gz'))
    utils.save_volume(lab, brain_generator.aff, brain_generator.header,
                      os.path.join(result_dir, f'labels_{n+1}.nii.gz'))

#im, lab = brain_generator.generate_brain()

# save output image and label map
#utils.save_volume(im, brain_generator.aff, brain_generator.header, './generated_examples/image_default.nii.gz')
#utils.save_volume(lab, brain_generator.aff, brain_generator.header, './generated_examples/labels_default.nii.gz')
