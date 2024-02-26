# Multimodal data integration for OVCA survival prediction

## Abstract

In the field of oncology, Overall Survival (OS, time between the diagnosis and the death of a patient) is a crucial piece of information used to schedule follow-up medical examinations and to define the correct type of treatment according to the characteristics and the specific conditions of each patient. (personalized treatment). The prediction of overall survival depends simultaneously on information coming both from the genetic characteristics of the tumor and from the appearance of the cancer-affected tissue.
To date, most overall survival prediction models mainly use molecular data (multi-omics). Still, simultaneously, integrating molecular data and histological images could be advantageous. Our work analyzes the SOTA models in both tasks and then creates a novel integration method to maximize the interaction between the different information contained in the modalities. In particular, two architectures have been explored: a Siamese network that tries to separate specific and shared information, and an autoencoder that is aimed to minimize the hazard risk error.
We were able to reach an accuracy of 0.69 in overall survival binary classification.

## Structure of the repo

- `/Data` contains the script for data manipulation as well as the creation of the patches.
- `/PLIP` contains the script for embeddings generation as well as zero-shot classification.
- `/siamese_shared_specific` contains the scripts of the siamese network.
- `/autoencoders` contains the scripts from other autoencoders models.
- `/misc` contains miscellaneous single omics baselines test using GNN and MLPs.

**All the input and output paths must be specified from the file `Data/config.ini` (see the example for the structure).**

### Data download and preprocessing

Molecular data from ovarian cancer patients are taken from [NCI's Genomic Data Commons](https://gdc.cancer.gov/) and elaborated with the `/Data/<omic>/create_table.py` files (a script for each omic).

Then, the script `/Data/download_batch.py` downloads the images and creates the patches using CLAM. The images are then removed for the sake of storage optimization. The outputs of the script are the embeddings of the images. The dataset is created using firstly `/PLIP/create_dataset_from_embedding.py` and finally `/PLIP/create_dataset_annotation.py`.

To select the best embeddings using the zero-shot classifier the script `/PLIP/zero_shot_classifier.py` and then `/PLIP/classification_visualizer.ipynb` can be run. 

### Models

The models need the embeddings both of the omics and the images set in the config file. 
Then, to reproduce the results, the several models present in the `/siamese_shared_specific` can be used. 

In particular, each `/siamese_shared_specific/train_iters_<n>_inputs.py` executes the training and evaluation of the respective models, where `n` is the number of inputs in the model. Automatically a 20-iter run is made both for the training and the testing.

## Acknowledgment 

Part of the code present in this work was taken from [Customics](https://github.com/HakimBenkirane/CustOmics/tree/main) and [ComprehensiveSurvival](https://github.com/githyr/ComprehensiveSurvival) as well as [CLAM](https://github.com/mahmoodlab/CLAM).

