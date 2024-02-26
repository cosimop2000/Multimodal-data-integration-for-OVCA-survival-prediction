# Multimodal data integration for OVCA survival prediction

## Abstract

In the field of oncology, Overall Survival (OS, time between the diagnosis and the death of a patient) is a crucial piece of information used to schedule follow-up medical examinations and to define the correct type of treatment according to the characteristics and the specific conditions of each patient. (personalized treatment). The prediction of overall survival depends simultaneously on information coming both from the genetic characteristics of the tumor and from the appearance of the cancer-affected tissue.
To date, most overall survival prediction models mainly use molecular data (multi-omics). Still, simultaneously, integrating molecular data and histological images could be advantageous. Our work analyzes the SOTA models in both tasks and then creates a novel integration method to maximize the interaction between the different information contained in the modalities. In particular, two architectures have been explored: a Siamese network that tries to separate specific and shared information, and an autoencoder that is aimed to minimize the hazard risk error.
We were able to reach an accuracy of 0.69 in overall survival binary classification.

## Data

Images and molecular data from ovarian cancer patients are taken from https://gdc.cancer.gov/

## Structure of the repo

- `/Data` contains the script for data manipulation as well as the creation of the patches.
- `/PLIP` contains the script for embeddings generation as well as zero shot classification.
- `/siamese_shared_specific` contains the scripts of the siamese network.
- `/autoencoders` contains the scripts from other autoencoders models.
- `/misc` contains miscellaneous single omics baselines test using GNN and MLPs.

## Acknowledgment 

Part of the code present in this work was taken from [Customics](https://github.com/HakimBenkirane/CustOmics/tree/main) and [ComprehensiveSurvival](https://github.com/githyr/ComprehensiveSurvival).

