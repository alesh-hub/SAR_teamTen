# TO DO LIST

- create notebook 'showroom.ipynb' to display custom transformations

- create code lines to zip the best checkpoint at the end of training and then unzip during the evaluation mode.

- modifying the preprocessing functions to work with tensors and gpu

### Idea: keep each model in paper_replication and project_extension, create "backbone" folder in both to keep data_module, and custom_transforms

## PAPER REPLICATION

- fare data_module per paper replication -> RUNNARE SUBITO SENZA SPLIT + TUTTE AUGMENTATIONS + NO NORMALIZE:
  - Con pesi classi already computed
  - Con pesi smussati
  - Senza pesi
  
- lightning module - TRAIN + Test

### RUNS TO DO 

- DeepLab:
  - 

## PROJECT EXTENSION

- fixare commenti data_module

- 2nd version probabilistic focus crop:

    - p + (p * labels present in image)

## To do in the repo:

- create different folders for the training of each model, containing:
    - train/test file for paper preprocessing
    - train/test file for custom preprocessing
