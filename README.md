# SAR_teamTen

Marine pollution, particularly oil spills, poses a significant threat to marine ecosystems. Accurate and timely detection of oil spills is crucial for mitigating their impact.

The project is supervised by LINKS Foundation, within the context of the MARIS project for marine pollution monitoring.

## Problem

Automated, SAR-based oil spill segmentation:
- The importance of an AI solution lies in its ability to provide rapid, accurate, and cost-effective  monitoring of marine environments, compared to the human counterpart.

## Data

- The datasets for this project is derived from **Sentinel-1 satellite**, provided by the European Space Agency.
- **Synthetic Aperture Radar** (SAR) is a form of radar used to create two-dimensional images or three-dimensional reconstructions of landscapes.
- The data includes images depicting oil spills, look-alikes, land, ships, and sea areas.

## Task

- Oil-spill segmentation based on SAR images
- **Segmentation task**: label each pixel of input image with a class label, i.e., _oil spills, look-alikes, land, ships, sea areas_.
- Develop and train well-known machine learning models for segmentation tasks, i.e., UNet, LinkNet, PSPNet, DeepLabv2, CBD-Net.
- Evaluate the performance of the models using standard metrics, i.e.,
Intersection over Union (IoU).
