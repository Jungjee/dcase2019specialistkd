# Distilling the Knowledge of Specialist DNNs in Acoustic Scene Classification
This repository contains script and DNN models that was used for the DCASE2019 challenge task1-a.
Currently, there are only codes for raw waveform model. 
Overall description of the system is in the [Workshop paper] (currently under review) and implementation details are further dealt in the [Technical report].  
(for now, the Workshop paper link is connected to our previous work on Knowledge distillation in acoustic scene classification, which will be presented at Interspeech 2019)


## Introduction
Common acoustic properties among different acoustic scenes were pointed as one of the causes for performance degradation in acoustic scene classification (ASC) task. <sup>1</sup>
These common properties resulted in a few pairs of acoustic scenes that are frequently misclassified (see the left confusion matrix in below image). 
In our [Workshop paper] <sup>2</sup>, we use the concept of specialist models that is in Hinton et al.'s paper <sup>3</sup>, modifying for ASC. 

## Specialist Knowledge Distillation

![aa][Overall Process Pipeline]
![aa][Conf mats]

## How to use scripts


## Reference
===
1: H. Heo, J. Jung, H. Shim and H. Yu, *Acoustic scene classification using teacher-student learning with soft-labels*, Interspeech 2019 (accepted)
2: J. Jung, H. Heo, H. Shim and H. Yu, *DISTILLING THE KNOWLEDGE OF SPECIALIST DEEP NEURAL NETWORKS IN ACOUSTIC SCENE CLASSIFICATION*, DCASE 2019 Workshop (under review)
3: G. Hinton, O. Vinyals, and J. Dean, Distilling the Knowledge in a Neural Network, NIPS 2014 deep learning workshop

[Interspeech 2019 paper]: https://arxiv.org/abs/1904.10135
[Workshop paper]: https://arxiv.org/abs/1904.10135
[Technical report]: https://dcase.community/documents/.../DCASE2019_Jung_98.pdf
[Overall Process Pipeline]: ./overall_flow.png
[Conf mats]: ./confusion_mat_exp.png
