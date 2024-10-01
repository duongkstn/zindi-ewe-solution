# TechCabal Ewè Audio Translation Challenge

## Introduction
This competition aims to evaluate an audio classification system for Ewe speakers in various contexts. The solution will be deployed on edge devices.
For more details, please visit https://zindi.africa/competitions/techcabal-ewe-audio-translation-challenge.


## My solution
#### I trained models using 3 following steps:
1. First, I trained a `efficientnet-b0` model achieving 0.964 accuracy (public LB)
2. Second, I trained a `mobilenet_v3_small` model, reaching a 0.959 accuracy (public LB)
3. Third, since model size of the first model is bigger than 10 MB, but it has higher accuracy, so I decided to use pseudo-labels from test set and merge with training set to train the better `mobilenet_v3_small` model, which achieved 0.962 accuracy (public LB)

#### Feature engineering:
I limited audio time to 2 seconds and used Logmel features (`n_mels = 128`). With model 2 and 3, Delta features were used.

#### Augmentation:
MaskFreq

## Folder structure
- `dataset` is folder of raw dataset, `dataset/train` includes 5335 wav files, `dataset/test` includes 2947 wav files.
- `train_0.964.py` is python script for training the first step.
- `train_0.959.py` is python script for training the second step.
- `train_0.959_v2.py` is python script for training the last step.
- `saved_model` is folder of models from all three steps
- `result` is folder of submissions. The `result/Submission_cnn_0.962_from_0.959_v2.csv` represents my final submission (0.962 public LB)
- `inference.py`: If you solely wish to execute inference, run this file!


## Things I learnt from this competition
- I experimented various ways of feature engineering (MFCC, Logmels), but Logmels `n_mels=128` gave me the best results. Decreasing `n_mels` resulted in decreased accuracy.
- I tried `mobilenet_v4` but it not worked
- I also tried to develop my own CNN model, but the result is worsen.
- I explored other augmentation techniques like PitchShift, Add random noise but these did not improve results significantly.
- I used other ML training frameworks like Pytorch, huggingface, tf.keras, but none of them surpassed fastai

Thanks for this interesting challenge, This is my first time joining Zindi <3