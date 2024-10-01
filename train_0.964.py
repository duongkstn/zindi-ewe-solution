from fastai.vision.all import *
from fastaudio.core.all import *
from fastaudio.augment.all import *
import numpy as np
import torch
import random
import pandas as pd
from efficientnet_pytorch import EfficientNet
import torch.nn as nn
import matplotlib.pyplot as plt


def random_seed(seed_value):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_model(dls):
    # Using efficientnet b0
    model = EfficientNet.from_name('efficientnet-b0')
    model._fc = nn.Linear(model._fc.in_features, dls.c)
    model._conv_stem.in_channels = 1
    model._conv_stem.weight = nn.Parameter(model._conv_stem.weight[:,1,:,:].unsqueeze(1))
    return model

if __name__ == "__main__":
    random_seed(42)
    df = pd.read_csv(f"dataset/Train.csv")
    classes = sorted(list(set(df['class'].values.tolist())))
    class2id = {x: i for i, x in enumerate(classes)}
    num_classes = len(classes)
    df = df[['id', 'class']]
    df['id'] = df['id'].apply(lambda x: os.path.join("dataset/train", f"{x}.wav"))

    df_test = pd.read_csv(f"dataset/SampleSubmission.csv")
    df_test = df_test[['id']]
    df_test['id'] = df_test['id'].apply(lambda x: os.path.join("dataset/test", f"{x}.wav"))


    DBMelSpec = SpectrogramTransformer(mel=True, to_db=True)
    aud2spec = DBMelSpec(n_mels=128, f_max=10000, n_fft=1024, hop_length=128, top_db=100)
    aud_digit = DataBlock(blocks=(AudioBlock(force_mono=True),
                                  CategoryBlock),
                          splitter=TrainTestSplitter(0.05,
                                                     stratify=df["class"],
                                                     shuffle=True),
                          get_x=ColReader(0),
                          get_y=ColReader(1),
                          item_tfms=[
                                  RemoveSilence(),
                                  ResizeSignal(2000),
                                  aud2spec,
                                  MaskFreq(size=10)
                              ]
                          )
    dls = aud_digit.dataloaders(df, bs=128, num_workers=4)

    learn = Learner(dls, get_model(dls),
                    metrics=[accuracy],
                    loss_func=CrossEntropyLossFlat())
    learn.lr_find()
    learn.fine_tune(epochs=30,
                    cbs=SaveModelCallback(monitor='valid_loss'))
    test_dl = dls.test_dl(df_test)
    preds, _ = learn.tta(dl=test_dl)
    preds = np.argmax(preds, axis=1)
    df_test["class"] = [classes[x] for x in preds]
    df_test["id"] = df_test["id"].apply(lambda x: x[13:-4])
    df_test.to_csv("result/Submission_cnn_0.964.csv", index=False)

    learn.export('saved_model/model_0.964.pkl')







