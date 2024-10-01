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
    # Using mobilenet v3 small
    model = mobilenet_v3_small(pretrained=False, num_classes=dls.c)
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

    # merge train and test
    df_test_phase1 = pd.read_csv("result/Submission_cnn_0.96.csv")
    df_test_phase1 = df_test_phase1[['id', 'class']]
    df_test_phase1['id'] = df_test_phase1['id'].apply(lambda x: os.path.join("dataset/test", f"{x}.wav"))
    df = pd.concat([df, df_test_phase1])


    DBMelSpec = SpectrogramTransformer(mel=True, to_db=True)
    aud2spec = DBMelSpec(n_mels=128, f_max=10000, n_fft=1024, hop_length=128, top_db=100)
    aud_digit = DataBlock(blocks=(AudioBlock(force_mono=True),
                                  CategoryBlock),
                          splitter=TrainTestSplitter(0.2,
                                                     stratify=df["class"],
                                                     shuffle=True),
                          get_x=ColReader(0),
                          get_y=ColReader(1),
                          item_tfms=[
                                  RemoveSilence(),
                                  ResizeSignal(2000),
                                  aud2spec,
                                  MaskFreq(size=10),
                                  Delta(),
                              ]
                          )
    dls = aud_digit.dataloaders(df, bs=128, num_workers=4)



    learn = load_learner('saved_model/model_0.959.pkl')

    torch.save(learn.model.state_dict(), 'saved_model/model_0.959.pt')

    model = get_model(dls)
    model.load_state_dict(torch.load(f"saved_model/model_0.959.pt"))
    learn = Learner(dls, model,
                    metrics=[accuracy],
                    loss_func=CrossEntropyLossFlat())
    learn.fine_tune(epochs=30,
                    cbs=SaveModelCallback(monitor='valid_loss'))
    test_dl = dls.test_dl(df_test)
    preds, _ = learn.tta(dl=test_dl)
    preds = np.argmax(preds, axis=1)
    df_test["class"] = [classes[x] for x in preds]
    df_test["id"] = df_test["id"].apply(lambda x: x[13:-4])
    df_test.to_csv("result/Submission_cnn_0.962_from_0.959_v2.csv", index=False)

    learn.export('saved_model/model_0.962_from_0.959_v2.pkl')







