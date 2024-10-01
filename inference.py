from fastai.vision.all import *
import numpy as np
import torch
import random
import pandas as pd
import torch.nn as nn
import time


def random_seed(seed_value):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_model(dls):
    model = mobilenet_v3_small(pretrained=False, num_classes=dls.c)
    return model

if __name__ == "__main__":
    start = time.time()
    random_seed(42)
    classes = ['down', 'go', 'left', 'no', 'right', 'stop', 'up', 'yes']

    df_test = pd.read_csv(f"dataset/SampleSubmission.csv")
    df_test = df_test[['id']]
    df_test['id'] = df_test['id'].apply(lambda x: os.path.join("dataset/test", f"{x}.wav"))

    learn = load_learner('saved_model/model_0.962_from_0.959_v2.pkl')
    test_dl = learn.dls.test_dl(df_test)
    preds, _ = learn.tta(dl=test_dl)
    preds = np.argmax(preds, axis=1)
    df_test["class"] = [classes[x] for x in preds]
    df_test["id"] = df_test["id"].apply(lambda x: x[13:-4])
    df_test.to_csv("result/Submission_cnn.csv", index=False)
    print(time.time() - start)







