# pip install ftfy regex tqdm scikit-learn datasets git+https://github.com/openai/CLIP.git
# pip install datasets

import datasets
import clip
import torch
import pandas as pd
from tqdm.notebook import tqdm
from matplotlib import pyplot as plt
from PIL import Image
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
