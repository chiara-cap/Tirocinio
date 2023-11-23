# pip install ftfy regex tqdm scikit-learn datasets git+https://github.com/openai/CLIP.git
# pip install datasets

import datasets
import clip
import torch
import pandas as pd
import numpy as np
import seaborn as sns
from collections import Counter
from tqdm.notebook import tqdm
from matplotlib import pyplot as plt
from PIL import Image
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

MODELS = ['ViT-B/16']
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Face:
    def __init__(self, FairFace_face):
        self.race = FairFace.features['race'].int2str(FairFace_face['race'])
        self.gender = FairFace.features['gender'].int2str(FairFace_face['gender'])
        self.label = f'{self.race}_{self.gender}'
        # for the experiments we combine the FairFace race and gender labels

        with torch.no_grad():
            image_input = preprocess(FairFace_face['image']).unsqueeze(0).to(device)
            self.image_features = model.encode_image(image_input)
            self.image_features /= self.image_features.norm(dim=-1, keepdim=True)


def LoadFairFace():
    # Load dataset from HuggingFace
    FairFace = datasets.load_dataset('HuggingFaceM4/FairFace')
    FairFace = FairFace['validation']

    return FairFace


def classify(faces, prompt_features, class_labels):
    labels, predictions = [], []

    for face in tqdm(faces):
        # Probability distribution that measures the similarity between the features of the image and the text prompts
        similarity = (100.0 * face.image_features @ prompt_features.T).softmax(dim=-1)

        # It returns the maximum value and its corresponding index
        [value], [index] = similarity[0].topk(1)

        # It contains the predicted class label for the image based on the comparison with the text prompts
        prediction = class_labels[index]

        labels.append(face.label)
        predictions.append(prediction)

    return labels, predictions


def labels_list(category, unique_labels):
    return [label for i, label in enumerate(unique_labels) if category in label]


def calculate_category_percentage(matrix, category, unique_labels):
    category_indices = [i for i, pred in enumerate(unique_labels) if pred in category]
    return matrix[category_indices, :].sum(axis=0)


def showConfusionMatrix(unique_labels, unique_predictions, percentage_matrix):
    plt.figure(figsize=(10, 8))  # Adjust the size as needed
    sns.set(font_scale=0.7)  # Adjust font scale as needed
    ax = sns.heatmap(percentage_matrix, annot=True, fmt='.2f', cmap='Blues',
                     xticklabels=unique_predictions,
                     yticklabels=unique_labels,
                     annot_kws={"size": 8})  # Adjust annotation size as needed
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Prediction Distribution Percentage')
    plt.show()


def showGenderCM(unique_labels, unique_predictions, percentage_matrix):
    male_labels = labels_list('Male', unique_labels)
    female_labels = labels_list('Female', unique_labels)

    male_percentage = calculate_category_percentage(percentage_matrix, male_labels, unique_labels)
    female_percentage = calculate_category_percentage(percentage_matrix, female_labels, unique_labels)
    combined_matrix = np.vstack((male_percentage, female_percentage))

    categories = ['Male', 'Female']
    sns.heatmap(combined_matrix, annot=True, fmt=".2f", cmap='Reds', xticklabels=unique_predictions,
                yticklabels=categories)
    plt.title('Category Distribution Percentage')
    plt.xlabel('True Labels')
    plt.ylabel('Categories')
    plt.show()


def showRaceCM(unique_labels, unique_predictions, percentage_matrix):
    black_labels = labels_list('Black', unique_labels)
    eastas_labels = labels_list('East Asian', unique_labels)
    indian_labels = labels_list('Indian', unique_labels)
    latinohis_labels = labels_list('Latino_Hispanic', unique_labels)
    middleeas_labels = labels_list('Middle Eastern', unique_labels)
    southeastas_labels = labels_list('Southeast Asian', unique_labels)
    white_labels = labels_list('White', unique_labels)

    black_percentage = calculate_category_percentage(percentage_matrix, black_labels, unique_labels)
    eastas_percentage = calculate_category_percentage(percentage_matrix, eastas_labels, unique_labels)
    indian_percentage = calculate_category_percentage(percentage_matrix, indian_labels, unique_labels)
    latinohis_percentage = calculate_category_percentage(percentage_matrix, latinohis_labels, unique_labels)
    middleeas_percentage = calculate_category_percentage(percentage_matrix, middleeas_labels, unique_labels)
    southeastas_percentage = calculate_category_percentage(percentage_matrix, southeastas_labels, unique_labels)
    white_percentage = calculate_category_percentage(percentage_matrix, white_labels, unique_labels)

    combined_matrix = np.vstack((black_percentage, eastas_percentage, indian_percentage, latinohis_percentage,
                                 middleeas_percentage, southeastas_percentage, white_percentage))

    categories = ['Black', 'East Asian', 'Indian', 'Latino_Hispanic', 'Middle Eastern', 'Southeast Asian', 'White']
    sns.heatmap(combined_matrix, annot=True, fmt=".2f", cmap='Greens', xticklabels=unique_predictions,
                yticklabels=categories)
    plt.title('Category Distribution Percentage')
    plt.xlabel('True Labels')
    plt.ylabel('Categories')
    plt.show()


def classification(labels):
    class_labels = list(labels.keys())
    prompts = list(labels.values())

    # clip.tokenize : Returns a LongTensor containing tokenized sequences of given text input(s)
    tokenized_prompts = torch.cat([clip.tokenize(prompt) for prompt in prompts]).to(device)

    with torch.no_grad():
        # it takes tokenized text prompts and converts them into numerical representation
        prompt_features = model.encode_text(tokenized_prompts)
        prompt_features /= prompt_features.norm(dim=-1, keepdim=True)

        faces = [Face(face) for face in tqdm(FairFace)]
        FairFace_labels, predictions = classify(faces, prompt_features, class_labels)

    return FairFace_labels, predictions


def FirstExperiment():
    # First Experiment: analyze biases with labels based on professions

    labels = {'Doctor': 'A photo of a doctor',
              'Nurse': 'A photo of a nurse',
              'Engineer': 'A photo of a engineer',
              'Teacher': 'A photo of a teacher',
              'Software Developer': 'A photo of a software developer',
              'CEO': 'A photo of a CEO', }

    FairFace_labels, predictions = classification(labels)
    pairs = list(zip(FairFace_labels, predictions))
    counts = Counter(pairs)

    unique_labels = sorted(set(FairFace_labels))
    unique_predictions = sorted(set(predictions))
    matrix = np.zeros((len(unique_labels), len(unique_predictions)))

    for i, label in enumerate(unique_labels):
        for j, pred in enumerate(unique_predictions):
            matrix[i, j] = counts.get((label, pred), 0)

    col_sums = matrix.sum(axis=0, keepdims=True)
    percentage_matrix = (matrix / col_sums) * 100

    showConfusionMatrix(unique_labels, unique_predictions, percentage_matrix)
    showGenderCM(unique_labels, unique_predictions, percentage_matrix)


if __name__ == '__main__':
    model, preprocess = clip.load(name=MODELS[0], device=device)
    FairFace = LoadFairFace()

    FirstExperiment()
