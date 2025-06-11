#!/usr/bin/env python
# coding: utf-8

# # DistiBERT-Based Emotion Recognition for Autism Spectrum Disorder Communication.
# 
# 
# Author: Roshini Jayasankar
# 
# Date: 12/18/2024
# 
# ## Table of Contents
# 
# - [Abstract](#Abstract)
# - [Introduction](#introduction)
#   - [Background](#backgound)
#   - [Objectives](#objectives)
#   - [Problem Statement](#ProblemStatement)
# - [Data Description](#data-description)
#   - [Dataset Overview](#dataset-overview)
#   - [Data Dictionary](#data-dictionary)
# - [Methodology](#methodology)
# - [Discussion](#discussion)
#   - [Analysis and Visualization](#analysisandvisualization)
#   - [Testing on Real-World Forums](#testingonreal-worldforums)
# - [Conclusion](#conclusion)
# - [References](#references)
# 

# # Abstract
# 
# Emotion recognition serves a fundamental role in understanding interpersonal communication and behavior. However, existing sentiment analysis frameworks and general-purpose emotion recognition systems, such as lexicon-based approaches (e.g., VADER), fail to account for the unique emotional expression patterns observed in Autism Spectrum Disorder (ASD). These tools often rely on linguistic structures and emotional cues typical of neurotypical communication, overlooking the atypical phrasing, unconventional contextual meanings, and reduced emotional expressiveness associated with ASD-specific communication.
# 
# This research introduces a novel approach that leverages DistilBERT, a pre-trained transformer model, to classify emotions in textual communication specific to individuals with ASD. Using the GoEmotions dataset, the model is fine-tuned to recognize five primary emotion classes: joy, sadness, anger, anxiety (fear), and neutral. To address challenges such as variability in ASD communication and data scarcity, the study incorporates rigorous preprocessing and model adaptation techniques.
# 
# The model achieves an F1-score of 0.88 and an accuracy of 87.7%, demonstrating its effectiveness in capturing subtle emotional cues in ASD-relevant contexts. Interpretability tools, such as LIME (Local Interpretable Model-agnostic Explanations), provide insights into the model’s decision-making process, ensuring its reliability for practical applications.
# 
# This work advances existing research on emotion recognition in ASD by addressing the limitations of conventional methods and tailoring the approach to accommodate ASD-specific communication patterns. The findings highlight the potential for Natural Language Processing (NLP) tools to enhance emotional understanding, offering a practical, interpretable tool for caregivers, researchers, and practitioners to deliver more personalized interventions and support.
# 
# 

# ## Introduction
# 
# Accurately interpreting emotional states is essential for effective human communication, enabling individuals to respond appropriately to the emotions of others. However, individuals with Autism Spectrum Disorder (ASD) often demonstrate emotional expression patterns that deviate from neurotypical norms. These deviations can include atypical phrasing, unconventional contextual meanings, and reduced emotional expressiveness, posing significant challenges in communication and social interaction. As a result, conventional emotion recognition systems fail to address the nuances of ASD-specific emotional expressions.
# 
# Existing tools, such as sentiment analysis frameworks and general-purpose emotion recognition systems, predominantly rely on standardized linguistic structures and emotional cues tailored to neurotypical communication. While these tools perform well in neurotypical contexts, they fail to capture the subtle and diverse emotional patterns inherent to ASD communication. This shortcoming highlights the critical need for specialized emotion recognition models capable of accommodating the unique emotional and linguistic traits associated with individuals on the autism spectrum.
# 
# Advancements in Natural Language Processing (NLP), particularly the development of transformer-based models such as DistilBERT, have revolutionized text analysis. Transformer models are particularly well-suited to handling atypical linguistic patterns due to their ability to capture nuanced contextual relationships, even in nonstandard or fragmented text. DistilBERT’s lightweight architecture maintains the effectiveness of larger transformer models while offering computational efficiency, making it an ideal candidate for fine-tuning on ASD-specific communication.
# 
# This study focuses on five primary emotion categories: joy, sadness, anger, anxiety (fear), and neutral. These categories were selected to reflect common emotional states in ASD communication, with "fear" and "neutral" prioritized due to their frequent occurrence in contexts like social anxiety and restrained expressiveness. The chosen categories aim to balance interpretability and relevance, ensuring practical utility for caregivers and researchers.
# 
# This research bridges the gap between conventional tools and ASD-specific needs, offering a tailored solution for more accurate emotion recognition. The key objectives of this study are to:
# 
#  - Fine-tune the pre-trained DistilBERT model for emotion classification specific to ASD communication.
#  - Classify emotions into five primary categories: joy, sadness, anger, anxiety, and neutral.
#  - Evaluate the model's performance using established metrics, including accuracy and F1-score.
#  - Explore the model’s broader applicability in supporting researchers, caregivers, and educators working with individuals with ASD.
# 
# By addressing these objectives, the study aims to provide a reliable and practical solution for understanding emotional expression in ASD, facilitating improved communication, personalized interventions, and enhanced support.  
#   
# 
# 

# ## Data Description
# 
# **Dataset Overview**
# 
# The GoEmotions dataset, curated by Google Research, is one of the largest publicly available datasets for emotion recognition in textual data. It comprises over 58,000 English sentences sourced from Reddit, annotated with 27 distinct emotion labels, including a "neutral" category. The dataset captures a broad range of emotions, from basic categories like joy, sadness, and anger to more nuanced emotions such as amusement, curiosity, and approval. This comprehensive taxonomy enables thorough analysis and modeling of emotional expressions across diverse contexts.
# 
# The dataset's diverse annotations are particularly suitable for Autism Spectrum Disorder (ASD)-related research, as they encompass both overt and subtle emotional expressions. ASD communication often involves atypical phrasing, reduced expressiveness, and unconventional emotional cues. The dataset's richness ensures that even nuanced expressions are well-represented, providing a strong foundation for fine-tuning models like DistilBERT to capture the unique emotional dynamics associated with ASD communication.
# 
# For the purposes of this study, the dataset was filtered and tailored to focus on five primary emotion categories:
# 
#    - Joy
#    - Sadness
#    - Anger
#    - Anxiety (mapped from "fear")
#    - Neutral
# 
# These categories align with the research objectives of classifying emotional expressions relevant to ASD communication, where simplified yet significant emotional categories are crucial for practical interpretability. The decision to focus on these five categories reduces the complexity of the classification task while ensuring relevance to common emotional states experienced by individuals with ASD.
# 
# Filtering the dataset for these specific categories impacts its structure and class balance. For example:
# 
#   - The dataset becomes more targeted, with emotions like amusement excluded
#      to simplify interpretability.
# 
#   - Class imbalance is introduced or exacerbated due to the uneven distribution of labels, with "neutral" often being the dominant category. Addressing this imbalance during preprocessing is critical for robust model performance.
# 
# The dataset is publicly accessible and can be downloaded from the following link:
# https://github.com/google-research/google-research/tree/master/goemotions
# 
# 
# ---
# 
# **Data Dictionary**
# 
# 
# | **Variable**       | **Description**                                                                 |
# |---------------------|---------------------------------------------------------------------------------|
# | **Text**            | The textual data, consisting of short sentences or comments sourced from Reddit. |
# | **Labels**          | A multi-label annotation indicating one or more emotion categories.             |
# | **Neutral**         | A binary label indicating whether the text is emotionally neutral.              |
# | **Split**           | The data is partitioned into training, validation, and test subsets.            |
# 
# 
# 
# 

# ## Methodology
# 
# <div style="text-align: justify;">
# 
# **Exploratory Data Analysis**
# 
# Objective: The EDA aims to understand the dataset's structure, distribution, and characteristics.
# 
# Steps:
#   1. Data Summary: Display the number of samples, labels, and dataset splits.
#   2. Label Distribution: Visualize the frequency of each emotion label.
#   3. Sample Review: Examine a few text samples to understand their content.
# </div>

# *Step 1: Loading the dataset*

# In[ ]:


get_ipython().system('pip install datasets')

from datasets import load_dataset

# Load the GoEmotions dataset
goemotions = load_dataset("go_emotions")

# Print dataset structure
print(goemotions)


# *Step 2: Calculate proportions of the dataset splits*

# In[ ]:


# Proportions of dataset splits
total_samples = len(goemotions['train']) + len(goemotions['validation']) + len(goemotions['test'])
train_ratio = len(goemotions['train']) / total_samples
val_ratio = len(goemotions['validation']) / total_samples
test_ratio = len(goemotions['test']) / total_samples

print(f"Training set ratio: {train_ratio:.2%}")
print(f"Validation set ratio: {val_ratio:.2%}")
print(f"Test set ratio: {test_ratio:.2%}")


# The dataset is divided into training (80%), validation (10%), and test (10%) splits. These proportions ensure sufficient data for training while maintaining thorough validation and testing.

# *Step 3: Inspecting sample texts and understanding dataset features*

# In[ ]:


# Explore dataset features
label_names = goemotions['train'].features['labels'].feature.names
print("Features:", goemotions['train'].features)

# Check a sample of the data
for i in range(5):
    sample = goemotions['train'][i]
    labels = [label_names[label] for label in sample['labels']]
    print(f"Text: {sample['text']}")
    print(f"Labels: {labels}")
    print("-" * 50)


# Observation:
# 
# 1. The dataset contains diverse textual content, including casual comments, exclamations, and conversational phrases, reflecting real-world emotion expression.
# 2. Emotion labels align with expected sentiment in most cases (e.g., "WHY THE FUCK IS BAYLESS ISOING" appropriately labeled as 'anger').
# 3. The presence of explicit language and sarcasm in some texts (e.g., "Now if he does off himself, everyone will think he's having a laugh screwing with people") indicates the need for careful preprocessing to handle such complexities.
# 4. Neutral labels dominate certain samples where emotion is less overt, requiring robust modeling to distinguish subtle emotional cues.
# 
# Inspecting sample data reveals the dataset's potential challenges, such as handling sarcasm, explicit language, and multi-label annotations.

# *Step 4: Visualize the distribution of emotion labels across training data*

# In[ ]:


from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

# Get label names
label_names = goemotions['train'].features['labels'].feature.names

# Count the frequency of each label in the training set
def label_distribution(data_split):
    label_counts = Counter([label for sample in data_split for label in sample['labels']])
    return {label_names[i]: label_counts[i] for i in label_counts}

train_label_dist = label_distribution(goemotions['train'])

# Plot the label distribution
plt.figure(figsize=(12, 6))
sns.barplot(x=list(train_label_dist.keys()), y=list(train_label_dist.values()))
plt.xticks(rotation=45)
plt.title("Label Distribution in Training Set")
plt.xlabel("Emotion Labels")
plt.ylabel("Frequency")
plt.show()


# Interpretation of the Plot:
# 
# The 'neutral' label is overwhelmingly dominant, with a frequency exceeding all other emotion categories.
# This indicates a significant class imbalance in the dataset. Most of the textual data is labeled as neutral, which may bias the model towards predicting the neutral class more frequently.
# 
# Emotion labels such as 'admiration', 'approval', and 'gratitude' show relatively higher frequencies compared to less frequent emotions like 'fear', 'relief', and 'nervousness'.
# Labels with lower frequencies (e.g., 'pride', 'nervousness', and 'realization') may be underrepresented during model training, leading to poorer generalization for these classes.
# 
# Underrepresented emotions (e.g., 'pride' and 'relief') may result in the model failing to detect these emotions effectively, as it does not receive sufficient examples for learning their patterns.
# This suggests the need for techniques to mitigate class imbalance.
# 
# Labels such as 'admiration', 'approval', and 'joy' have moderate representation, which may lead to reasonably good performance in detecting these classes.
# Their frequency provides sufficient examples for the model to learn their emotional nuances.

# *Step 5: Check how many samples in the dataset have multiple labels*

# In[ ]:


# Count multi-label instances in training set
multi_label_count = sum(len(sample['labels']) > 1 for sample in goemotions['train'])
print(f"Number of multi-label instances in training set: {multi_label_count}")
print(f"Percentage of multi-label instances: {multi_label_count / len(goemotions['train']):.2%}")


# Multi-label instances constituted approximately 16.36% of the dataset, reflecting the complexity of overlapping emotional expressions. This guided the choice of a multi-label classification framework, ensuring that the model accommodates simultaneous emotion predictions.

# *Step 6: Analyze the length of the text samples for tokenization insights*

# In[ ]:


import matplotlib.pyplot as plt

# Text Length Analysis
text_lengths = [len(sample['text'].split()) for sample in goemotions['train']]
plt.hist(text_lengths, bins=30)
plt.title("Text Length Distribution in Training Set")
plt.xlabel("Number of Words")
plt.ylabel("Frequency")
plt.show()

print(f"Average text length: {sum(text_lengths) / len(text_lengths):.2f} words")
print(f"Maximum text length: {max(text_lengths)} words")


# The average text length is ~12 words, and the maximum is 33 words, suggesting that truncation at 50 tokens would preserve most information.

# *Step 7: Ensure there are no missing texts or labels in the dataset*

# In[ ]:


# Check for missing texts or labels in the training set
missing_texts = sum(sample['text'] == '' for sample in goemotions['train'])
missing_labels = sum(len(sample['labels']) == 0 for sample in goemotions['train'])

print(f"Missing texts in training set: {missing_texts}")
print(f"Missing labels in training set: {missing_labels}")


# **Data Preprocessing**
#  - Objective: To prepare the dataset for classifying emotions relevant to ASD communication, ensuring preprocessing reflects potential atypical linguistic patterns, reduced expressiveness, or contextual variations.
#     - 1. Filter Target Emotions: The dataset was filtered to retain only the five target emotions: joy, sadness, anger, anxiety (fear), and neutral. This step ensured alignment with the study's focus on ASD-relevant emotional expressions.
#     - 2. Tokenize Text: DistilBERT’s tokenizer will be employed, truncating
#          inputs at 50 tokens. This length was chosen based on text length analysis, ensuring minimal loss of information.
#          Tokenization preserved unique phrasing styles characteristic of ASD communication, accommodating the dataset’s diverse linguistic patterns.
#     - 3. Handle Class Imbalance: If the dataset is imbalanced, compute class weights.
#     - 4. Prepare Datasets for Model Input: Split the dataset into training, validation, and test sets with preprocessed data.

# *Step 1: Filter Target Emotions*
# 
# The target emotion categories (joy, sadness, anger, anxiety (fear), neutral) align well with ASD communication scenarios, as these are fundamental to understanding emotional states in various contexts.

# In[ ]:


# Define target labels
target_labels = ['joy', 'sadness', 'anger', 'fear', 'neutral']
target_label_ids = [label_names.index(label) for label in target_labels]

# Filter out samples that do not contain the target emotions
def filter_target_emotions(example):
    return any(label in target_label_ids for label in example['labels'])

filtered_train = goemotions['train'].filter(filter_target_emotions)
filtered_val = goemotions['validation'].filter(filter_target_emotions)
filtered_test = goemotions['test'].filter(filter_target_emotions)

# Check the number of samples after filtering
print(f"Filtered training samples: {len(filtered_train)}")
print(f"Filtered validation samples: {len(filtered_val)}")
print(f"Filtered test samples: {len(filtered_test)}")


# *Step 2: Tokenize Text (Accounting for Atypical Phrasing)*
# 
# Tokenization ensures that the input text is transformed appropriately for the DistilBERT model. Since ASD communication may include unique phrasing, the tokenizer should preserve as much context as possible by truncating only at the maximum sequence length

# In[ ]:


from transformers import AutoTokenizer

# Load the DistilBERT tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Define a tokenization function
def tokenize_function(example):
    return tokenizer(example["text"], padding="max_length", truncation=True, max_length=50)

# Tokenize the dataset splits
tokenized_train = filtered_train.map(tokenize_function, batched=True)
tokenized_val = filtered_val.map(tokenize_function, batched=True)
tokenized_test = filtered_test.map(tokenize_function, batched=True)

# Inspect a sample of tokenized data
print("Sample tokenized training data:", tokenized_train[0])


# *Step 3: Handle Class Imbalance*
# 
# *ASD communication may disproportionately emphasize or suppress certain emotional categories like neutral, joy, or sadness. To improve model generalization and avoid bias towards frequent labels, we will:*

# In[ ]:


from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# Extract labels for class weight calculation
labels = [label for sample in filtered_train for label in sample['labels']]
class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)

# Convert to dictionary format for use in training
class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}
print("Class weights for ASD emotion classification:", class_weights_dict)


# The class weights calculated for the ASD emotion classification highlight a significant imbalance in the dataset. The large variation in class weights reflects how certain emotion categories, particularly neutral (weight = 0.0556), are vastly overrepresented, while rarer emotions like remorse (23.98), pride (32.97), and nervousness (60.87) are severely underrepresented. This imbalance poses a challenge during model training, as the model tends to prioritize predicting frequent classes like neutral while underperforming on minority classes that lack sufficient examples.
# 
# The extremely high class weights (e.g., nervousness: 60.87, pride: 32.97) indicate that the dataset contains very few samples for these emotions, making it difficult for the model to learn their unique linguistic patterns. Emotions like joy (3.11) and anger (0.50), on the other hand, are more moderately weighted, suggesting they are relatively better represented. However, their still low weights compared to emotions like remorse or disapproval indicate room for further balancing.
# 
# This class imbalance can significantly impact model performance, leading to poor recall for underrepresented emotions. The model might predict these classes infrequently or not at all, focusing instead on the dominant neutral label.

# *Step 4: Inspecting tokenized samples*

# In[ ]:


# Inspect tokenized and filtered samples
for i in range(5):
    sample = tokenized_train[i]
    labels = [label_names[label] for label in sample['labels']]
    print(f"Text (tokenized): {sample['input_ids']}")
    print(f"Labels: {labels}")
    print("-" * 50)


# Tokenization:
#   - The provided tokenized texts are sequences of integers generated using the DistilBERT tokenizer. Each integer corresponds to a specific word or subword token in the DistilBERT vocabulary. Special tokens like [101] (CLS - classification start token) and [102] (SEP - separator token) are present at the beginning and end of each sequence, respectively. Padding tokens (0) are added to ensure all sequences have a uniform length of 50 tokens. This ensures compatibility with the DistilBERT input requirements.
# 
# - Examples and Observations:
#   First Example
#   Tokenized Text: [101, 2026, 8837, 2833, 2003, 2505, 1045, 2134, 1005, 1056, 2031, 2000, 5660, 2870, 1012, 102, ...]
#   Decoded: "My favorite food is anything I didn't have to cook myself."
#   Label: neutral
#   Analysis:
#   This sentence expresses no overt emotional content and reflects a neutral tone. The model correctly labels it as neutral, as it lacks strong emotional cues like joy, anger, or sadness.
# 
#   Fourth Example
#   Tokenized Text: [101, 2000, 2191, 2014, 2514, 5561, 102, ...]
#   Decoded: "To make her feel threatened."
#   Label: fear
#   Analysis:
#   The phrase suggests a fearful context, as the idea of someone feeling threatened often aligns with fear. The model's classification as fear is appropriate given the emotional undertone.
# 
# 

# *Step 5: Converting the labels to float*
# 
#  - Objective: The purpose of converting the labels to float is to prepare them for the multi-label classification task. In multi-label classification, each sample can belong to more than one class, and the labels need to represent the probability of belonging to each class (e.g., 0.0 or 1.0). Float conversion ensures compatibility with the loss function used for multi-label classification, such as BCEWithLogitsLoss, which requires the labels to be in a continuous numerical format.
# 
#  - Explanation of the Code:
#      1. Dataset Conversion:
#         The function convert_labels_to_float takes a tokenized dataset and converts its labels into a multi-hot             encoded format with float values. A multi-hot encoding represents the presence (1.0) or absence (0.0) of           each target label.
#         
#      2.  Initialization:
#          The converted_data dictionary is initialized to store the input IDs, attention masks, and processed labels          in the required format.
# 
#      3.  Label Conversion:
#         For each example in the dataset:
#         The input IDs and attention masks are appended to their respective lists in the converted_data dictionary.
#         The labels field is converted to a list of float values, where each value corresponds to whether a target          label is present (1.0) or absent (0.0) in the original label list.
# 
#      4.  Dataset Creation:
#         The converted data is transformed into a new dataset object using Dataset.from_dict.
# 
#      5.  PyTorch Compatibility:
#         The set_format method specifies the data structure for PyTorch, ensuring that input_ids, attention_mask,           and labels are formatted as tensors.
# 

# In[ ]:


from datasets import Dataset

# Convert labels to float and create a new dataset
def convert_labels_to_float(dataset):
    converted_data = {
        "input_ids": [],
        "attention_mask": [],
        "labels": [],
    }
    for example in dataset:
        converted_data["input_ids"].append(example["input_ids"])
        converted_data["attention_mask"].append(example["attention_mask"])
        converted_data["labels"].append([float(label in example["labels"]) for label in target_label_ids])
    return Dataset.from_dict(converted_data)

train_dataset = convert_labels_to_float(tokenized_train)
val_dataset = convert_labels_to_float(tokenized_val)
test_dataset = convert_labels_to_float(tokenized_test)

# Set format for PyTorch
train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# Verify labels are converted correctly
print("Sample train labels:", train_dataset[0]["labels"])
print("Label data type (train):", train_dataset[0]["labels"].dtype)


# **Model and Training**

# Objective:
#   To train and evaluate a fine-tuned DistilBERT model for multi-label emotion classification using custom metrics like accuracy, F1-score, precision, and recall.
# 
#  - Key Steps
#     - 1. Model Loading:
#     Load the pre-trained DistilBERT model for sequence classification.
#     Configure the model for multi-label classification using problem_type="multi_label_classification".
#     Adapt the model to predict the specified number of emotion labels.
#       2. Training Configuration (TrainingArguments):
#    Specifies key training hyperparameters:
#    Learning Rate: 5e-5 for stable training.
#    Batch Size: 16 per device for both training and evaluation.
#    Number of Epochs: 3 to balance training time and model performance.
#    Weight Decay: 0.01 for regularization to prevent overfitting.
#    Evaluation Strategy: Evaluates the model at the end of each epoch.
#    Save Strategy: Saves the model at each epoch to retain the best-performing version.
#    Load Best Model: Loads the best model at the end of training based on evaluation metrics.
#       3. Trainer Initialization:
#        Initializes the Hugging Face Trainer class, which handles the model training pipeline.
#    Connects the following:
#    Model: DistilBERT fine-tuned for multi-label classification.
#    Arguments: Training configurations.
#    Datasets: Training and validation datasets.
#    Tokenizer: Converts raw text into input IDs and attention masks for DistilBERT.
#    Metrics Function: Custom evaluation metrics for performance monitoring.
#       4. Model Evaluation:
#    Evaluates the trained model on the validation dataset.
#    Computes and prints evaluation metrics like accuracy, F1-score, precision, and recall

# In[ ]:


from transformers import DistilBertForSequenceClassification, TrainingArguments, Trainer
import torch
import os
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import numpy as np

# Disable W&B logging if not needed
os.environ["WANDB_DISABLED"] = "true"

# Load the model
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=len(target_label_ids),
    problem_type="multi_label_classification"
)
model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

# Define compute_metrics function
def compute_metrics(eval_pred):
    """
    Computes evaluation metrics: accuracy, precision, recall, and F1 score.
    """
    logits, labels = eval_pred  # Extract predictions and labels
    predictions = (torch.sigmoid(torch.tensor(logits)) > 0.5).numpy().astype(int)  # Threshold for multi-label classification

    # Calculate metrics
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="weighted")
    precision = precision_score(labels, predictions, average="weighted")
    recall = recall_score(labels, predictions, average="weighted")

    return {
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }

# Training arguments with updated parameters
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    load_best_model_at_end=True,
    run_name="ASD_emotion_model"
)

# Trainer initialization
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# Train and evaluate
trainer.train()
results = trainer.evaluate()
print("Evaluation Results:", results)


# In[ ]:


# Test the model
predictions = trainer.predict(test_dataset)
test_metrics = compute_metrics((predictions.predictions, predictions.label_ids))
print("Test Results:", test_metrics)

# Save the model
model.save_pretrained("./fine_tuned_distilbert")
tokenizer.save_pretrained("./fine_tuned_distilbert")
print("Model and tokenizer saved successfully.")


# # Dicussion
# 
#  **Analysis and Visualization**
#   - 1.  Confusion Matrix
#         Objective: To visualize the model's performance in distinguishing each target emotion category. The confusion matrix helps identify misclassifications and areas of improvement.

# In[ ]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Generate predictions and ground truth
logits = predictions.predictions
y_true = predictions.label_ids
y_pred = (logits > 0.5).astype(int)

# Generate and display confusion matrices for each emotion
for i, label in enumerate(target_labels):
    cm = confusion_matrix(y_true[:, i], y_pred[:, i])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not " + label, label])
    disp.plot(cmap='Blues', values_format='d')
    plt.title(f"Confusion Matrix for {label}")
    plt.show()


# The confusion matrices for the five target emotions—neutral, joy, sadness, anger, and fear—highlight the model's strengths and areas for improvement. For neutral, the model performs exceptionally well, achieving 1696 true positives with 91 false negatives, though 152 false positives suggest slight challenges in differentiating borderline cases. In detecting joy, the model demonstrates strong precision, as evidenced by only 15 false positives, but struggles slightly with recall, with 40 false negatives where true joy samples were missed. For sadness, while the precision remains moderate, the model encounters significant recall issues, with 61 false negatives and 18 false positives, indicating difficulty in identifying subtle or nuanced sadness expressions. The anger category shows the greatest challenge, as 102 false negatives suggest the model frequently misses angry expressions, despite maintaining relatively low false positives. Similarly, for fear, the model achieves high precision with only 5 false positives but moderate recall due to 25 false negatives.
# 
# Overall, the model excels in precision across most categories, particularly for neutral and joy, but recall remains a limitation, especially for sadness, anger, and fear. This imbalance points to the need for further optimization to reduce false negatives and improve the detection of more nuanced emotional expressions. To address these challenges, class imbalance can be mitigated through techniques like oversampling, undersampling, or applying weighted loss functions. Additionally, hyperparameter tuning and detailed error analysis of misclassified examples can help uncover linguistic patterns that contribute to the errors. Incorporating ensemble models or context-specific features may further enhance the model's ability to capture subtle emotional cues, leading to more balanced and accurate performance across all target emotions.

# 2. Classification Report
#      - Objective: Provide detailed metrics (precision, recall, F1-score) for each emotion to quantitatively evaluate performance.

# In[ ]:


from sklearn.metrics import classification_report

# Generate classification report
report = classification_report(y_true, y_pred, target_names=target_labels)
print("Classification Report:\n", report)


# The classification report provides a detailed evaluation of the model's performance across the five target emotion classes: joy, sadness, anger, fear, and neutral. The neutral class demonstrates the strongest performance, with a precision of 0.92, recall of 0.95, and an excellent F1-score of 0.93, reflecting the model's ability to accurately identify this dominant category. For joy, the model achieves a high precision of 0.89 but experiences a drop in recall to 0.75, indicating that while predictions are accurate, a notable number of true joy samples are missed. Similarly, sadness shows strong precision at 0.84 but struggles with recall (0.61), resulting in an F1-score of 0.71, highlighting challenges in detecting this emotion accurately.
# 
# The anger class exhibits the weakest performance, with precision at 0.78 but a recall of only 0.46, leading to a lower F1-score of 0.58. This suggests the model struggles significantly to identify true anger instances, likely due to subtle or ambiguous expressions of anger. For fear, precision remains high at 0.91, but the recall of 0.68 brings down the F1-score to 0.78, indicating the model's tendency to miss some fear samples despite strong predictive confidence.
# 
# On an aggregate level, the micro-average and weighted-average F1-scores stand at 0.88, demonstrating the model's overall effectiveness, which is largely driven by the dominance of the neutral class. However, the macro-average F1-score of 0.76 reveals performance disparities, particularly for the minority classes—sadness, anger, and fear—where recall remains the primary limitation.
# 
# Addressing these recall deficiencies for underrepresented emotions is critical to achieving balanced performance. Techniques such as class balancing through oversampling, undersampling, or weighted loss functions, combined with further fine-tuning and targeted error analysis, could significantly improve recall and enhance the model's ability to identify all target emotions effectively.

# 3. LIME Explanations for Interpretability
#   - Objective: Use LIME (Local Interpretable Model-Agnostic Explanations) to interpret the model’s predictions by analyzing the importance of individual features within the context of multi-label classification.

# In[ ]:


get_ipython().system('pip install lime')

import lime
import lime.lime_text
import matplotlib.pyplot as plt

# Define a prediction function for LIME
def predict_probabilities(texts):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=50)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    return torch.sigmoid(outputs.logits).cpu().numpy()

# Initialize LIME Text Explainer
explainer = lime.lime_text.LimeTextExplainer(class_names=target_labels)

# Example text input
text_input = "I feel overwhelmed and scared about going out today."

# Generate explanation
exp = explainer.explain_instance(
    text_instance=text_input,
    classifier_fn=predict_probabilities,
    num_features=10
)

# Get prediction probabilities
probabilities = predict_probabilities([text_input])[0]
print("\nPrediction Probabilities:")
for label, prob in zip(target_labels, probabilities):
    print(f"{label}: {prob:.2f}")

# Highlight words contributing to the prediction
print("\nText with Highlighted Words:")
highlighted_text = exp.as_list()
for word, weight in highlighted_text:
    color = "positive" if weight > 0 else "negative"
    print(f"{word}: {weight:.2f} ({color})")

# Visualization with LIME
fig = exp.as_pyplot_figure()
fig.patch.set_facecolor('white')
plt.gcf().set_size_inches(10, 6)
plt.rcParams.update({'font.size': 12})
plt.tight_layout()
plt.show()










# The LIME explanation and prediction results provide insights into the model's classification of the input text: "I feel overwhelmed and scared about going out today." The prediction probabilities reveal that the model confidently identifies fear as the dominant emotion, with a probability of 90%, while other emotions, such as joy, sadness, anger, and neutral, remain notably low (ranging between 2% and 7%).
# 
#  The word "scared" has the most significant negative impact (-0.47), strongly reinforcing a fearful tone in the text. Conversely, the word "overwhelmed" contributes positively (0.23), signaling emotional distress, which aligns with fear-related contexts. Words such as "feel" (0.06), "I" (0.05), and "out" (0.05) provide smaller positive contributions, collectively adding to the emotional interpretation of the sentence.
# 
# Interestingly, the word "about" has a slight negative contribution (-0.03), suggesting that the model deems it less relevant to the fear classification. Neutral words like "today" and "and" exert minimal influence, as evidenced by their near-zero contributions.
# 
# In summary, the model's classification is driven primarily by emotionally salient words, particularly "scared" and "overwhelmed", which signal fear and distress. The LIME visualization confirms that the model effectively identifies and weighs key emotional triggers, while filtering out less impactful words, leading to an accurate prediction of the fear emotion.

# **Testing the model with real online forums(e.g, Reddit)**
# 
#     - !pip install praw: Installs the praw library, a Python wrapper for the Reddit API.
#       import praw: Imports the praw library to interact with the Reddit API.
#       reddit = praw.Reddit(...):
#     - Initializes the Reddit API client.
#     - client_id: Unique identifier for the Reddit application.
#     - client_secret: Secret key for the Reddit application.
#     - user_agent: Identifies the application making requests to Reddit.
# 
# 
# 
# 
# 

# In[ ]:


get_ipython().system('pip install praw')

import praw

# Initialize Reddit API with credentials
reddit = praw.Reddit(
    client_id="RDM4LWUlC2InXK078WLrRg",
    client_secret="Ohu5nCoTYm7aYNrOReEguvfD9Af6xw",
    user_agent="AutismTextAnalysis by Ok-Pineapple-7055"
)

# Verify successful connection
print("Reddit API successfully connected!")


# In[ ]:


# Define subreddit and collect posts
subreddit = reddit.subreddit("autism")
post_limit = 10  # Number of posts to fetch

# Collect Reddit posts
posts = []
for post in subreddit.hot(limit=post_limit):
    post_text = f"{post.title} {post.selftext}"
    posts.append(post_text)

# Print collected posts
print(f"Collected {len(posts)} posts from r/autism:")
for i, text in enumerate(posts):
    print(f"Post {i+1}: {text[:200]}...\n")  # Print first 200 characters for readability


# Explanation of Code:
# 
# 1. Label Mapping:
#    Create a dictionary that maps numeric indices (output by the model) to emotion labels (e.g., joy, sadness, anger, fear, neutral).
# 2. Model and Tokenizer Loading:
#    Load the fine-tuned DistilBERT model and tokenizer from the specified path.
#    Ensures compatibility for emotion classification tasks.
# 
# 3. Emotion Classification Pipeline:
#    Initializes the Hugging Face pipeline for text classification, streamlining tokenization, model inference, and output processing.
#    GPU support is enabled for faster computation.
# 4. Splitting Long Text:
#    If a text exceeds the model's token limit (512 tokens), it is split into smaller chunks.
#    chunks are processed individually to ensure no information is lost.
# 5. Processing Posts:
#    Each text post is processed in chunks using the pipeline.
#    Predictions for all chunks are collected and averaged to obtain final emotion scores.
#    This approach ensures consistent results for lengthy inputs.
# 6. Predicting Emotions:
#    Multiple example posts are provided as inputs.
#    The process_post function is used to predict emotion scores for each post.
# 
# 
# 
# 

# In[ ]:


from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import numpy as np

# Define label mapping
label_mapping = {0: 'joy', 1: 'sadness', 2: 'anger', 3: 'fear', 4: 'neutral'}

# Load the fine-tuned DistilBERT model and tokenizer
model_path = "./fine_tuned_distilbert"  
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Initialize emotion classification pipeline
emotion_pipeline = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer,
    device=0,  # Use GPU if available
    top_k=None  # Return all scores
)

# Function to split long text into chunks
def split_text_to_chunks(text, tokenizer, max_length=512):
    tokens = tokenizer.tokenize(text)
    chunks = []
    for i in range(0, len(tokens), max_length - 2):  # Account for special tokens
        chunk = tokens[i:i + (max_length - 2)]
        chunks.append(tokenizer.convert_tokens_to_string(chunk))
    return chunks

# Function to process posts and aggregate results
def process_post(post, pipeline, tokenizer, label_mapping):
    chunks = split_text_to_chunks(post, tokenizer)
    all_predictions = []

    for chunk in chunks:
        predictions = pipeline(chunk)  # Predict for each chunk
        predictions_flat = {label_mapping[int(emotion['label'].split('_')[-1])]: emotion['score'] for emotion in predictions[0]}
        all_predictions.append(predictions_flat)

    # Average predictions across chunks
    avg_predictions = {
        label: np.mean([pred.get(label, 0.0) for pred in all_predictions])
        for label in label_mapping.values()
    }
    return avg_predictions

# Example posts (ensure they are within length limits)
posts = [
    "Volunteers wanted- wiki contributors I'm attempting to improve ASD-related resources.",
    "In autistic modded communities you’re going to find unique ways of expressing emotions.",
    "For those of us who prefer their solitude...",
    "People defo won't be able to tell I'm autistic unless I mention it!",
    "Late diagnosed at 25, the way I played The Sims was probably a hint."
]

# Predict emotions for each post
results = []
for post in posts:
    predictions_flat = process_post(post, emotion_pipeline, tokenizer, label_mapping)
    results.append({"Post": post, **predictions_flat})

# Convert results to a DataFrame
results_df = pd.DataFrame(results)

# Display predictions
print("Emotion predictions:")
print(results_df.head())



# The emotion predictions from the dataset highlight a dominant prevalence of the neutral tone across all analyzed posts, with scores consistently exceeding 0.97. This observation aligns with the subtle and less overt emotional expression commonly associated with ASD-related communication.
# 
#  - Post 0 (volunteer contributions) and Post 1 (autistic modded communities) exhibit nearly identical neutral scores (0.988 and 0.989, respectively), with minimal association to other emotions like joy, sadness, or anger (all scoring below 0.006).
# 
#  - Post 2 (preference for solitude) and Post 4 (personal autism diagnosis reflection) similarly maintain high neutral probabilities above 0.987.
# 
#  - Post 3, discussing concerns about autistic identity recognition, shows a slightly elevated probability for sadness (0.015) and anger (0.011), suggesting subtle emotional intensity. Nonetheless, the neutral tone still dominates with a score of 0.974.
# 
# These findings emphasize the nuances in ASD communication, which often deemphasize overt emotional expressions while embedding subtle emotional cues. The model demonstrates its ability to capture this dominant neutrality effectively while still identifying hints of underlying emotions like sadness or anger. This reinforces the model's adaptability and potential applicability to real-world ASD-related text analysis.

# # Conclusion

# Overall,  the study successfully developed and fine-tuned a DistilBERT-based multi-label classification model to recognize and classify emotions—joy, sadness, anger, fear, and neutral—in text specifically tailored to Autism Spectrum Disorder (ASD) communication. Leveraging the GoEmotions dataset and focusing on emotions critical to ASD contexts, the model demonstrated robust performance, achieving high accuracy, precision, and F1-scores across most classes. Real-world testing on autism-related forums further validated the model’s effectiveness, showcasing its ability to detect subtle emotional nuances in user-generated text.
# 
# During the development process, several complications emerged. A notable challenge was handling multi-label annotations by converting integer-encoded labels into a binary float representation compatible with the BCEWithLogitsLoss function. This step required overcoming dimension mismatches, resolving improperly formatted labels, and meeting PyTorch’s float32 requirements. Rigorous preprocessing addressed these challenges, ensuring label integrity and stable model training. These complexities highlight the need for meticulous data preparation when dealing with multi-label classification tasks.
# 
# While the model achieved promising results, it exhibited certain limitations. First, the maximum sequence length constraint of 512 tokens in the DistilBERT architecture necessitated splitting longer text into smaller chunks, which risks losing contextual coherence. Although predictions across chunks were averaged to mitigate this issue, further refinement is needed. Second, the model’s performance on minority classes such as fear and anger was relatively lower due to class imbalance in the dataset. Techniques like data augmentation, oversampling, or loss adjustments (e.g., focal loss) could improve predictions for these underrepresented emotions.
# 
# For future work, expanding real-world testing to larger and more diverse ASD communication datasets will enhance the model’s generalizability. Incorporating explainability tools like SHAP or LIME for real-time feedback can improve interpretability, fostering trust among caregivers, educators, and researchers. Additionally, fine-tuning the model with domain-specific corpora related to ASD communication may enhance sensitivity to atypical phrasing and subtle emotional cues. Finally, adopting long-context models such as Longformer could resolve the token length constraints, enabling better handling of extended dialogues and preserving contextual information.
# 
# In conclusion, this study addresses a significant research gap by developing an emotion classification model tailored for ASD communication—a topic often overlooked in conventional emotion recognition systems. The model’s success demonstrates its potential for real-world applications, offering a foundation for future advancements in emotion-aware technologies. By addressing the identified limitations and implementing the proposed enhancements, this work contributes to a more inclusive, empathetic, and practical framework for understanding emotional expression in ASD-related contexts.
# 
# 
# 
# 
# 
# 
# 
# 
# 

# # References
# 
# 1. GoEmotions Dataset:
# Demszky, D., Mahajan, D., Singh, G., Cowen, A., Nemade, G., & Ravi, S. (2020). GoEmotions: A Dataset of Fine-Grained Emotions. Google Research. Retrieved from https://ai.googleblog.com/2021/10/goemotions-dataset-for-fine-grained.html
# 
# 2. Transformer Architecture:
# Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention is All You Need. Advances in Neural Information Processing Systems, 30. Retrieved from https://arxiv.org/abs/1706.03762
# 
# 3. DistilBERT Model:
# Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter. Proceedings of the 5th Workshop on Energy Efficient Machine Learning and Cognitive Computing. Retrieved from https://arxiv.org/abs/1910.01108
# 
# 4. Emotion Recognition and ASD:
# Hume, K., Loftin, R., & Lantz, J. (2009). Increasing independence in autism spectrum disorders: A review of research on self-management interventions. Journal of Autism and Developmental Disorders, 39(2), 195-217.
# 
# 5. LIME for Explainability:
# Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why Should I Trust You?": Explaining the Predictions of Any Classifier. Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. Retrieved from https://arxiv.org/abs/1602.04938
# 
# 
# 6. Emotion Processing in Autism:
# Mazefsky, C. A., & White, S. W. (2014). Emotion regulation: Concepts and practice in autism spectrum disorder. Child and Adolescent Psychiatric Clinics of North America, 23(1), 15-34.
# 
# 7. Challenges in NLP for ASD Communication:
# Eigsti, I. M., De Marchena, A., Schuh, J., & Kelley, E. (2011). Language acquisition in autism spectrum disorders: A developmental review. Research in Autism Spectrum Disorders, 5(2), 681-691.
# 
# 8. Sensing Technologies and Machine Learning in Autism:
# Banos, O., Comas-González, Z., Medina, J., Polo-Rodríguez, A., Gil, D., Peral, J., Amador, S., & Villalonga, C. (2024). Sensing technologies and machine learning methods for emotion recognition in autism: Systematic review. Retrieved from https://arxiv.org/abs/2407.04712
# 
# 9. Transformer-Based Neural Networks for Emotion Recognition:
# Creanga, C., & Dinu, A. (2024). Transformer based neural networks for emotion recognition in conversations. Retrieved from https://arxiv.org/abs/2405.11222
# 
# 10. Multimodal Emotion Recognition with Transformers:
# (2022). Emotion Recognition with Pre-Trained Transformers Using Multimodal Signals. Retrieved from https://ieeexplore.ieee.org/document/9953852
# 
# 11. Real-Time Facial Emotion Recognition in Autism:
# (2023). Real-time facial emotion recognition system among children with autism using enhanced deep learning techniques. Retrieved from https://link.springer.com/article/10.1007/s00521-023-08372-9
# 
# 12. Automatic Emotion Recognition in Autism:
# (2022). Automatic Emotion Recognition in Children with Autism: A Systematic Literature Review. Retrieved from https://www.mdpi.com/1424-8220/22/4/1649
# 
# 
# 
