# genaiproj
 <h1>ECG Heartbeat Arrhythmia Classification</h1>

Classify individual heartbeats from raw ECG signals into four cardiac rhythm categories.

Overview
In this competition, you are given short segments of electrocardiogram (ECG) signal, each segment centered on a single heartbeat. Your task is to classify each heartbeat into one of four clinically meaningful arrhythmia categories.

This is a supervised multiclass classification problem on one-dimensional time-series data. Each row in the dataset represents one heartbeat extracted from a real hospital ECG recording.

Submissions are evaluated using Macro F1-Score.

The F1-Score for a single class is the harmonic mean of precision and recall for that class:

F1_class = 2 × (Precision × Recall) / (Precision + Recall)
where:

Precision = (true positives) / (true positives + false positives)

Recall = (true positives) / (true positives + false negatives)

The Macro F1-Score is the unweighted arithmetic mean of the per-class F1-Scores across all four classes:

Macro F1 = (F1_class0 + F1_class1 + F1_class2 + F1_class3) / 4

This means every class contributes equally to your score, regardless of how many examples that class has in the test set.

A score of 1.0 indicates perfect classification. A score near 0.25 indicates performance close to random guessing

Submission Format
For each beat in the test set, you must predict its label (0, 1, 2, or 3). Your submission file must contain exactly two columns: id and label.

id,label
te_0000000,0
te_0000001,2
te_0000002,0
te_0000003,1
...
The id column must match the id values in test.csv exactly
The label column must contain integer values: 0, 1, 2, or 3
The file must contain a header row
Every row in test.csv must have a corresponding prediction
The file sample_submission.csv is provided as a template with the correct format and all required id values.

---
Citation
AVINASH SINGH913. NPPE2-T2-26–ECG Heartbeat Arrhythmia Classification. https://www.kaggle.com/competitions/nppe-2-t-2-26-ecg-heartbeat-arrhythmia-classification, 2026. Kaggle.
