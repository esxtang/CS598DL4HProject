# CS598 Deep Learning for Healthcare Reproducibility Project
Final project for UIUC CS598 Deep Learning for Healthcare, Spring 2023

This code is developed as part of the final project for the CS598 Deep Learning for Healthcare course at UIUC. The goal of this project is to reproduce the results of a recently published research paper related to deep learning in the healthcare sector.

The paper I have chosen to reproduce is [_Imputation Strategies Under Clinical Presence: Impact on Algorithmic Fairness_](https://proceedings.mlr.press/v193/jeanselme22a/jeanselme22a.pdf) by Jeanselme et al. (2022)[^1]. The original code provided by the authors is available [here](https://github.com/Jeanselme/ClinicalPresenceFairness).

### To run the experiments either use the [Descriptive Notebook](https://github.com/esxtang/CS598DL4HProject/blob/master/Descriptive%20Notebook.ipynb) or follow these steps:
1. Obtain MIMIC-III data. The data required to run the experiments is in the MIMIC-III dataset, which is available on [PhysioNet](https://physionet.org/content/mimiciii/1.4/). Once you have obtained access to the MIMIC-III dataset, place the ADMISSIONS.csv, LABEVENTS.csv, and PATIENTS.csv files in the MimicData folder. 
2. Obtain the itemid_to_variable_map.csv file from [MIMIC_Extract](https://github.com/MLforHealth/MIMIC_Extract/blob/master/resources/itemid_to_variable_map.csv). Add this file to the MimicData folder.
3. Install dependencies using `pip install -r requirements.txt`
4. Run the [Preprocessing.py](https://github.com/esxtang/CS598DL4HProject/blob/master/Preprocessing.py) code. This will produce two files (preprocessed_labeled_outcomes.csv and preprocessed_labs.csv) that will be stored in the PreprocessedData folder.
5. Run the [Experiment.py](https://github.com/esxtang/CS598DL4HProject/blob/master/Experiment.py) code. This will produce 4 files (experiment_results_Median.csv, experiment_results_MICE.csv, experiment_results_Group MICE.csv, and experiment_results_Group MICE Missing.csv) that will be stored in the ExperimentResults folder.
6. Run the [Analysis.py](https://github.com/esxtang/CS598DL4HProject/blob/master/Analysis.py) code to view the results of the experiments.

### These are the results obtained by running my code:

Average number of lab events and tests per patient, stratified by demographic group:

![Labs by Demographic - Bar](https://user-images.githubusercontent.com/63872692/236654911-3dc937e8-6ae2-4b02-9660-6deff289a8b8.png)

Average AUC ROC score, prioritized percentage, and wrongly not prioritized rate (FNR), stratified by demographic group:

![Population Results Table](https://user-images.githubusercontent.com/63872692/236652580-a5a38a4e-863a-4760-bf00-365af1eca6c8.png)

![Population Results Bars](https://user-images.githubusercontent.com/63872692/236652617-d2edb2bc-569f-40d2-b5c8-2f4678bc1c38.png)

Differences in average AUC ROC score, prioritized percentage, and wrongly not prioritized rate (FNR) for the minority population versus the majority population:

ETHNICITY: Minority = Black, Majority = Non Black

GENDER: Minority = Female, Majority = Male

INSURANCE: Minority = Public, Majority = Private

![Population Delta Table](https://user-images.githubusercontent.com/63872692/236652901-912df800-e8cd-4895-b00a-ecebe487d759.png)

![Population Delta Bars](https://user-images.githubusercontent.com/63872692/236652862-202ce6b5-6803-4402-874b-6ea1d1de91f6.png)


[^1]: Original paper: Jeanselme, V., De-Arteaga, M., Zhang, Z., Barrett, J., & Tom, B. (2022, November). Imputation Strategies Under Clinical Presence: Impact on Algorithmic Fairness. In _Machine Learning for Health_ (pp. 12-34). PMLR.
