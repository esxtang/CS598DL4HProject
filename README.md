# CS598 Deep Learning for Healthcare Reproducibility Project
Final project for UIUC CS598 Deep Learning for Healthcare, Spring 2023

This code is developed as part of the final project for the CS598 Deep Learning for Healthcare course at UIUC. The goal of this project is to reproduce the results of a recently published research paper related to deep learning in the healthcare sector.

The paper I have chosen to reproduce is [_Imputation Strategies Under Clinical Presence: Impact on Algorithmic Fairness_](https://proceedings.mlr.press/v193/jeanselme22a/jeanselme22a.pdf) by Jeanselme et al. (2022)[^1]. The original code provided by the authors is available [here](https://github.com/Jeanselme/ClinicalPresenceFairness).

### To run the experiments, please do the following:
1. Obtain MIMIC-III data. The data required to run the experiments is in the MIMIC-III dataset, which is available on [PhysioNet](https://physionet.org/content/mimiciii/1.4/). Once you have obtained access to the MIMIC-III dataset, place the ADMISSIONS.csv, LABEVENTS.csv, and PATIENTS.csv files in the MimicData folder. 
2. Obtain the itemid_to_variable_map.csv file from [MIMIC_Extract](https://github.com/MLforHealth/MIMIC_Extract/blob/master/resources/itemid_to_variable_map.csv). Add this file to the MimicData folder.
3. Run the Preprocessing.py code. This will produce two files (preprocessed_labeled_outcomes.csv and preprocessed_labs.csv) that will be stored in the PreprocessedData folder.
4. Run the Experiment.py code. This will produce 4 files (experiment_results_Median.csv, experiment_results_MICE.csv, experiment_results_Group MICE.csv, and experiment_results_Group MICE Missing.csv) that will be stored in the ExperimentResults folder.
5. Run the Analysis.py code to view the results of the experiments.

### These are the results obtained by running my code:

Average number of lab events and tests per patient, stratified by demographic group:

![image](https://user-images.githubusercontent.com/63872692/235328520-13d41ddb-7a32-41a5-9413-eae2a44fc858.png)

Average AUC ROC score, prioritized percentage, and wrongly not prioritized rate (FNR), stratified by demographic group:

<img src="https://user-images.githubusercontent.com/63872692/235328816-c3949395-cec2-49dd-a053-aa05ca02ff07.jpg" width="60%">

![image](https://user-images.githubusercontent.com/63872692/235328775-c1071db2-fedc-4872-b7bc-cd1959b7fe6b.png)

Differences in average AUC ROC score, prioritized percentage, and wrongly not prioritized rate (FNR) for the minority population versus the majority population:

ETHNICITY: Minority = Black, Majority = Non Black

GENDER: Minority = Female, Majority = Male

INSURANCE: Minority = Public, Majority = Private

<img src="https://user-images.githubusercontent.com/63872692/235328812-fb212240-69dd-4b18-863e-05da941c0e7b.jpg" width="60%">

![image](https://user-images.githubusercontent.com/63872692/235328602-ec7e64f6-291b-49ef-a971-ebfe085c30d3.png)


[^1]: Original paper: Jeanselme, V., De-Arteaga, M., Zhang, Z., Barrett, J., & Tom, B. (2022, November). Imputation Strategies Under Clinical Presence: Impact on Algorithmic Fairness. In _Machine Learning for Health_ (pp. 12-34). PMLR.
