# CS598DL4H Project
Final project for UIUC CS598 Deep Learning for Healthcare, Spring 2023

This code is developed as part of the final project for the CS598 Deep Learning for Healthcare course at UIUC. The goal of this project is to reproduce the results of a recently published research paper related to deep learning in the healthcare sector.

The paper I have chosen to reproduce is [_Imputation Strategies Under Clinical Presence: Impact on Algorithmic Fairness_](https://proceedings.mlr.press/v193/jeanselme22a/jeanselme22a.pdf) by Jeanselme et al. (2022). The original code provided by the authors is available [here](https://github.com/Jeanselme/ClinicalPresenceFairness).

To run the experiments, please do the following:
1. Obtain MIMIC-III data. The data required to run the experiments is the MIMIC-III dataset, which is available on [PhysioNet](https://physionet.org/content/mimiciii/1.4/). Once you have obtained access to the MIMIC-III dataset, place the ADMISSIONS.csv, LABEVENTS.csv, and PATIENTS.csv files in the MimicData folder. 
2. Obtain the itemid_to_variable_map.csv file from [MIMIC_Extract](https://github.com/MLforHealth/MIMIC_Extract/blob/master/resources/itemid_to_variable_map.csv). Add this file to the MimicData folder.
3. Run the Preprocessing.py code. This will produce two files (preprocessed_labeled_outcomes.csv and preprocessed_labs.csv) that will be stored in the PreprocessedData folder.
4. Run the Experiment.py code. This will produce 4 files (experiment_results_Median.csv, experiment_results_MICE.csv, experiment_results_Group MICE.csv, and experiment_results_Group MICE Missing.csv) that will be stored in the ExperimentResults folder.
5. Run the Analysis.py code to view the results of the experiments.

Please note that the code is still under development. The code for displaying results is very similar to the authors' original code to enable easy comparison across results.
