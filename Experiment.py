### Import libraries
import time
import tracemalloc
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import ParameterSampler, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from IPython.display import display


def main():
  # Start tracking time and memory usage 
  start = time.time()
  tracemalloc.start()

  # Select model parameters and run experiment
  imputation_iterations = {
    'Median': 100,
    'MICE': 10,
    'Group MICE': 10,
    'Group MICE Missing': 10
  }
  hyperparameters = {
    'logistic_regression': {
        'penalty': ['l2'],
        'C': [0.01, 0.1, 1., 10],
        'solver': ['sag'], 
        'max_iter': [1000],
        'n_jobs': [-1]
    },
    'mlp_classifier': {
        'hidden_layer_sizes': [(5, 2)],
        'activation': ['logistic'],
        'solver': ['adam'], 
        'max_iter': [1000]
    }
  }
  # Ablation: To remove regularization, call experiment method with regularization = False
  # Experiment: To test MLP Classifier, call experiment method with model_name = 'mlp_classifier'
  experiment(imputation_iterations, hyperparameters, regularization = True, model_name = 'logistic_regression')

  # Report time and memory usage
  end = time.time()
  display(pd.DataFrame(
    data = {'Experiment': [(end - start)/60, tracemalloc.get_traced_memory()[1]/10**9]},
    index = ['Time (min)', 'Memory Usage (GB)']))
  tracemalloc.stop()

if __name__ == "__main__":
  main()


### Experiment method
def experiment(imputation_iterations, hyperparameters, regularization, model_name):
  # Set path for getting data
  PREPROCESSED_DATA_PATH = './PreprocessedData/'
  EXPERIMENT_RESULTS_PATH = './ExperimentResults/'

  # Get preprocessed data
  labs = pd.read_csv(PREPROCESSED_DATA_PATH + 'preprocessed_labs.csv', index_col = [0, 1], header = [0, 1])
  outcomes = pd.read_csv(PREPROCESSED_DATA_PATH + 'preprocessed_labeled_outcomes.csv', index_col = 0)
  outcomes['Outcome'] = outcomes['Outcome'] == 'Death' # Translate values to True if Death, False if Alive

  # Select Patient training set, Labs, Demographic data, and Missingness Indicators
  train_indices = outcomes.sample(frac = 0.8, random_state = 0).index.sort_values()
  print('Total patients: {}'.format(len(outcomes)))
  print('Training patients: {}'.format(len(train_indices)))

  # Use only latest lab record for each patient
  latest_labs = labs.groupby('SUBJECT_ID').last()

  # Impute data, train model, and store predictions for each imputation strategy
  for strategy, iterations in imputation_iterations.items():
    predictions = []
    for i in range(iterations):
      # Impute missing data
      imputed_labs = imputation(strategy, latest_labs, outcomes, train_indices)
      # Train model and store predictions
      prediction = train(imputed_labs, outcomes, train_indices, model_name, hyperparameters, regularization)
      predictions.append(prediction)
    # Average and store predictions
    last_train_test_labels = [p['Train Test Label'] for p in predictions][-1]
    outcome_predictions = pd.concat([p[1] for p in predictions], axis = 1)
    avg_predictions = pd.concat([outcome_predictions.mean(axis = 1).rename('Mean'), last_train_test_labels], axis = 1)
    # Write to CSV
    avg_predictions.to_csv(EXPERIMENT_RESULTS_PATH + 'experiment_results_' + strategy + '.csv')


### Imputation method
def imputation(imputation_strategy, latest_labs, outcomes, train_indices, max_iterations = 10):
  # Single median imputation
  if imputation_strategy == 'Median':
    return latest_labs.fillna(latest_labs.loc[train_indices].median())

  # Multiple Imputation using Chained Equation (MICE), Group MICE, and Group MICE Missing
  if 'MICE' in imputation_strategy:
    if 'Missing' in imputation_strategy:
      counts = latest_labs.isna().add_suffix('_count') # Count columns for Group MICE Missing
      latest_labs = pd.concat([latest_labs, counts], axis = 1)
    if 'Group' in imputation_strategy: 
      minority_population = pd.concat([outcomes['ETHNICITY'] == 'Black', outcomes['GENDER'] == 'Female', outcomes['INSURANCE'] == 'Public'], axis = 1) # Flag patients in minority population for Group MICE and Group MICE Missing (TRUE = patient is in minority group)
      latest_labs = pd.concat([latest_labs, minority_population], axis = 1)
      
    missing_lab_values = latest_labs.isna()
    labs_with_missing_values = missing_lab_values.sum().sort_values()
    labs_with_missing_values = labs_with_missing_values[labs_with_missing_values > 0]

    # Start with median imputation
    mice_imputed_labs = pd.DataFrame(SimpleImputer(strategy = "median").fit(latest_labs.loc[train_indices].values).transform(latest_labs.values), index = latest_labs.index, columns = latest_labs.columns)

    # Impute for each lab using linear regression until convergence
    for _ in range(max_iterations):
      for lab in labs_with_missing_values.index:
        # Use train data rows (excluding rows where there is a missing value for this lab)
        train_data_for_lab = mice_imputed_labs.loc[train_indices][~missing_lab_values.loc[train_indices][lab]]
        # Train data (all lab data excluding the current lab)
        x_train = train_data_for_lab.loc[:, mice_imputed_labs.columns != lab].values
        # Train data for current lab
        y_train = train_data_for_lab[lab].values

        # Fit linear regression
        linear_reg_model = LinearRegression().fit(x_train, y_train)
        residuals = np.abs(linear_reg_model.predict(x_train) - y_train)

        # Data excluding current lab column, including only rows where this lab is missing values
        x = mice_imputed_labs.loc[:, mice_imputed_labs.columns != lab][missing_lab_values[lab]].values
        epsilon = np.random.normal(scale = np.std(residuals), size = missing_lab_values[lab].sum())
        # Impute data for missing rows in this lab column
        mice_imputed_labs[lab][missing_lab_values[lab]] = linear_reg_model.predict(x) + epsilon
    
    if 'Group' in imputation_strategy:
      mice_imputed_labs = mice_imputed_labs.rename(columns={'ETHNICITY': ('VALUENUM', 'ETHNICITY'), 'GENDER': ('VALUENUM', 'GENDER'), 'INSURANCE': ('VALUENUM', 'INSURANCE')})
    return mice_imputed_labs


### Model training method
def train(imputed_labs, outcomes, all_train_indices, model_name, hyperparameters, regularization):
  # Split training data into train, validation, and test
  train_index, test_index = train_test_split(all_train_indices, train_size = 0.9, random_state = 0)
  train_index, validation_index = train_test_split(train_index, train_size = 0.9, random_state = 0)

  # Label outcome rows as train or test
  train_test_labels = pd.Series("Test", index = outcomes.index, name = "Train Test Label")
  train_test_labels[validation_index] = "Train"
  train_test_labels[train_index] = "Train"

  # Get training and validation lab data and outcomes
  train_imputed_labs = imputed_labs.loc[train_index]
  validation_imputed_labs = imputed_labs.loc[validation_index]
  train_outcomes = outcomes['Outcome'].loc[train_index]
  validation_outcomes = outcomes['Outcome'].loc[validation_index]
  
  # Normalization
  normalizer = StandardScaler().fit(train_imputed_labs)
  train_imputed_labs = pd.DataFrame(normalizer.transform(train_imputed_labs), index = train_imputed_labs.index)
  validation_imputed_labs = pd.DataFrame(normalizer.transform(validation_imputed_labs), index = validation_imputed_labs.index)
  imputed_labs = pd.DataFrame(normalizer.transform(imputed_labs), index = imputed_labs.index)
  
  # Regularization for Logisitic Regression
  if (model_name == 'logistic_regression') & (not regularization):
    hyperparameters[model_name]['penalty'] = [None] 
    hyperparameters[model_name]['C'] = [10000000000] # Setting penalty=None will ignore the C parameter
   
  # Search for best model and hyperparameters
  best_neg_log_likelihood = np.inf
  best_hyperparameter = None
  best_model = None
  hyper_grid = list(ParameterSampler(hyperparameters[model_name], n_iter = 100, random_state = 0))
  for _, hyperparams in enumerate(hyper_grid):
    np.random.seed(0)
    if model_name == 'logistic_regression':
      model = LogisticRegression(**hyperparams).fit(train_imputed_labs, train_outcomes)
    elif model_name == 'mlp_classifier':
      model = MLPClassifier(**hyperparams).fit(train_imputed_labs, train_outcomes)
    if model:
      neg_log_likelihood = -model.score(validation_imputed_labs, validation_outcomes)
      if neg_log_likelihood < best_neg_log_likelihood:
        best_hyperparameter = hyperparams
        best_model = model
        best_neg_log_likelihood = neg_log_likelihood
  
  # Predict using best model
  prediction = pd.DataFrame(best_model.predict_proba(imputed_labs), index = outcomes.index)
  result = pd.concat([prediction, train_test_labels], axis = 1)
  return result