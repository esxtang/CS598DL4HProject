### EXPERIMENT

## Start tracking time and memory usage
# Import time & memory modules
import time
import tracemalloc
 
# Starting the monitoring
start = time.time()
tracemalloc.start()

## Import libraries
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import ParameterSampler, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

## Set data paths
PREPROCESSED_PATH = '/home/etang/Desktop/CS598DL4HProject/PreprocessedData/'  # Edit to your path
EXPERIMENT_PATH = '/home/etang/Desktop/CS598DL4HProject/ExperimentResults/'  # Edit to your path

## Get preprocessed data
labs = pd.read_csv(PREPROCESSED_PATH + 'preprocessed_labs.csv', index_col = [0, 1], header = [0, 1])
labs.head()

outcomes = pd.read_csv(PREPROCESSED_PATH + 'preprocessed_labeled_outcomes.csv', index_col = 0)
outcomes['Outcome'] = outcomes['Outcome'] == 'Death' # Translate values to True if Death, False if Alive
outcomes.head()

## Select training data
train_indices = outcomes.sample(frac = 0.8, random_state = 0).index.sort_values()
# train_outcomes = pd.Series(outcomes.index.isin(train_indices), index = outcomes.index)
# print(train_indices)

print('Total patients: {}'.format(len(outcomes)))
print('Training patients: {}'.format(len(train_indices)))

## Imputation
# Use only latest lab record for each patient
latest_labs = labs.groupby('SUBJECT_ID').last()

minority_population = pd.concat([outcomes['ETHNICITY'] == 'Black', outcomes['GENDER'] == 'Female', outcomes['INSURANCE'] == 'Public'], axis = 1)

# Count columns for Group MICE Missing
counts = latest_labs.isna().add_suffix('_count')

def imputation(imputation_strategy, latest_labs, train_indices, max_iterations = 10):
  # Single median imputation
  if imputation_strategy == 'Median':
    return latest_labs.fillna(latest_labs.loc[train_indices].median())

  # Multiple Imputation using Chained Equation (MICE), Group MICE, and Group MICE Missing
  if 'MICE' in imputation_strategy:
    if 'Missing' in imputation_strategy:
      latest_labs = pd.concat([latest_labs, counts], axis = 1)
    if 'Group' in imputation_strategy: 
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
      # Remove the group columns
      # mice_imputed_labs = mice_imputed_labs.iloc[:, :-3] # original only removes 1 col? .iloc[:, :-1]
      # mice_imputed_labs = mice_imputed_labs.loc[:, ~mice_imputed_labs.columns.isin(['ETHNICITY', 'GENDER', 'INSURANCE'])]
      mice_imputed_labs = mice_imputed_labs.rename(columns={'ETHNICITY': ('VALUENUM', 'ETHNICITY'), 'GENDER': ('VALUENUM', 'GENDER'), 'INSURANCE': ('VALUENUM', 'INSURANCE')}) ### MY FIX   
    print(mice_imputed_labs)
    return mice_imputed_labs

## Training
def train(imputed_labs, outcomes, all_train_indices, model_name, hyperparameters, regularization, normalization = True):
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
  if normalization:
    normalizer = StandardScaler().fit(train_imputed_labs)
    train_imputed_labs = pd.DataFrame(normalizer.transform(train_imputed_labs), index = train_imputed_labs.index)
    validation_imputed_labs = pd.DataFrame(normalizer.transform(validation_imputed_labs), index = validation_imputed_labs.index)
    imputed_labs = pd.DataFrame(normalizer.transform(imputed_labs), index = imputed_labs.index)
  
  # Regularization for Logisitic Regression
  if (model_name == 'logistic_regression') & (not regularization):
    hyperparameters[model_name]['penalty'] = [None] 
    hyperparameters[model_name]['C'] = [10000000000] # Setting penalty=None will ignore the C parameter
    
  print('xxxxx Initial Model Info xxxxx')
  print('Model:', model_name, '-- Normalization:', normalization, '-- Regularization:', regularization)
  print(hyperparameters[model_name])
  
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
  print("============= Best Parameters =============")
  print("Best Hyperparameter:")
  print(best_hyperparameter)
  print("Best model:")
  print(best_model)
  print("Best Neg Log Likelihood:")
  print(best_neg_log_likelihood)
  
  # Predict using best model
  prediction = pd.DataFrame(best_model.predict_proba(imputed_labs), index = outcomes.index)
  result = pd.concat([prediction, train_test_labels], axis = 1)
  return result

## Parameters
# Set parameters
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
        # 'alpha': [0, 0.0001, 0.01],
        'max_iter': [1000]
    }
}

model_name = 'logistic_regression'
# model_name = 'mlp_classifier'

regularization = True
# regularization = False

## Run experiments
for strategy, iterations in imputation_iterations.items():
  print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Imputation strategy: ' + strategy + ' <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
  print('Iterations: ' + str(iterations))

  predictions = []
  for i in range(iterations):
    print('.............................', strategy, i, '.............................')
    imputed_labs = imputation(strategy, latest_labs, train_indices)
    prediction = train(imputed_labs, outcomes, train_indices, model_name, hyperparameters, regularization)
    predictions.append(prediction)
  
  # Average multiple imputations models
  last_train_test_labels = [p['Train Test Label'] for p in predictions][-1]
  predictions_death_values = pd.concat([p[1] for p in predictions], axis = 1)
  result = pd.concat([predictions_death_values.mean(axis = 1).rename('Mean'), predictions_death_values.std(axis = 1).rename('Std'), last_train_test_labels], axis = 1)
  # Write to CSV
  result.to_csv(EXPERIMENT_PATH + 'experiment_results_' + strategy + '.csv')
  
## Report time and memory usage
# Record end time
end = time.time()

# Print the difference between start and end time in milli. secs
print("The time of execution of the above program is :", (end - start) * 10**3, "ms")

# Display memory usage
print("The memory usage of the above program (current memory, peak memory) is :", tracemalloc.get_traced_memory())

computational_requirements = pd.DataFrame(
    data = {'Experiment': [(end - start)/60, tracemalloc.get_traced_memory()[1]/10**9]},
    index = ['Time (min)', 'Memory Usage (GB)'])
display(computational_requirements)
 
# Stopping the memory trace
tracemalloc.stop()
