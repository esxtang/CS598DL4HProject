### ANALYSIS

## Start tracking time and memory usage
# Import time & memory modules
import time
import tracemalloc
 
# Starting the monitoring
start = time.time()
tracemalloc.start()

## Import libraries
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score, brier_score_loss
from IPython.display import display
import matplotlib.pyplot as plt
from matplotlib import colors as plt_colors

## Set data paths
PREPROCESSED_PATH = '/home/etang/Desktop/CS598DL4HProject/PreprocessedData/'  # Edit to your path
EXPERIMENT_PATH = '/home/etang/Desktop/CS598DL4HProject/ExperimentResults/'  # Edit to your path

## Get Experiment data
labs = pd.read_csv(PREPROCESSED_PATH + 'preprocessed_labs.csv', index_col = [0, 1], header = [0, 1])
outcomes = pd.read_csv(PREPROCESSED_PATH + 'preprocessed_labeled_outcomes.csv', index_col = 0)
outcomes['Outcome'] = outcomes['Outcome'] == 'Death' # Translate values to True if Death, False if Alive

## Imputations and Demographic groups
imputations = ['Median', 'MICE', 'Group MICE', 'Group MICE Missing']
              
predictions = {}
for strategy in imputations:
  predictions[strategy] = pd.read_csv(EXPERIMENT_PATH + 'experiment_results_' + strategy + '.csv', index_col = 0)

demographics = {
    'GENDER': {'data': outcomes['GENDER'], 'populations': np.unique(outcomes['GENDER'].values)}, 
    'ETHNICITY': {'data': outcomes['ETHNICITY'], 'populations': np.unique(outcomes['ETHNICITY'].values)}, 
    'INSURANCE': {'data': outcomes['INSURANCE'], 'populations': np.sort(np.unique(outcomes['INSURANCE'].values))[::-1]}, 
    'OVERALL': {'data': outcomes, 'populations': ['Overall']}
}

## Data set statistics
# Hypothesis 1: There are non-random missingness patterns in real data.

# More lab tests are ordered for patients who end up dying than for those who survive, 
# which aligns with the confirmation bias missingness pattern.
print('----- SURVIVAL -----')
deceased_subj_ids = outcomes[outcomes['Outcome'] == True].index
num_labs_deceased = labs.loc[deceased_subj_ids].notna().sum().sum()
surviving_subj_ids = outcomes[outcomes['Outcome'] == False].index
num_labs_surviving = labs.loc[surviving_subj_ids].notna().sum().sum()

survival_stats = pd.DataFrame(
    data = {'Deceased': [len(deceased_subj_ids), num_labs_deceased / len(deceased_subj_ids)],
            'Surviving': [len(surviving_subj_ids), num_labs_surviving / len(surviving_subj_ids)]},
    index = ['# of Patients', 'Average # of Labs']).round(2)
display(survival_stats)

# Fewer lab tests are ordered for marginalized groups (Black, Female, Public Insurance)
marginalized_population = {
    'ETHNICITY': {'Minority': 'Black', 'Majority': 'Non Black'},
    'GENDER': {'Minority': 'Female', 'Majority': 'Male'},
    'INSURANCE': {'Minority': 'Public', 'Majority': 'Private'}
    }
for demographic, population in marginalized_population.items():
  print('----- ' + demographic + ' -----')
  minority_subj_ids = outcomes[outcomes[demographic] == population['Minority']].index
  minority_num_labs = labs.loc[minority_subj_ids].notna().sum().sum()
  majority_subj_ids = outcomes[outcomes[demographic] == population['Majority']].index
  majority_num_labs = labs.loc[majority_subj_ids].notna().sum().sum()

  population_stats = pd.DataFrame(
      data = {population['Minority']: [len(minority_subj_ids), minority_num_labs / len(minority_subj_ids)],
              population['Majority']: [len(majority_subj_ids), majority_num_labs / len(majority_subj_ids)]},
      index = ['# of Patients', 'Average # of Labs']).round(2)
  display(population_stats)

## Evaluation and Analysis (NOTE: This code is currently still very similar to the original authors' code)
def evaluate(y_pred, y_true, demo_data, demo_populations, threshold_percent = 0.3, iterations = 100):
  # Results placeholders (format is {'Black': [], 'Non Black': []})
  all_br = {population: [] for population in demo_populations}
  all_fpr = {population: [] for population in demo_populations}
  all_tpr = {population: [] for population in demo_populations}
  all_roc = {population: [] for population in demo_populations}
  screened = {population: [] for population in demo_populations}
  screened_fpr = {population: [] for population in demo_populations}
  screened_fnr = {population: [] for population in demo_populations}

  fpr, tpr, thresholds = roc_curve(y_true, y_pred)
  fpr_threshold = np.interp(0.9, tpr[np.argsort(tpr)], thresholds[np.argsort(tpr)])
  tpr_threshold = np.interp(0.1, fpr[np.argsort(fpr)], thresholds[np.argsort(fpr)])

  threshold_top = pd.Series(y_pred).nlargest(int(len(y_pred) * threshold_percent), keep = 'all').min()

  for population in demo_populations:
    # Get y_pred and y_true data for this population
    if population == 'Overall':
      y_pred_pop = y_pred
      y_true_pop = y_true
    else:
      y_pred_pop = y_pred[demo_data == population]
      y_true_pop = y_true[demo_data == population]

    # Use bootstrap method for validation
    for i in range(iterations):
      bootstrap = np.random.choice(np.arange(len(y_pred_pop)), size = len(y_pred_pop), replace = True) ### This isn't taking a smaller sample? just changing order?
      y_pred_iteration = y_pred_pop[bootstrap]
      y_true_iteration = y_true_pop[bootstrap]

      # Get metrics for boostrapped sample
      all_br[population].append(brier_score_loss(y_true_iteration, y_pred_iteration))
      fpr_iteration, tpr_iteration, thresholds_iteration = roc_curve(y_true_iteration, y_pred_iteration)
      all_fpr[population].append(np.interp(fpr_threshold, thresholds_iteration[np.argsort(thresholds_iteration)], fpr_iteration[np.argsort(thresholds_iteration)]))
      all_tpr[population].append(np.interp(tpr_threshold, thresholds_iteration[np.argsort(thresholds_iteration)], tpr_iteration[np.argsort(thresholds_iteration)]))
      all_roc[population].append(roc_auc_score(y_true_iteration, y_pred_iteration))

      # Percentage screened-out
      selected = y_pred_iteration >= threshold_top
      screened[population].append(np.mean(selected)) # Percentage of patients in this demo_group that are prioritized
      screened_fnr[population].append((y_true_iteration[~selected]).sum() / y_true_iteration.sum()) # Wrongly not prioritized
      screened_fpr[population].append((1 - y_true_iteration[selected]).sum() / (1 - y_true_iteration).sum()) # Wrongly prioritized
  
  # Collect results
  result = {}
  if demo_populations[0] != 'Overall':
    difference_label = 'Difference ' + demo_populations[0] + ' - ' + demo_populations[1]
    result = {
        (difference_label, "Brier Score", 'Mean'): np.mean(np.array(all_br[demo_populations[0]]) - np.array(all_br[demo_populations[1]])),
        (difference_label, "Brier Score", 'Std'): np.std(np.array(all_br[demo_populations[0]]) - np.array(all_br[demo_populations[1]])),
        (difference_label, "AUC ROC", 'Mean'): np.mean(np.array(all_roc[demo_populations[0]]) - np.array(all_roc[demo_populations[1]])),
        (difference_label, "AUC ROC", 'Std'): np.std(np.array(all_roc[demo_populations[0]]) - np.array(all_roc[demo_populations[1]])),

        (difference_label, "FPR @ 90% TPR", 'Mean'): np.mean(np.array(all_fpr[demo_populations[0]]) - np.array(all_fpr[demo_populations[1]])),
        (difference_label, "FPR @ 90% TPR", 'Std'): np.std(np.array(all_fpr[demo_populations[0]]) - np.array(all_fpr[demo_populations[1]])),
        (difference_label, "TPR @ 10% FPR", 'Mean'): np.mean(np.array(all_tpr[demo_populations[0]]) - np.array(all_tpr[demo_populations[1]])),
        (difference_label, "TPR @ 10% FPR", 'Std'): np.std(np.array(all_tpr[demo_populations[0]]) - np.array(all_tpr[demo_populations[1]])),

        (difference_label, "Prioritized", 'Mean'): np.mean(np.array(screened[demo_populations[0]]) - np.array(screened[demo_populations[1]])),
        (difference_label, "Prioritized", 'Std'): np.std(np.array(screened[demo_populations[0]]) - np.array(screened[demo_populations[1]])),
        (difference_label, "Wrongly prioritized (FPR)", 'Mean'): np.mean(np.array(screened_fpr[demo_populations[0]]) - np.array(screened_fpr[demo_populations[1]])),
        (difference_label, "Wrongly prioritized (FPR)", 'Std'): np.std(np.array(screened_fpr[demo_populations[0]]) - np.array(screened_fpr[demo_populations[1]])),
        (difference_label, "Wrongly not prioritized (FNR)", 'Mean'): np.mean(np.array(screened_fnr[demo_populations[0]]) - np.array(screened_fnr[demo_populations[1]])),
        (difference_label, "Wrongly not prioritized (FNR)", 'Std'): np.std(np.array(screened_fnr[demo_populations[0]]) - np.array(screened_fnr[demo_populations[1]]))
    }
  for population in demo_populations:
    result.update({
        (population, "Brier Score", 'Mean'): np.mean(all_br[population]),
        (population, "Brier Score", 'Std'): np.std(all_br[population]),
        (population, "AUC ROC", 'Mean'): np.mean(all_roc[population]),
        (population, "AUC ROC", 'Std'): np.std(all_roc[population]),

        (population, "FPR @ 90% TPR", 'Mean'): np.mean(all_fpr[population]),
        (population, "FPR @ 90% TPR", 'Std'): np.std(all_fpr[population]),
        (population, "TPR @ 10% FPR", 'Mean'): np.mean(all_tpr[population]),
        (population, "TPR @ 10% FPR", 'Std'): np.std(all_tpr[population]),

        (population, "Prioritized", 'Mean'): np.mean(screened[population]),
        (population, "Prioritized", 'Std'): np.std(screened[population]),
        (population, "Wrongly prioritized (FPR)", 'Mean'): np.mean(screened_fpr[population]),
        (population, "Wrongly prioritized (FPR)", 'Std'): np.std(screened_fpr[population]),
        (population, "Wrongly not prioritized (FNR)", 'Mean'): np.mean(screened_fnr[population]),
        (population, "Wrongly not prioritized (FNR)", 'Std'): np.std(screened_fnr[population])
    })

  return pd.Series(result)
  
## Run evaluation
performances = {}
for demo_group, demo_group_data in demographics.items():
  print('--------------------------------------------')
  print('Group: ' + demo_group)
  performance_group = {}
  for strategy in predictions:
    print('Imputation Strategy: ' + strategy)

    np.random.seed(42)
    strategy_preds = predictions[strategy]

    test_indices = strategy_preds[strategy_preds['Train Test Label'].str.contains('Test')].index
    y_pred_test = strategy_preds.loc[test_indices]['Mean'].values
    y_true_test = outcomes['Outcome'].loc[test_indices].values
    demo_group_data_test = demo_group_data['data'].loc[test_indices]
    performance_group[strategy] = evaluate(y_pred_test, y_true_test, demo_group_data_test, demo_group_data['populations'])

  performances[demo_group] = pd.concat(performance_group, axis = 1).T  
  
## Display performance
metrics = ['Prioritized', 'AUC ROC', 'Wrongly not prioritized (FNR)'] 
for metric in metrics:  
  print(metric) 
  for demo_group, demo_group_data in demographics.items():
      perf_group = performances[demo_group][demo_group_data['populations']]
      perf_group = perf_group.loc[:, perf_group.columns.get_level_values(1) == metric].droplevel(1, 1)
      perf_group = pd.DataFrame.from_dict({model: ["{:.3f} ({:.3f})".format(perf_group.loc[model].loc[i].Mean, perf_group.loc[model].loc[i].Std) for i in perf_group.loc[model].index.get_level_values(0).unique()] for model in perf_group.index}, columns = perf_group.columns.get_level_values(0).unique(), orient = 'index')
      # print(perf_group.T.to_latex())
      display(perf_group.T)

## Comparison across groups
for metric in metrics:
  print(metric) 
  # Difference in FNR
  comparison = {}
  for model in performances['OVERALL'].index:
      comparison[model] = pd.concat({
          'INSURANCE': performances['INSURANCE'].loc[model]['Difference Public - Private'][metric],
          'GENDER': performances['GENDER'].loc[model]['Difference Female - Male'][metric],
          'ETHNICITY': performances['ETHNICITY'].loc[model]['Difference Black - Non Black'][metric],
      }, axis = 1).T
  metrics_short = {
      "Brier Score": "Brier",
      "AUC ROC": "AUC",
      "FPR @ 90% TPR": "False Positive Rate",
      "TPR @ 10% FPR": "True Positive Rate",
      "Prioritized": "Prioritisation",
      "Wrongly prioritized (FPR)": "False Positive Rate",
      "Wrongly not prioritized (FNR)": "False Negative Rate"
  }
  pd.concat(comparison, axis = 1)
  comparison = pd.concat(comparison, axis = 1).swaplevel(0, axis = 1)
  comparison
  ax = comparison.Mean.plot.barh(xerr = 1.96 * comparison.Std / np.sqrt(100), width = 0.7, legend = 'FNR' in metric, figsize = (6.4, 4.8))
  hatches = ['', 'ooo', 'xx', '//', '||', '***', '++']
  for i, thisbar in enumerate(ax.patches):
      c = list(plt_colors.to_rgba('tab:blue'))
      c[3] = 0.35 if i // len(comparison) < 2 else 1
      thisbar.set(edgecolor = '#eaeaf2', facecolor = c, linewidth = 1, hatch = hatches[i // len(comparison)])

  if 'FNR' in metric:
      patches = [ax.patches[i * len(comparison)] for i in range(len(comparison.Mean.columns))][::-1]
      labels = comparison.Mean.columns.tolist()[::-1]
      ax.legend(patches, labels, loc='upper left', bbox_to_anchor=(1.15, 1.04),
          title = 'Imputation strategies', handletextpad = 0.5, handlelength = 1.0, columnspacing = -0.5,)
      # ax.set_yticklabels([])
  plt.xlim(-0.30, 0.30)
  plt.axvline(0, ls = '--', alpha = 0.5, c = 'k')
  plt.xlabel('$\Delta$ {}'.format(metrics_short[metric]))
  plt.show()
  # print(pd.DataFrame.from_dict({group: ["{:.3f} ({:.3f})".format(comparison.loc[group].loc[('Mean', i)], comparison.loc[group].loc[('Std', i)]) for i in comparison.loc[group].index.get_level_values(1).unique()] for group in comparison.index}, columns = comparison.columns.get_level_values(1).unique(), orient = 'index').to_latex())
  display(pd.DataFrame.from_dict({group: ["{:.3f} ({:.3f})".format(comparison.loc[group].loc[('Mean', i)], comparison.loc[group].loc[('Std', i)]) for i in comparison.loc[group].index.get_level_values(1).unique()] for group in comparison.index}, columns = comparison.columns.get_level_values(1).unique(), orient = 'index'))
 
## Report time and memory usage
# Record end time
end = time.time()

# Print the difference between start and end time in milli. secs
print("The time of execution of the above program is :", (end-start) * 10**3, "ms")

# Display memory usage
print("The memory usage of the above program (current memory, peak memory) is :", tracemalloc.get_traced_memory())
computational_requirements = pd.DataFrame(
    data = {'Analysis': [(end - start)/60, tracemalloc.get_traced_memory()[1]/10**9]},
    index = ['Time (min)', 'Memory Usage (GB)'])
display(computational_requirements)

# Stopping the memory trace
tracemalloc.stop()

