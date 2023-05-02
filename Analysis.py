### Start tracking time and memory usage
# Import time & memory modules
import time
import tracemalloc
 
# Starting the monitoring
start = time.time()
tracemalloc.start()


### Import libraries
import numpy as np
import pandas as pd
import scipy.stats
from sklearn.metrics import roc_auc_score

from IPython.display import display
import matplotlib.pyplot as plt
from matplotlib import colors as plt_colors
import seaborn as sns


### Set chart defaults
# Set dots per inches
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
sns.set_theme(style = "darkgrid", font_scale = 1, rc = {'figure.dpi':300, 'savefig.dpi':300})


### Set path for getting data
PREPROCESSED_DATA_PATH = './PreprocessedData/'
EXPERIMENT_RESULTS_PATH = './ExperimentResults/'

labs = pd.read_csv(PREPROCESSED_DATA_PATH + 'preprocessed_labs.csv', index_col = [0, 1], header = [0, 1])
outcomes = pd.read_csv(PREPROCESSED_DATA_PATH + 'preprocessed_labeled_outcomes.csv', index_col = 0)

predictions = {}
for imputation in ['Median', 'MICE', 'Group MICE', 'Group MICE Missing']:
  predictions[imputation] = pd.read_csv(EXPERIMENT_RESULTS_PATH + 'experiment_results_' + imputation + '.csv', index_col = 0)

demographics = {
    'GENDER': {'data': outcomes['GENDER'], 'populations': ['Female', 'Male']}, 
    'ETHNICITY': {'data': outcomes['ETHNICITY'], 'populations': ['Black', 'Non Black']}, 
    'INSURANCE': {'data': outcomes['INSURANCE'], 'populations': ['Public', 'Private']}, 
    'Outcome': {'data': outcomes['Outcome'], 'populations': ['Death', 'Alive']}
}


### Dataset population statistics
# Average # of lab events by population
# Data for line plot
bins = np.linspace(0, 24, endpoint = True)
evolution = labs.groupby('SUBJECT_ID').apply(lambda x: pd.DataFrame({
    'Cumulative number of Lab Events': np.histogram(24 * x.index.get_level_values('LOS_LAB_EVENT'), bins)[0].cumsum(), 
    'Hour after admission': bins[1:]}))

# Display lab events over time for each demographic
fig, axes = plt.subplots(1, 4, figsize = (18, 3))
for index, (d_name, d_values) in enumerate(demographics.items()):
  sns.lineplot(ax = axes[index], data = evolution.join(d_values['data']),
               x = "Hour after admission", y = "Cumulative number of Lab Events", 
               hue = d_name, hue_order = d_values['populations'][::-1]).set(title = 'Cumulative Lab Events by ' + d_name)

# Set precision
pd.set_option('display.precision', 1)

# Data for bar plot
lab_event_count = labs.groupby('SUBJECT_ID').size()

# Display average lab events and tests for each demographic
plt.subplots(figsize=(18, 3))
for index, (d_name, d_values) in enumerate(demographics.items()):
  minority_subj_ids = outcomes[outcomes[d_name] == d_values['populations'][0]].index
  minority_num_events = lab_event_count[minority_subj_ids].mean()
  minority_num_labs = labs.loc[minority_subj_ids].notna().sum().sum()
  
  majority_subj_ids = outcomes[outcomes[d_name] == d_values['populations'][1]].index
  majority_num_events = lab_event_count[majority_subj_ids].mean()
  majority_num_labs = labs.loc[majority_subj_ids].notna().sum().sum()

  lab_stats = pd.DataFrame(
      data = {d_values['populations'][1]: [len(majority_subj_ids), majority_num_events, majority_num_labs / len(majority_subj_ids)],
              d_values['populations'][0]: [len(minority_subj_ids), minority_num_events, minority_num_labs / len(minority_subj_ids)]},
      index = ['# of Patients', 'Avg # Lab Events', 'Avg # Lab Tests'])
  axes = plt.subplot(1, 4, index + 1)
  lab_stats.loc[['Avg # Lab Events', 'Avg # Lab Tests']].plot.barh(ax = axes, width = 0.7)
  if index != 0:
    plt.yticks([])
  for c in axes.containers:
    plt.bar_label(c, fmt='%.1f', label_type = 'center', color = 'white')
  
  display(lab_stats)


### Analyze experiment results
def calculateMetrics(y_pred, y_true, demo_data, demo_populations, threshold_percent = 0.3, iterations = 100):
  # Threshold for % of patients to prioritize (default is top 30%)
  threshold_top = pd.Series(y_pred).nlargest(int(len(y_pred) * threshold_percent), keep = 'all').min()

  # Prepare to collect calculated metrics
  metric_list = ['AUC ROC', 'Prioritized Percentage', 'Wrongly not prioritized (FNR)']
  bootstrap_values = {}
  average_metrics = {}
  gap_metrics = {}

  for pop_index, population in enumerate(demo_populations):
    # Placeholders for storing results
    bootstrap_values[population] = {}
    average_metrics[population] = {}
    for metric in metric_list:
      bootstrap_values[population][metric] = []
    
    # Get y_pred and y_true data for this population
    if population == 'Overall':
      y_pred_pop = y_pred
      y_true_pop = y_true
    else:
      y_pred_pop = y_pred[demo_data == population]
      y_true_pop = y_true[demo_data == population]

    # Use bootstrap method for validation
    for i in range(iterations):
      # Select bootstrap sample
      bootstrap = np.random.choice(np.arange(len(y_pred_pop)), size = len(y_pred_pop), replace = True)
      this_y_pred = y_pred_pop[bootstrap]
      this_y_true = y_true_pop[bootstrap]

      # Get metrics for boostrapped sample
      bootstrap_values[population]['AUC ROC'].append(roc_auc_score(this_y_true, this_y_pred))
      selected = this_y_pred >= threshold_top
      bootstrap_values[population]['Prioritized Percentage'].append(np.mean(selected)) # Percentage of patients in this demo group that are prioritized
      bootstrap_values[population]['Wrongly not prioritized (FNR)'].append((this_y_true[~selected]).sum() / this_y_true.sum()) # Wrongly not prioritized

    # Aggregate metrics using averages
    for metric in metric_list:
      # Average metrics
      average_metrics[population][metric] = np.mean(bootstrap_values[population][metric])
      # Population gap metrics
      if pop_index == 1:
        gap_metrics[metric] = average_metrics[demo_populations[0]][metric] - average_metrics[demo_populations[1]][metric]

  return average_metrics, gap_metrics


### Calculate metrics for each population and imputation strategy
average_metrics = {}
gap_metrics = {}
for demo_key, demo_val in demographics.items():
  # Overall stats only need to be calculated once
  if demo_key == 'Outcome':
    demo_key = 'OVERALL'
    demo_val = {'data': outcomes, 'populations': ['Overall']}
  
  print('Demographic:', demo_key)
  average_metrics[demo_key] = {}
  gap_metrics[demo_key] = {}
  for imputation in predictions:
    print('Imputation Strategy:', imputation)
    np.random.seed(42)
    imputation_predictions = predictions[imputation]
    test_indices = imputation_predictions[imputation_predictions['Train Test Label'].str.contains('Test')].index
    y_pred_test = imputation_predictions.loc[test_indices]['Mean'].values
    y_true_test = outcomes['Outcome'].loc[test_indices].values == 'Death'
    average_metrics[demo_key][imputation], gap_metrics[demo_key][imputation] = calculateMetrics(y_pred_test, y_true_test, demo_val['data'].loc[test_indices], demo_val['populations'])

metric_list = ['AUC ROC', 'Prioritized Percentage', 'Wrongly not prioritized (FNR)']


### Compare results for minority vs majority population
# Set precision
pd.set_option('display.precision', 3)

# Format gap metric results into clean dataframe
gap_metrics_df = pd.DataFrame([[demographic, strategy, metric, value]
                               for demographic, d in gap_metrics.items()
                               for strategy, s in d.items()
                               for metric, value in s.items()],
                              columns = ['Demographic', 'Strategy', 'Metric', 'Value'])

# Display results for each gap metric
plt.subplots(figsize=(18, 5))
for index, metric in enumerate(metric_list):
  # Display tables for each metric
  this_gap_metric = gap_metrics_df[['Demographic', 'Strategy', 'Value']][gap_metrics_df['Metric'] == metric].pivot(index = 'Demographic', columns = 'Strategy')
  this_gap_metric.columns = this_gap_metric.columns.droplevel(0)
  this_gap_metric = this_gap_metric[['Group MICE Missing', 'Group MICE', 'MICE', 'Median']]
  print('-------', metric, '(Minority vs Majority Gap)', '-------')
  display(this_gap_metric[this_gap_metric.columns[::-1]])

  # Display bar graph for each metric
  axes = plt.subplot(1, 3, index + 1)
  this_gap_metric.plot.barh(ax = axes, width = 0.7, color = ['tab:blue', 'tab:cyan', 'tab:pink', 'tab:gray'])
  # Format & hide legend
  patches = [axes.patches[i * len(this_gap_metric)] for i in range(len(this_gap_metric.columns))]
  labels = this_gap_metric.columns.tolist()
  axes.legend(patches, labels, ncol = 4, loc='upper left', bbox_to_anchor=(0.0, 1.2), title = 'Imputation strategies')
  if index != 0:
    axes.get_legend().remove()
  # Format plot
  plt.ylabel('')
  if index != 0:
    plt.yticks([])
  plt.xlim(-0.30, 0.30)
  plt.axvline(0, ls = '--', alpha = 0.5, c = 'k')
  plt.xlabel('$\Delta$ {}'.format(metric))
  for c in axes.containers:
    plt.bar_label(c, fmt = '%.3f')
  plt.gca().invert_yaxis()


### Display results for each metric for each population
# Format average metric results into clean dataframe
average_metrics_df = pd.DataFrame([[demographic, strategy, population, metric, value]
                               for demographic, d in average_metrics.items()
                               for strategy, s in d.items()
                               for population, p in s.items()
                               for metric, value in p.items()],
                              columns = ['Demographic', 'Strategy', 'Population', 'Metric', 'Value'])

# Display results for each metric
plt.subplots(figsize=(30, 4))
plt.subplots_adjust(wspace = 0.1)

for index, metric in enumerate(metric_list):
  # Display tables for each metric
  this_avg_metric = average_metrics_df[['Demographic', 'Population', 'Strategy', 'Value']][average_metrics_df['Metric'] == metric].pivot(index = ['Demographic', 'Population'], columns = 'Strategy')
  this_avg_metric.columns = this_avg_metric.columns.droplevel(0)
  this_avg_metric = this_avg_metric[['Median', 'MICE', 'Group MICE', 'Group MICE Missing']].reindex(['Black', 'Non Black', 'Female', 'Male', 'Public', 'Private', 'Overall'], level = 1)
  print('-------', metric, '(Average)', '-------')
  display(this_avg_metric)

  # Display bar graph for each metric
  axes = plt.subplot(1, 3, index + 1)
  this_avg_metric.plot.bar(ax = axes, width = 0.8, color = ['tab:gray', 'tab:pink', 'tab:cyan', 'tab:blue'])
  # Format & hide legend
  patches = [axes.patches[i * len(this_avg_metric)] for i in range(len(this_avg_metric.columns))]
  labels = this_avg_metric.columns.tolist()
  axes.legend(patches, labels, ncol = 4, loc='upper left', bbox_to_anchor=(0.0, 1.25), title = 'Imputation strategies')
  if index != 0:
    axes.get_legend().remove()
  # Format plot
  plt.ylabel('')
  plt.xlabel(metric)
  for c in axes.containers:
    plt.bar_label(c, fmt = '%.3f', rotation = 'vertical', label_type = 'center', color = 'white')


### Report time and memory usage
# Record end time
end = time.time()

display(pd.DataFrame(
    data = {'Analysis': [(end - start)/60, tracemalloc.get_traced_memory()[1]/10**9]},
    index = ['Time (min)', 'Memory Usage (GB)']))
 
# Stopping the memory trace
tracemalloc.stop()