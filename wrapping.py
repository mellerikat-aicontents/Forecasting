import argparse
import time
import os
import pandas as pd
import numpy as np
import copy
os.chdir("..")
from src.alo import ALO
from src.alo import AssetStructure
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.stats import skew, kurtosis
import matplotlib.font_manager as fm

import subprocess
import os




# 폰트 파일 경로와 사용자 폰트 디렉토리 경로 설정
font_path = './solution/.fonts/NanumGothic.ttf'
user_font_dir = os.path.expanduser('~/.fonts/')

# 사용자 폰트 디렉토리가 없으면 생성
if not os.path.exists(user_font_dir):
    os.makedirs(user_font_dir)

# 폰트 파일을 사용자 폰트 디렉토리로 복사
subprocess.run(['cp', font_path, user_font_dir])

# 폰트 캐시를 갱신
subprocess.run(['fc-cache', '-fv'])

class Wrapper(object):
    def __init__(self):
        self.alo = ALO(); 
        self.pipelines = list(self.alo.asset_source.keys())

    def run(self, step, pipeline, asset_structure):
        # 반복되는 작업을 함수로 변환
        asset_config = self.alo.asset_source[pipeline]
        return self.pipeline.process_asset_step(asset_config[step], step)

    def run_train_pipeline(self):
        pipe = self.pipelines[0]
        self.alo._external_load_data(pipe)
        self.alo.set_metadata(pipeline_type=pipe.split('_')[0])
        self.pipeline = self.alo.pipeline(pipeline_type=pipe)
        self.pipeline._set_asset_structure()
        #pipeline.load()

    def run_inference_pipeline(self):
        pipe = self.pipelines[1]
        self.alo._external_load_data(pipe)
        self.alo.set_metadata(pipeline_type=pipe.split('_')[0])
        self.pipeline = self.alo.pipeline(pipeline_type=pipe)
        self.pipeline._set_asset_structure()

    def get_asset_structure(self, pipeline, step):
        asset_config = self.pipeline.asset_source[pipeline][step]
        self.pipeline.asset_structure.args[asset_config['step']] = self.pipeline.get_parameter(asset_config['step'])
        self.pipeline.process_asset_step(asset_config, step)
        asset_structure = copy.deepcopy(self.pipeline.asset_structure)
        return asset_structure
        
    def get_train_asset_output(self):
        cv_result_dict = None
        cv_score_df = None

        if os.path.exists('train_artifacts/extra_output'):
            filelist = os.listdir('train_artifacts/extra_output/train')
            cv_result_dict = {}
            for file in filelist:
                cv_result_dict[file.split('.')[0]] = pd.read_csv(os.path.join('train_artifacts/extra_output/train', file))
        
        if os.path.exists('train_artifacts/output'):
            cv_score_df = pd.read_csv('train_artifacts/output/train_score.csv')

        return cv_result_dict, cv_score_df
    
    def get_inference_asset_output(self):
        inference_prediction_df = None
        if os.path.exists('inference_artifacts/output'):
            filelist = os.listdir('inference_artifacts/output')
            inference_prediction_df = pd.read_csv(os.path.join('inference_artifacts/output', filelist[0]))
        return inference_prediction_df


class EvaluationReport:
    def __init__(self, asset_structure):
        
        self.asset_structure = asset_structure
        self.config = self.asset_structure.config
        self.dataset = self.asset_structure.data['dataframe']
        plt.rc('font', family='NanumGothic')
        plt.rcParams['axes.unicode_minus'] =False
        
    def summarize_variable_composition(self, dataset, config):
        total_columns = {"Component": "Total columns", "Count": len(dataset.columns), "List": list(dataset.columns)}
        y_column = {"Component": "Target column", "Count": 1, "List": [config['y_column']]}
        time_column = {"Component": "Time column", "Count": 1, "List": [config['time_column']]}
        if (config['groupkey_column'] != '') | (config['groupkey_column'] is not None):
            groupkey_column = {"Component": "Groupkey column", "Count": 1, "List": [config['groupkey_column']]}
        else:
            groupkey_column = {"Component": "Groupkey column", "Count": "-", "List": "-"}
        if len(config['x_covariates']) > 0:
            x_covariates = {"Component": "X covaiates", "Count": len(config['x_covariates']), "List": config['x_covariates']}
        else:
            groupkey_column = {"Component": "Static covariates", "Count": "-", "List": "-"}
        if len(config['static_covariates']) > 0:
            static_covariates = {"Component": "Static covariates", "Count": len(config['static_covariates']), "List": config['static_covariates']}
        else:
            static_covariates = {"Component": "Static covariates", "Count": "-", "List": "-"}
        return pd.DataFrame([total_columns, y_column, time_column, x_covariates, groupkey_column, static_covariates])

    def time_length_per_groupkey(self, dataset, config, show_boxplot=False):
        if (config['groupkey_column'] is not None) | (config['groupkey_column'] != ''):
            length_by_group = dataset.groupby([config['groupkey_column']]).count()[config['time_column']].reset_index().sort_values([config['time_column']])
            Statitistics = ['Min', 'Q1', 'Median', 'Q3', 'Max', 'Count', 'Mean', 'Std', 'Skewness', 'Kurtosis']
            min_value = np.min(length_by_group[config['time_column']])
            Q1_value = length_by_group[config['time_column']].quantile(0.25, interpolation = 'nearest')
            median_value = int(np.median(length_by_group[config['time_column']]))
            Q3_value = length_by_group[config['time_column']].quantile(0.75, interpolation = 'nearest')
            max_value = np.max(length_by_group[config['time_column']])
            count_value = len(length_by_group[config['time_column']])
            mean_value = round(np.mean(length_by_group[config['time_column']]), 3)
            std_value = round(np.std(length_by_group[config['time_column']]), 3)
            skewness_value = round(skew(length_by_group[config['time_column']]), 3)
            kurtosis_value = round(kurtosis(length_by_group[config['time_column']]), 3)
            Values = [min_value, Q1_value, median_value, Q3_value, max_value, count_value, mean_value, std_value, skewness_value, kurtosis_value]
            Values = [str(val) for val in Values]
            min_example = list(length_by_group[config['groupkey_column']][length_by_group[config['time_column']] == min_value])[0]
            Q1_example = list(length_by_group[config['groupkey_column']][length_by_group[config['time_column']] == Q1_value])[0]
            median_example = list(length_by_group[config['groupkey_column']][length_by_group[config['time_column']] == median_value])[0]
            Q3_example = list(length_by_group[config['groupkey_column']][length_by_group[config['time_column']] == Q3_value])[0]
            max_example = list(length_by_group[config['groupkey_column']][length_by_group[config['time_column']] == max_value])[0]
            Examples = [min_example, Q1_example, median_example, Q3_example, max_example, '-', '-', '-', '-', '-']
            result = pd.DataFrame({"Statitistic": Statitistics, "Value": Values, "Example": Examples})
            print(result)
            if show_boxplot:
                sns.set_theme(style="whitegrid") 
                plt.rc('font', family='NanumGothic')
                plt.rcParams['axes.unicode_minus'] =False
                ax = sns.boxplot(x=0, y=config['time_column'], data=length_by_group)
                ax = sns.swarmplot(x=0, y=config['time_column'], data=length_by_group, color="red")
                ax.set_ylabel("Timeseries Length")
                ax.set_xticks([])
                plt.show()
        else:
            print("Your dataset has no groupkey.")

    def timeseries_plot(self, dataset, config):
        plt.figure(figsize=(16,8)) 
        if (config['groupkey_column'] is not None) | (config['groupkey_column'] != ""):
            sns.lineplot(data=dataset, x=config['time_column'], y=config['y_column'], hue=config['groupkey_column'])
        else:
            sns.lineplot(data=dataset, x=config['time_column'], y=config['y_column'])

    def moving_average_plot(self, dataset, config, groupkey=None, moving_windows=[5, 10, 15], show_each=False):
        if groupkey is not None:
            dataset = dataset.loc[dataset[config['groupkey_column']] == groupkey, :]        
        moving_average1 = pd.Series.rolling(dataset[config['y_column']], window=moving_windows[0], center = False).mean()
        moving_average2 = pd.Series.rolling(dataset[config['y_column']], window=moving_windows[1], center = False).mean()
        moving_average3 = pd.Series.rolling(dataset[config['y_column']], window=moving_windows[2], center = False).mean()

        if not show_each:
            fig = plt.figure(figsize = (20, 12))
            chart = fig.add_subplot(1,1,1)

            chart.plot(dataset[config['time_column']], dataset[config['y_column']], color='blue' , label='Original')
            chart.plot(dataset[config['time_column']], moving_average1, color='red', label='Moving average with {} windows'.format(moving_windows[0]))
            chart.plot(dataset[config['time_column']], moving_average2, color='orange', label='Moving average with {} windows'.format(moving_windows[1]))
            chart.plot(dataset[config['time_column']], moving_average3, color='green', label='Moving average with {} windows'.format(moving_windows[2]))
            plt.legend(loc = 'best')
        else:
            fig, axes = plt.subplots(2, 2, figsize=(20,12))
            axes[0,0].plot(dataset[config['time_column']], dataset[config['y_column']], color='blue' )
            axes[0,0].tick_params(axis='x', labelrotation=45)
            axes[0,0].set_title("Original")
            axes[0,1].plot(dataset[config['time_column']], moving_average1, color='red' )
            axes[0,1].tick_params(axis='x', labelrotation=45)
            axes[0,1].set_title('Moving average with {} windows'.format(moving_windows[0]))
            axes[1,0].plot(dataset[config['time_column']], moving_average2, color='orange')
            axes[1,0].tick_params(axis='x', labelrotation=45)
            axes[1,0].set_title('Moving average with {} windows'.format(moving_windows[1]))
            axes[1,1].plot(dataset[config['time_column']], moving_average3, color='green' )
            axes[1,1].tick_params(axis='x', labelrotation=45)
            axes[1,1].set_title('Moving average with {} windows'.format(moving_windows[2]))
        
    def timeseries_acf_pacf_plot(self, dataset, config, groupkey=None, N_LAGS=20, pval=0.05):
        if groupkey is not None:
            dataset = dataset.loc[dataset[config['groupkey_column']] == groupkey, :]
        auto = pd.Series(dataset[config['y_column']])
        for i in range(0, N_LAGS+1):
            scatter1 = pd.DataFrame()
            scatter1['lags'] = [i for i in range (1, N_LAGS +1)]
            scatter1['autocorrelation'] = [auto.autocorr(lag=i) for i in range(1, N_LAGS +1)]

        for i in range(0, N_LAGS+1):
            scatter2 = pd.DataFrame()
            scatter2['lags'] = [i for i in range (1, N_LAGS +1)]
            scatter2['Partial autocorrelation'] = [pacf(auto, alpha=.05)[0][i] for i in range(1, N_LAGS +1)]
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,8))
        plot_acf(dataset[config['y_column']], lags=N_LAGS, alpha=pval, ax=ax1)
        ax1.scatter(x=scatter1['lags'], y=scatter1['autocorrelation'], edgecolors='red',linewidth=1, s=200, alpha = .5)
        plot_pacf(dataset[config['y_column']], lags=N_LAGS, alpha=pval, method='ywm', ax=ax2)
        ax2.scatter(x=scatter2['lags'], y=scatter2['Partial autocorrelation'], edgecolors='red',linewidth=1, s=200, alpha = .5)
        if groupkey is not None:
            plt.suptitle("Group: {}".format(groupkey))
        plt.show()

    def timeseries_decomposition_plot(self, dataset, config, groupkey=None, period=30):
        if groupkey is not None:
            dataset = dataset.loc[dataset[config['groupkey_column']] == groupkey, :]
        res = seasonal_decompose(dataset[config['y_column']], period=period)

        plt.figure(figsize=(12,6))
        plt.subplot(411)
        plt.xticks([])
        plt.ylabel('observed')
        plt.plot(res.observed)

        plt.subplot(412)
        plt.xticks([])
        plt.ylabel('trend')
        plt.plot(res.trend)

        plt.subplot(413)
        plt.xticks([])
        plt.ylabel('seasonal')
        plt.plot(res.seasonal)

        plt.subplot(414)
        plt.xticks([])
        plt.ylabel('residual')
        plt.plot(res.resid)

        if groupkey is not None:
            plt.suptitle('Timeseries Decomposition (Groupkey: {})'.format(groupkey))
        else:
            plt.suptitle('Timeseries Decomposition')
        plt.tight_layout()
        plt.show()

    def train_cv_score_plot(self, config, cv_score_df, chart_type='bar'):
        if (config['groupkey_column'] is not None) | (config['groupkey_column'] != ''):
            if chart_type == 'bar':
                plt.figure(figsize=(12,6))
                sns.barplot(x="CV", y=config['metric_to_compare'], data=cv_score_df, hue=config['groupkey_column'])
                plt.show()
            elif chart_type == 'box':
                plt.figure(figsize=(12,6))
                sns.boxplot(x="CV", y=config['metric_to_compare'], data=cv_score_df)
                plt.show()
        else:
            plt.figure(figsize=(12,6))
            sns.barplot(x="CV", y=config['metric_to_compare'], data=cv_score_df)
            plt.show()
        
    def train_cv_prediction_plot(self, config, cv_result_dict, cv_score_df, cv='CV1', groupkey=None):
        result = cv_result_dict[cv]
        result[config['time_column']] = pd.to_datetime(result[config['time_column']])
        if groupkey is not None:
            result = result.loc[result[config['groupkey_column']] == groupkey, :]

        fig, axes = plt.subplots(2, 1, figsize=(20,12))
        axes[0].plot(result[config['time_column']], result[config['y_column']], color='blue')
        axes[0].set_title(cv + ': ' + 'Train Period (' + list(cv_score_df['Train Period'][cv_score_df['CV'] == cv])[0] + ') → Valid Period (' + list(cv_score_df['Valid Period'][cv_score_df['CV'] == cv])[0] + ')')
        train_period = result[config['time_column']][pd.isna(result['predicted'])]
        span_start = min(train_period)
        span_end = max(train_period)
        axes[0].axvspan(span_start, span_end, facecolor='gray', alpha=0.2, label='Train Period')
        valid_period = result[config['time_column']][-pd.isna(result['predicted'])]
        span_start = span_end
        span_end = max(valid_period)
        axes[0].axvspan(span_start, span_end, facecolor='red', alpha=0.3, label='Valid Period')
        axes[0].legend(loc='best')

        result = result.dropna(subset=['predicted'])
        axes[1].plot(result[config['time_column']], result[config['y_column']], color='green' , label='Ground Truth')
        axes[1].plot(result[config['time_column']], result['predicted'], color='blue', label='Predicted')
        axes[1].set_title('Valid Period : Ground Truth vs Predicted')
        plt.legend(loc = 'best')
        if groupkey is not None:
            plt.suptitle("Group: {}".format(groupkey))
        plt.show()

    def inference_prediction_plot(self, dataset, config, inference_prediction_df, groupkey=None):
        dataset['targetdate'] = pd.to_datetime(dataset['targetdate'])
        inference_prediction_df['targetdate'] = pd.to_datetime(inference_prediction_df['targetdate'])

        if groupkey is not None:
            dataset = dataset.loc[dataset[config['groupkey_column']] == groupkey, :]
            inference_prediction_df = inference_prediction_df.loc[inference_prediction_df[config['groupkey_column']] == groupkey, :]
            inference_prediction_df = inference_prediction_df.loc[:, ['targetdate', config['y_column']]]

        inference_dataset = dataset.loc[:, ['targetdate', config['y_column']]]
        predicted_dataset = pd.concat([inference_dataset.iloc[-1:, :], inference_prediction_df])

        fig, axes = plt.subplots(1, 2, figsize=(20,12), gridspec_kw={"width_ratios": [2, 1]})
        axes[0].plot(inference_dataset['targetdate'], inference_dataset[config['y_column']], color='green' , label='Raw')
        axes[0].plot(predicted_dataset['targetdate'], predicted_dataset[config['y_column']], color='blue', label='Predicted')
        axes[0].set_title('Prediction Graph with Total')
        axes[0].legend(loc='best')

        axes[1].plot(inference_dataset['targetdate'][-config['input_chunk_length']:], inference_dataset[config['y_column']][-config['input_chunk_length']:], color='green' , label='Raw')
        axes[1].plot(predicted_dataset['targetdate'], predicted_dataset[config['y_column']], color='blue', label='Predicted')
        axes[1].set_title('Prediction Graph with Input chunk length')
        axes[0].legend(loc='best')
        if groupkey is not None:
            plt.suptitle('(Group: {}) Forecasting Periods: {} ~ {}'.format(groupkey, predicted_dataset['targetdate'][0].strftime(config['time_format']), max(predicted_dataset['targetdate']).strftime(config['time_format'])))
        else:
            plt.suptitle('Forecasting Periods: {} ~ {}'.format(predicted_dataset['targetdate'][0].strftime(config['time_format']), max(predicted_dataset['targetdate']).strftime(config['time_format'])))
        plt.show()


if __name__ == '__main__':
    wrapper = Wrapper()
    wrapper.run_train_pipeline()
    pipeline = 'train_pipeline'
    input_asset_structure = wrapper.get_input_asset(pipeline)
    print("Input Data Size: {} Rows, {} Columns".format(input_asset_structure.data['dataframe'].shape[0], input_asset_structure.data['dataframe'].shape[1]))