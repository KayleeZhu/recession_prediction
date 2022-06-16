import pandas as pd
import numpy as np
import os
import pickle
import itertools
from pathlib import Path

from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score
from sklearn import metrics
import matplotlib.pyplot as plt

import config
import data
from model_pipeline import ModelPipeline, get_train_val_indices


def check_directory_exist(directory):
    # Create folder if not exist. If not then create the directory
    if not os.path.isdir(directory):
        os.mkdir(directory)


def get_model_metrics(y_true, y_pred):
    """
    This function calculate different performance metrics of the predicted value by passing the predicted y & actual y
    :param y_true: The actual y values
    :param y_pred: The predicted y values
    :return: A dictionary that stores all the performance metrics
    """
    metrics_dict = dict(accuracy=metrics.accuracy_score(y_true, y_pred),
                        balanced_accuracy=metrics.balanced_accuracy_score(y_true, y_pred),
                        average_precision=metrics.average_precision_score(y_true, y_pred),
                        f1_binary=metrics.f1_score(y_true, y_pred, average='binary'),
                        f1_micro=metrics.f1_score(y_true, y_pred, average='micro'),
                        f1_macro=metrics.f1_score(y_true, y_pred, average='macro'),
                        f1_weighted=metrics.f1_score(y_true, y_pred, average='weighted'),
                        precision=metrics.precision_score(y_true, y_pred),
                        recall=metrics.recall_score(y_true, y_pred),
                        roc_auc_micro=metrics.roc_auc_score(y_true, y_pred, average='micro'),
                        roc_auc_macro=metrics.roc_auc_score(y_true, y_pred, average='macro'),
                        roc_auc_weighted=metrics.roc_auc_score(y_true, y_pred, average='weighted')
                        )
    return metrics_dict


def features_experiment(model_type, data_dir, exp_dir):
    """
    This function will explore different features combinations by choosing one feature in each of the economic category
    to see which set has the best predictability.
    This will only run for one type of ML model, specified by the "model_type" parameter.
    :param model_type: str, type of ML model want to run on
    :param data_dir: directory of the processed data csv file
    :param exp_dir: directory (folder) that you want to save all the experiment results
    """

    # Read processed data from directory
    all_df = pd.read_csv(data_dir).set_index('date')
    # Check if the folder directory exists for saving all the experiment results
    check_directory_exist(exp_dir)
    # Create sub folder for the model type
    save_folder = Path(f'{exp_dir}') / Path(f'{model_type}')
    check_directory_exist(save_folder)

    score_method = 'f1_macro'
    random_state = 1

    # Sub features
    emp_col = ['payroll_diff_12m_1m', 'payroll_diff_12m_3m', 'payroll_diff_12m_6m']
    mon_col = ['policy_rate_delta_1m', 'policy_rate_delta_3m', 'policy_rate_delta_6m', 'policy_rate_delta_12m']
    inf_col = ['CPI_delta_1m', 'CPI_delta_3m', 'CPI_delta_6m', 'CPI_delta_12m']
    yield_col = ['yield_10y_delta1m', 'yield_10y_delta3m', 'yield_10y_delta6m', 'yield_10y_delta12m']
    spread_col = ['yield_spread']
    stock_col = ['sp500_delta_1m', 'sp500_delta_3m', 'sp500_delta_6m', 'sp500_delta_12m']
    vol_col = ['vol_30d']
    y_col = ['recession_3m', 'recession_6m', 'recession_12m']

    # Get all possible sub features
    iter_list = [emp_col, mon_col, inf_col, yield_col, spread_col, stock_col, vol_col]
    all_sub_fea = list(itertools.product(*iter_list))

    counter = 0
    for fea in all_sub_fea:
        fea_list = list(fea)
        for y_name in y_col:
            counter += 1
            print(f'Running Experiment {counter}:'
                  f'Y is {y_name}, '
                  f'X are {fea_list}'
                  f'')
            sub_all_df = all_df[fea_list + [y_name]].copy().reset_index()
            sub_all_df = sub_all_df.dropna()

            # X = sub_all_df[fea_list].copy()
            y_true = sub_all_df[y_name].copy().reset_index()

            # Train Model
            model = ModelPipeline(model_type=model_type,
                                  feature_list=fea_list,
                                  y_var_name=y_name,
                                  score_method=score_method,
                                  random_state=random_state,
                                  recession_period=y_name[10:]
                                  )
            model.hyperparameter_tuning(sub_all_df, sub_all_df)
            model.model_training(sub_all_df, sub_all_df)
            y_data = model.get_prediction(sub_all_df, y_true)

            # Save Info
            exp_folder = Path(f'{save_folder}') / Path(f'exp_{counter}')
            check_directory_exist(exp_folder)

            # Model Metrics & Other Info
            metrics_dict = get_model_metrics(y_data[y_name], y_data['y_pred'])
            metrics_dict['exp_num'] = counter
            pd.DataFrame.from_dict(metrics_dict, orient='index').to_csv(
                Path(f'{exp_folder}') / Path('model_metrics.csv'))

            info_dict = dict(exp_number=counter,
                             model_type=model_type,
                             features=fea_list,
                             y_variable=y_name,
                             score_method=score_method,
                             random_state=random_state,
                             )
            pd.DataFrame.from_dict(info_dict, orient='index').to_csv(f'{exp_folder}/info.csv')

            # Save Model
            filename = f'{exp_folder}/model.pkl'
            with open(filename, 'wb') as f:
                pickle.dump(model.pipeline, f)


def out_sample_nested_experiments(model_type, exp_list, insample_exp_dir, outsample_exp_dir):
    df = pd.read_csv('all_data.csv')  # .set_index('date').reset_index()
    # df = df.reset_index().copy()
    score_method = 'f1_macro'
    random_state = 1
    check_directory_exist(outsample_exp_dir)
    # save_folder = f'data_folder/experiments_out_sample/{model_type}'
    save_folder = Path(outsample_exp_dir) / Path(f'{model_type}')
    check_directory_exist(save_folder)

    # Get Experiment info of the top models
    all_info, all_metrics = read_experiment_results(model_type, exp_list, insample_exp_dir)

    for exp_num in exp_list:
        exp_num = int(exp_num)
        mask = all_info['exp_number'] == str(exp_num)
        feas = all_info.loc[mask, 'features'][0].replace('[', '').replace(']', '').replace(' ', '').replace("'",
                                                                                                            '').split(
            ',')
        y_name = all_info.loc[mask, 'y_variable'][0]
        print(f'For exp {exp_num}:')
        print(f'Y is {y_name} & Features are: {feas}')
        print(" ")
        sub_df = df[['date'] + feas + [y_name]].copy().dropna()

        # Get k fold indices
        fold_indices = get_train_val_indices(sub_df, lag=5, recession_period='3m')
        for k in range(0, len(fold_indices)):
            train_ind = fold_indices[k][0].copy()
            test_ind = fold_indices[k][1].copy()
            # Get train & test data set
            train_set = sub_df.iloc[train_ind].copy()
            test_set = sub_df.iloc[test_ind].copy()
            X_train = train_set[['date'] + feas].copy()
            y_train = train_set[y_name].reset_index()
            X_test = test_set[feas].copy()
            y_test = test_set[y_name].reset_index()

            # Train Model on train set
            model = ModelPipeline(model_type=model_type,
                                  feature_list=feas,
                                  y_var_name=y_name,
                                  score_method=score_method,
                                  random_state=random_state,
                                  recession_period=y_name[10:]
                                  )
            model.hyperparameter_tuning(X_train, y_train)
            model.model_training(X_train, y_train)

            # Make prediction on test set
            y_data = model.get_prediction(X_test, y_test)

            # Save Info
            exp_folder = Path(save_folder) / Path(f'exp_{exp_num}')
            fold_path = Path(save_folder) / Path(f'exp_{exp_num}') / Path(f'fold_{k}')
            check_directory_exist(exp_folder)
            check_directory_exist(fold_path)
            y_data.to_csv(Path(fold_path) / Path('y_data.csv'))

            # Model Metrics & Other Info
            metrics_dict = get_model_metrics(y_data[y_name], y_data['y_pred'])
            metrics_dict['exp_num'] = exp_num
            metrics_dict['fold_num'] = k

            model_metrics_save_path = Path(fold_path) / Path('model_metrics.csv')
            pd.DataFrame.from_dict(metrics_dict, orient='index').to_csv(model_metrics_save_path)

            info_dict = dict(exp_number=exp_num,
                             fold_num=k,
                             model_type=model_type,
                             features=feas,
                             y_variable=y_name,
                             score_method=score_method,
                             random_state=random_state,
                             train_ind=train_ind,
                             test_ind=test_ind
                             )
            info_save_path = Path(fold_path) / Path('info.csv')
            pd.DataFrame.from_dict(info_dict, orient='index').to_csv(info_save_path)
            # pd.DataFrame(info_dict).to_csv(f'{save_folder}/info.csv')

            # Save Model
            filename = Path(fold_path) / Path('model.pkl')
            with open(filename, 'wb') as f:
                pickle.dump(model, f)

    print('done')


def read_experiment_results(model_type, exp_list, exp_dir):
    info_list = []
    metrics_list = []
    for i in exp_list:
        i = int(i)
        save_folder = Path(exp_dir) / Path(f'{model_type}') / Path(f'exp_{i}')
        info_path = Path(save_folder) / Path('info.csv')
        metrics_path = Path(save_folder) / Path('model_metrics.csv')

        info_df = pd.read_csv(info_path).set_index('Unnamed: 0').T
        metrics_df = pd.read_csv(metrics_path).set_index('Unnamed: 0').T
        info_list.append(info_df)
        metrics_list.append(metrics_df)

    all_info = pd.concat(info_list)
    all_metrics = pd.concat(metrics_list)
    return all_info, all_metrics


def read_out_sample_results(model_type, exp_list, exp_dir):
    info_list = []
    metrics_list = []
    for i in exp_list:
        i = int(i)
        for k in range(0, 5):
            save_folder = Path(exp_dir) / Path(model_type) / Path(f'exp_{i}') / Path(f'fold_{k}')
            info_path = Path(save_folder) / Path('info.csv')
            metrics_path = Path(save_folder) / Path('model_metrics.csv')

            info_df = pd.read_csv(info_path).set_index('Unnamed: 0').T
            metrics_df = pd.read_csv(metrics_path).set_index('Unnamed: 0').T
            info_list.append(info_df)
            metrics_list.append(metrics_df)

            all_info = pd.concat(info_list)
            all_metrics = pd.concat(metrics_list)
    return all_info, all_metrics


if __name__ == '__main__':
    pass
