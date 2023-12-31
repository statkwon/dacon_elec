import numpy as np
import pandas as pd
import xgboost as xgb
from tqdm import tqdm
from typing import Tuple
import matplotlib.pyplot as plt


def smape(predt: np.ndarray, dtrain: xgb.DMatrix) -> Tuple[str, float]:
    y = dtrain.get_label()
    return 'SMAPE', 100 * np.mean(np.abs(predt - y) / ((np.abs(y) + np.abs(predt)) / 2))


def sudo_smape(predt: np.ndarray, dtrain: xgb.DMatrix) -> Tuple[np.ndarray, np.ndarray]:
    def gradient(predt: np.ndarray, dtrain: xgb.DMatrix) -> np.ndarray:
        y = dtrain.get_label()
        res = predt - y
        return np.where(res > 0, 2 * res, 1.1 * 2 * res)

    def hessian(predt: np.ndarray, dtrain: xgb.DMatrix) -> np.ndarray:
        y = dtrain.get_label()
        res = predt - y
        return np.where(res > 0, 2, 1.1 * 2)

    grad = gradient(predt, dtrain)
    hess = hessian(predt, dtrain)
    return grad, hess


def fpreproc(dtrain, dtest, param):
    train_data = pd.DataFrame(dtrain.get_data().toarray(), columns=dtrain.feature_names)
    train_label = dtrain.get_label()
    train_data['target'] = train_label
    test_data = pd.DataFrame(dtest.get_data().toarray(), columns=dtest.feature_names)
    test_label = dtest.get_label()

    gb_h_target = train_data.groupby(by='h').agg({'target': ['mean', 'std']})
    gb_h_target.columns = gb_h_target.columns.droplevel()
    gb_h_target.reset_index(inplace=True)
    gb_h_target.rename({'mean': 'gbht_mean', 'std': 'gbht_std'}, axis=1, inplace=True)
    train_data = pd.merge(train_data, gb_h_target, how='left')
    test_data = pd.merge(test_data, gb_h_target, how='left')

    bd_no = train_data['bd_no'][0]

    if bd_no in [9, 87, 88, 89, 90, 92]:
        train_data['sun24'] = (train_data['w'].isin([2, 4])) & (train_data['wd'] == 6)
        test_data['sun24'] = (test_data['w'].isin([2, 4])) & (test_data['wd'] == 6)
    elif bd_no in [7, 12]:
        train_data['sun13'] = (train_data['w'].isin([1, 3])) & (train_data['wd'] == 6)
        test_data['sun13'] = (test_data['w'].isin([1, 3])) & (test_data['wd'] == 6)
    elif bd_no == 5:
        train_data['fss'] = train_data['wd'].isin([4, 5, 6])
        test_data['fss'] = test_data['wd'].isin([4, 5, 6])
    elif bd_no in [2, 3, 54, 91]:
        train_data['mon'] = train_data['wd'] == 0
        test_data['mon'] = test_data['wd'] == 0

    train_data.drop(['bd_no', 'h', 'target'], axis=1, inplace=True)
    test_data.drop(['bd_no', 'h'], axis=1, inplace=True)

    dtrain = xgb.DMatrix(data=train_data, label=train_label)
    dtest = xgb.DMatrix(data=test_data, label=test_label)

    return dtrain, dtest, param


class Model:
    def __init__(self, train, test, feature_names=None, cv_results=None, answer=None):
        self.train = train
        self.test = test
        self.feature_names = feature_names
        self.cv_results = cv_results
        self.answer = answer

    def cross_validation(self, bd_no, params, num_boost_round, folds, early_stopping_rounds):
        data = self.train.loc[self.train['bd_no'] == bd_no, [*self.feature_names, 'bd_no']]
        label = self.train.loc[self.train['bd_no'] == bd_no, 'target']
        dtrain = xgb.DMatrix(data=data, label=label)

        cv = xgb.cv(params=params,
                    dtrain=dtrain,
                    num_boost_round=num_boost_round,
                    folds=folds,
                    obj=sudo_smape,
                    maximize=False,
                    early_stopping_rounds=early_stopping_rounds,
                    fpreproc=fpreproc,
                    shuffle=False,
                    custom_metric=smape)

        return cv

    def grid_search(self, config, num_boost_round, folds, early_stopping_rounds):
        result = []

        for bd_no in tqdm(range(1, 101)):
            for max_depth in config['max_depth']:
                for subsample in config['subsample']:
                    for colsample_bytree in config['colsample_bytree']:
                        params = {'eta': 1e-2,
                                  'max_depth': max_depth,
                                  'subsample': subsample,
                                  'colsample_bytree': colsample_bytree,
                                  'random_state': 0}
                        cv = self.cross_validation(bd_no, params, num_boost_round, folds, early_stopping_rounds)
                        best_smape_mean = cv['test-SMAPE-mean'].min()
                        best_num_boost_round = cv['test-SMAPE-mean'].argmin()
                        result.append([bd_no, max_depth, subsample, colsample_bytree, best_num_boost_round, best_smape_mean])

        result_df = pd.DataFrame(result, columns=['bd_no', 'max_depth', 'subsample', 'colsample_bytree', 'best_num_boost_round', 'test-SMAPE-mean'])
        return result_df

    def draw_last_fold_prediction(self, bd_no, params, best_num_boost_round):
        train_data = self.train.loc[self.train['bd_no'] == bd_no, self.feature_names][0:1872]
        test_data = self.train.loc[self.train['bd_no'] == bd_no, self.feature_names][1872:]
        train_label = self.train.loc[self.train['bd_no'] == bd_no, 'target'][0:1872]
        test_label = self.train.loc[self.train['bd_no'] == bd_no, 'target'][1872:]
        dtrain = xgb.DMatrix(data=train_data, label=train_label)
        dtest = xgb.DMatrix(data=test_data)

        model = xgb.train(params=params,
                          dtrain=dtrain,
                          num_boost_round=best_num_boost_round,
                          obj=sudo_smape,
                          custom_metric=smape)

        y_pred = model.predict(dtest)

        train_dt = pd.date_range('2022-06-01 00:00:00', '2022-08-17 23:00:00', freq='H')
        test_dt = pd.date_range('2022-08-18 00:00:00', '2022-08-24 23:00:00', freq='H')

        fig, ax = plt.subplots(figsize=(20, 4))
        ax.plot(train_dt, train_label)
        ax.plot(test_dt, test_label, label='true')
        ax.plot(test_dt, y_pred, label='pred')
        ax.legend()
        plt.show()

    def run_cv(self, params, num_boost_round, folds, early_stopping_rounds):
        smape_means = []
        smape_stds = []
        best_num_boost_rounds = []

        for bd_no in tqdm(range(1, 101)):
            cv = self.cross_validation(bd_no, params, num_boost_round, folds, early_stopping_rounds)
            smape_mean = cv['test-SMAPE-mean'].min()
            smape_std = cv['test-SMAPE-std'].min()
            best_num_boost_round = cv['test-SMAPE-mean'].argmin()
            smape_means.append(smape_mean)
            smape_stds.append(smape_std)
            best_num_boost_rounds.append(best_num_boost_round)

        self.cv_results = pd.DataFrame({'smape_mean': smape_means, 'smape_std': smape_stds, 'best_num_boost_round': best_num_boost_rounds})
        return self.cv_results

    def draw_cv_scores(self):
        if self.cv_results is not None:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.bar(range(1, 101), self.cv_results['smape_mean'], width=0.2)
            ax.axhline(y=5, c='r', linestyle='--')
            ax.set_title('Mean: %f | Std: %f' % (self.cv_results['smape_mean'].mean(), self.cv_results['smape_std'].mean()))
        else:
            raise Exception('run_cv() should be runned first!')

    def predict(self, params, best_num_boost_rounds, labels):
        answer = []

        gb_h_target = self.train.groupby(by=['bd_no', 'h']).agg({'target': ['mean', 'std']})
        gb_h_target.columns = gb_h_target.columns.droplevel()
        gb_h_target.reset_index(inplace=True)
        gb_h_target.rename({'mean': 'gbht_mean', 'std': 'gbht_std'}, axis=1, inplace=True)
        self.train = pd.merge(self.train, gb_h_target, how='left')
        self.test = pd.merge(self.test, gb_h_target, how='left')

        for bd_no in tqdm(range(1, 101)):
            train_data = self.train[self.train['bd_no'] == bd_no].drop([*labels, 'target'], axis=1)
            test_data = self.test[self.test['bd_no'] == bd_no].drop(labels, axis=1)
            train_label = self.train.loc[self.train['bd_no'] == bd_no, 'target']

            if bd_no in [9, 87, 88, 89, 90, 92]:
                train_data['sun24'] = (train_data['w'].isin([2, 4])) & (train_data['wd'] == 6)
                test_data['sun24'] = (test_data['w'].isin([2, 4])) & (test_data['wd'] == 6)
            elif bd_no in [7, 12]:
                train_data['sun13'] = (train_data['w'].isin([1, 3])) & (train_data['wd'] == 6)
                test_data['sun13'] = (test_data['w'].isin([1, 3])) & (test_data['wd'] == 6)
            elif bd_no == 5:
                train_data['fss'] = train_data['wd'].isin([4, 5, 6])
                test_data['fss'] = test_data['wd'].isin([4, 5, 6])
            elif bd_no in [2, 3, 54, 91]:
                train_data['mon'] = train_data['wd'] == 0
                test_data['mon'] = test_data['wd'] == 0

            dtrain = xgb.DMatrix(data=train_data, label=train_label)
            dtest = xgb.DMatrix(data=test_data)

            model = xgb.train(params=params,
                              dtrain=dtrain,
                              num_boost_round=best_num_boost_rounds[bd_no - 1],
                              obj=sudo_smape,
                              custom_metric=smape)
            model.save_model('./model/model%d.model' % bd_no)

            y_pred = model.predict(dtest)
            answer.extend(y_pred.tolist())
            self.answer = answer

        return answer

    def predict_with_best_params(self, best_params, labels):
        answer = []

        gb_h_target = self.train.groupby(by=['bd_no', 'h']).agg({'target': ['mean', 'std']})
        gb_h_target.columns = gb_h_target.columns.droplevel()
        gb_h_target.reset_index(inplace=True)
        gb_h_target.rename({'mean': 'gbht_mean', 'std': 'gbht_std'}, axis=1, inplace=True)
        self.train = pd.merge(self.train, gb_h_target, how='left')
        self.test = pd.merge(self.test, gb_h_target, how='left')

        for bd_no in tqdm(range(1, 101)):
            train_data = self.train[self.train['bd_no'] == bd_no].drop([*labels, 'target'], axis=1)
            test_data = self.test[self.test['bd_no'] == bd_no].drop(labels, axis=1)
            train_label = self.train.loc[self.train['bd_no'] == bd_no, 'target']

            if bd_no in [9, 87, 88, 89, 90, 92]:
                train_data['sun24'] = (train_data['w'].isin([2, 4])) & (train_data['wd'] == 6)
                test_data['sun24'] = (test_data['w'].isin([2, 4])) & (test_data['wd'] == 6)
            elif bd_no in [7, 12]:
                train_data['sun13'] = (train_data['w'].isin([1, 3])) & (train_data['wd'] == 6)
                test_data['sun13'] = (test_data['w'].isin([1, 3])) & (test_data['wd'] == 6)
            elif bd_no == 5:
                train_data['fss'] = train_data['wd'].isin([4, 5, 6])
                test_data['fss'] = test_data['wd'].isin([4, 5, 6])
            elif bd_no in [2, 3, 54, 91]:
                train_data['mon'] = train_data['wd'] == 0
                test_data['mon'] = test_data['wd'] == 0

            dtrain = xgb.DMatrix(data=train_data, label=train_label)
            dtest = xgb.DMatrix(data=test_data)

            params = {'eta': 1e-2,
                      'max_depth': int(best_params.loc[bd_no - 1, 'max_depth']),
                      'subsample': best_params.loc[bd_no - 1, 'subsample'],
                      'colsample_bytree': best_params.loc[bd_no - 1, 'colsample_bytree'],
                      'random_state': 0}

            model = xgb.train(params=params,
                              dtrain=dtrain,
                              num_boost_round=int(best_params.loc[bd_no - 1, 'best_num_boost_round']),
                              obj=sudo_smape,
                              custom_metric=smape)
            model.save_model('./model/model%d.model' % bd_no)

            y_pred = model.predict(dtest)
            answer.extend(y_pred.tolist())
            self.answer = answer

        return answer

    def draw_prediction(self, bd_no):
        if self.answer is not None:
            fig, ax = plt.subplots(figsize=(20, 4))
            ax.plot('dt', 'target', data=self.train[self.train['bd_no'] == bd_no])
            ax.plot(self.test.loc[self.test['bd_no'] == bd_no, 'dt'], self.answer[(168 * bd_no - 168):(168 * bd_no)])
            ax.set_title('Building %d' % bd_no)
            plt.show()
        else:
            raise Exception('predict() should be runned first!')
