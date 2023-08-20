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

    gb_m_target = train_data.groupby(by=['m', 'h']).agg({'target': ['min', 'mean', 'max', 'std']})
    gb_m_target.columns = gb_m_target.columns.droplevel()
    gb_m_target.reset_index(inplace=True)
    gb_m_target.rename({'min': 'gbmt_min', 'mean': 'gbmt_mean', 'max': 'gbmt_max', 'std': 'gbmt_std'}, axis=1, inplace=True)
    train_data = pd.merge(train_data, gb_m_target, how='left')
    test_data = pd.merge(test_data, gb_m_target, how='left')

    gb_w_target = train_data.groupby(by=['w', 'h']).agg({'target': ['min', 'mean', 'max', 'std']})
    gb_w_target.columns = gb_w_target.columns.droplevel()
    gb_w_target.reset_index(inplace=True)
    gb_w_target.rename({'min': 'gbwt_min', 'mean': 'gbwt_mean', 'max': 'gbwt_max', 'std': 'gbwt_std'}, axis=1, inplace=True)
    train_data = pd.merge(train_data, gb_w_target, how='left')
    test_data = pd.merge(test_data, gb_w_target, how='left')

    gb_wd_target = train_data.groupby(by=['wd', 'h']).agg({'target': ['min', 'mean', 'max', 'std']})
    gb_wd_target.columns = gb_wd_target.columns.droplevel()
    gb_wd_target.reset_index(inplace=True)
    gb_wd_target.rename({'min': 'gbwdt_min', 'mean': 'gbwdt_mean', 'max': 'gbwdt_max', 'std': 'gbwdt_std'}, axis=1, inplace=True)
    train_data = pd.merge(train_data, gb_wd_target, how='left')
    test_data = pd.merge(test_data, gb_wd_target, how='left')

    gb_hd_target = train_data.groupby(by=['hd', 'h']).agg({'target': ['min', 'mean', 'max', 'std']})
    gb_hd_target.columns = gb_hd_target.columns.droplevel()
    gb_hd_target.reset_index(inplace=True)
    gb_hd_target.rename({'min': 'gbhdt_min', 'mean': 'gbhdt_mean', 'max': 'gbhdt_max', 'std': 'gbhdt_std'}, axis=1, inplace=True)
    train_data = pd.merge(train_data, gb_hd_target, how='left')
    test_data = pd.merge(test_data, gb_hd_target, how='left')

    gb_h_target = train_data.groupby(by='h').agg({'target': ['min', 'mean', 'max', 'std']})
    gb_h_target.columns = gb_h_target.columns.droplevel()
    gb_h_target.reset_index(inplace=True)
    gb_h_target.rename({'min': 'gbht_min', 'mean': 'gbht_mean', 'max': 'gbht_max', 'std': 'gbht_std'}, axis=1, inplace=True)
    train_data = pd.merge(train_data, gb_h_target, how='left')
    test_data = pd.merge(test_data, gb_h_target, how='left')

    gb_mon_target = train_data.groupby(by=['mon', 'h']).agg({'target': ['min', 'mean', 'max', 'std']})
    gb_mon_target.columns = gb_mon_target.columns.droplevel()
    gb_mon_target.reset_index(inplace=True)
    gb_mon_target.rename({'min': 'gbmont_min', 'mean': 'gbmont_mean', 'max': 'gbmont_max', 'std': 'gbmont_std'}, axis=1, inplace=True)
    train_data = pd.merge(train_data, gb_mon_target, how='left')
    test_data = pd.merge(test_data, gb_mon_target, how='left')

    gb_sun_target = train_data.groupby(by=['sun', 'h']).agg({'target': ['min', 'mean', 'max', 'std']})
    gb_sun_target.columns = gb_sun_target.columns.droplevel()
    gb_sun_target.reset_index(inplace=True)
    gb_sun_target.rename({'min': 'gbsunt_min', 'mean': 'gbsunt_mean', 'max': 'gbsunt_max', 'std': 'gbsunt_std'}, axis=1, inplace=True)
    train_data = pd.merge(train_data, gb_sun_target, how='left')
    test_data = pd.merge(test_data, gb_sun_target, how='left')

    train_data.drop(['m', 'wd', 'h', 'w', 'hd', 'mon', 'sun', 'target'], axis=1, inplace=True)
    test_data.drop(['m', 'wd', 'h', 'w', 'hd', 'mon', 'sun'], axis=1, inplace=True)

    dtrain = xgb.DMatrix(data=train_data, label=train_label)
    dtest = xgb.DMatrix(data=test_data, label=test_label)

    return dtrain, dtest, param


class Model:
    def __init__(self, train, test, feature_names, cv_results=None, answer=None):
        self.train = train
        self.test = test
        self.feature_names = feature_names
        self.cv_results = cv_results
        self.answer = answer

    def cross_validation(self, bd_no, params, num_boost_round, folds, early_stopping_rounds):
        data = self.train.loc[self.train['bd_no'] == bd_no, self.feature_names]
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

    def draw_train_test_smape(self, train_smape, test_smape):
        fig, ax = plt.subplots()
        ax.plot(train_smape, label='train')
        ax.plot(test_smape, label='test')
        ax.legend()
        plt.show()

    def run_cv(self, params, num_boost_round, folds, early_stopping_rounds):
        # runtime: about 48 min.
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

    def predict(self, params, best_num_boost_rounds):
        answer = []
        for bd_no in tqdm(range(1, 101)):
            train_data = self.train.loc[self.train['bd_no'] == bd_no, self.feature_names]
            test_data = self.test.loc[self.test['bd_no'] == bd_no, self.feature_names]
            train_label = self.train.loc[self.train['bd_no'] == bd_no, 'target']
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

    def draw_prediction(self, bd_no):
        if self.answer is not None:
            fig, ax = plt.subplots(figsize=(20, 4))
            ax.plot('dt', 'target', data=self.train[self.train['bd_no'] == bd_no])
            ax.plot(self.test.loc[self.test['bd_no'] == bd_no, 'dt'], self.answer[(168 * bd_no - 168):(168 * bd_no)])
            ax.set_title('Building %d' % bd_no)
            plt.show()
        else:
            raise Exception('predict() should be runned first!')
