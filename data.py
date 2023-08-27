import holidays
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Data:
    def __init__(self, preproc=True):

        self.preproc = preproc

        self.train = pd.read_csv('./data/train.csv')
        self.test = pd.read_csv('./data/test.csv')
        self.building_info = pd.read_csv('./data/building_info.csv')
        self.col_map = {'건물번호': 'bd_no',
                        '일시': 'dt',
                        '기온(C)': 'temp',
                        '강수량(mm)': 'pcpn',
                        '풍속(m/s)': 'wn_spd',
                        '습도(%)': 'hmd',
                        '전력소비량(kWh)': 'target',
                        '건물유형': 'bd_type'}
        if self.preproc:
            self.preprocessing()

    def preprocessing(self):
        # 테스트 데이터에 존재하지 않는 변수인 '일조', '일사' 제거
        self.train.drop(['num_date_time', '일조(hr)', '일사(MJ/m2)'], axis=1, inplace=True)
        self.test.drop('num_date_time', axis=1, inplace=True)

        # '일시' 변수 pandas Timestamp 형식으로 변경
        self.train['일시'] = pd.to_datetime(self.train['일시'])
        self.test['일시'] = pd.to_datetime(self.test['일시'])

        # 결측률이 각각 64%, 95%, 95%인 '태양광용랑', 'ESS저장용량', 'PCS용량' 제거
        self.building_info.drop(['태양광용량(kW)', 'ESS저장용량(kWh)', 'PCS용량(kW)'], axis=1, inplace=True)

        # 사용하지 않는 '연면적', '냉방면적', '건물유형' 제거
        self.building_info.drop(['연면적(m2)', '냉방면적(m2)', '건물유형'], axis=1, inplace=True)

        self.train = pd.merge(self.train, self.building_info, how='left')
        self.test = pd.merge(self.test, self.building_info, how='left')

        self.train.rename(self.col_map, axis=1, inplace=True)
        self.test.rename(self.col_map, axis=1, inplace=True)

        # 월, 주, 요일, 시간 변수 생성
        self.train['m'] = self.train['dt'].dt.month
        self.test['m'] = self.test['dt'].dt.month

        self.train['w'] = [
            dt.isocalendar().week - 21 if dt.month == 6
            else dt.isocalendar().week - 25 if dt.month == 7
            else dt.isocalendar().week - 30
            for dt in self.train['dt']
        ]
        self.test['w'] = [
            dt.isocalendar().week - 21 if dt.month == 6
            else dt.isocalendar().week - 25 if dt.month == 7
            else dt.isocalendar().week - 30
            for dt in self.test['dt']
        ]

        self.train['wd'] = self.train['dt'].dt.weekday
        self.test['wd'] = self.test['dt'].dt.weekday

        self.train['h'] = self.train['dt'].dt.hour
        self.test['h'] = self.test['dt'].dt.hour

        # 공휴일 변수 추가
        kr_holidays = holidays.KR(years=2022)
        self.train['hd'] = [dt.day_of_week in [5, 6] or dt in kr_holidays for dt in self.train['dt']]
        self.test['hd'] = [dt.day_of_week in [5, 6] or dt in kr_holidays for dt in self.test['dt']]

        # Cyclic Encoding
        self.train['sinh'] = np.sin((2 / 24 * np.pi) * self.train['h'])
        self.train['cosh'] = np.cos((2 / 24 * np.pi) * self.train['h'])
        self.test['sinh'] = np.sin((2 / 24 * np.pi) * self.test['h'])
        self.test['cosh'] = np.cos((2 / 24 * np.pi) * self.test['h'])

        # 불쾌지수 변수 추가
        self.train['di'] = (9 / 5) * self.train['temp'] - 0.55 * (1 - self.train['hmd'] / 100) * ((9 / 5) * self.train['temp'] - 26) + 32
        self.test['di'] = (9 / 5) * self.test['temp'] - 0.55 * (1 - self.test['hmd'] / 100) * ((9 / 5) * self.test['temp'] - 26) + 32

        # 30도 이상/이하 변수 추가
        self.train['thirty'] = self.train['temp'] >= 30
        self.test['thirty'] = self.test['temp'] >= 30

        # 강수량 결측치 0으로 대체
        self.train['pcpn'].fillna(0.0, inplace=True)

        # 풍속 및 습도 결측치 한 시점 이전의 값으로 대체
        self.train.fillna(method='ffill', inplace=True)

        # 백화점 및 아울렛 휴일 제거
        self.train['date'] = self.train['dt'].dt.date
        target_max = self.train.groupby(by=['bd_no', 'date'], as_index=False).agg({'target': 'max'})
        cutoff = [3500, 1000, 1500, 2500, 3500, 1500]
        for i, bd_no in enumerate(range(37, 43)):
            for date in target_max.loc[(target_max['bd_no'] == bd_no) & (target_max['target'] < cutoff[i]), 'date']:
                idx = self.train[(self.train['bd_no'] == bd_no) & (self.train['dt'].dt.date == date)].index
                self.train.drop(idx, inplace=True)
        self.train.drop('date', axis=1, inplace=True)

    def draw_target(self, bd_no):
        fig, ax = plt.subplots(figsize=(20, 4))
        ax.plot('dt', 'target', data=self.train[self.train['bd_no'] == bd_no])
        ax.set_title('Building %d' % bd_no)
        plt.show()

    def get_submission(self, answer, fname):
        submission = pd.read_csv('./data/sample_submission.csv')
        submission['answer'] = answer
        submission.to_csv(fname, index=False)
