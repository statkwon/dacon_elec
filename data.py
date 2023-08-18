import pickle
import holidays
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

        # 재정의한 '건물유형' 추가
        with open('bd_type.pkl', 'rb') as f:
            bd_type = pickle.load(f)
        self.building_info['건물유형'] = bd_type

        self.train = pd.merge(self.train, self.building_info, how='left')
        self.test = pd.merge(self.test, self.building_info, how='left')

        self.train.rename(self.col_map, axis=1, inplace=True)
        self.test.rename(self.col_map, axis=1, inplace=True)

        # 월, 요일, 시간 변수 생성
        self.train['m'] = self.train['dt'].dt.month
        self.test['m'] = self.test['dt'].dt.month
        self.train['wd'] = self.train['dt'].dt.weekday
        self.test['wd'] = self.test['dt'].dt.weekday
        self.train['h'] = self.train['dt'].dt.hour
        self.test['h'] = self.test['dt'].dt.hour

        # 주 변수 생성
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

        # 불쾌지수 변수 추가
        self.train['di'] = (9 / 5) * self.train['temp'] - 0.55 * (1 - self.train['hmd'] / 100) * ((9 / 5) * self.train['temp'] - 26) + 32
        self.test['di'] = (9 / 5) * self.test['temp'] - 0.55 * (1 - self.test['hmd'] / 100) * ((9 / 5) * self.test['temp'] - 26) + 32

        # 공휴일 변수 추가
        kr_holidays = holidays.KR(years=2022)
        self.train['hd'] = [dt.day_of_week in [5, 6] or dt in kr_holidays for dt in self.train['dt']]
        self.test['hd'] = [dt.day_of_week in [5, 6] or dt in kr_holidays for dt in self.test['dt']]

        # 월요일, 일요일 변수 추가
        self.train['mon'] = [dt.day_of_week == 0 for dt in self.train['dt']]
        self.test['mon'] = [dt.day_of_week == 0 for dt in self.test['dt']]
        self.train['sun'] = [dt.day_of_week == 6 for dt in self.train['dt']]
        self.test['sun'] = [dt.day_of_week == 6 for dt in self.test['dt']]

        # 건물 / 월 / 주 / 요일 / 시간별 전력소비량 평균
        gb_m_target = self.train.groupby(by=['bd_no', 'm'], as_index=False).agg({'target': 'mean'})
        gb_m_target.rename({'target': 'gbmt'}, axis=1, inplace=True)
        self.train = pd.merge(self.train, gb_m_target, how='left')
        self.test = pd.merge(self.test, gb_m_target, how='left')

        gb_w_target = self.train.groupby(by=['bd_no', 'w'], as_index=False).agg({'target': 'mean'})
        gb_w_target.rename({'target': 'gbwt'}, axis=1, inplace=True)
        self.train = pd.merge(self.train, gb_w_target, how='left')
        self.test = pd.merge(self.test, gb_w_target, how='left')

        gb_wd_target = self.train.groupby(by=['bd_no', 'wd'], as_index=False).agg({'target': 'mean'})
        gb_wd_target.rename({'target': 'gbwdt'}, axis=1, inplace=True)
        self.train = pd.merge(self.train, gb_wd_target, how='left')
        self.test = pd.merge(self.test, gb_wd_target, how='left')

        gb_hd_target = self.train.groupby(by=['bd_no', 'hd'], as_index=False).agg({'target': 'mean'})
        gb_hd_target.rename({'target': 'gbhdt'}, axis=1, inplace=True)
        self.train = pd.merge(self.train, gb_hd_target, how='left')
        self.test = pd.merge(self.test, gb_hd_target, how='left')

        gb_h_target = self.train.groupby(by=['bd_no', 'h'], as_index=False).agg({'target': 'mean'})
        gb_h_target.rename({'target': 'gbht'}, axis=1, inplace=True)
        self.train = pd.merge(self.train, gb_h_target, how='left')
        self.test = pd.merge(self.test, gb_h_target, how='left')

        gb_mon_target = self.train.groupby(by=['bd_no', 'mon'], as_index=False).agg({'target': 'mean'})
        gb_mon_target.rename({'target': 'gbmont'}, axis=1, inplace=True)
        self.train = pd.merge(self.train, gb_mon_target, how='left')
        self.test = pd.merge(self.test, gb_mon_target, how='left')

        gb_sun_target = self.train.groupby(by=['bd_no', 'sun'], as_index=False).agg({'target': 'mean'})
        gb_sun_target.rename({'target': 'gbsunt'}, axis=1, inplace=True)
        self.train = pd.merge(self.train, gb_sun_target, how='left')
        self.test = pd.merge(self.test, gb_sun_target, how='left')

        # 강수량 결측치 0으로 대체
        self.train['pcpn'].fillna(0.0, inplace=True)

        # 풍속 및 습도 결측치 한 시점 이전의 값으로 대체
        self.train.fillna(method='ffill', inplace=True)

    def draw_target(self, bd_no):
        fig, ax = plt.subplots(figsize=(20, 4))
        ax.plot('dt', 'target', data=self.train[self.train['bd_no'] == bd_no])
        ax.set_title('Building %d' % bd_no)
        plt.show()

    def get_submission(self, answer, fname):
        submission = pd.read_csv('./data/sample_submission.csv')
        submission['answer'] = answer
        submission.to_csv(fname, index=False)
