# dacon_elec
## Submissions
|        Date         | CV SMAPE (mean/std) |   Test SMAPE    | Features                                                                                                    |
|:-------------------:|:-------------------:|:---------------:|-------------------------------------------------------------------------------------------------------------|
| 2023-08-14 00:07:06 | 7.319213 / 0.027626 |  7.3584366253   | ['temp', 'pcpn', 'wn_spd', 'hmd', 'area', 'c_area', 'm', 'h', 'di', 'gbmht', 'hd']                          |
| 2023-08-14 14:58:54 |    7.295225 / -     |  7.3430932744   | ['temp', 'pcpn', 'wn_spd', 'hmd', 'area', 'c_area', 'm', 'sinh', 'cosh', 'di', 'gbmht', 'hd']               |
| 2023-08-15 21:50:38 | 6.459990 / 0.027497 |  7.5985445132   | ['temp', 'pcpn', 'wn_spd', 'hmd', 'm', 'h', 'di', 'gbmwdht', 'hd']                                          |
| 2023-08-16 22:07:38 | 6.874945 / 0.026166 | **7.090952714** | ['temp', 'pcpn', 'wn_spd', 'hmd', 'm', 'h', 'di', 'gbmt', 'gbwt', 'gbwdt', 'gbht', 'hd']                    |
| 2023-08-17 23:22:46 | 7.721417 / 0.026394 |  7.2037221875   | ['temp', 'pcpn', 'wn_spd', 'hmd', 'di', 'gbmt', 'gbwt', 'gbwdt', 'gbht', 'gbhdt', 'gbmont', 'gbsunt', 'hd'] |

## SMAPE
$$\text{SMAPE}=\dfrac{100}{n}\sum_{i=1}^n\dfrac{\vert \hat{y}_i-y_i \vert}{(\vert y_i \vert+\vert \hat{y}_i \vert)/2}$$
```python
import numpy as np
import matplotlib.pyplot as plt

def draw_sape(y_pred, y_true):
    sape = 100 * np.abs(y_pred - y_true) / ((np.abs(y_true) + np.abs(y_pred)) / 2)
    fig, ax = plt.subplots()
    ax.plot(y_pred - y_true, sape)
    ax.axvline(x=0, c='r', linestyle='--')
    ax.set_xlabel('$r=\hat{y}-y$')
    plt.show()

draw_sape(np.linspace(data.train['target'].min(), data.train['target'].max(), 1000), data.train['target'].mean())
```
![](./figures/smape.png)

## References
- [SMAPE](https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error)
- [Discomfort Index](https://news.samsungdisplay.com/32491)
- [XGBoost - Custom Objective & Evaluation Metric](https://xgboost.readthedocs.io/en/stable/tutorials/custom_metric_obj.html)
- [sklearn - TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)
- [Pandas - Moving Average](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.rolling.html)
