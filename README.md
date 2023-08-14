# dacon_elec
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

## Submissions
|        Date         | CV SMAPE (mean/std) |  Test SMAPE  | Features                                                              |
|:-------------------:|:-------------------:|:------------:|-----------------------------------------------------------------------|
| 2023-08-14 00:07:06 | 7.319213 / 0.027626 | 7.3584366253 | [temp, pcpn, wn_spd, hmd, area, c_area, m, h, di, gbmht, hd]          |
| 2023-08-14 14:58:54 |                     | 7.3430932744 | [temp, pcpn, wn_spd, hmd, area, c_area, m, sinh, cosh, di, gbmht, hd] |

## References
- [SMAPE](https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error)
- [Discomfort Index](https://news.samsungdisplay.com/32491)
- [XGBoost - Custom Objective & Evaluation Metric](https://xgboost.readthedocs.io/en/stable/tutorials/custom_metric_obj.html)
- [sklearn - TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)
- [Pandas - Moving Average](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.rolling.html)
