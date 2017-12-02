from pandas.tools.plotting import autocorrelation_plot
from statsmodels.tsa.arima_model import ARIMA
from godiva_data import *

csvdata = pd.DataFrame.from_dict(data())
