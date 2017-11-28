from scipy.stats import randint as rd
from scipy.stats import uniform as uni


def xgb_params():
    para_dict = {
        "max_depth": rd(3, 10),
        "learning_rate": uni(loc=0, scale=1),
        "n_estimators": rd(50, 200),
        'objective': ['reg:linear'],
    }
    return para_dict;
