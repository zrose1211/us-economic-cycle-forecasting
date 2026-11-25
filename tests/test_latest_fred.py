import pandas as pd
import numpy as np
from latest_fred import prepare, run

def sample():
    dates=pd.date_range('1990-01-01',periods=120,freq='MS')
    n=np.arange(len(dates))
    return pd.DataFrame({'date':dates,'UNRATE':5+np.sin(n/5),'CPIAUCSL':100+n,'INDPRO':100+n/2,'FEDFUNDS':3+n/100})

def test_lagged_inputs_and_target():
    raw=sample()
    d=prepare(raw)
    t=pd.Period('1992-01','M')
    assert d.loc[t,'policy_rate']==raw.loc[23,'FEDFUNDS']
    assert d.loc[t,'target']==raw.loc[25,'UNRATE']
    assert d.loc[t,'target_month']==t+1

def test_refresh_outputs(tmp_path):
    source=tmp_path/'fred.csv'
    sample().to_csv(source,index=False)
    run(source,tmp_path/'out')
    assert (tmp_path/'out'/'metrics.json').exists()
    assert (tmp_path/'out'/'forecast.png').exists()
