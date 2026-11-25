"""Monthly unemployment forecast using the latest available FRED series."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error,mean_squared_error

SERIES=['UNRATE','CPIAUCSL','INDPRO','FEDFUNDS']
URL='https://fred.stlouisfed.org/graph/fredgraph.csv?id='+','.join(SERIES)

def prepare(raw):
    d=raw.copy()
    date=d.columns[0]
    d[date]=pd.to_datetime(d[date])
    d=d.set_index(date).sort_index()
    d.index=d.index.to_period('M')
    if d.index.duplicated().any(): raise ValueError('Duplicate month')
    d=d.reindex(pd.period_range(d.index.min(),d.index.max(),freq='M'))
    for s in SERIES:d[s]=pd.to_numeric(d[s],errors='coerce')
    # Forecast next month from unemployment at t and other series at t-1.
    d['unrate_change']=d.UNRATE.diff()
    d['inflation']=d.CPIAUCSL.pct_change(12,fill_method=None).shift(1)
    d['production_growth']=d.INDPRO.pct_change(3,fill_method=None).shift(1)
    d['policy_rate']=d.FEDFUNDS.shift(1)
    d['target']=d.UNRATE.shift(-1)
    d['target_month']=d.index+1
    return d.loc['1990-01':].replace([np.inf,-np.inf],np.nan).dropna(subset=['UNRATE','unrate_change','inflation','production_growth','policy_rate','target'])

def run(source,out):
    d=prepare(pd.read_csv(source))
    if d.empty: raise ValueError('No valid observations')
    end=d.target_month.max();test_start=end-35;val_start=test_start-24
    train=d[d.target_month<val_start];val=d[(d.target_month>=val_start)&(d.target_month<test_start)];test=d[d.target_month>=test_start]
    if min(map(len,(train,val,test)))==0: raise ValueError('Insufficient chronological coverage')
    x=['UNRATE','unrate_change','inflation','production_growth','policy_rate']
    scores={'persistence':float(mean_absolute_error(val.target,val.UNRATE))}
    for alpha in (.1,1,10,100):
        m=make_pipeline(StandardScaler(),Ridge(alpha=alpha)).fit(train[x],train.target)
        scores[str(alpha)]=float(mean_absolute_error(val.target,m.predict(val[x])))
    winner=min(scores,key=scores.get);dev=pd.concat([train,val])
    pred=test.UNRATE.to_numpy() if winner=='persistence' else make_pipeline(StandardScaler(),Ridge(alpha=float(winner))).fit(dev[x],dev.target).predict(test[x])
    metrics=dict(source=str(source),last_target_month=str(end),test_start=str(test_start),selected=winner,validation_mae=scores,test_mae=float(mean_absolute_error(test.target,pred)),baseline_mae=float(mean_absolute_error(test.target,test.UNRATE)),test_rmse=float(mean_squared_error(test.target,pred)**.5),test_rows=len(test))
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2))
    p=test[['target_month','target','UNRATE']].copy();p['prediction']=pred;p.to_csv(out/'predictions.csv',index=False)
    import matplotlib.pyplot as plt
    p['date']=p.target_month.dt.to_timestamp()
    ax=p.set_index('date')[['target','prediction','UNRATE']].rename(columns={'target':'Actual','prediction':'Selected model','UNRATE':'Persistence'}).plot(figsize=(9,4),title='Next-month unemployment: chronological holdout')
    ax.set_ylabel('Unemployment rate (%)');ax.set_xlabel('Target month')
    ax.figure.tight_layout();ax.figure.savefig(out/'forecast.png',dpi=150);plt.close(ax.figure)
    print(json.dumps(metrics,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--csv',default=URL);p.add_argument('--out',default='reports_fred');a=p.parse_args();run(a.csv,a.out)
