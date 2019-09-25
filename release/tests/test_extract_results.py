import pylab as plt
import pandas as pd
stats_df = pd.read_pickle('release/data/{}.pkl'.format('channels1_bands1_splitedTrue_thresholds17'))

for k in range(40):
    plt.plot(stats_df.query('subj_id=={} & metric_type=="n_spindles" & threshold_factor==2'.format(k))['metric'].values)