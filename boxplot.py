import matplotlib.pyplot as plt
import pandas as pd
import os

"""This script plots boxplots for different confidence levels, undersampling factors and noise levels"""

#load data
path1 = 'results/test_results_it_net_30per_sigma60_noisy/'
path2 = 'results/test_results_it_net_40per_sigma60_noisy/'
path3 = 'results/test_results_it_net_60per_sigma60_noisy/'

data30 = pd.read_csv(os.path.join(path1, 'dataframe_CI_0.05.dat'))
data3090 = pd.read_csv(os.path.join(path1, 'dataframe_CI_0.1.dat'))
data40 = pd.read_csv(os.path.join(path2, 'dataframe_CI_0.05.dat'))
data4090 = pd.read_csv(os.path.join(path1, 'dataframe_CI_0.1.dat'))
data60 = pd.read_csv(os.path.join(path3, 'dataframe_CI_0.05.dat'))
data6090 = pd.read_csv(os.path.join(path1, 'dataframe_CI_0.1.dat'))

#put data wanted for boxplot into a dataframe
data = [data6090]
df = pd.concat(data)



# Create boxplot
fig = plt.plot(1)
df.boxplot(column=['hitrates all gauss', 'hitrates all asymp'])
legend = plt.legend()
plt.ylabel('')
plt.xlabel('')
plt.savefig('CI_boxplot_itnet_sigma60_60per_all_90.pdf')
plt.show()