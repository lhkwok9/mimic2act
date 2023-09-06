import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

lift_100_1 = pd.read_csv('/home/jk/robomimic/csv/lift_100_1.csv')
lift_100_2 = pd.read_csv('/home/jk/robomimic/csv/lift_100_2.csv')
lift_100_3 = pd.read_csv('/home/jk/robomimic/csv/lift_100_3.csv')

lift_50_1 = pd.read_csv('/home/jk/robomimic/csv/lift_50_1.csv')
lift_50_2 = pd.read_csv('/home/jk/robomimic/csv/lift_50_2.csv')
lift_50_3 = pd.read_csv('/home/jk/robomimic/csv/lift_50_3.csv')

lift_20_1 = pd.read_csv('/home/jk/robomimic/csv/lift_20_1.csv')
lift_20_2 = pd.read_csv('/home/jk/robomimic/csv/lift_20_2.csv')
lift_20_3 = pd.read_csv('/home/jk/robomimic/csv/lift_20_3.csv')

can_100_1 = pd.read_csv('/home/jk/robomimic/csv/can_100_1.csv')
can_100_2 = pd.read_csv('/home/jk/robomimic/csv/can_100_2.csv')

can_filtered_100_1 = pd.read_csv('/home/jk/robomimic/csv/can_filtered_100_1.csv')

# ploting bar graph

# lift_100_max = max(lift_100_1['Value'].max('index'), lift_100_2['Value'].max('index'), lift_100_3['Value'].max('index'))
# lift_50_max = max(lift_50_1['Value'].max('index'), lift_50_2['Value'].max('index'), lift_50_3['Value'].max('index'))
# lift_20_max = max(lift_20_1['Value'].max('index'), lift_20_2['Value'].max('index'), lift_20_3['Value'].max('index'))

# df = pd.DataFrame({'Size': {0: '20%', 1: '50%', 2: '100%'},
#                    'Testing Success Rate': {0: lift_20_max,1: lift_50_max,2: lift_100_max}})

# sns.set_theme()
# sns.barplot(data=df, x='Size', y='Testing Success Rate')
# plt.show()

# ploting lift seed graph

# lift_100_2.drop(columns=['Wall time'], inplace=True)
# lift_100_2.columns = ['Step', '100%']

# lift_50_3.drop(columns=['Wall time'], inplace=True)
# lift_50_3.columns = ['Step', '50%']

# lift_20_2.drop(columns=['Wall time'], inplace=True)
# lift_20_2.columns = ['Step', '20%']

# lift_100_2 = lift_100_2.merge(lift_50_3, on='Step')
# lift_100_2 = lift_100_2.merge(lift_20_2, on='Step')

# print(pd.melt(lift_100_1, ['Step']))
# sns.set_theme()
# sns.lineplot(x='Step', y='Testing Success Rate', hue='Size', data=pd.melt(lift_100_2, ['Step'], var_name='Size', value_name='Testing Success Rate'))
# plt.show()

# ploting can seed graph

can_100_1.drop(columns=['Wall time'], inplace=True)
can_100_1.columns = ['Step', '1']

# can_100_2.drop(columns=['Wall time'], inplace=True)
# can_100_2.columns = ['Step', '2']

# can_100_3.drop(columns=['Wall time'], inplace=True)
# can_100_3.columns = ['Step', '3']

# can_100_1 = can_100_1.merge(can_100_2, on='Step')
# can_100_1 = can_100_1.merge(can_100_3, on='Step')

print(pd.melt(can_100_1, ['Step']))
# sns.set_theme()
# sns.lineplot(x='Step', y='Testing Success Rate', data=pd.melt(can_100_1, ['Step'], var_name='Seed', value_name='Testing Success Rate'))
# plt.show()