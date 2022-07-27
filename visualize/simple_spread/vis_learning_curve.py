import pickle
import matplotlib.pyplot as plt
import numpy as np
import copy
import pandas as pd
plt.style.use('seaborn')


with open('/home/smile/iros22_darl1n/result/simple_spread/darl1n/1agents/1agents_1/good_agent.pkl','rb') as f:
    data_darl1n=pickle.load(f)

with open('/home/smile/iros22_darl1n/result/simple_spread/darl1n/1agents/1agents_1/global_time.pkl','rb') as f:
    time_darl1n=pickle.load(f)


with open('/home/smile/iros22_darl1n/result/simple_spread/maddpg/1agents/1agents_1/good_agent.pkl','rb') as f:
    data_maddpg=pickle.load(f)

with open('/home/smile/iros22_darl1n/result/simple_spread/maddpg/1agents/1agents_1/global_time.pkl','rb') as f:
    time_maddpg=pickle.load(f)



plt.figure(figsize=(8,5.3))
plt.plot(time_darl1n, data_darl1n, '-.', linewidth=4, label='DARL1N')
plt.plot(time_maddpg, data_maddpg, '-.', linewidth=4, label='MADDPG')
plt.ylabel('Reward', fontsize=18)
plt.xlabel('Training iteration', fontsize=18)
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.legend(fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
# plt.xlim([0, 2000])
plt.show()
