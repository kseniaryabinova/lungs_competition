import os
import shutil
import time

import numpy as np

from torch.utils.tensorboard import SummaryWriter

os.makedirs('tensorboard_runs', exist_ok=True)
shutil.rmtree('tensorboard_runs')
writer = SummaryWriter(log_dir='tensorboard_runs', filename_suffix=str(time.time()))

for k in range(11):
    for i in range(10):
        data = np.random.random(10)
        # for j in range(data.shape[0]):
        #     writer.add_scalars('ROC curve/{}_data'.format(k), {str(i): data[j]}, j / 10)
        writer.add_pr_curve('ROC curve/{}_data'.format(k), data, data)

writer.flush()
