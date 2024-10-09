import numpy as np
top1_acc = [0.335, 0.524, 0.525, 0.560, 0.405, 0.595, 0.505, 0.680, 0.495, 0.625]
top5_acc = [0.595, 0.810, 0.815, 0.830, 0.690, 0.830, 0.820, 0.865, 0.775, 0.890]
a1 = np.mean(top1_acc)
a2 = np.mean(top5_acc)
print(f'top1 acc:{a1:.3f}, top5 acc:{a2:.3f}')