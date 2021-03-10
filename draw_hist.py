import matplotlib.pyplot as plt

x = [.982, .94, .7869, .693, .9796, .9137, .3742, .5965, .899, .6325, .9053, .8482, .975]

plt.style.use('ggplot')
plt.hist(x, bins=[x/10 for x in range(11)])

plt.xlabel('Uncertainty')
plt.ylabel('Number of Detections')
plt.title('Uncertainty of the False Detections')
plt.show()