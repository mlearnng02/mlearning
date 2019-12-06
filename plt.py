import matplotlib.pyplot as plt

fig, ax = plt.subplots(1,4)
for i, j in enumerate(['p1','p2','p3','p4']):
    ax[i].text(0.5,0.5, str('{}{}'.format(i,j)))
plt.savefig('a')
