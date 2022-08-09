import numpy as np
import matplotlib.pyplot as plt
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPDF

val = np.loadtxt('runs/loss_val_21h47m.txt',delimiter=',')
train = np.loadtxt('runs/loss_train_21h47m.txt',delimiter=',')
val_loss = np.zeros(val.shape[0])
train_loss = np.zeros(val.shape[0])

for i in range(val.shape[0]):
    train_loss[i] = np.mean(train[i,:])
    val_loss[i] = np.sum(val[i, val[i,:]>0])/np.sum(val[i,:]>0)


plt.figure()
plt.plot(train_loss, '-k', label='Train loss')
plt.plot(val_loss, '-r', label='Val loss')
plt.ylim([0,0.1])
plt.legend(loc='best')
ax=plt.gca() 
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig('runs/pic_curve_error.svg', format="svg")
plt.show()
drawing = svg2rlg('runs/pic_curve_error.svg')
renderPDF.drawToFile(drawing, 'runs/pic_curve_error.pdf')

plt.show()

