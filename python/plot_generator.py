
import sklearn.naive_bayes as nb
import matplotlib.pyplot as plt
import sklearn.metrics as skm
import matplotlib as mpl
import rust_bayes as rb
import pandas as pd
import numpy as np

# Setup classifiers
bayes = rb.GaussianOptimal()
naive = nb.GaussianNB()


# Parameters for for generation
spreading = 5
classes = 5
    

def posdefizer(M) :
    return M*M.T

X = pd.read_csv('python/X_export.csv')
y = pd.read_csv('python/y_export.csv')

# USING SELF-MADE
bayes.fit(X,y) 
y_pred = bayes.predict(X)
conf = skm.confusion_matrix(y.unstack(),y_pred)
accuracy1 = np.sum(np.diag(conf))/np.sum(conf)*100

## USING SKLEARN
naive.fit(np.array(X),np.array(y.unstack()))
y_pred2 = naive.predict(np.array(X))
conf2 = skm.confusion_matrix(np.array(y.unstack()),np.array(y_pred2))
accuracy2 = np.sum(np.diag(conf2))/np.sum(conf2)*100

# Plot of optimal bayes classifier
fig,axs = plt.subplots(1,2,figsize=(10,5))

# Hack for getting natural x and y limits
for c in range(classes):
    axs[0].scatter(X[y['target']==c]['x'],X[y['target']==c]['y'],alpha=0.5,s = 15,color = mpl.cm.get_cmap('tab20')((c*2)/20))
    x_lim = axs[0].get_xlim()
    y_lim = axs[0].get_ylim()
    
# Setup classification region grid
N = 2000
grid_X,grid_Y = np.meshgrid(np.linspace(x_lim[0], x_lim[1], N), np.linspace(y_lim[0], y_lim[1], N))

grid = np.array([np.ravel(grid_X),np.ravel(grid_Y)]).T
grid_pred = bayes.predict(grid)
grid_pred2 = naive.predict(grid)

# For optimal bayes, plot background and points
axs[0].set_title(f"Optimal Bayes prediction ({round(accuracy1,2)}% accurate)")
for c in range(classes):
    axs[0].scatter(grid[grid_pred==c][:,0],grid[grid_pred==c][:,1],alpha=1, s = .025, color = mpl.cm.get_cmap('tab20')((c*2 + 1)/20))
for c in range(classes):
    axs[0].scatter(X[y['target']==c]['x'],X[y['target']==c]['y'],alpha=0.5,s = 15,color = mpl.cm.get_cmap('tab20')((c*2)/20))

# For naive bayes, plot background and points
axs[1].set_title(f"Naive Bayes prediction ({round(accuracy2,2)}% accurate)")
for c in range(classes):
    axs[1].scatter(grid[grid_pred2==c][:,0],grid[grid_pred2==c][:,1],alpha=1, s = .025, color = mpl.cm.get_cmap('tab20')((c*2 + 1)/20))
for c in range(classes):
    axs[1].scatter(X[y['target']==c]['x'],X[y['target']==c]['y'],alpha=0.5,s = 15, color = mpl.cm.get_cmap('tab20')((c*2)/20))

# Remove unneeded ticks
axs[0].tick_params(left = False , bottom = False,labelleft = False ,labelbottom = False)
axs[1].tick_params(left = False , bottom = False,labelleft = False ,labelbottom = False)

# Apply axis limits
axs[0].set_ylim(y_lim)
axs[0].set_xlim(x_lim)
axs[1].set_ylim(y_lim)
axs[1].set_xlim(x_lim)

# Export
fig.tight_layout()
plt.savefig(f"image/comparison_export.png",bbox_inches=0)
plt.show()