import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import confusion_matrix

df = pd.read_csv(r"D:\ML data\digit prediction\train.csv")
df.head()


# Visualize dataset 
image0 = df.iloc[3,1:]
image0                                                
plt.imshow(np.array(image0).reshape(28,28))

x = df.iloc[:,1:]
y = df.iloc[:,0]
xtrain , xtest ,ytrain, ytest = train_test_split(x,y,test_size =0.2,shuffle = False,random_state =7)


dtree = DecisionTreeClassifier()
dtree.fit(xtrain , ytrain)


cmdtree = confusion_matrix(ytest,ypred)
cmdtree ,dtree.score(xtest , ytest)