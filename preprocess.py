# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 19:33:41 2017

@author: Riyad Imrul
"""

import pandas as pd
import numpy as np

a=pd.read_csv('x_values_dataset2.csv')
b=a.as_matrix()
index_sensitive=4
x_control=b[:,index_sensitive]
for i in range(len(x_control)):
    if x_control[i]!=0:
        x_control[i]=1    
b=np.delete(b,index_sensitive,1)
x=b[:,1:]

a=pd.read_csv('y_values_dataset2.csv')
b=a.as_matrix()
y=b[:,1]


np.save('x2.npy',x)
np.save('x_control2.npy',x_control)
np.save('y2.npy',y)