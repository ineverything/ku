#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np

#1. Create a null vector of size 10    创建一个空向量
x1 = np.empty(10)
print(x1)

#2. Create a null vector of size 10 but the fifth value which is 1 
x2 = np.zeros(10)
x2[4] = 1
print(x2)

#3. Create a vector with values ranging from 10 to 49 
x3 = np.arange(10,50)
print(x3)

#4. Create a 3x3 matrix with ranging from 0 to 8 
x4 = np.arange(9).reshape(3,3)
print(x4)

#5. Create a 10x10 array with random values and find the minimum and maximum values
x5 = np.random.random((10,10))
x5max,x5min = x5.max(),x5.min()
print(x5max,x5min)

#6. Create a 2d array with 1 on the border and 0 inside 
x6 = np.ones((10,10))
x6[1:-1,1:-1] = 0
print(x6)

#7. Multiply a 5x3 matrix(矩阵) by a 3x2 matrix (real matrix product) 
x7 = np.dot(np.ones((5,3)), np.ones((3,2)))
print(x7)


# In[17]:


import numpy as  np
import pandas as pd

#1. Create df with labels as index
data = {'animal': ['cat', 'cat', 'snake', 'dog', 'dog', 'cat', 'snake', 'cat', 'dog', 'dog'],
        'age': [2.5, 3, 0.5, np.nan, 5, 2, 4.5, np.nan, 7, 3],
        'visits': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],
        'priority': ['yes', 'yes', 'no', 'yes', 'no', 'no', 'no', 'yes', 'no', 'no']}

labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
df = pd.DataFrame(data,index=labels)
print(df)

#2. Show df basic information and its data
df.info()

#3. Select only the 'animal' and 'age' columns
print(df.loc[:, ['animal', 'age']])

#4. Select the row with missing value
print(df[df['age'].isnull()])

#5. Sort DF in descending age and ascending visit order  年龄下降，参观量上升
print(df.sort_values(by=['age', 'visits'], ascending=[False, True]))

#6. In the 'animal' column, replace 'Snake' with 'Python'
df['animal'] = df['animal'].replace('snake', 'python')
print(df)


# In[16]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as mpl
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

data = sns.load_dataset("iris")
data.head() #读取前五行数据
# 萼片长度，萼片宽度，花瓣长度，花瓣宽度，种类

#1. Size relationship between sepal and petal (scatter diagram)花萼和花瓣的散点图
data['sepal_size'] = data['sepal_length'] * data['sepal_width']
data['petal_size'] = data['petal_length'] * data['petal_width']
plt.scatter(data['sepal_size'],data['petal_size'])

#2. The size relationship between sepals and petals of iris of    different species
t = data.groupby(['species']).size()#3种
t.index
#Index(['setosa', 'versicolor', 'virginica'], dtype='object', name='species')
data[data['species'].values == 'setosa']['sepal_size']

plt.figure()
flag = 1
for name in data.groupby(['species']).size().index:
    sepal_size = data[data['species'].values == name]['sepal_size']
    petal_size = data[data['species'].values == name]['petal_size']
    plt.subplot(2,2,flag)
    plt.scatter(sepal_size.values,petal_size.values)
    flag += 1
plt.show()

#3. Distribution of sepals and petal sizes of different Iris species (box diagram)
plt.figure(figsize=(20,20))
flag = 1
for name in data.groupby(['species']).size().index:
    sepal_size = data[data['species'].values == name]['sepal_size']
    petal_size = data[data['species'].values == name]['petal_size']
    plt.subplot(3,3,flag)
    plt.boxplot(sepal_size.values
                ,patch_artist = True
               # 中位数线颜色
               , medianprops = {'color': 'b'}
               # 箱子颜色设置，color：边框颜色，facecolor：填充颜色
               , boxprops = {'color': 'b', 'facecolor': 'r'}
               # 猫须颜色whisker
               , whiskerprops = {'color': 'r'}
               # 猫须界限颜色whisker cap
               , capprops = {'color': 'b'})
    plt.title(name +'+sepal_size')
    plt.subplot(3,3,flag * 2)
    plt.boxplot(sepal_size.values,
               patch_artist = True
               # 中位数线颜色
               , medianprops = {'color': 'b'}
               # 箱子颜色设置，color：边框颜色，facecolor：填充颜色
               , boxprops = {'color': 'b', 'facecolor': 'r'}
               # 猫须颜色whisker
               , whiskerprops = {'color': 'r'}
               # 猫须界限颜色whisker cap
               , capprops = {'color': 'b'})
    plt.title(name +'+petal_size')
    flag += 1
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




