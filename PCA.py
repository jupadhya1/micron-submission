#!/usr/bin/env python
# coding: utf-8

# In[2]:


# code for loading the format for the notebook
import os

# path : store the current path to convert back to it later
path = os.getcwd()
os.chdir(os.path.join('..', 'notebook_format'))
from formats import load_style
load_style(plot_style = False)


# In[ ]:


os.chdir(path)

# 1. magic for inline plot
# 2. magic to print version
# 3. magic so that the notebook will reload external python modules
# 4. magic to enable retina (high resolution) plots
# https://gist.github.com/minrk/3301035
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('load_ext', 'watermark')
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from pyspark.conf import SparkConf
from pyspark.sql import SparkSession 
from pyspark.ml.feature import VectorAssembler, StandardScaler, PCA

# create the SparkSession class,
# which is the entry point into all functionality in Spark
# The .master part sets it to run on all cores on local, note
# that we should leave out the .master part if we're actually
# running the job on a cluster, or else we won't be actually
# using the cluster
spark = (SparkSession.
         builder.
         master('local[*]').
         appName('PCA').
         config(conf = SparkConf()).
         getOrCreate())

get_ipython().run_line_magic('watermark', "-a 'Ethen' -d -t -v -p numpy,pandas,matplotlib,sklearn,pyspark")


# Run a local spark session to test your installation:

# In[2]:


import findspark
findspark.init()
from pyspark.sql import SparkSession
spark = SparkSession.builder.master("local[*]").getOrCreate()


# # Data Exploration & Imputation

# In[7]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from pyspark.conf import SparkConf
from pyspark.sql import SparkSession 
from pyspark.ml.feature import VectorAssembler, StandardScaler, PCA
from pyspark.sql.functions import col,sum, udataFrame, lit
from pyspark.sql.types import *


# In[ ]:


"The starting point most data analysis problems is to perform some sort of exploratory data analysis on our raw data. This step is important because until we have basic understanding of the structure of our raw data, it might be hard to know whether the data is suitable for the task at hand or even derive insights that we can later share. During this exploratory process, unsupervised methods such as dimensionality reduction can help us identify simpler and more compact representations of the original raw data to either aid our understanding or provide useful input to other stages of analysis. Here, we'll be focusing on a specific dimensionality reduction technique called Principal Component Analysis (PCA)"


# In[ ]:


dataFrame = spark.read.options(header='True', inferSchema='True', delimiter=',').csv("data.csv")
dataFrame.printSchema()

dataFrame.select(*(sum(col(c).isNull().cast("int")).alias(c) for c in dataFrame.columns)).show()
dataFrame.groupby('target').count().show()



def Generate_Imputation(value):
   if   value == 0: return None
   else: return 0

udataFrameGenerate_Imputation = udataFrame(Generate_Imputation, IntegerType())

dataFrame = dataFrame.withColumn("helper", udataFrameGenerate_Imputation("target"))


# In[ ]:


dataFrame.select(*(sum(col(c).isNull().cast("int")).alias(c) for c in dataFrame.columns)).show()


# Now lets drop the relevant 0 target missing rows pairwise with helper, by rerunning below two cells with combinations (40, helper), (69, helper), (75, helper), (80, helper), (111, helper)

# In[ ]:


noMissing = dataFrame


# In[ ]:


noMissing=noMissing.dropna(how='all',subset=['111','helper'])
noMissing.select(*(sum(col(c).isNull().cast("int")).alias(c) for c in noMissing.columns)).show()


# In[ ]:


noMissing.groupby('target').count().show()

noMissing = noMissing.drop('helper')


# Use KNN imputation

# In[ ]:


from fancyimpute import KNN
knn_imputed = noMissing.toPandas().copy(deep=True)
knn_imputer = KNN()
knn_imputed.iloc[:, :] = knn_imputer.fit_transform(knn_imputed)


# In[ ]:


knn_imputed.to_csv("imputed_data.csv")


# # PCA

# In[9]:


dataFrame = spark.read.options(header='True', inferSchema='True', delimiter=',').csv("imputed_data.csv")


# In[10]:


dataFrame.select(*(sum(col(c).isNull().cast("int")).alias(c) for c in dataFrame.columns)).show()


# In[11]:


dataFrame.groupby('target').count().show()


# In[12]:


dataFrame = dataFrame.withColumn("target", dataFrame["target"].cast(IntegerType()))
dataFrame.printSchema()


# In[13]:


dataFrame.groupby('target').count().show()


# In[14]:


X = dataFrame.select([c for c in dataFrame.columns if c not in ['target']])


# In[15]:


X.show()


# In[16]:


X = X.drop('_c0')


# In[17]:


Y = dataFrame.select('target')
Y.show(5)


# In[18]:


X.show(5)


# Create a vector assembler to pass list of columns to PCA task

# In[19]:


assembler = VectorAssembler(
    inputCols = X.columns, outputCol = 'features')
dataFrame_X = assembler.transform(X).select('features')
dataFrame_X.show(5)


# PCA must have scaled dataset

# In[20]:


scaler = StandardScaler(
    inputCol = 'features', 
    outputCol = 'scaledataFrameeatures',
    withMean = True,
    withStd = True
).fit(dataFrame_X)
dataFrame_scaled = scaler.transform(dataFrame_X)
dataFrame_scaled.show(5)


# In[21]:


n_components = 110
pca = PCA(
    k = n_components, 
    inputCol = 'scaledataFrameeatures', 
    outputCol = 'pcaFeatures'
).fit(dataFrame_scaled)

dataFrame_pca = pca.transform(dataFrame_scaled)
print('Explained Variance Ratio', pca.explainedVariance.toArray().sum())
dataFrame_pca.show(6)




X_pca = dataFrame_pca.rdd.map(lambda row: row.pcaFeatures).collect()
X_pca = np.array(X_pca)
np.savetxt("data_new.csv", X_pca, delimiter=",")


# In[28]:


get_ipython().system('zip -r /content/PCAtransformer.zip /content/Result/')


# In[29]:


from google.colab import files
files.download("/content/reduceFeatures.zip")

