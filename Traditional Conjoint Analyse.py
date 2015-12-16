
# coding: utf-8

# In[2]:

import numpy as np
import pandas as pd


# In[3]:

caInputeDF = pd.read_csv("ConjointInput.csv", sep = ";")


# In[4]:

caInputeDF


# In[5]:

caInputeDF


# ## First step is to introduce dummy variables for every stimulus<p>
# There are in total 9 differen stimuli, and 18 different combinations
# 

# In[6]:

ConjointDummyDF = pd.DataFrame(np.zeros((18,9)), columns=["Rank","A1", "A2", "A3",
                                                    "B1","B2", 
                                                    "C1", "C2",
                                                    "C3"])


# In[7]:

ConjointDummyDF.Rank = caInputeDF.Rank

for index, row in caInputeDF.iterrows(): 
    stimuli1, stimuli2, stimuli3 = caInputeDF["Stimulus"].ix[index][:2],     caInputeDF["Stimulus"].ix[index][2:4], caInputeDF["Stimulus"].ix[index][4:6]
    
    
    ConjointDummyDF.ix[index, [stimuli1,stimuli2,stimuli3]] = 1


# In[8]:

ConjointDummyDF.head()


# ## Insert the proper Stimulus names

# In[9]:

fullNames = {"Rank":"Rank",            "A1": "32\" (81cm)","A2": "37\" (94cm)","A3": "42\" (107cm)",           "B1": "Plasma", "B2":"LCD",            "C1":"Silver", "C2":"Black", "C3": "Anthrazit",          }

ConjointDummyDF.rename(columns=fullNames, inplace=True)


# In[10]:

ConjointDummyDF.head()


# ## Estimate Main Effects with a linear regression
# <p>
# 
# There are different ways for parameter estimation beside linear regression depending on what kind of rating you have.<br>
# For example using Probit or Logit is the output is not a rank but a decision (1=chose stimulus, 0 = no choice).

# In[11]:

import statsmodels.api as sm


# In[12]:

ConjointDummyDF.columns


# In[13]:

X = ConjointDummyDF[[u'32" (81cm)', u'37" (94cm)', u'42" (107cm)', u'Plasma',       u'LCD', u'Silver', u'Black', u'Anthrazit']]
X = sm.add_constant(X)
Y = ConjointDummyDF.Rank
linearRegression = sm.OLS(Y, X). fit()
linearRegression.summary()


# ## Part worth values & relative importance of the stimuli
# <p>
# Importance of Stimuli= Max(beta) - Min(beta)
# <br>
# Relative Importance of Stimuli = Importance of Stim / Sum(Importance of all Stimuli)

# In[48]:

importance = []
relative_importance = []

rangePerFeature = []

begin = "A"
tempRange = []
for stimuli in fullNames.keys():
    if stimuli[0] == begin:
        tempRange.append(linearRegression.params[fullNames[stimuli]])
    elif stimuli == "Rank":
        rangePerFeature.append(tempRange)
    else:
        rangePerFeature.append(tempRange)
        begin = stimuli[0]
        tempRange = [linearRegression.params[fullNames[stimuli]]]
        


# In[49]:

for item in rangePerFeature:
    importance.append( max(item) - min(item))

for item in importance:
    relative_importance.append(100* round(item/sum(importance),3))


# ### Summary & Results

# In[55]:

print "Relative Importance of Feature:\n\nMonitor Size:",relative_importance[0], "%","\nType of Monitor:", relative_importance[1], "%", "\nColor of TV:", relative_importance[2], "%"


# What would be the optimal product bundle? <p>
# 42", LCD, Black

# In[62]:

#As array that looks like X
#Must include Constant!

optBundle = [1,0,0,1,0,1,0,1,0]
print "The best possible Combination of Stimuli would have the highest rank:",linearRegression.predict(optBundle)[0]


# In[ ]:



