
# coding: utf-8

# # Analysing the Titanic Passengers Data
# 

# ## Taking a look at the data itself : Let's try and answer the following questions.
# 1. What demographics made it possible for the passengers to survive?
# 2. How did being a male or a female affect survival? What numbers of men and women survived?
# 3. Were survivors from a higher class?
# 4. Did having siblings and parents in the ship affect survival?

# In[1]:

import numpy as np
import pandas as pd
import matplotlib as mt
import seaborn as sn


# In[2]:

titanic = pd.read_csv("titanic-data.csv")
titanic


# Opening up the data set to see what columns of data are available to be analyzed for any conclusions. 

# In[4]:

titanic.head()


# The information about various fields here from the website.
# 
# Data Dictionary
# 
#     Variable	Definition	Key
#     survival	Survival	0 = No, 1 = Yes
#     pclass	Ticket class	1 = 1st, 2 = 2nd, 3 = 3rd
#     sex	Sex	
#     Age	Age in years	
#     sibsp	# of siblings / spouses aboard the Titanic	
#     parch	# of parents / children aboard the Titanic	
#     ticket	Ticket number	
#     fare	Passenger fare	
#     cabin	Cabin number	
#     embarked	Port of Embarkation	C = Cherbourg, Q = Queenstown, S = Southampton
#     Variable Notes
# 
# pclass: A proxy for socio-economic status (SES)
# 1st = Upper
# 2nd = Middle
# 3rd = Lower
# 

# Looking at the data types in the data set for analysis
# 
# 

# In[5]:

titanic.dtypes


# In[6]:

ageSeries = pd.Series(titanic['Age'])
ageSeries.isnull().sum()


# On looking at the age column a 177 NaNs were found. Obviously it would not make sense to have this data, as our predictions on age correlations with survivors would be skewed. Therefore we need to remove these values from the dataset. Therefore we remove the rows in the data set that has NaN values using the Age column only by selecting finite values of Age from the this Column and creating a new dataset for those with the finite values.
# 
# 

# In[7]:

titanic = titanic[np.isfinite(titanic['Age'])]
titanic



# Checking number of survivors and the total number of passengers , to get an Idea of survivor:passenger ratio
# 
# 

# In[8]:

titanic.shape


# Now that the data set has been cleaned of the NaN values in age we can perform some stats.

# The count only gives me the total number of entries 891. This is the length of the data with 12 columns. The next step is to get out the numbers of survivors, since the key mentions 1 = Survived, 0 = Did not survive.

# In[9]:

titanic['Survived'].sum()


# In[10]:

ratio  = 342.0/ 891.0 
print ratio 


# Close to 40% survived.

# Let's try and look at the various correlations for the survivors and the non survivors by first writing a correlation function and then applying it to various variables. 
# 

# In[11]:

def correlation(x, y):
    meanx = x.mean()
    meany = y.mean()
    newx = (x - meanx) / x.std(ddof=0)
    newy = (y - meany) / y.std(ddof=0)
    corr = (newx * newy).mean()
    return corr


# In[12]:

ageAndSurviveCorr = correlation(titanic['Age'], titanic['Survived'])
PclassAndSurviveCorr= correlation(titanic['Pclass'], titanic['Survived'])
ParentAndSurviveCorr = correlation(titanic['Parch'], titanic['Survived'])
SiblingsAndSurviveCorr = correlation(titanic['SibSp'], titanic['Survived'])
print ageAndSurviveCorr , ' : Age and Survive Corr'
print PclassAndSurviveCorr, ' : Class and Survive Corr'
print ParentAndSurviveCorr, ' : Parents and Survive Corr'
print SiblingsAndSurviveCorr, ': Siblings and Survive Corr'


# In order to do a correlation study between sex and survival it would make sense to convert sex into a binary and then add this to the original data set. I changed 'male' to 1  and 'female' to 0. 
# 

# In[13]:

newTitanic = titanic.replace(to_replace =['male', 'female'],value =[1.0,0.0])

newTitanic.head()


# In[14]:

SexAndSurviveCorr = correlation(newTitanic['Sex'], newTitanic['Survived'])
print SexAndSurviveCorr, ': Sex and Survive Corr'


# In[15]:

newTitanic['Sex'].sum()


# 453 out of the 714 were men.

# In[16]:

men = (float(453)/714)*100
women = 100 - 63.44

print men ,'% Men'
print women, ' % Women'
print (714 -453) , ': Number of Women' 


# Writing code to check for all Sex and Survived columns (if they both are 1, these are men that survived)

# In[17]:

newTitanic['sex&survive'] = np.where(( newTitanic['Sex'] == newTitanic['Survived']) & (newTitanic['Sex'] == 1), newTitanic['Sex'],np.nan)
newTitanic['sex&survive'].sum()


# In[18]:

93.0 / 453 *100


# 93 of the 453 men survived, that's a survival chance of 20.52 % for men.

# Writing code to check for sex and survived if sex = 0 and survived = 1, these are women that survived)

# In[19]:

newTitanic['sex&survive2'] = np.where(( newTitanic['Sex'] == 0) & (newTitanic['Survived'] == 1), 1, np.nan)
newTitanic['sex&survive2'].sum()


# In[20]:

197.0 / 261


# 197 of the 261 women in the data set survived, that's a survival chance of 75%.

# From the correlation study we see that Class and Survival , and Sex and survival, seem to be extremely negatively correlated. Lets make a reduced data set from this newTitanic data and apply the coorelation function from Pandas again to see any other correlations. Also modify embarcation data to number for making the analysis easier.
# 
# 

# In[21]:

redTitanic = newTitanic.drop(newTitanic.columns[[3, 8, 10]], axis=1) 

modTitanic = redTitanic.replace(to_replace = ['C', 'Q', 'S'], value = [1, 2, 3])


# In[22]:

modTitanic.head()


# Now that we have a reduced dataset from the original with some modifications such as sex changed to binary and embarkation value changed to number it would make sense to apply the correlation function on this. 
# 
# 
# This will give us the pairwise correlation for each variable.
# 

# In[23]:

modTitanic.corr(method='pearson', min_periods=1)


# Looking at the Survived column it is immediately clear (and it confirms our previous study) : 
# 1. Survival is strongly negatively correlated with Sex (male = 1, female = 0) which means that women survived more often than men.
# 2. Survival is strongly positively correlated with Fare : higher the fare class, higher the chances one survives. 
# 3. Class and fare are strongly negatively correlated : Lower the fare, lower (higher in number value) the class (3rd being lower class compared to 1st). 
# 4. Number of parents and number of siblings are strong positively correlated : more the parents, more the siblings. There is also the correlation that the more the parents, the higher the fare. 
# 
# 
# ##THE NET qualitative understanding from this would be : Higher paying women with higher class seats survived the most. It also helped positively towards survival if one was a girl of a higher class with her parents on the ship. 
# 
# 
# Next we can separate the data set into two separate sets one for survivors and one for non-survivors to get quantitative data about survival. 
# 

# In[24]:

groupSurvive = modTitanic.groupby('Survived')


# Here we have made a group based on the key 'Survived'. Creating two groups. One with the Key 1 'group1' = Survivors and the other with the key 0, 'group 0 ' = Dead. 

# In[25]:

group1 = groupSurvive.get_group(1)
group1.head()


# In[26]:


group0 = groupSurvive.get_group(0)
group0.head()



# This group1 consists of all the survivors and the group0 consists of the the ones who did not survive.

# Let's take some statistics on this. 
# 

# In[27]:

group1Class= group1['Pclass'].mean()
print group1Class , 'Class average : survivors'
group0Class = group0['Pclass'].mean()
print group0Class, 'Class average: dead'

group1Age = group1['Age'].mean()
print group1Age, 'Age average : suvivors'
group0Age = group0['Age'].mean()
print group0Age, 'Age average : dead'

group1SibSp = group1['SibSp'].mean()
print group1SibSp, 'SibSp average : suvivors'
group0SibSp = group0['SibSp'].mean()
print group0SibSp, 'SibSp average : dead'

group1Sex = group1['Sex'].mean()
print group1Sex, 'Sex average : survivors'
group0Sex = group0['Sex'].mean()
print group0Sex, 'Sex average : dead'

group1Parch = group1['Parch'].mean()
print group1Parch, 'Parch average : suvivors'
group0Parch = group0['Parch'].mean()
print group0Parch, 'Parch average : dead'

group1Fare = group1['Fare'].mean()
print group1Fare, 'Fare average : suvivors'
group0Fare = group0['Fare'].mean()
print group0Fare, 'Fare average : dead'
                    


# 1. Being in 2nd Class and above proved positive to survival.
# 2. Being below the age of 30 was negative towards survival.
# 3. Being a woman was positive to survival.
# 4. Having parents was positive to survival. 
# 5. Paying 48 and above was positive to survival.
# 

# In[28]:

get_ipython().magic(u'pylab inline')
import matplotlib.pyplot as plt


# In[29]:

#plt.hist((group1['Sex'], group0['Sex']), bins = 10, label=['Men', 'Women'])
#plt.locator_params(axis='x', nticks=2)
#plt.xlabel('Survival Boolean')
#plt.ylabel('Numbers in that category')

#
#plt.legend()
sex = [261, 453]
plt.legend()
plt.title('Sex representation of the passengers on the titanic on the titanic')
plt.pie(sex, explode=None, labels=['men', 'women'], colors=['r', 'g'], autopct=None, pctdistance=0.6, shadow=True, labeldistance=1.1, startangle=None, radius=1, counterclock=True, wedgeprops=None, textprops=None, center=(0, 0), frame=False, hold=None, data=None)


# In[30]:

surviveMen = [93, 360]
plt.legend()
plt.title('Male representation of the passengers on the titanic')
plt.pie(surviveMen, explode=None, labels=['Survived', 'Died'], colors=None, autopct=None, pctdistance=0.6, shadow=True, labeldistance=1.1, startangle=None, radius=1, counterclock=True, wedgeprops=None, textprops=None, center=(0, 0), frame=False, hold=None, data=None)


# In[31]:

surviveWomen = [197 , 64]
plt.legend()
plt.title('Female representation of the passengers on the titanic')
plt.pie(surviveWomen, explode=None, labels=['Survived', 'Died'], colors=None, autopct=None, pctdistance=0.6, shadow=True, labeldistance=1.1, startangle=None, radius=1, counterclock=True, wedgeprops=None, textprops=None, center=(0, 0), frame=False, hold=None, data=None)


# These pie charts shows the skewed demographic of the ones who survived and the one who didn't. Among the ones that survived women had the highest chances. Among the ones that died men had the highest chance.  

# In[32]:

plt.hist((group1['Fare'], group0['Fare']),bins = 15, label=['Survivors','Dead'])
plt.xlabel('Fare Value')
plt.ylabel('Number of people who paid')
plt.title(r'Fare representation of survivors -"Green" and dead "Blue"')
plt.legend()


# People who survived seem to be in the categories that paid more. Approximately around 2X as much.
# 

# In[33]:

modTitanic['Fare'].max()
(0.527586206897 - 0.36) / 0.527586206897


# The survivor who paid maximum was paying as much 512.329 to be on the Titanic.
# 
# ## Conclusions:
# Q1. What demographics made it possible for the passengers to survive?
# A1. We were able to get qualitative understanding of the people who survived the disaster, It is pretty clear that being a young woman in the upper class having paid more have the highest rates of survival.
# 
# Q2. How did being a male or a female affect survival? What numbers of men and women survived?
# A2. We were also able to get quantitive values of the people who survived the disaster, mainly women survived more often and by significant margin. 197 of the 261 women in the data set survived, that's a survival chance of 75% when compared to a survival chance of 20.52% for men.
# 
# Q3. Were survivors from a higher class?
# A3. Indeed it was found that survivors on an average paid almost 2X as much compared to their dead counterparts to buy their tickets to survival into a higher class. People who survived has a class average of 1.8. Close to 2nd Class but also many in class 1. 
# 
# Q4. Did having siblings and parents in the ship affect survival?
# A4. Having a parent also increased the chances of survival by about 30%. 
# 
# 
# 
# ### Possible Biases in the report :
# Based on the data it is quite possible that there were a lot of mistakes in the age data and we had to take them out. Having this would have given me better predictions on age groups that were more likely to survive. 
# 
# Also another thing that came across as strange was the fact that having siblings did not affect the chances of survival much, if one sibling was saved there was a good chance that the other would be saved as well considering rescue efforts are always targeted mainly at women and children first.
# 
# 
# #### Future research :
# I quite enjoyed investigating this data set, powerful results could be derived by simple but beautiful manipulation using easy stats. I was quite excited about how I could derive conclusions for the reason that Kate survived in the movie 'Titanic' was not just a chance but an actual fact that was highly probable. I also told all my friends at work about it.
# 
# Further reasearch would be machine learning based where given this information of the passengers I could predict wether they would survive or not. This would be done by training a machine learning algorithm on the exisiting data and then making predictions of survival on the test set. 
# 
# ##### Note of thanks :
# Thanks for your valuable feedback. I enjoy making things better and cleared (esp when you have data around).
# 
# 
