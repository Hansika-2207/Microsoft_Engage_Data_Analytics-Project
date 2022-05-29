#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# #  DATA ANALYSIS OF CARS DATASET
# 

# In[ ]:





# ####  In this Analysis , we have dataset of cars containing all specifications of car and car price .So by using this I am going to predict the expected price of car if any company decides to manufacture a new car. There can be many features  in any car but we first try to find out the featuresof car  which affect the car pricing most then try to predict the price of car. Prediction also depends on the dataset available , larger the dataset more accurate the prediction 

# ### Importing Essential Libraries of Python 

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Reading the csv file

# In[2]:


df=pd.read_csv('cars_engage_2022.csv',index_col=0)


# In[4]:


df.shape


# Car Dataset containing the 1276 rows only and 140 columns(Features of Car).As Dataset is very small there is large probability of getting inaccurate prediction.
# But let's try to make our predictions with this small dataset only.

# In[5]:


# To make all columns Visible


# In[6]:


pd.set_option('display.max_columns',None)


# In[7]:


df.head(1)


# #### To check how much data is not present we are using heatmap 

# In[8]:


sns.heatmap(df.isnull())


# ###   % of null values present in specific column

# In[ ]:





# % of missing values in columns the dataset 

# In[9]:


missing_data_percent=df.isnull().sum()/df.shape[0] *100
missing_data_percent


# ###  Data Cleaning 

# In[10]:


# remove the colums which contains more unknown values 
# here I am removing the columns which contains moe than 15% null values so that it will be more easy to analyse data
# which is given 


# In[11]:


more_null_value_col=[var for var in df.columns if df[var].isnull().sum()/df.shape[0] *100>15]


# In[ ]:


# function to delete columns from dataset which contains more missing values more tha 15%


# In[12]:


for col in more_null_value_col:
    del df[col]


# In[13]:


df.head(1)


# In[14]:


# columns left ater deleting columns in more_null_value_cols


# In[15]:


df.columns


# In[16]:


# Heatmap after removing some columns 


# In[17]:


sns.heatmap(df.isnull())


# Our data has both numberical and categorical data but most of numerical data is of object.
# For data cleaning of numerical data we first need to convert its dataType.

# To find the data Type of columns

# In[18]:


df.info()


# In[19]:


# function to change the format and  data type of currancy


# In[20]:


def change_currancy_format(x):
    x=x.replace('Rs.','')
    x=x.replace(',','')
    return x


# In[21]:


df['Ex-Showroom_Price']=df['Ex-Showroom_Price'].apply(change_currancy_format)


# In[22]:


df['Ex-Showroom_Price']=df['Ex-Showroom_Price'].astype('float')


# In[23]:


df['Ex-Showroom_Price'][0]


# In[24]:


# Function to change the format of other numerical data columns which contains any floating number and some 
# units in string


# In[25]:


import re


# In[26]:


def change_format(x):
    if type(x)==str:
        x=re.sub(r"[A-Za-z]", '',x, flags = re.I)
        x=x.replace(',','')
        x=x.replace('/','')
    return float(x)


# In[27]:


change_format_col=['Displacement','Fuel_Tank_Capacity','Height','Length','Width','Wheelbase']


# In[28]:


for col in change_format_col:
    df[col]=df[col].apply(change_format)


# After changing data types of some columns our data frame looks like:

# In[29]:


df.head(1)


# In[32]:


df.rename(columns = {'Ex-Showroom_Price':'Ex-Showroom_Price_Rs',
                     'Displacement':'Displacement_cc',
                     'Fuel_Tank_Capacity':'Fuel_Tank_Capacity_litres',
                     'Height':'Height_mm',
                     'Length':'Length_mm',
                     'Width':'Width_mm',
                     'Wheelbase':'Wheelbase_mm'
                     }, inplace = True)


# After Renaming some column names Our dataset looks like:

# In[33]:


df.head(1)


# ### Filling the NaN values  of numerical columns

# In[34]:


df['Ex-Showroom_Price_Rs'].isnull().sum()


# In[35]:


df['Displacement_cc'].isnull().sum()


# In[36]:


sns.histplot(df['Displacement_cc'],kde=True)


# #### Error Found in the dataset 
Here min Displacement of engine is 72 which is not in actual .So from here we can conclude that there may be any error in the Displacement column of this data. This may generate some issuses in predicting any parameter dependent on displacement.These types of error will cause inaccuracy in prediction
# In[37]:


df['Displacement_cc'].describe()


# ### Before filling nan values correlation tables is as follows
# 

# In[38]:


df.corr()


# #### Correlation between Ex-Showroom_Price_Rs and Displacement_cc is 0.794293

# In[39]:


# First finding the mean displacment of engine according to manufacturing company then fill the nan values
# with mean displacemnt according to the 'Make' (Company)


# In[40]:


avg_displacement=df[['Make','Displacement_cc']].groupby('Make').mean()


# In[41]:


df = df.merge(avg_displacement, on = 'Make')


# In[42]:


df.head(1)


# In[43]:


df['Displacement_cc_x']=df['Displacement_cc_x'].fillna(df['Displacement_cc_y'])


# In[44]:


del df['Displacement_cc_y']


# In[45]:


df.rename(columns = {'Displacement_cc_x':'Displacement_cc'}, inplace = True)


# In[47]:


df['Displacement_cc'].isnull().sum()


# In[48]:


df['Ex-Showroom_Price_Rs'].corr(df['Displacement_cc'])


# #### After filling NaN values the correlation there is very less difference in correlation between other parameters and displacement .
# #### The difference is large if we fill the values with mean , mode or median of displcement of that column

# In[49]:


sns.histplot(df['Displacement_cc'],kde=True)


# Plot is very similar to the before filling nan values

# In[ ]:





# In[50]:


# Filling missing values of Cylinders


# In[51]:


df['Cylinders'].isnull().sum()


# Cylinders is stronly related with Displacement so we will try to fill missing values using Cylinders data column

# In[53]:


df['Cylinders'].corr(df['Displacement_cc'])


# In[54]:


df['Cylinders'].value_counts()


# In[55]:


sns.catplot(x='Cylinders',data=df,kind='count')


# In[56]:


avg_cyl=df[['Cylinders','Displacement_cc']].groupby('Cylinders').mean()


# In[57]:


avg_cyl


# In[58]:


avg_cyl.reset_index(level=0,inplace=True)


# In[59]:


avg_cyl


# In[60]:


sns.relplot(x='Cylinders',y='Displacement_cc',data=avg_cyl)


# In[ ]:





# In[61]:


# To fill the cylinder columns 


# In[62]:


m1 = (df['Displacement_cc'] > 0) & (df['Displacement_cc'] <= 624.0)
m2 = (df['Displacement_cc'] > 624.0) & (df['Displacement_cc'] <= 1159)
m3 = (df['Displacement_cc'] > 1159) & (df['Displacement_cc'] <= 1618)
m4 = (df['Displacement_cc'] >1618) & (df['Displacement_cc'] <= 2141)
m5 = (df['Displacement_cc'] > 2141) & (df['Displacement_cc'] <= 2670)
m6 = (df['Displacement_cc'] >2670) & (df['Displacement_cc'] <= 4206)
m7 = (df['Displacement_cc'] > 4206) & (df['Displacement_cc'] <= 5204)
m8 = (df['Displacement_cc'] > 5204) & (df['Displacement_cc'] <= 5968)
m9 = (df['Displacement_cc'] > 5968)

df.loc[m1,'Cylinders'] = df.loc[m1,'Cylinders'].fillna(2)
df.loc[m2,'Cylinders'] = df.loc[m2,'Cylinders'].fillna(3)
df.loc[m3,'Cylinders'] = df.loc[m3,'Cylinders'].fillna(4)
df.loc[m4,'Cylinders'] = df.loc[m4,'Cylinders'].fillna(5)
df.loc[m5,'Cylinders'] = df.loc[m5,'Cylinders'].fillna(6)
df.loc[m6,'Cylinders'] = df.loc[m6,'Cylinders'].fillna(8)
df.loc[m7,'Cylinders'] = df.loc[m7,'Cylinders'].fillna(10)
df.loc[m8,'Cylinders'] = df.loc[m8,'Cylinders'].fillna(12)
df.loc[m9,'Cylinders'] = df.loc[m9,'Cylinders'].fillna(16)


# In[63]:


df['Cylinders'].isnull().sum()


# In[64]:


df['Cylinders'].corr(df['Displacement_cc'])


# After filling all missing values in cylinder columns ,the correlations remains the approx same

# In[65]:


# Filling Missing values of Valves_Per_cylinder


# In[66]:


df['Valves_Per_Cylinder'].isnull().sum()


# In[67]:


df.corr()


# From the corrrelation Table we can see that Vales_Per_Cylinder is not correlated with any other parameter so we can use mode to fill this parameter  

# In[ ]:





# In[68]:


sns.catplot(x='Valves_Per_Cylinder',data=df,kind='count')


# In[ ]:





# In[69]:


sns.displot(df['Valves_Per_Cylinder'],kde=True)


# In[70]:


df['Valves_Per_Cylinder']=df['Valves_Per_Cylinder'].fillna(df['Valves_Per_Cylinder'].mode()[0])


# In[71]:


df['Valves_Per_Cylinder'].isnull().sum()


# In[ ]:





# Analysing the height, width and length of cars from dataset

# In[ ]:





# In[72]:


sns.displot(data=df, x='Height_mm',kde=True)
sns.displot(data=df, x='Width_mm',kde=True)
sns.displot(data=df, x='Length_mm',kde=True)


# In[73]:


df[['Height_mm','Width_mm','Length_mm']].describe()


# From above discription we can found error in length , height and Width columns in the dataset .
# Minimum Values of Length, Height and Width of Car can't be too less as in the dataset. 

# #### Error found in the dataset 

# In[74]:


# function to remove observed errors from length,height and width column


# In[75]:


def remove_errors(x):
    if x<10:
        return float(x*1000);
    return float(x)


# In[76]:


df['Length_mm']=df['Length_mm'].apply(remove_errors)


# In[77]:


df['Height_mm']=df['Height_mm'].apply(remove_errors)


# In[78]:


df['Width_mm']=df['Width_mm'].apply(remove_errors)


# In[79]:


df[['Height_mm','Width_mm','Length_mm']].describe()


# After removing errors

# In[80]:


sns.displot(data=df, x='Height_mm',kde=True)
sns.displot(data=df, x='Width_mm',kde=True)
sns.displot(data=df, x='Length_mm',kde=True)


# In[81]:


# Checking to missing values


# In[82]:


df['Height_mm'].isnull().sum()


# In[83]:


df['Width_mm'].isnull().sum()


# In[84]:


df['Length_mm'].isnull().sum()


# In[85]:


# As missing values are very less we can use mode value directly


# In[86]:


df['Height_mm']=df['Height_mm'].fillna(df['Height_mm'].mean())
df['Width_mm']=df['Width_mm'].fillna(df['Width_mm'].mean())


# In[87]:


df['Height_mm'].isnull().sum()


# In[88]:


df['Width_mm'].isnull().sum()


# In[ ]:





# In[89]:


# filling missing values of doors


# In[90]:


df['Doors'].isnull().sum()


# In[91]:


sns.displot(data=df, x='Doors',kde=True)


# In[92]:


df['Doors'].mean()


# In[93]:


df['Doors'].mode()


# In[94]:


df['Doors']=df['Doors'].fillna(df['Doors'].mode()[0])


# In[95]:


df['Doors'].isnull().sum()


# In[ ]:





# In[96]:


# Filling the missing values of Seating Capacity 


# In[97]:


df['Seating_Capacity'].isnull().sum()


# In[98]:


sns.displot(x=df['Seating_Capacity'])


# In[99]:


sns.relplot(x='Seating_Capacity',y='Ex-Showroom_Price_Rs',data=df)


# In[ ]:





# Missing values of Seating Capacity is very less so we can fill this with mode value as there is less chance of getting errors 
# 
# 

# In[100]:


df['Seating_Capacity']=df['Seating_Capacity'].fillna(df['Seating_Capacity'].mode()[0])


# In[101]:


df['Seating_Capacity'].isnull().sum()


# In[ ]:





# In[102]:


# Filling the missing values of Wheelbase


# In[103]:


df['Wheelbase_mm'].isnull().sum()


# In[104]:


sns.displot(x=df['Wheelbase_mm'],kde=True)


# Wheelbase is normally distributed so we can use mean value to fill its missing values and number of missin values is also less so it will not prodoce more error

# In[105]:


df['Wheelbase_mm']=df['Wheelbase_mm'].fillna(df['Wheelbase_mm'].mean())


# In[106]:


sns.displot(x=df['Wheelbase_mm'],kde=True)


# After filling missing values distribution is nearly same as before

# In[ ]:





# In[125]:


df.head(1)


# In[ ]:





# In[107]:


# Checking Airbags data 


# In[108]:


df['Number_of_Airbags'].isnull().sum()


# In[109]:


df['Airbags'].isnull().sum()


# In[110]:


(df['Number_of_Airbags'].isnull() & df['Airbags'].isnull()).sum()


# In[111]:


sns.displot(x=df['Number_of_Airbags'],kde=True)


# In[ ]:





# In[112]:


sns.lineplot(x='Number_of_Airbags',y='Ex-Showroom_Price_Rs',data=df)


# In[113]:


df['Number_of_Airbags'].corr(df['Ex-Showroom_Price_Rs'])


# As number of airbags and price of car is not strongly related we will not select this feature ib our prediction 

# In[ ]:





# In[114]:


# Filling the missing value of Fuel tank Capacity column


# In[115]:


df['Fuel_Tank_Capacity_litres'].isnull().sum()


# In[116]:


df['Fuel_Tank_Capacity_litres'].describe()


# In[117]:


df.corr()


# In[ ]:





# In[118]:


sns.displot(x='Fuel_Tank_Capacity_litres',data=df,kde=True)


# In[119]:


sns.lineplot(x='Fuel_Tank_Capacity_litres',y='Ex-Showroom_Price_Rs',data=df)


# In[120]:


sns.relplot(x='Fuel_Tank_Capacity_litres',y='Ex-Showroom_Price_Rs',data=df)


# **********

# For filling missing values of fuel tank capacity I am filling the values with different methods , the method which 
# is more likely correlated as before will be choosen

# In[122]:


# Filling missing vallues with mean 


# In[123]:


df2=df[df.columns]


# In[124]:


df2['Fuel_Tank_Capacity_litres']=df2['Fuel_Tank_Capacity_litres'].fillna(df2['Fuel_Tank_Capacity_litres'].mean())


# In[125]:


sns.lineplot(x='Fuel_Tank_Capacity_litres',y='Ex-Showroom_Price_Rs',data=df2)


# In[126]:


df2.corr()


# Using mean produces huge correlation difference as before 

# In[127]:


# Using median


# In[128]:


df3=df[df.columns]


# In[129]:


df3.corr()


# In[130]:


df3['Fuel_Tank_Capacity_litres']=df3['Fuel_Tank_Capacity_litres'].fillna(df3['Fuel_Tank_Capacity_litres'].median())


# In[131]:


sns.lineplot(x='Fuel_Tank_Capacity_litres',y='Ex-Showroom_Price_Rs',data=df3)


# In[133]:


df3.corr()


# Using median also produces huge correlation difference as before 

# In[ ]:





# #### Using Above Approaches we can find that there will be huge error if we try  to fill Fuel_Tank_Capacity With mean, mode or median values as missing values are also more in number so we don't use these for further good Analysis

# #### Using Regression Model we are going to fill these NaN values 

# In[ ]:





# In[134]:


# Using any temporary dataset to find if it is good to use regression model to fill nan values


# In[135]:


df4=df[df.columns]


# In[136]:


df4.corr()


# We can observe from above that Ex-Showroom_Price_Rs	and Displacement_cc are more related with Fuel_Tank_Capacity so
# I am going to use these parameters to find regression line than fill nan values according to the line, By this there is less cahnce of getting errors

# In[137]:


df5=df[['Ex-Showroom_Price_Rs','Displacement_cc','Fuel_Tank_Capacity_litres']]


# In[138]:


df5=df5.dropna()


# In[139]:


df5.info()


# In[140]:


x=df5[['Ex-Showroom_Price_Rs','Displacement_cc']]


# In[141]:


y=df5['Fuel_Tank_Capacity_litres']


# In[142]:


from sklearn.linear_model import LinearRegression
lr = LinearRegression()


# In[143]:


lr.fit(x,y)


# In[144]:


lr.coef_


# In[145]:


lr.intercept_


# In[146]:


# y=-3.50579018e-07*df['Ex-Showroom_Price_Rs']+1.63121218e-02*df['Displacement_cc'+23.551868076838996]


# In[147]:


df4['Fuel_Tank_Capacity_litres']=df4['Fuel_Tank_Capacity_litres'].fillna(-3.50579018e-07*df4['Ex-Showroom_Price_Rs']+1.63121218e-02*df['Displacement_cc']+23.551868076838996)


# In[148]:


df4.corr()


# In[149]:


sns.regplot(y='Ex-Showroom_Price_Rs',x='Fuel_Tank_Capacity_litres',data=df4)


# In[150]:


# From above observation we can find that using regression model to fill nan values of Fuel_Tank_Capacity_litres.


# In[151]:


df['Fuel_Tank_Capacity_litres']=df['Fuel_Tank_Capacity_litres'].fillna(-3.50579018e-07*df['Ex-Showroom_Price_Rs']+1.63121218e-02*df['Displacement_cc']+23.551868076838996)


# In[152]:


df['Fuel_Tank_Capacity_litres'].isnull().sum()


# In[ ]:





# Doors 

# In[157]:


sns.catplot(y='Ex-Showroom_Price_Rs',x='Doors',data=df)


# In[160]:


sns.regplot(y='Ex-Showroom_Price_Rs',x='Doors',data=df)


# In[ ]:





# Seating_Capacity

# In[161]:


sns.catplot(y='Ex-Showroom_Price_Rs',x='Seating_Capacity',data=df)


# In[162]:


sns.regplot(y='Ex-Showroom_Price_Rs',x='Seating_Capacity',data=df)


# In[ ]:





# ###  Analysing Numerical Data Columns

# In[163]:


sns.pairplot(df)


# In[ ]:





# In[164]:


df.head(1)


# In[ ]:





# #### Visualising columns which do not contain numeric data

# In[ ]:


# Visualization of data for some queries for available data only , not filling the nan values as they are of Non-Numeric and if we 
# try to fill them then it can cause error in observation 


# In[ ]:





# In[165]:


df['Make'].value_counts()


# ###  Analysing the average price of car made by any specific company . 

# In[ ]:


# Average price of car made by any company


# In[166]:


avg=df[['Make','Ex-Showroom_Price_Rs']].groupby('Make').mean()
avg


# In[ ]:





# In[168]:


avg['Ex-Showroom_Price_Rs'].plot(kind='hist' ,bins=10, density = 1, color ='blue', edgecolor="white",alpha=0.8)


# In[ ]:





# In[169]:


plt1=df[['Make','Ex-Showroom_Price_Rs']].groupby('Make').mean().plot(kind='bar',legend=False)
plt1.set_xlabel("Make")
plt1.set_ylabel("Avg Price (Rupees)")
# xticks(rotation = 0)
plt.show()


# In[ ]:





# Drivetrain data

# In[170]:


df['Drivetrain'].value_counts()


# In[ ]:





# #####  Most of the car have FWD (Front Wheel Drive)

# In[171]:


sns.catplot(x='Drivetrain',kind='count',data=df,legend=False)


# In[ ]:





# In[172]:


sns.catplot(x='Drivetrain',y='Ex-Showroom_Price_Rs',data=df,legend=False)


# Drivetrain does not affect Ex-Showroom Price more strongy as we can observe from above plot

# In[173]:


df['Cylinder_Configuration'].value_counts()


# In[174]:


sns.catplot(x='Cylinder_Configuration',kind='count',data=df,legend=False)


# In[175]:


sns.catplot(x='Cylinder_Configuration',y='Ex-Showroom_Price_Rs',data=df,legend=False)

Cylinder Configuration may affect Ex-Showroom Price as we can observe from above plot. But we can't say it surely because of the size of available data is very small 
# In[ ]:





# In[176]:


df.head(1)


# In[177]:


df['Emission_Norm'].value_counts()


# In[178]:


sns.catplot(x='Emission_Norm',kind='count',data=df,legend=False)


# In[179]:


sns.catplot(x='Emission_Norm',y='Ex-Showroom_Price_Rs',data=df,legend=False)

Emission_Norm and Ex-Showroom Price does not have any strong relaton as we can observe from above plot.
# In[180]:


df.head(1)


# In[181]:


df['Engine_Location'].value_counts()


# In[182]:


sns.catplot(x='Engine_Location',kind='count',data=df,legend=False)


# Widely Engine locations choosed for cars are:
# Front, Transverse      
# Front, Longitudinal

# In[ ]:





# In[183]:


df.head(1)


# In[ ]:





# In[184]:


df['Fuel_System'].value_counts()


# In[185]:


sns.catplot(x='Fuel_System',kind='count',data=df,legend=False)


# In[186]:


sns.catplot(x='Fuel_System',y='Ex-Showroom_Price_Rs',data=df,legend=False)


# We can conclude that mostly Fuel_System used is Injection but in some Expensive Cars Fuel system PGM-Fi is used

# In[ ]:





# In[187]:


df.head(1)


# In[ ]:





# In[188]:


df['Fuel_Type'].value_counts()


# In[189]:


sns.catplot(x='Fuel_Type',kind='count',data=df,legend=False)


# In[190]:


sns.catplot(x='Fuel_Type',y='Ex-Showroom_Price_Rs',data=df,legend=False)


# We can conclude that most car have fuel type petrol or diesel 
# And in expensive cars are mostly Petrol Fuel type

# In[191]:


df.head(1)


# In[192]:


df.rename(columns = {'Ex-Showroom_Price_Rs':'ExShowroom_Price_Rs'}, inplace = True)


# In[193]:


df.head(1)


# In[194]:


df['Body_Type'].value_counts()


# In[195]:


sns.catplot(x='Body_Type',y='ExShowroom_Price_Rs',data=df,legend=False)


# In[196]:


# try to figure out for  cars less than 5 crores


# In[197]:


sns.catplot(x='Body_Type',y='ExShowroom_Price_Rs',data=df.query(" ExShowroom_Price_Rs<50000000 "),legend=False)

We can conclude that Price of car is dependent on Body Type of car also
# In[ ]:





# #### If any company want to manufacture any car of specific body type then what can be its dimensions  

# In[198]:


dfl=df[['Body_Type','Length_mm']].groupby('Body_Type').mean()


# In[199]:


dfw=df[['Body_Type','Width_mm']].groupby('Body_Type').mean()


# In[200]:


dfh=df[['Body_Type','Height_mm']].groupby('Body_Type').mean()


# In[201]:


dfnew=dfl.merge(dfw,on='Body_Type')


# In[202]:


dfnew=dfnew.merge(dfh,on='Body_Type')


# In[204]:


dfnew


# From the available dataset we can find the average length , width and height of car of specific body type.
# This can be used in manufacturing any new car of any specific body_type

# ####  Length and Width Ratio according to Body type of car

# In[205]:


sns.relplot(x = 'Length_mm', y = 'Width_mm', data = dfnew,hue='Body_Type')


# #### Length and Height Ratio according to Body type of car 

# In[206]:


sns.relplot(x = 'Length_mm', y = 'Height_mm', data = dfnew,hue='Body_Type')


# In[ ]:





# Seat Material

# In[207]:


sns.catplot(x='Seats_Material',y='ExShowroom_Price_Rs',data=df)


# In[208]:


# For closer visualization of less expensive cars


# In[209]:


sns.catplot(x='Seats_Material',y='ExShowroom_Price_Rs',data=df.query(" ExShowroom_Price_Rs<50000000 "))


# Mostly Seat material of Cars is Leather and in most Expensive cars Leather seats are used only
# 

# In[ ]:





# HandBrake

# In[210]:


sns.catplot(x='Handbrake',y='ExShowroom_Price_Rs',data=df)


# In[211]:


# For Clear visualization select rows where ExShowroom_Price_Rs<50000000 as most of data lies in this range


# In[212]:


sns.catplot(x='Handbrake',y='ExShowroom_Price_Rs',data=df.query(" ExShowroom_Price_Rs<50000000 "))


# Manual Brakes are widly used in less Expensive Cars 

# In[ ]:





# Gears:

# In[213]:


df['Gears'].value_counts()


# In[214]:


sns.catplot(x='Gears',y='ExShowroom_Price_Rs',data=df,order=['4','5','6','7','8','9','Single Speed Reduction Gear ','7 Dual Clutch'])


# In[215]:


# For Clear Visualization select rows where ExShowroom_Price_Rs<50000000 only 


# In[216]:


sns.catplot(x='Gears',y='ExShowroom_Price_Rs',data=df.query(" ExShowroom_Price_Rs<50000000 "),order=['4','5','6','7','8','9','Single Speed Reduction Gear ','7 Dual Clutch'])


# Single Speed Reduction Gear      
# 7 Dual Clutch   
# Thses gears are used in most Expensive Cars only

# In[ ]:





# In[ ]:





# In[217]:


tempdf=df[df.columns]


# In[252]:


t2=tempdf[tempdf.columns]


# In[253]:


t2.head(1)


# In[254]:


df=tempdf[tempdf.columns]


# In[ ]:





# In[319]:


# Some More plots to find the relation between categorical data and Price of car


# In[322]:


plt.figure(figsize=(10, 20))
plt.subplot(2,2,1)
sns.boxplot(x = 'Fuel_Type', y = 'ExShowroom_Price_Rs', data = df)
plt.subplot(2,2,2)
sns.boxplot(x = 'Fuel_System', y = 'ExShowroom_Price_Rs', data = df)
plt.subplot(2,2,3)
sns.boxplot(x = 'Emission_Norm', y = 'ExShowroom_Price_Rs', data = df)
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ## Prediction of Car Price using some features only

# In[ ]:





# Making prdictions using simple linear Regression model 
# As the dataset is very small we we will not make predictions highly accuratly  

# In[ ]:





# In[288]:


df.head(1)


# In[289]:


df.corr()


# In[ ]:




From Above Analysis we find some features which affect the Price of car most .
We can get these feature selection from above analysis only and some more features are discared as they contain more missing values
# In[294]:


d2=df[['Make','ExShowroom_Price_Rs','Displacement_cc','Cylinders','Fuel_Tank_Capacity_litres','Length_mm','Width_mm','Doors','Gears']]


# In[296]:


d2.head(1)


# In[297]:


d2.info()


# In[298]:


d2['Gears']=d2['Gears'].fillna('5')


# In[ ]:





# In[299]:


d2.info()


# In[ ]:


# to convert categorial data into numeric data


# In[300]:


Company_cat = pd.get_dummies(d2['Make'])


# In[301]:


d2 = pd.concat([d2, Company_cat], axis = 1)


# In[302]:


del d2['Make']


# In[303]:


d2.head(2)


# In[304]:


g = pd.get_dummies(d2['Gears'])


# In[305]:


d2 = pd.concat([d2, g], axis = 1)


# In[306]:


del d2['Gears']


# In[307]:


d2.head(2)


# In[ ]:





# In[308]:


all_cols=[d2.columns]
all_cols


# In[309]:


X=d2[[ 'Displacement_cc', 'Cylinders',
        'Fuel_Tank_Capacity_litres', 'Length_mm', 'Width_mm', 'Doors',
        'Aston Martin', 'Audi', 'Bajaj', 'Bentley', 'Bmw', 'Bugatti', 'Datsun',
        'Dc', 'Ferrari', 'Fiat', 'Force', 'Ford', 'Honda', 'Hyundai', 'Icml',
        'Isuzu', 'Jaguar', 'Jeep', 'Kia', 'Lamborghini', 'Land Rover',
        'Land Rover Rover', 'Lexus', 'Mahindra', 'Maruti Suzuki',
        'Maruti Suzuki R', 'Maserati', 'Mg', 'Mini', 'Mitsubishi', 'Nissan',
        'Porsche', 'Premier', 'Renault', 'Skoda', 'Tata', 'Toyota',
        'Volkswagen', 'Volvo', '4', '5', '6', '7', '7 Dual Clutch', '8', '9',
        'Single Speed Reduction Gear']]


# In[310]:


y=d2['ExShowroom_Price_Rs']


# In[312]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.1,random_state=10) 


# In[313]:


X_train.head(2)


# In[314]:


X_test.head(2)


# In[315]:


y_train.head(2)


# In[316]:


y_test.head(2)


# In[317]:


from sklearn.linear_model import LinearRegression
clf = LinearRegression()
clf.fit(X_train, y_train)


# In[318]:


clf.score(X_test, y_test)


# In[ ]:





# ## Conclusion :
# 

# #### Accuracy in prediction is 94%. There is many reasons of getting this inaccurate result. 
# #### 1. The size of data is very small . For a good estimation we need large dataset.
# #### 2. Datset initailly contains more missing values .
# #### 3. There is many errors in dataset. Many errors are observed above
# #### 4. Using of only linear regression model in the prediction

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




