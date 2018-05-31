# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 02:19:02 2018

@author: Gurudeo
"""
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import sklearn.linear_model
import sklearn.ensemble
data=pd.read_csv('data.csv')
data_Missing=data.isnull().sum()
data_Missing




#step 1: Missing values treatement
#Hypothesis :Size of Outlet will depent on Outlet_Type
#outlet_size by small if Outlet_Type is of Grocery Shop
data.Outlet_Size.value_counts()
plt.scatter(data['Outlet_Size'],data['Outlet_Type'])
twowaytable=pd.crosstab(data['Outlet_Size'],data['Outlet_Type'])
twowaytable
#Assumption from plot
#fill missing values
d={'Grocery Store':'Small'}
s=data.Outlet_Type.map(d)
data.Outlet_Size=data.Outlet_Size.combine_first(s)
d={'Grocery Store':'Small'}

#Hypothesis :Outlet_Size depends on Outlet_Location_Type
#Similariy for plot of outlet_Size and Outlet_Location Type
plt.scatter(data['Outlet_Size'],data['Outlet_Location_Type'])
twowaytable=pd.crosstab(data['Outlet_Size'],data['Outlet_Location_Type'])
twowaytable
#from plot and table it is observed that Tier2 Location type will have Outlet_Size of small
d={'Tier 2':'Small'}
s=data.Outlet_Location_Type.map(d)
data.Outlet_Size=data.Outlet_Size.combine_first(s)
data.Outlet_Size.value_counts()

data_Missing_temp=data.isnull().sum()

data_Missing_temp
#Update Weight of Item by taking average of corresponding item type and  
#Missing values in corresponding locations
Mean_values_Item_Type_data=data.groupby('Item_Type')['Item_Weight'].mean()

data.Item_Type.value_counts()
#Fruits and Vegetables
d={'Fruits and Vegetables':Mean_values_Item_Type_data['Fruits and Vegetables']}
s=data.Item_Type.map(d)
data.Item_Weight=data.Item_Weight.combine_first(s)

#Snack Foods
d={'Snack Foods':Mean_values_Item_Type_data['Snack Foods']}
s=data.Item_Type.map(d)
data.Item_Weight=data.Item_Weight.combine_first(s)

#Household
d={'Household':Mean_values_Item_Type_data['Household']}
s=data.Item_Type.map(d)
data.Item_Weight=data.Item_Weight.combine_first(s)

#Frozen Foods
d={'Frozen Foods':Mean_values_Item_Type_data['Frozen Foods']}
s=data.Item_Type.map(d)
data.Item_Weight=data.Item_Weight.combine_first(s)

#Dairy
d={'Dairy':Mean_values_Item_Type_data['Dairy']}
s=data.Item_Type.map(d)
data.Item_Weight=data.Item_Weight.combine_first(s)

#Canned
d={'Canned':Mean_values_Item_Type_data['Canned']}
s=data.Item_Type.map(d)
data.Item_Weight=data.Item_Weight.combine_first(s)

#Banking Goods
d={'Baking Goods':Mean_values_Item_Type_data['Baking Goods']}
s=data.Item_Type.map(d)
data.Item_Weight=data.Item_Weight.combine_first(s)

#Health and Hygiene 
d={'Health and Hygiene':Mean_values_Item_Type_data['Health and Hygiene']}
s=data.Item_Type.map(d)
data.Item_Weight=data.Item_Weight.combine_first(s)

#Soft Drinks
d={'Soft Drinks':Mean_values_Item_Type_data['Soft Drinks']}
s=data.Item_Type.map(d)
data.Item_Weight=data.Item_Weight.combine_first(s)

#Meat
d={'Meat':Mean_values_Item_Type_data['Meat']}
s=data.Item_Type.map(d)
data.Item_Weight=data.Item_Weight.combine_first(s)

#Breads
d={'Breads':Mean_values_Item_Type_data['Breads']}
s=data.Item_Type.map(d)
data.Item_Weight=data.Item_Weight.combine_first(s)

#Hard Drinks
d={'Hard Drinks':Mean_values_Item_Type_data['Hard Drinks']}
s=data.Item_Type.map(d)
data.Item_Weight=data.Item_Weight.combine_first(s)

#Others
d={'Others':Mean_values_Item_Type_data['Others']}
s=data.Item_Type.map(d)
data.Item_Weight=data.Item_Weight.combine_first(s)

#Starchy Foods
d={'Starchy Foods':Mean_values_Item_Type_data['Starchy Foods']}
s=data.Item_Type.map(d)
data.Item_Weight=data.Item_Weight.combine_first(s)

#Breakfast
d={'Breakfast':Mean_values_Item_Type_data['Breakfast']}
s=data.Item_Type.map(d)
data.Item_Weight=data.Item_Weight.combine_first(s)

#Seafood
d={'Seafood':Mean_values_Item_Type_data['Seafood']}
s=data.Item_Type.map(d)
data.Item_Weight=data.Item_Weight.combine_first(s)
#recheck missing values again
data_temp=data.isnull().sum()
data_temp

#Add New feature Outlet operation years
data['Outlet_years']=2017-data['Outlet_Establishment_Year']



#converting catgorical into numberical varibles


#step 2: Categorical into numerical variable
number=LabelEncoder()
#Item_Type

data['Item_Type'] = number.fit_transform(data['Item_Type'].astype(str))

#Outlet_Type
data['Outlet_Type']=number.fit_transform(data['Outlet_Type'].astype(str))

#Outlet_Location_Type
data['Outlet_Location_Type']=number.fit_transform(data['Outlet_Location_Type'].astype(str))

#Outlet_Size
data['Outlet_Size']=number.fit_transform(data['Outlet_Size'].astype(str))

#Fat_Contents

data['Item_Fat_Content']=number.fit_transform(data['Item_Fat_Content'].astype(str))

#for convience multiply Item_Visiblty by 100
data['Item_Visibility']=data['Item_Visibility']*1000

##Feature creations
#Item which is visible and having Lower MRP will be prefered to purchased 
#So create feature such that ItemMRP*Item_Visiblity
#In addition replace Item_Visiblity 0.000 by some minimum amount half of second minimum
#find second minimum by logic
data.Item_Visibility.value_counts()
data['Item_Visibility'].nsmallest(527)
data['Item_Visibility'].replace(0.000000,0.178735,inplace=True)
#Check mean(),median() and mode then found distribution item visiblity is 
#stable thus takes log
plt.boxplot(data['Item_Visibility'])
data['Item_Visibility_log']=np.log(data['Item_Visibility'])
plt.boxplot(data['Item_Visibility_log'])
###features creation 
data['Item_MRP_Visibility_log']=data['Item_MRP']*data['Item_Visibility_log']

#extrem values
#by using boxplots and checking mean,median and  of Item_MRP and Item_Visiblity
#apply log transform on them
#Item_Visibility
plt.boxplot(data['Item_Weight'])
plt.boxplot(data['Item_MRP'])
plt.boxplot(data['Item_Visibility'])

#
data['Item_Visibility_log']=np.log(data['Item_Visibility'])

###features creation 
data['Item_MRP_Visibility_log']=data['Item_MRP']*data['Item_Visibility_log']


#deciding predictos
predictors=['Item_Weight','Item_Fat_Content','Item_Visibility','Item_MRP_Visibility_log','Item_Visibility_log','Item_Type','Item_MRP','Outlet_Establishment_Year','Outlet_Size','Outlet_Location_Type','Outlet_Type']

x_data = data[predictors].values
y_data = data[''].values



#feature importance
model = sklearn.ensemble.RandomForestRegressor()
model.fit(x_data, y_data)
featimp = pd.Series(model.feature_importances_, index=predictors).sort_values(ascending=False)
print (featimp)

#
X=data[data.columns[1:]]
y=data['Item_Outlet_Sales']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=42, stratify=y)



