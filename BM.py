# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 04:53:11 2018

@author: Gurudeo
"""
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import sklearn.linear_model
import sklearn.ensemble
data=pd.read_csv('data.csv')

#lets looks at number od types of differnet categorical variable
data.Item_Fat_Content.value_counts()
data.Item_Type.value_counts()
data.Outlet_Size.value_counts()
data.Outlet_Type.value_counts()
data.Outlet_Location_Type.value_counts()

#imputing Missing value
#check which are missings
data_Missing=data.isnull().sum()
data_Missing

#Outlet_Size
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
data.Outlet_Size.value_counts()
plt.scatter(data['Outlet_Size'],data['Outlet_Location_Type'])
twowaytable=pd.crosstab(data['Outlet_Size'],data['Outlet_Location_Type'])
twowaytable
#from plot and table it is observed that Tier2 Location type will have Outlet_Size of small
d={'Tier 2':'Small'}
s=data.Outlet_Location_Type.map(d)
data.Outlet_Size=data.Outlet_Size.combine_first(s)
data.Outlet_Size.value_counts()
data.Outlet_Size.isnull().any()

#Item_Weight

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

data.Item_Weight.isnull().any()

##Feature Engineering

#combining Fat_Content_Type such as LF and Low Fat into single Types
data.Item_Fat_Content.value_counts()
#Fat_Content showing redudancy of differnt types
data['Item_Fat_Content']=data['Item_Fat_Content'].replace({'low fat':'Low Fat','reg':'Regular','LF':'Low Fat'})
data.Item_Fat_Content.value_counts()

#No of years of Outlet 

data['Outlet_Years']=2017-data['Outlet_Establishment_Year']


#Item_Visiblity can not be zero .so impute it with corresponding aveerage value of visiblity
#of particular Item_Identifier Type

data['Item_Visibility']=data['Item_Visibility'].replace(0.0,np.nan)
data['Item_Visibility'].fillna(data.groupby('Item_Identifier')['Item_Visibility'].mean())

#
Mean_Visibility=data['Item_Visibility'].mean()


data['Item_Visibility_MeanRatio']=data.apply(lambda x:x['Item_Visibility']/Mean_Visibility,axis=1)

var_mod=['Item_Fat_Content','Outlet_Location_Type','Outlet_Size','Outlet_Type','Item_Type']

number=LabelEncoder()

for i in var_mod:
      data[i]=number.fit_transform(data[i])
      


