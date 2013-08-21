#using ipython notebook - data science stuff

import pandas as pd
import numpy as np
from pandas import Series, DataFrame
import statsmodels.api as sm
import matplotlib.pyplot as plt

!head -n 5 'subsaharan_africa.csv'

#making this empty due to testing
life = []
life = pd.read_csv('subsaharan_africa.csv', index_col=3, na_values=None)
jk = pd.read_csv('subsaharan_africa.csv')

life.columns

#create an array with all the years
all_year = [str(x) for x in range(1960, 2000)]
#Check what's inside all_year
print all_year.count

#drop all columns with no values
#life = life.dropna(axis=0)
print life.head(4)

life.shape

#Clean the data by removing the columns that have no value in any row
#new_data = total_data2.drop(['open', 'close'], axis=1)
life = life.drop([str(x) for x in range(1960, 2000)], axis=1)
life = life.drop('2012', axis=1)

life = life.fillna(0)
print life.head(3) #use 14 to confirm that there are 13 indicators

#holds the description of the Indicators
ind_desc = life['Indicator Name'][0:14]
ind_desc

type(life)

life.columns

#data.columns
#ind_code = life['Indicator Code'][:14]
#ind_code

#life.columns = ['Country_Code','Indicator_Name', 'Indicator_Code', '2000', '2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011']
print life.columns


life = life.drop('Indicator Name', axis=1)
life = life.drop('Country Name', axis=0)

print life.columns

life.shape

#import statsmodels.formula.api as sm
y = life['2000']
plt.plot(y, 'g--', label='DOT'); plt.legend(loc='best')
#plt.axis(y, 'g-', label='DOT')

#life['Indicator Code']
#plt.xlabel('Counted')
plt.ylabel('People')
h = life['2010']
plt.figure();
h.plot(kind='bar',figsize=(18,8))
#plot(h, label=life.Country Code); plt.legend(loc='best')

#model = sm.ols(formula='Gas ~ Temp', data=whiteside, subset = whiteside['Insul']=="Before")
#fitted = model.fit()
#print fitted.summary()


from sklearn.cluster import KMeans
kmeans = KMeans(k=3, init='random') #initialization
kmeans.fit(life) #actual execution


life['Country Code']


model = sm.OLS(formula='2011 ~ 2010', data=life, subset = life['Country_Code'] == "NGA")


life.ix[14]
life.columns[2:]
type(life)
life.keys()


life['Indicator Code'][0:27]


life['Indicator_Name'][:14]


#select particular indicator in every country
print life['Indicator Code'][::13]


life['Indicator_Name'][0::13]


plt.xlabel(life['Indicator_Name'][13])
plt.ylabel('Yearly figures')
ari = life['2010'][::13]
ari1 = life['2011'][::13]
ari.plot(subplots=True); ari1.plot(subplots=True);plt.legend(loc='best')


#plt.xlabel('Counted')
plt.ylabel('People')
h = life['2005']
plt.figure();
plot(h, figsize=(10,10))


cols = life.columns


#tickers = pd.DataFrame(['MSFT', 'AAPL'], columns= ['Ticker'])
col_trim = cols [2:]
#picks = pd.DataFrame(['SH.MED.CMHW.P3','SH.XPD.PUBL.ZS'], columns=['2000'])


hf = pd.DataFrame(life,columns=col_trim, index = life.index)


print col_trim


print hf.head(3)


life.columns


ang = hf[:13]


#bol = pd.DataFrame(data=life, index=None,columns=col_trim)
print ang.head(3)


ang.shape
ang.keys



#Clustering a single country
from sklearn.cluster import KMeans
kmeans = KMeans(k=3, init='random')
kmeans.fit(ang)


c = kmeans.predict(ang)
print c


import statsmodels.api as sm
x =ang[['2006','2007','2008','2009','2010']]
y = range(len(x))
print x


print y


ols_model = sm.OLS(y,x)


fit = ols_model.fit()
print fit.summary




#using Cross validation to split and train data
t = ang
from sklearn import cross_validation
train, test, t_train, t_test = cross_validation.train_test_split(ang, t, test_size=0.3, random_state=0)
print len(train), len(test), len(t_train), len(t_test)


from sklearn.ensemble import RandomForestRegressor
clf = RandomForestRegressor()
clf.



from sklearn.svm import SVR
cf = SVR()
cf.fit(train, t_train)
classifier = cross_validation.fit()
classifier.fit(train, t_train)
print classifier.score(test, t_test)


#Correlation
from numpy import corrcoef
corr = corrcoef(ang.T)
print corr


from pylab import pcolor, colorbar, xticks, yticks
from numpy import arange
pcolor(corr)


ang.columns


x = ang['2010']
y = ang['2011']


x = ang.columns[0:]
print x


y = ang.index
print y


ang.plot(kind='bar', figsize=(20,10)); plt.legend(loc='best')


#Regression
from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
x = ang.columns[:13]
y = ang.columns[15]
linreg.fit(ang,x)

# <markdowncell>

# life2 = life.copy()
# life2.columns[0] = 'Country Name'


gy = hf['2010']
plt.figure()
gy.plot(subplots=False, figsize=(20,10),kind='bar'); plt.legend(loc='best');


ang.plot()


plt.figure(); gy.plot(style='k--', label='All Countries'); plt.legend(loc='best')


ang.plot(figsize(20,10), kind='bar'); plt.legend(loc='best')


#2,3,4,5,6,7,8,9,10,11,12 (loads of countries) with data
hf[13::13][:4].plot(kind(pie)


from sklearn import svm
import statsmodels.api as sm
life.columns[2][0:]


ang.columns
life['Country Code'][410:]


x = ang['Indicator Code'][0:]
y = ang[1:]
ols_model = sm.OLS(x,y)


#model = sm.ols(formula='Gas ~ Temp', data=whiteside, subset = whiteside['Insul']=="Before")
mod = pd.ols(y : Series, x : dict of DataFrame -> OLS)
#model = sm.regression(formula='Indicator Code ~ 1999', data=life, subset = life['Country Code']=="AGO")
fitted = mod.fit()
print fitted.summary()


#plt.plot(life['Country Name'], life['2010'], '.', label = 'Country', mew = 0, mfc='coral', alpha = .1)


#model = sm.OLS(life.endog, life.exog).fit().resid
#sm.OLS(data.endog, data.exog).fit().resid


#Regression models : run multiple regression models and select the best fit - Linear, Logistic, Polynomial regression


#Use some clustering to group data
#Split data into training and test set
#Train data
#Run algorithm on test data(predict)
#present results


cf = hf['Indicator_Code'] = hf.index[-9]


print cf


#nt = hf.groupby(['Country Name'])['2010'].apply
#grouped = obj.groupby(key)
grped = life.groupby(life.Country_Code)


print type(grped)


ang.plot()


from sklearn import cross_validation, datasets, linear_model
#import matplotlib.pyplot as plt
#ngr = hf.icol(hf.where(life.Country_Code =='NGA'))
ngr = Series(


#SH.XPD.OOPC.TO.ZS # sample Indicator Code 
#X,Y = data['Country Code'], data['Indicator Code']
#plt.plot(X,Y,'bo')
#plt.plot(data2[1,2,3,4])
#plt.ylabel('Country Code')
#plt.show()


life.columns


#df.CatA.where(df.CatA == 'a') 
nga = hf.where(life.Country_Code == 'NGA')


#cups = hf['2004'][::13]
cups = hf[col_trim[0]][::13]


print col_trim[1]


ind = 13
yr = col_trim[1:]
ct = len(col_trim)
print col_trim[1:]


print hf.all.where(hf[])


t = 1
plt.xlabel('year')
plt.ylabel('count')
while t < ct:
    for f in yr:
        cups = hf[col_trim[t]][::f]
        plot(cups)
        t +=1


plt.xlabel('year')
plt.ylabel('count')
h = life['2010']
plot(h, label=life.Country_Code)


#plot(h, kind='barh', label=life.Country_Code)
plt.figure();
ry = life.boxplot()
h.plot(kind='barh', stacked=True, label=life.Country_Code)


life


from pandas.tools.plotting import radviz


#rn = range(0,56)
#for n in rn:
 #   print df.values[-n]
plt.figure()


radviz(ang, '2011')


pops = pd.read_csv('subsaharan_africa.csv', index_col=17,na_values=None)


pops.columns


from sklearn import cross_validation
pops = pops.dropna(axis=0)


pops.describe


X_train, X_test, y_train, y_test = cross_validation.train_test_split(pops[['Country Code', 'Indicator Code', '2005', '2006', '2007', '2008', '2009', '2010']], pops[['2011']], test_size=0.3, random_state=0)
#check the size of each set in the training and test set
print len(X_train), len(X_test), len(y_train), len(y_test)


X_train, X_test, y_train, y_test = cross_validation.train_test_split(ang[['2005', '2006', '2007', '2008', '2009', '2010']], ang[['2011']], test_size=0.3, random_state=0)
#check the size of each set in the training and test set
print len(X_train), len(X_test), len(y_train), len(y_test)


from sklearn import svm
cf = svm.SVC()


cf.fit(X_train, y_train)


x_predict = cf.predict(X_test)


from pandas.tools.plotting import scatter_matrix
#scatter_matrix()
cups.plot()


ang.plot()



    

