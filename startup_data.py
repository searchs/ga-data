
# coding: utf-8

# In[6]:

import pandas as pd

pd.set_option('display.line_width', 5000)
pd.set_option('display.max_columns', 60)

df = pd.read_csv('TechCrunchcontinentalUSA.csv',index_col='fundedDate', parse_dates=['fundedDate'], dayfirst=True,)
# df = pd.read_csv('TechCrunchcontinentalUSA.csv')
print("First five rows:\n", df[:5])


# In[7]:

# Raised Funds Sample
raised = df['raisedAmt'][:5] 
print("Funding Raised by Companies over time:\n", raised)


# In[25]:

# Visualization using Seaborn
import pandas as pd 
from matplotlib import pyplot as plt
import seaborn as sns 
 
plt.style.use('default') 
pd.set_option('display.line_width', 5000) 
pd.set_option('display.max_columns', 60) 
 
df = pd.read_csv('TechCrunchcontinentalUSA.csv') 
print("First five rows:\n", df[:5] )
 
df = pd.read_csv('TechCrunchcontinentalUSA.csv'
                 ,index_col='fundedDate'
                 ,parse_dates=['fundedDate']
                 , dayfirst=True,) 
print("Top five rows:\n", df[:5])
raised = df['raisedAmt'][:5] 
print("Funding Raised by Companies over time:\n", raised)

# Now generate the visualization
sns.set_style("darkgrid") 
sns_plot = df['raisedAmt'].plot() 
plt.ylabel("Amount Raised in USD");
plt.xlabel("Funding Year") 
sns.plt.show()
plt.savefig('amountRaisedOverTime.pdf')


# In[12]:

# Select another subset
import pandas as pd 
from matplotlib import pyplot as plt 
plt.style.use('default') 
pd.set_option('display.line_width', 5000) 
pd.set_option('display.max_columns', 60) 

fundings = pd.read_csv('TechcrunchcontinentalUSA.csv') 

print("Type of funding:\n", fundings[:5]['round'] )
 
# Selecting multiple columns 
print("Selected company, category and date of funding:\n"
          ,fundings[['company', 'category', 'fundedDate']][600:650] )


# In[17]:

fundings[600:650].groupby('category').count()


# In[29]:

# sns.set_style("darkgrid") 
# sns_plot = fundings.plot() 
# plt.ylabel("Category");
# plt.xlabel("Number Funded") 
# sns.plt.show()
# plt.savefig('categoryFunding.pdf')


# In[26]:

# Most common category of company that got funded 
counts = fundings['category'].value_counts() 
print("Count of common categories of company that raised funds:\n", counts) 


# In[31]:

# Simple plot
counts.plot()
sns.plt.show()


# In[30]:

# Horizontal bar plot
counts.plot(kind='barh')
plt.xlabel("Count of categories")
plt.savefig('categoriesFunded.pdf') 
sns.plt.show()


# In[54]:

# Data Aggregation and Filtering
#Web fundings in CA 

# funding = pd.read_csv('TechCrunchcontinentalUSA.csv'
#                  ,index_col='fundedDate'
#                  ,parse_dates=['fundedDate']
#                  , dayfirst=True,)
plt.style.use('default')
funding = pd.read_csv(
                  'TechcrunchcontinentalUSA.csv', 
                  index_col='fundedDate',
                  parse_dates=['fundedDate'], dayfirst=True)

web_funding = funding['category'] == 'web' 
in_CA = funding['state'] == 'CA' 
print(in_CA.count())
in_city = funding['city'].isin(['Palo Alto','San Francisco', 'San Mateo','Los Angeles', 'Redwood City'])


# In[55]:

print(funding.count())


# In[56]:

web_funding = funding[web_funding & in_CA & in_city]
web_counts = web_funding['city'].value_counts() 
print("Funding rounds for companies in 'web'category by cities in CA:\n", web_counts )


# In[57]:

# More Aggregations
total_funding = funding[in_CA & in_city] 
total_counts = total_funding['city'].value_counts() 
print("Funding rounds for companies in 'all' categories by cities in CA:\n",total_counts)


# In[58]:

# Plot Web Funding as a percent of overall funding
sns.set_style("darkgrid") 
sns_plot = (web_counts*100/total_counts.astype(float)).plot(kind='barh') 
plt.xlabel("(Funding Rounds in Web Category) / (Funding Rounds in All Categories) * (100)") 
plt.savefig('webFundedByCity.pdf') 

plt.show()


# In[71]:

funds = funding.copy()
funds = funding[['raisedAmt', 'round']]
funds['month'] = funds.index.month 
print("Funding Rounds with Month Index:\n", funds)
plt.xlabel("(Funding Rounds in Web Category) / (Funding Rounds in All Categories) * (100)") 
plt.savefig('webFundedByCity.pdf') 
sns.plt.show()


# In[72]:

funding_by_month = funds.groupby('month').aggregate('sum') 
funding_by_month.index = ['Jan', 'Feb', 'Mar','Apr', 'May', 'June', 'July', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec'] 
print("Funding Rounds Grouped By Month:\n", funding_by_month)


# In[73]:

funding_by_month.plot()
plt.show()


# In[74]:

# Amount raised by month
funds['month'] = funds.index.month
funding_by_stage = funds.groupby(['month','round']).aggregate('sum') 
print(funding_by_stage)


# In[76]:

funding_by_stage.plot()
sns.plt.show()


# In[ ]:



