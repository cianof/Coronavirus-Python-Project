#!/usr/bin/env python
# coding: utf-8

# # COMP30920 Software & Data Project

# Cian Ferriter

# <font color = blue> Covid-19 Data Analysis
# 

# In[1]:


#Relevant Module imports
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np


# ##### <font color=red> Importing the Datasets into dataframes

# In[2]:


confirmed_url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"
recovered_url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv"
deaths_url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv"

confirmed_df  = pd.read_csv(confirmed_url)
recovered_df = pd.read_csv(recovered_url)
deaths_df = pd.read_csv(deaths_url)


# ##### Quick check to make sure data imported correctly

# In[3]:


confirmed_df.head()


# In[4]:


recovered_df.head()


# In[5]:


deaths_df.head()


# All looks to be ok so can get started on the Data Cleansing side of things.

# ## <font color = red> Data Cleansing

# Immediately from the above we can see issues with the data in the 'Province/State' column. this will require further investigation to see if it needs to be dropped or can be adequately populated. <br>
# I will perform some standard data checks on the columns to see if the data requires some alterations.

# In[6]:


confirmed_df.shape


# In[7]:


recovered_df.shape


# In[8]:


deaths_df.shape


# It appears that not all of the datasets are the same size. The recovered set seems to have 13 less rows than the other 2 datasets. This will require some investigation

# In[9]:


recovered_df.tail()


# In[10]:


confirmed_df.tail()


# In[11]:


counts=[]
counts=confirmed_df.isnull().sum()
print(counts)


# From this it appears that the the only compromised column is the Province/State column. I will investigate this column further below

# In[12]:


df = pd.DataFrame(confirmed_df["Province/State"].dropna())
df


# Out of 266 rows, only 81 are populated in the Province/State column. There is simply not enough data contained here to justify keeping this column. I will drop this column but may revisit this later if I need to do some region specific analysis

# In[13]:


confirmed_df.drop(['Province/State'],axis=1, inplace=True)


# In[14]:


confirmed_df.head()


# In[15]:


recovered_df.isnull().sum()


# In[16]:


recovered_df.drop(['Province/State'],axis=1, inplace=True)


# In[17]:


recovered_df.head()


# In[18]:


deaths_df.isnull().sum()


# In[19]:


deaths_df.drop(['Province/State'],axis=1, inplace=True)


# In[20]:


deaths_df.head()


# I have dropped the useless column but I still think the dataframe could use some alterations. <br> Right now the dataframe is very wide, ~150 columns wide. This is not very practical. I will try do some work on this to improve its usability.

# In[21]:


def meltData(df,name):
    #I dont want the Lat and Long data in this frame so I will drop this also
    df_melt=df.drop(['Lat', 'Long'],axis=1)
    #melt the dateframe to go from wide to short
    df_melt=df_melt.melt(id_vars=['Country/Region'],value_name=name,var_name='Date')
    #convert the date column into a datetime object
    df_melt['Date']= pd.to_datetime(df_melt['Date'])
    df_melt['date'] = df_melt['Date'].dt.date
    df_melt.drop('Date', axis=1,inplace=True)
    df_melt=df_melt.rename(columns={'date':'Date'})
    #set the index of the frame
    df_melt=df_melt.set_index(['Country/Region', 'Date'])
    return df_melt 


# Get all 3 dataframes into the melted format

# In[22]:


confirmed_df_melt=meltData(confirmed_df, "Cases")
recovered_df_melt=meltData(recovered_df, "Recoveries")
deaths_df_melt=meltData(deaths_df, "Deaths")


# In[23]:


confirmed_df_melt


# In[24]:


recovered_df_melt


# In[25]:


deaths_df_melt.tail()


# This is much better but in its current format it will be next to impossible to gather information about individual countries in this format. I will therefore group the countries together with the entire date range so they can be analysed easier.

# In[26]:


def countryClean(df, name):
    #get the countries arranged by country, date and then sum up the cases for that country on that date
    df_countries = df.groupby(['Country/Region','Date'])[name].sum()
    #the above returns a series so need to convert back into a df
    df_countries=pd.DataFrame(df_countries)
    #reset index
    df_countries=df_countries.reset_index()
    #set the index to be country and date
    df_countries=df_countries.set_index(['Country/Region', 'Date'])
    #set levels to alow for multilevel indexing later on
    df_countries.index=df_countries.index.set_levels([df_countries.index.levels[0], pd.to_datetime(df_countries.index.levels[1])])
    df_countries=df_countries.sort_values(['Country/Region','Date'],ascending=True)
    
    return df_countries


# In[27]:


confirmed_df_countries = countryClean(confirmed_df_melt, "Cases")
recovered_df_countries = countryClean(recovered_df_melt, "Recoveries")
deaths_df_countries = countryClean(deaths_df_melt, "Deaths")


# In[28]:


confirmed_df_countries


# In[29]:


recovered_df_countries


# In[30]:


deaths_df_countries


# In[31]:


confirmed_df_countries.loc["China", : ]


# ### Data Validation

# I want to make sure the data I have formatted is still intact from its original state

# In[32]:


verified_df=confirmed_df_countries.groupby(['Date']).sum()
verified_df.max()


# In[33]:


verified_df=deaths_df_countries.groupby(['Date']).sum()
verified_df.max()


# In[34]:


verified_df=recovered_df_countries.groupby(['Date']).sum()
verified_df.max()


# Comparing the max value to the worldometers (https://www.worldometers.info/coronavirus/ ) sum totals for Cases, Deaths, Recoveries I am confident the data is uncorrrupted (within a reasonable margin of error - there are different collection methods which leads to differing figures on a worldwide scale - particulary with the 'Recovered' cases)

# #### With the data now cleaned and verified I can begin digging in to some of the data to produce visulisations

# ## <font color=red> Visualisation 1

# The first set of visualisations we need to produce is a sum of all the cases, deaths and recoveries worldwide placed on a log scale.

# In[35]:


def logPlot(df, name):
    p=df.groupby(['Date']).sum().plot(title="Total " + name + "\nLog Scale",figsize=(10,6), logy=True)
    p.set_xlabel("Date")
    p.set_ylabel("Total " + name + " (Log)")


# In[36]:


logPlot(confirmed_df_countries, "Cases")


# In[37]:


logPlot(recovered_df_countries, "Recoveries")


# In[38]:


logPlot(deaths_df_countries, "Deaths")


# # <font color = red> Visualisation 1 (b)

# The next set of visualisations we need to produce is a bar chart of daily cases worldwide on a linear scale with a 3 day rolling average placed over.

# Here I will use the summed dataframe and use the .diff() function to get the daily cases

# In[39]:


def getDailyData(df, name):
    
    daily = df.groupby('Date')[name].sum()
    daily = pd.DataFrame(daily)
    daily[name] = daily[name].diff().fillna(0)
    daily["Rolling"] = daily.rolling(window=3).mean()

    return daily


# In[40]:


temp = getDailyData(confirmed_df_melt, "Cases")
temp


# In[ ]:





# In[41]:


def dailyPlot(daily, name): 
    dates=daily.index
#     print(dates)
    daily=daily.reset_index()
    #create axes and figure
    fig, ax = plt.subplots()

    ax = daily[name].plot(kind="bar", figsize=(12,6))
    ax.plot(daily["Rolling"], color='r')
    ax.set_title('Daily Worldwide ' + name + '\n 3 day MA')
    
    
    #set x-axis
    x_range = np.arange( 0, len(dates) , 3)
    plt.xticks( x_range, rotation = 60 )
    ax.set_xticklabels(dates)
    plt.xlabel("Date")
    
    plt.ylabel("Total " + name + " Worldwide")


# In[42]:


dailyPlot(getDailyData(confirmed_df_melt, "Cases"), 'Cases')


# In[43]:


dailyPlot(getDailyData(deaths_df_melt, "Deaths"), 'Deaths')


# In[44]:


dailyPlot(getDailyData(recovered_df_melt, 'Recoveries'), 'Recoveries')


# # <font color= red> Visualisation 2

# This visualisation presents the countries with days since the 100th confirmed case against total cases and likewise for Deaths and Recoveries after 10th.

# In[45]:


def getXDay(df, i, var):
    
    #use the apply function to create a new column which states the point at which a country had >= 100 cases and use a boolean to determine this
    df["True/False"] = df[var].apply(lambda x: x >=i)
    #remove false values so only rpws with values of cases >= 100 remain
    df = df.loc[df['True/False'], :]
    df=df.drop('True/False', axis=1)
   
    
    return df


# In[46]:


#remove days previous to 100th/10th 
confirmed_100_df=getXDay(confirmed_df_countries, 100, 'Cases')
deaths_10_df=getXDay(deaths_df_countries, 10, 'Deaths')
recovered_10_df=getXDay(recovered_df_countries, 10, 'Recoveries')


# In[47]:


confirmed_100_df


# In[48]:


#create a list of all the countries we want to color and focus on in the plots
special_countries = ['Ireland','US', 'Italy', 'Spain', 'Germany', 'France', 'United Kingdom', 'Canada', 'Korea, South', 'Japan', 'Singapore', 'China']


# In[49]:


def makeSpecialDf(df, country_list):
    
    special_df=pd.DataFrame()
    #loop through df and append countries in list
    for country in special_countries:
        special_df=special_df.append(df.loc(axis=0)[country, :,], ignore_index=False)
    return special_df


# In[50]:


#create dataframes of the special countries
special_confirmed_df = makeSpecialDf(confirmed_100_df, special_countries)
special_deaths_df = makeSpecialDf(deaths_10_df, special_countries)
special_recovered_df = makeSpecialDf(recovered_10_df, special_countries)


# In[51]:


special_deaths_df


# In[52]:


def addDaysCol(df):
    
    #create a new column called 'Days' which will contain the days since 100th case data - filling with nan values for now
    df['Days'] = np.nan
    #loop through the level 0 country column add create a numpy list of integers - represent days since 100/10th case/death/recovery
    
    for row in df.iterrows():
        df.loc[row[0][0], 'Days'] = np.arange(0, len(df.loc[row[0][0], 'Days']), 1)
        
    return df


# In[53]:


#create a dataframe of the 'special' countries with the days column
special_confirmed_df=addDaysCol(special_confirmed_df)
special_deaths_df=addDaysCol(special_deaths_df)
special_recovered_df=addDaysCol(special_recovered_df)


# In[54]:


#quick look at a random country from the list to eyeball the data
special_confirmed_df.loc(axis=0)['US', :,]


# In[55]:


#add in the days column for all dataframes
confirmed_100_df = addDaysCol(confirmed_100_df)
deaths_10_df = addDaysCol(deaths_10_df)
recovered_10_df =addDaysCol(recovered_10_df)


# In[56]:


#get data for one country to eyeball data
deaths_10_df.loc(axis=0)['Chile', :,]


# Testing out plotting just Ireland to begin with - will use this as a base for plotting all other countries then

# In[57]:


fig, ax = plt.subplots()
special_confirmed_df.loc["Ireland", : ].plot(x = 'Days', y = 'Cases', ax = ax, figsize=(12,8), color='g') 
# remove the legend from the plot
ax.legend_.remove()
#get the co-ordinates of the plot
# coord_list=[]
# ax = plt.gca()
# line = ax.lines[0]
# coord_list = line.get_xydata()

x=special_confirmed_df.loc["Ireland", : ]['Days'][-1]
y=special_confirmed_df.loc["Ireland", : ]['Cases'].values[-1]
# add country annotation based on last coordinate pair in list
ax.annotate('Ireland', (x,y))
print(x,y)
plt.margins(0)





# The above is a good starting point I can use this code as a basis for the main plot

# In[58]:


#I need to make a list of all the countries in the dataset, this function does that
def countryList(df):
    countries=set()
    #loop through the df and add the countries to a set
    for row in df.iterrows():
        countries.add(row[0][0])

    return countries


# In[59]:


#remove duplicates using set
con_countries = set(countryList(confirmed_100_df))
deaths_countries = set(countryList(deaths_10_df))
rec_countries = set(countryList(recovered_10_df))


# In[60]:


len(deaths_countries)


# In[61]:


def consolidatedList(lists):
    #create a list of the countries but exclude the 'special countries'
    item_list = [e for e in lists if e not in special_countries]
    
    return item_list


# In[62]:


#create the lists for each type using the above function
con_countries_cons = consolidatedList(con_countries)
deaths_countries_cons = consolidatedList(deaths_countries)
rec_countries_cons = consolidatedList(rec_countries)


# In[63]:


len(deaths_countries_cons)


# ###### test run of doubling lines

# In[64]:


fig, ax = plt.subplots()
x = list(range(confirmed_100_df['Days'].astype(int).max()))
#Double every day
double = [(2**(val))*100 for val in x] 
plt.plot(x[:17], double[:17], 'k--', alpha=.9) 
plt.annotate('Daily Doubling' , xy=(x[8], double[10]), color='Black', rotation= 35)

ax.set_yscale('log')


# The above works well and gives me a good line on the log scale. I will use this as a basis for all the other lines

# In[65]:


def specialPlot(df_s, df, var, special_countries, item_list):
    
    #create axes and figure
    fig, ax = plt.subplots()


    for country in item_list:
        if(country=="Ireland"):
            df_s.loc[country, : ].plot(x = 'Days', y = var, ax = ax, figsize=(15,10), color='tab:blue',marker="o",ls='dashed')
            x=df_s.loc["Ireland", : ]['Days'][-1]
            y=df_s.loc["Ireland", : ][var].values[-1]
            # add country annotation based on last coordinate pair in list
            ax.annotate('Ireland', (x,y))

        if(country=="US"):
            df_s.loc[country, : ].plot(x = 'Days', y = var, ax = ax, figsize=(15,10), color='r',marker="o",ls='dashed')
            x=df_s.loc["US", : ]['Days'][-1]
            y=df_s.loc["US", : ][var].values[-1]
            # add country annotation based on last coordinate pair in list
            ax.annotate('US',  (x,y))

        if(country=="Italy"):
            df_s.loc[country, : ].plot(x = 'Days', y = var, ax = ax, figsize=(15,10), color='tab:orange',marker="o",ls='dashed')

            x=df_s.loc["Italy", : ]['Days'][-1]
            y=df_s.loc["Italy", : ][var].values[-1]
            # add country annotation based on last coordinate pair in list
            ax.annotate('Italy',  (x,y))

        if(country=="Spain"):
            df_s.loc[country, : ].plot(x = 'Days', y = var, ax = ax, figsize=(15,10), color='tab:green',marker="o",ls='dashed')
            x=df_s.loc["Spain", : ]['Days'][-1]
            y=df_s.loc["Spain", : ][var].values[-1]
            # add country annotation based on last coordinate pair in list
            ax.annotate('Spain',  (x,y))



        if(country=="United Kingdom"):
            df_s.loc[country, : ].plot(x = 'Days', y = var, ax = ax, figsize=(15,10), color='#337192',marker="o",ls='dashed')
            x=df_s.loc["United Kingdom", : ]['Days'][-1]
            y=df_s.loc["United Kingdom", : ][var].values[-1]
            # add country annotation based on last coordinate pair in list
            ax.annotate('United Kingdom',  (x,y))

        if(country=="Canada"):
            df_s.loc[country, : ].plot(x = 'Days', y = var, ax = ax, figsize=(15,10), color='tab:purple',marker="o",ls='dashed')
            x=df_s.loc["Canada", : ]['Days'][-1]
            y=df_s.loc["Canada", : ][var].values[-1]
            # add country annotation based on last coordinate pair in list
            ax.annotate('Canada',  (x,y))



        if(country=="Germany"):
            df_s.loc[country, : ].plot(x = 'Days', y = var, ax = ax, figsize=(15,10), color='#dfff82',marker="o",ls='dashed')
            x=df_s.loc["Germany", : ]['Days'][-1]
            y=df_s.loc["Germany", : ][var].values[-1] 
            # add country annotation based on last coordinate pair in list
            ax.annotate('Germany',  (x,y))

        if(country=="China"):
            df_s.loc[country, : ].plot(x = 'Days', y = var, ax = ax, figsize=(15,10), color='tab:brown',marker="o",ls='dashed')
            x=df_s.loc["China", : ]['Days'][-1]
            y=df_s.loc["China", : ][var].values[-1] 
            # add country annotation based on last coordinate pair in list
            ax.annotate('China',  (x,y))


        
    #add the rest of the countries in unlabeld and grey
    for countries in item_list:
            df.loc[countries, : ].plot(x = 'Days', y = var, ax = ax, figsize=(15,10), color='#D8D8D8', alpha=0.2)
            


    ax.set_yscale('log')
    ax.legend_.remove()
    if(var == 'Cases'):
        plt.title("Cases By Time")
        plt.xlabel('Days Since 100th Case')
        plt.ylabel('Total Cases')
        doubler = 100
        ylimit = 10000000
    else:
        plt.title(var+" By Time")
        plt.xlabel('Days Since 10th ' +var)
        plt.ylabel('Total '+var)
        doubler = 10
        ylimit = 1000000

    x = list(range(df['Days'].astype(int).max()))


    #Double every day
    #uses formula y = y0*(2**x) to build y coordinates list
    ydailydouble = [(2**(val))*doubler for val in x] 
    plt.plot(x[:20], ydailydouble[:20], 'k--', alpha=.7) 
    plt.annotate('Doubles Daily' , xy=(x[8], ydailydouble[10]), color='Black', rotation= 80)

    #Double every 2 days
    y2double = [(2**(val/2)) * doubler for val in x] 
    plt.plot(x[:40], y2double[:40], 'k--', alpha=.7) 
    plt.annotate('Doubles Every 2 Days', xy =(x[19], y2double[21]), color='Black', rotation=70)

    # double every 5 days
    y5double = [(2**(val/5))*doubler for val in x] 
    plt.plot(x[:89], y5double[:89],'k--' , alpha=.7) 
    plt.annotate('Doubles Every 5 Days', xy=(x[68], y5double[70]), color='Black', rotation= 52) 

    #7day double 
    y7double = [(2**(val/7))*doubler for val in x]
    plt.plot(x[:125], y7double[:125], 'k--', alpha=.7) 
    plt.annotate("Doubles Every Week", xy=(x[100], y7double[102]), color ='Black', rotation=40)

    # fortnightly double
    y14double = [(2**(val/14))*doubler for val in x]
    plt.plot(x, y14double, 'k--', alpha=.7) 
    plt.annotate("Doubles Fortnightly", xy=(x[140], y14double[142]), color='Black', rotation=24) 
    #set the y limit so that doubling lines extend the full length of graph
    plt.ylim(doubler, ylimit)
    plt.margins(0)
    plt.grid(True)
    plt.show()


# In[66]:


specialPlot(special_confirmed_df, confirmed_100_df, 'Cases', special_countries, con_countries)


# In[67]:


specialPlot(special_deaths_df, deaths_10_df, 'Deaths', special_countries, deaths_countries )


# In[68]:


specialPlot(special_recovered_df, recovered_10_df, 'Recoveries', special_countries, rec_countries)


# # <font color = red> Visualisation 3

#  Plots (log) new daily cases on the y -axis vs. (log) total cases on the x-axis

# Example given only highlights certain countries so I will select a few countries that I am interested in and plot them. 

# Need to get a dataframe with column of countries, column of daily new cases (use previous function from earlier)and use existing cumulative column  

# In[69]:


#re-clean the df
confirmed_df_countries=confirmed_df_countries.drop('True/False', axis=1)
deaths_df_countries=deaths_df_countries.drop('True/False', axis=1)
recovered_df_countries=recovered_df_countries.drop('True/False', axis=1)


# In[70]:


drip_df_confirmed = confirmed_df_countries
drip_df_deaths = deaths_df_countries
drip_df_recoveries = recovered_df_countries


# In[71]:


drip_df_deaths


# In[72]:


def vis3Df(df, var, countries):
    
    df['Daily'] = df[var].diff().fillna(0)
    #REMOVE THE FIRST ROW OF EACH COUNTRY AS THERE IS NO 'Previous Day Data' to compare it too
    #Also remove negative values
    df["True/False"] = df['Daily'].apply(lambda x: x >= 0)
    #remove false values so only rows with values >0 remain
    df = df.loc[df['True/False'], :]
    df=df.drop('True/False', axis=1)
    
    return df


# In[73]:


#create a list of countries from the dataframes created
con_countries_drip = set(countryList(drip_df_confirmed))
deaths_countries_drip = set(countryList(drip_df_deaths))
recovered_countries_drip = set(countryList(drip_df_recoveries))


# In[75]:


#make the dataframes
drip_df_confirmed = vis3Df(drip_df_confirmed, 'Cases', con_countries_drip)
drip_df_deaths = vis3Df(drip_df_deaths, 'Deaths', deaths_countries_drip)
drip_df_recoveries = vis3Df(drip_df_recoveries, 'Recoveries', recovered_countries_drip)


# In[86]:


def dripPlot(df, var, countries):

    fig, ax = plt.subplots()

    

    for country in countries:
        
        df.loc[country, : ].plot(x = var, y = 'Daily', ax = ax, figsize=(15,10), color='lightgrey', alpha = 0.2)
        if(country=="Ireland"):
            df.loc[country, : ].plot(x = var, y = 'Daily', ax = ax, figsize=(15,10), color='tab:blue',marker="o",ls='dashed')
            x=df.loc["Ireland", : ][var][-1]
            y=df.loc["Ireland", : ]['Daily'].values[-1]
                # add country annotation based on last coordinate pair in list
            ax.annotate('Ireland', (x,y))
            
        if(country=="Germany"):
            df.loc[country, : ].plot(x = var, y = 'Daily', ax = ax, figsize=(15,10), color='#dfff82',marker="o",ls='dashed')

            x=df.loc["Germany", : ][var][-1]
            y=df.loc["Germany", : ]['Daily'].values[-1] 
                # add country annotation based on last coordinate pair in list
            ax.annotate('Germany',  (x,y))

        if(country=="Chile"):
            df.loc[country, : ].plot(x = var, y = 'Daily', ax = ax, figsize=(15,10), color='tab:brown',marker="o",ls='dashed')
            x=df.loc["Chile", : ][var][-1]
            y=df.loc["Chile", : ]['Daily'].values[-1] 
                # add country annotation based on last coordinate pair in list
            ax.annotate('Chile',  (x,y))
            
        if(country=="US"):
            df.loc[country, : ].plot(x = var, y = 'Daily', ax = ax, figsize=(15,10), color='r',marker="o",ls='dashed')
            x=df.loc["US", : ][var][-1]
            y=df.loc["US", : ]['Daily'].values[-1] 
                # add country annotation based on last coordinate pair in list
            ax.annotate('US',  (x,y))
            
        if(country=="United Kingdom"):
            df.loc[country, : ].plot(x = var, y = 'Daily', ax = ax, figsize=(15,10), color='#337192',marker="o",ls='dashed')
            x=df.loc["United Kingdom", : ][var][-1]
            y=df.loc["United Kingdom", : ]['Daily'].values[-1] 
                # add country annotation based on last coordinate pair in list
            ax.annotate('United Kingdom',  (x,y))
            
        if(country=="Canada"):
            df.loc[country, : ].plot(x = var, y = 'Daily', ax = ax, figsize=(15,10), color='tab:purple',marker="o",ls='dashed')
            x=df.loc["Canada", : ][var][-1]
            y=df.loc["Canada", : ]['Daily'].values[-1] 
                # add country annotation based on last coordinate pair in list
            ax.annotate('Canada',  (x,y))
        


    plt. annotate('', xy=(1000, 1000), xytext=(10,10), arrowprops=dict(facecolor='k', shrink=0), 
              horizontalalignment='right', verticalalignment='bottom') 
    plt.annotate("Exponential Growth", xy=(10, 10), xytext=(50,100), rotation=45)

    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.legend_.remove()
    plt.title("New "+ var+ " v Total " +var)
    plt.xlabel("Total "+ var + " (log) by Time")
    plt.ylabel("New Daily "+ var + " (log) by Time")
    plt.grid(True)
    plt.show()


# ### Cases

# In[87]:


dripPlot(drip_df_confirmed, 'Cases', con_countries)


# ### Deaths

# In[88]:


fig, ax = plt.subplots()


for country in deaths_countries_drip:
    if(country=="Ireland"):
        drip_df_deaths.loc[country, : ].plot(x = 'Deaths',y='Daily', ax = ax, figsize=(15,10), color='tab:blue',marker="o",ls='dashed')
        x=drip_df_deaths.loc["Ireland", : ]['Deaths'][-1]
        y=drip_df_deaths.loc["Ireland", : ]['Daily'].values[-1]
        # add country annotation based on last coordinate pair in list
        ax.annotate('Ireland', (x,y))
        
    if(country=="Germany"):
        drip_df_deaths.loc[country, : ].plot(x = 'Deaths', y = 'Daily', ax = ax, figsize=(15,10), color='#dfff82',marker="o",ls='dashed')

        x=drip_df_deaths.loc["Germany", : ]['Deaths'][-1]
        y=drip_df_deaths.loc["Germany", : ]['Daily'].values[-1] 
        # add country annotation based on last coordinate pair in list
        ax.annotate('Germany',  (x,y))

        
    if(country=="US"):
        drip_df_deaths.loc[country, : ].plot(x = 'Deaths', y = 'Daily', ax = ax, figsize=(15,10), color='r',marker="o",ls='dashed')
        x=drip_df_deaths.loc["US", : ]['Deaths'][-1]
        y=drip_df_deaths.loc["US", : ]['Daily'].values[-1] 
        # add country annotation based on last coordinate pair in list
        ax.annotate('US',  (x,y))
        
    if(country=="Italy"):
        drip_df_deaths.loc[country, : ].plot(x = 'Deaths', y = 'Daily', ax = ax, figsize=(15,10), color='g',marker="o",ls='dashed')
        x=drip_df_deaths.loc["Italy", : ]['Deaths'][-1]
        y=drip_df_deaths.loc["Italy", : ]['Daily'].values[-1] 
        # add country annotation based on last coordinate pair in list
        ax.annotate('Italy',  (x,y))
    else:
        drip_df_deaths.loc[country, : ].plot(x = 'Deaths', y = 'Daily', ax = ax, figsize=(15,10), color='#D8D8D8', alpha = 0.2)

plt. annotate('', xy=(1000, 1000), xytext=(10,10), arrowprops=dict(facecolor='k', shrink=0), 
              horizontalalignment='right', verticalalignment='bottom') 
plt.annotate("Exponential Growth", xy=(10, 10), xytext=(50,100), rotation=45)
    

ax.set_yscale('log')
ax.set_xscale('log')
ax.legend_.remove()
plt.title("New Deaths v Total Deaths" )
plt.xlabel("Total Deaths (log) by Time")
plt.ylabel("New Daily Deaths(log) by Time")
plt.grid(True)
plt.show()


# ### Recoveries

# In[89]:


fig, ax = plt.subplots()


for country in recovered_countries_drip:
    #plot most of the countries in grey
    drip_df_recoveries.loc[country, : ].plot(x = 'Recoveries', y = 'Daily', ax = ax, figsize=(15,10), color='#D8D8D8', alpha = 0.2)
        
    if(country=="Germany"):
        drip_df_recoveries.loc[country, : ].plot(x = 'Recoveries', y = 'Daily', ax = ax, figsize=(15,10), color='#dfff82',marker="o",ls='dashed')

        x=drip_df_recoveries.loc["Germany", : ]['Recoveries'][-1]
        y=drip_df_recoveries.loc["Germany", : ]['Daily'].values[-1] 
        # add country annotation based on last coordinate pair in list
        ax.annotate('Germany',  (x,y))

        
    if(country=="US"):
        drip_df_recoveries.loc[country, : ].plot(x = 'Recoveries', y = 'Daily', ax = ax, figsize=(15,10), color='r',marker="o",ls='dashed')
        x=drip_df_recoveries.loc["US", : ]['Recoveries'][-1]
        y=drip_df_recoveries.loc["US", : ]['Daily'].values[-1] 
        # add country annotation based on last coordinate pair in list
        ax.annotate('US',  (x,y))
        
    if(country=="Canada"):
        drip_df_recoveries.loc[country, : ].plot(x = 'Recoveries', y = 'Daily', ax = ax, figsize=(15,10), color='tab:purple',marker="o",ls='dashed')
        x=drip_df_recoveries.loc["Canada", : ]['Recoveries'][-1]
        y=drip_df_recoveries.loc["Canada", : ]['Daily'].values[-1] 
        # add country annotation based on last coordinate pair in list
        ax.annotate('Canada',  (x,y))
        
    if(country=="Italy"):
        drip_df_deaths.loc[country, : ].plot(x = 'Deaths', y = 'Daily', ax = ax, figsize=(15,10), color='g',marker="o",ls='dashed')
        x=drip_df_deaths.loc["Italy", : ]['Deaths'][-1]
        y=drip_df_deaths.loc["Italy", : ]['Daily'].values[-1] 
        # add country annotation based on last coordinate pair in list
        ax.annotate('Italy',  (x,y))
        
#     if(country=="United Kingdom"):
#         drip_df_recoveries.loc[country, : ].plot(x = 'Recoveries', y = 'Daily', ax = ax, figsize=(15,10), color='tab:blue',marker="o",ls='dashed')
#         x=drip_df_recoveries.loc["United Kingdom", : ]['Recoveries'][-1]
#         y=drip_df_recoveries.loc["United Kingdom", : ]['Daily'].values[-1] 
#         # add country annotation based on last coordinate pair in list
#         ax.annotate('United Kingdom',  (x,y))
        

plt. annotate('', xy=(1000, 1000), xytext=(10,10), arrowprops=dict(facecolor='k', shrink=0), 
              horizontalalignment='right', verticalalignment='bottom') 
plt.annotate("Exponential Growth", xy=(10, 10), xytext=(50,100), rotation=45)
    

ax.set_yscale('log')
ax.set_xscale('log')
ax.legend_.remove()
plt.title("New Recoveries v Total Recoveries" )
plt.xlabel("Total Recoveries (log) by Time")
plt.ylabel("New Daily Recoveries (log) by Time")
plt.grid(True)
plt.show()


# ### End

# Thanks for all your help over the last 6 weeks and accomodating us all! Have a great Summer.
