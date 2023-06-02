#!/usr/bin/env python
# coding: utf-8

# <div class="alert alert-block alert-success">
#     
# # Berkeley- PCMLAI: Module 6- Song Recommendation
#     
#  👨‍🏫 **Vikesh K**      
#  📅 **[24-Apr-2023]** ⏰ **8pm UTC**   
#   
# </div>

# ## Import Modules

# In[1]:


import pandas as pd
from IPython.display import HTML
import pickle
import os

# pandas options
pd.set_option('display.max_colwidth', None)


# ## Data 

# **Download the data from [Gdrive Link](https://drive.google.com/file/d/1TelyB-yXmvsyqUa1byONUA_kc3HentWk/view?usp=sharing)**
# 
# Save it in your working directory printed above

# In[2]:


os.getcwd()


# In[3]:


df = pd.read_csv('/home/administrator/AI_ML_EXERCISES/UC_BERKELEY_AI_ML/datasets/spotify_data.csv')


# ## Data Exploration

# In[62]:


df.shape


# In[123]:


df.query("name == 'Mambo Italiano' & mode == 0 & year == 2013")


# ## Data Cleaning

# In[64]:


df['name'].nunique()


# In[65]:


df.columns


# In[66]:


df.describe()


# In[67]:


# df.select_dtypes(['int', 'float']).drop(columns = ['year', 'explicit', 'mode'])


# In[6]:


numerical_df = df.select_dtypes(['int', 'float']).drop(columns = ['year', 'explicit', 'mode'])


# In[ ]:





# In[7]:


numerical_df.head()


# In[8]:


# numerical_df.plot(kind = 'kde', subplots = True, figsize = (22, 15));


# ## Standardizing the values

# In[9]:


from sklearn.preprocessing import StandardScaler


# In[10]:


scaler = StandardScaler()

# scaling the dataset
numerical_df_scaled = scaler.fit_transform(numerical_df)


# In[11]:


type(numerical_df_scaled)


# In[12]:


numerical_df_scaled = pd.DataFrame(numerical_df_scaled, columns = numerical_df.columns)


# In[13]:


numerical_df_scaled.iloc[169906, :]


# In[14]:


# numerical_df_scaled.plot(kind = 'kde', subplots = True, figsize = (22, 15));


# ## Fit the [`Nearest Neigbors`](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html#sklearn.neighbors.NearestNeighbors) model
# 
# **Unsupervised learner for implementing neighbor searches.**

# In[15]:


# calling the model 
from sklearn.neighbors import NearestNeighbors


# In[16]:


# initiating the model 
nn = NearestNeighbors(n_neighbors=5)


# In[17]:


# fit the model on data. This is unsupervised in nature
nn.fit(numerical_df_scaled)


# ### Testing the model to generate predictions

# In[18]:


select_song = df.query("name == 'How Would I Know' ").select_dtypes(['int', 'float']).drop(columns = ['year', 'explicit', 'mode'])




# In[37]:


my_song = df.query("name == 'Mambo Italiano' & mode == 0 & year == 2013")



# In[38]:


my_song


# In[21]:


select_song


# In[22]:


select_song_scaled = scaler.transform(my_song)


# In[23]:


select_song_scaled


# In[24]:


select_song_scaled = pd.DataFrame(select_song_scaled, columns = select_song.columns)


# In[25]:


select_song_scaled


# **Get the recommendations**

# In[26]:


nn.kneighbors(select_song_scaled,  return_distance=True)


# **Get only the row positions**

# In[27]:


# get only the array values
# it returns the index number of the recommended songs

nn.kneighbors(select_song_scaled,  return_distance=True)[1:]


# In[28]:


nn.kneighbors(select_song_scaled,  return_distance=False)


# **We will ignore the first recommendation as that is the song itself**

# In[29]:


recco_list = nn.kneighbors(select_song_scaled,  return_distance=False)[:,1:].tolist()[0]# Skip the first value 


# In[30]:


recco_list


# **Recommended Song Names**

# In[31]:


df.loc[recco_list,['name', 'id']]


# **Column Addition**

# In[32]:


df['song_url'] = 'https://open.spotify.com/track/' + df['id']


# In[33]:


neighbors = df.loc[recco_list,['name', 'song_url']]


# In[34]:


neighbors


# **We need to render the links dynamic**

# In[35]:


HTML(neighbors.to_html(render_links=True, escape=False))


# **Export the model for Streamlit**

# In[ ]:


filename = 'nn_model_prediction.sav' # name of the file 
pickle.dump(nn, open(filename, 'wb')) # exporting the file

print("file exported")

