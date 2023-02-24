#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import pandas as pd 
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
matplotlib.rcParams["figure.figsize"]=(20,10)


# In[6]:


df1=pd.read_csv("Bengaluru_House_Data.csv")


# In[7]:


df1.head()


# In[8]:


df1.shape


# In[9]:


df1.groupby('area_type')['area_type'].agg('count')


# In[10]:


df2=df1.drop(['area_type','availability','society','balcony'],axis='columns')
df2.head()


# In[11]:


df2.isnull().sum()


# In[12]:


df3=df2.dropna()
df3.isnull().sum()


# In[13]:


df3.shape


# In[14]:


df3['size'].unique()


# In[15]:


you = df3["size"].str.split(" ", n = 1, expand = True)
df3['bhk']=you[0].astype(int)


# In[16]:


df3.head(5)


# In[17]:


df3['bhk'].unique()


# In[18]:


df3[df3.bhk>20]


# In[18]:


df3.total_sqft.unique()


# In[19]:


def is_float(x):
    try:
        float(x)
    except:
        return False
    return True


# In[20]:


df3[~df3['total_sqft'].apply(is_float)].head(10)


# In[21]:


def convert_sqft_to_num(x):
    tokens=x.split("-")
    if len(tokens)==2:
        return((float(tokens[0])+(float(tokens[1]))))/2
    try:
        return float(x)
    except:
   
        return None
                
    


# In[22]:


df4=df3.copy()
df4['total_sqft']=df4['total_sqft'].apply(convert_sqft_to_num)
df4.head(10)


# In[23]:


df5=df4.copy()
df5['price_per_sqft']=df5['price']*100000/df5['total_sqft']
df5.head()


# In[24]:


len(df5.location.unique())


# In[25]:


df5.location=df5.location.apply(lambda x: x.strip())
location_stats=df5.groupby('location')['location'].agg('count').sort_values(ascending=False)
location_stats


# In[26]:


len(df5.location.unique())


# In[27]:


len(location_stats[location_stats<=10])


# In[28]:


location_stats_less_than_10=location_stats[location_stats<=10]


# In[29]:


df5.location=df5.location.apply(lambda x: 'other' if x in location_stats_less_than_10 else x )


# In[30]:


df5.head(10)


# In[31]:


len(df5.location.unique())


# In[32]:


df5[df5.total_sqft/df5.bhk<300].head()


# In[33]:


df5.shape


# In[34]:


df6=df5[~(df5.total_sqft/df5.bhk<300)]


# In[35]:


df6.shape


# In[36]:


df6.price_per_sqft.describe()


# In[37]:


def remove_pps_outliers(df):
    df_out=pd.DataFrame()
    for key,subdf in df.groupby('location'):
        m=np.mean(subdf.price_per_sqft)
        st=np.std(subdf.price_per_sqft)
        reduced_df=subdf[(subdf.price_per_sqft>(m-st))&(subdf.price_per_sqft<=(m+st))]
        df_out=pd.concat([df_out,reduced_df],ignore_index=True)
    return df_out
df7=remove_pps_outliers(df6)
df7.shape


# In[40]:


def plot_scatter_chart(df,location):
    bhk2 = df[(df.location==location) & (df.bhk==2)]
    bhk3 = df[(df.location==location) & (df.bhk==3)]
    matplotlib.rcParams['figure.figsize'] = (15,10)
    plt.scatter(bhk2.total_sqft,bhk2.price,color='blue',label='2 BHK', s=50)
    plt.scatter(bhk3.total_sqft,bhk3.price,marker='+', color='green',label='3 BHK', s=50)
    plt.xlabel("Total Square Feet Area")
    plt.ylabel("Price (Lakh Indian Rupees)")
    plt.title(location)
    plt.legend()
    
plot_scatter_chart(df7,"Rajaji Nagar")


# In[41]:


plot_scatter_chart(df7,"Hebbal")


# In[42]:


def remove_bhk_outliers(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby('location'):
        bhk_stats = {}
        for bhk, bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk] = {
                'mean': np.mean(bhk_df.price_per_sqft),
                'std': np.std(bhk_df.price_per_sqft),
                'count': bhk_df.shape[0]
            }
        for bhk, bhk_df in location_df.groupby('bhk'):
            stats = bhk_stats.get(bhk-1)
            if stats and stats['count']>5:
                exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)
    return df.drop(exclude_indices,axis='index')
df8 = remove_bhk_outliers(df7)
# df8 = df7.copy()
df8.shape


# In[43]:


plot_scatter_chart(df8,"Rajaji Nagar")


# In[44]:


import matplotlib
matplotlib.rcParams["figure.figsize"] = (20,10)
plt.hist(df8.price_per_sqft,rwidth=0.8)
plt.xlabel("Price Per Square Feet")
plt.ylabel("Count")


# In[45]:


df8.bath.unique()


# In[46]:


plt.hist(df8.bath,rwidth=0.8)
plt.xlabel("Number of bathrooms")
plt.ylabel("Count")


# In[47]:


df8[df8.bath>10]


# In[48]:


df8[df8.bath>df8.bhk+2]


# In[49]:


df9 = df8[df8.bath<df8.bhk+2]
df9.shape


# In[50]:


df9.head(2)


# In[51]:


df10 = df9.drop(['size','price_per_sqft'],axis='columns')
df10.head(3)


# In[52]:


dummies = pd.get_dummies(df10.location)
dummies.head(3)


# In[53]:


df11 = pd.concat([df10,dummies.drop('other',axis='columns')],axis='columns')
df11.head()



# In[54]:


df12 = df11.drop('location',axis='columns')
df12.head(2)


# In[55]:


df12.shape


# In[56]:


X = df12.drop(['price'],axis='columns')
X.head(3)


# In[57]:


X.shape


# In[58]:


y = df12.price
y.head(3)


# In[59]:


len(y)


# In[60]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=10)


# In[3]:


import time

def fib_recursive(n):
    if n <= 2:
        return 1
    
    else:
        return fib_recursive(n-1) + fib_recursive(n-2)

def fib_array(n):
    arr = [0,1]
    for i in range(n-1):
        s = arr[-1] + arr[-2]
        arr.append(s)
    return arr[-1]

def fib_itret(n):
    if n <= 1:
        return n
    prev_prev, prev = 0, 1
    for i in range(2, n+1):
        current = prev_prev + prev
        prev_prev = prev
        prev = current
    return current

def fib_dynamic_memoziation(n, memo={}):
    if n <= 1:
        return n
    if n in memo:
        return memo[n]
    memo[n] = fib_dynamic_memoziation(n-1, memo) + fib_dynamic_memoziation(n-2, memo)
    return memo[n]


def fib_greedy(n):
    if n <= 0:
        return []
    elif n == 1:
        return [0]
    elif n == 2:
        return [0, 1]
    else:
        fib = [0, 1]
        for i in range(2, n+1):
            fib.append(fib[i - 1] + fib[i - 2])
        return fib[-1]



if __name__ == '__main__':
    start = time.time()
    print(fib_recursive(25))
    print("execution time for Recursive method: " + str(time.time() - start))
    start2 = time.time()
    print(fib_array(25))
    print("execution time for array method: " + str(time.time() - start2))
    start3 = time.time()
    print(fib_itret(25))
    print("Execution time for iterative method: " + str(time.time() -start3))
    start4 = time.time()
    print(fib_dynamic_memoziation(25, memo={}))
    print("Execution time for dynamic memoziation method: " + str(time.time() -start4))
    start5 = time.time()
    print(fib_greedy(25))
    print("Execution time for dynamic memoziation method: " + str(time.time() -start5))


# In[1]:


# Python3 program for Bellman-Ford's single source
# shortest path algorithm.

# Class to represent a graph

class Graph:

  def init(self, vertices):
    self.V = vertices # No. of vertices
    self.graph = []

  # function to add an edge to graph
  def addEdge(self, u, v, w):
    self.graph.append([u, v, w])

  # utility function used to print the solution
  def printArr(self, dist):
    print("Vertex Distance from Source")
    for i in range(self.V):
      print("{0}\t\t{1}".format(i, dist[i]))

  # The main function that finds shortest distances from src to
  # all other vertices using Bellman-Ford algorithm. The function
  # also detects negative weight cycle
  def BellmanFord(self, src):

    # Step 1: Initialize distances from src to all other vertices
    # as INFINITE
    dist = [float("Inf")] * self.V
    dist[src] = 0

    # Step 2: Relax all edges |V| - 1 times. A simple shortest
    # path from src to any other vertex can have at-most |V| - 1
    # edges
    for _ in range(self.V - 1):
      # Update dist value and parent index of the adjacent vertices of
      # the picked vertex. Consider only those vertices which are still in
      # queue
      for u, v, w in self.graph:
        if dist[u] != float("Inf") and dist[u] + w < dist[v]:
          dist[v] = dist[u] + w

    # Step 3: check for negative-weight cycles. The above step
    # guarantees shortest distances if graph doesn't contain
    # negative weight cycle. If we get a shorter path, then there
    # is a cycle.

    for u, v, w in self.graph:
      if dist[u] != float("Inf") and dist[u] + w < dist[v]:
        print("Graph contains negative weight cycle")
        return

    # print all distance
    self.printArr(dist)


# Driver's code
if __name__ == '__main__':
  g = Graph()
  g.addEdge(0, 1, -1)
  g.addEdge(0, 2, 4)
  g.addEdge(1, 2, 3)
  g.addEdge(1, 3, 2)
  g.addEdge(1, 4, 2)
  g.addEdge(3, 2, 5)
  g.addEdge(3, 1, 1)
  g.addEdge(4, 3, -3)

  # function call
  g.BellmanFord(0)


# In[ ]:




