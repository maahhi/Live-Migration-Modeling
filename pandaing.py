import pandas as pd

profiling = ['working_set_pages','non_working_set_pages','zero_pages','VM_size','pdr','mwpp','wse','nwse']
df = pd.read_csv('data.csv')
pdfu = df[['total_pages','working_set_pages','non_working_set_pages','zero_pages','VM_size','pdr','mwpp','wse','nwse','technique']]
pdfu.drop_duplicates(keep='first')
pdfu.groupby(pdfu.columns.to_list(),as_index= False).size()

# to make data set from pandas data frame use :
df_p = df[profiling]
X = df_p.values

from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize, StandardScaler
import numpy as np

df_cluster = df.copy()
for p in profiling:
    x = df[p].values
    kmeans = KMeans(n_clusters=3, random_state=0).fit(x.reshape(-1,1))
    df_cluster[p] = kmeans.labels_
pdfu = df_cluster[['total_pages','working_set_pages','non_working_set_pages','zero_pages','VM_size','pdr','mwpp','wse','nwse','technique']]


scaler = StandardScaler()
X_ = []
X = df['working_set_pages'].values
X=X.reshape(-1,1)
X_scaled = scaler.fit_transform(X)
X_normalized = normalize(X_scaled)


tech_uniq = df.technique.unique()
df_per_tec = ['' for i in range(len(tech_uniq))]
for i in range(len(tech_uniq)):
    df_per_tec[i] = df.loc[df['technique'] == tech_uniq[i]].copy()

import scipy
df_post = df_per_tec[1]
df1 = df_post.iloc[[0]]
for i in range(len(tech_uniq)):
    df2 = df_per_tec[i]
    ary = scipy.spatial.distance.cdist(df2[profiling], df1[profiling], metric='euclidean')
    match = df2[ary==ary.min()]
    print(match)

#Table 1: The 20 input features of the ML model

inp_fit = ['VM_size','pdr','working_set_pages','wse','nwse','mwpp', #vm fitures
           'vm_perf_instructions',['max-bandwidth'],['SRC_cpu_system','SRC_cpu_user'] ,'SRC_net_manage_ifutil',#Source host
            ['SRC_cpu_system','SRC_cpu_user','DST_cpu_system','DST_cpu_user'],['SRC_net_manage_ifutil','DST_net_manage_ifutil'],#Src + dest host
            'RPTR', 'THR_benefit', 'DTC_benefit', 'DLTC_benefit', 'POST_benefit', 'ewss', 'enwss'#composed (nwss not found)
           ]