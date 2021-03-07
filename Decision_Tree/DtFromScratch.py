# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 07:47:51 2018
@author: utsav
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 13:37:03 2018
@author: utsav
"""
#%%
import numpy as np
import pandas as pd
from pprint import pprint
from sklearn import datasets
from pprint import pprint
#%%
df=datasets.load_iris()
#print(df)

x=df.data
y=df.target
y=y.reshape(len(y),1)
data=np.append(x,y,axis=1)

#print(x.shape," ",y.shape," ",data.shape)
data=pd.DataFrame(data)
column_headers=df.feature_names
column_headers.append("species")
data.columns=column_headers
data=data.rename(columns={"species":"label"})
species={0.0:'Iris-setosa', 1.0:'Iris-versicolor', 2.0:'Iris-virginica'}
data.label=[species[item] for item in data.label]
#print(data.head())
#%%
def check_purity(data):
    label_column=data[:,-1]
    unique_classes=np.unique(label_column)
    if(len(unique_classes)==1):
        return True
    else:
        return False

#%%
        
#for numpy data
def classify_data(data):
    label_column=data[:,-1]
    unique_labels,counts=np.unique(label_column,return_counts=True)
    index=np.argmax(counts)
    classification={"counts":counts,"unique_label":unique_labels[index]}
    return classification
#print(classify_data(np.array(data)))

#%%
 #for pandas DataFrame   
def determine_type_of_feature(data):
    threshold=15
    feature_types=[]
    for column in data.columns:
        if (column!="label"):   
            unique_values=data[column].unique()
            example_value=unique_values[0]
            if (len(unique_values)>threshold or isinstance(example_value,str)):
                feature_types.append("continuous")
            else:
                feature_types.append("categorical")
    return feature_types             
print(determine_type_of_feature(data))            
        
#%%
#for numpy data
def get_potential_split(data):
    potential_splits={}
    global FEATURE_TYPES
    FEATURE_TYPES=determine_type_of_feature(pd.DataFrame(data))
    _,n_columns=data.shape
    for col_index in range(n_columns-1):
        values=data[:,col_index]
        unique_vals=np.unique(values)
        
        if(FEATURE_TYPES[col_index]=="continuous"):
            potential_splits[col_index]=[]
            for index in range(len(unique_vals)):
                if(index!=0):
                    current_val=unique_vals[index]
                    previous_val=unique_vals[index-1]
                    mean_val=(current_val+previous_val)/2
                    potential_splits[col_index].append(mean_val)
        #feature is categorical            
        else: 
            potential_splits[col_index]=unique_vals
    
    return potential_splits   
#print(get_potential_split(np.array(data)))    
                    
#%%   
def split_data(data,split_col,split_val):
    split_col_values=data[:,split_col]
    FEATURE_TYPES=determine_type_of_feature(pd.DataFrame(data))
    type_of_feature=FEATURE_TYPES[split_col]
    if(type_of_feature=="continuous"):
        data_above=data[split_col_values>split_val]
        data_below=data[split_col_values<=split_val]
    else:
        data_above=data[split_col_values!=split_val]
        data_below=data[split_col_values==split_val]
    return data_below,data_above
data_below,data_above=split_data(np.array(data),1,2.5)
#print("Data Below..........",data_below,"Data Above................",data_above)  

#%%
def calculate_entropy(data):
    value=data[:,-1]
    unique_value,counts=np.unique(value,return_counts=True)
    total=counts.sum()
    probability=counts/total
    entropy=sum(probability*(-np.log2(probability)))
    return entropy
#print(calculate_entropy(np.array(data)))   
#%%
    
def overall_entropy(data_below,data_above):
    
    total=len(data_above)+len(data_below)
    count_below=len(data_below)/total
    count_above=len(data_above)/total
    overall_entropy=count_above*calculate_entropy(data_above)+count_below*calculate_entropy(data_below)
    return overall_entropy
#%%
    
def determine_gain_ratio(data_below,data_above):
    overall_entropy=overall_entropy(data_below,data_above)
    total=len(data_above)+len(data_below)
    count_below=len(data_below)/total
    count_above=len(data_above)/total
    split_info=(count_below*np.log2(count_below)+count_above*np.log2(count_above))*(-1)
    gain_ratio=overall_entropy/split_info
    return gain_ratio
#%%
    
def determine_best_split(data,potential_splits):
    optimal_entropy=999
    for index in potential_splits:
        for value in potential_splits[index]:
            data_below,data_above=split_data(data,index,value)
            current_entropy=overall_entropy(data_below,data_above)
            if(current_entropy<=optimal_entropy):
                optimal_entropy=current_entropy
                best_split_col=index
                best_split_val=value
    return optimal_entropy,best_split_col,best_split_val 

#print(determine_best_split(np.array(data),get_potential_split(np.array(data))))          
#%%
    
def decision_tree_algorithm(df,counter=0,min_samples=2,max_depth=3):
    if counter==0:
        global FEATURE_TYPES,COLUMN_HEADERS
        FEATURE_TYPES=determine_type_of_feature(df)
        COLUMN_HEADERS=df.columns
        data=df.values
        _,optimal_entropy,feature_name=required_values_for_subtree(data)
        print_statements([data,],feature_name,counter)
#        counter+=1
    else:
        data=df
        
        
    if(check_purity(data) or len(data)<min_samples or max_depth==counter):
        classification=classify_data(data)
        print("Level :",counter)
        max_index=np.argmax(classification["counts"])
        print("Count of "+classification["unique_label"]+"="+str(classification["counts"][max_index]))
        print("Current entropy = 0.0")
        print("Reached the leaf node \n")
        return 
    else:
        
        
        counter+=1
        
        datasets,_,feature_name=required_values_for_subtree(data)
       
        print_statements(datasets,feature_name,counter)
         
         
        
        decision_tree_algorithm(datasets[0],counter,min_samples)
        decision_tree_algorithm(datasets[1],counter,min_samples)
        
        
        
    return    
 
decision_tree_algorithm(data,min_samples=20,max_depth=10)            
#%%    
def print_statements(datasets,feature_name,counter):
     for element in datasets:
         if(check_purity(element)==False):
             print("Level :",counter)
             unique_elements,counts=np.unique(element[:,-1],return_counts=True)
             for index in range(len(unique_elements)):
#                 last_column_values=element[:,-1]
#                 last_column_values=np.array(last_column_values)
                 print("Count of "+unique_elements[index]+"="+ str(counts[index]))
#                 index_elements=element[last_column_values==unique_elements[index]]
             print("Current entropy is = "+str(calculate_entropy(element))) 
             potential_splits=get_potential_split(element)
             optimal_entropy,_,_=determine_best_split(element,potential_splits)
             print("Splitted on feature "+feature_name+" with gain ratio "+str(optimal_entropy) )   
             print()    
#%%
def required_values_for_subtree(data):
    potential_splits=get_potential_split(data)
    optimal_entropy,best_split_col,best_split_val=determine_best_split(data,potential_splits)
    datasets=split_data(data,best_split_col,best_split_val)
    feature_name=COLUMN_HEADERS[best_split_col]
    return datasets,optimal_entropy,feature_name
            
