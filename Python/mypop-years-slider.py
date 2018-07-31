# -*- coding: utf-8 -*-
"""
Created on Thu May 17 13:27:44 2018

@author: geoadom
"""

import numpy as np
import pandas as pd

#population years 2012 - 2021 files
files_list = ["data\pop2012.csv"]
""",
             "data\pop2013.csv",
             "data\pop2014.csv",
             "data\pop2015.csv",
             "data\pop2016.csv",
             "data\pop2017.csv",
             "data\pop2018.csv",
             "data\pop2019.csv",
             "data\pop2020.csv",
             "data\pop2021.csv"]
"""

#prepocessing each file and crate a new files with the follow files:
#Ethnicity,	Age_Range,	F,	M -- range Age & AllAges
#MSOA,	Ethnicity,	Age,	F,	M -- range Age & AllAges

for f in files_list:
    
    df = pd.read_csv(f,usecols =["MSOA","Sex","Age","Ethnicity"])
    output = f.split(".")[0]
     
    
    #create dataframe with count by Ethicity and each age range
    #1) Crate a dataframe with age range => Eth, age-range, F, M
    #2) Create final dataframe with counts => Eth, age-range,total, F,M
    #df_count = pd.DataFrame.from_dict( {'count': df.groupby(["Ethnicity","Age","Sex"]).size()} ).reset_index()
    
    #Count for Ethnicity and age (for each year)
    df_allAge = df.groupby(["Ethnicity","Age","Sex"]).size().unstack().reset_index()
    
    # mapping from age bands to age ranges
    #Create Age Range
    bins = [0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,100]
    labels = ["0-4","5-9","10-14","15-19","20-24","25-29","30-34","35-39","40-44","45-49","50-54","55-59","60-64","65-69","70-74","75-79","80-84","85+"]
    
    df_rangeAge = df[["Ethnicity","Age","Sex"]].copy()
    df_rangeAge['Age_Range'] = pd.cut(df_rangeAge['Age'], 
                         include_lowest=True,
                         right = False,
                         bins=bins,
                         labels=labels
                        )
    
    #Count for Ethnicity and age range
    df_rangeAge = df_rangeAge.groupby(["Ethnicity","Age_Range","Sex"]).size().unstack().reset_index()
    
    #Save population to csv files
    df_allAge.to_csv(output + "_allAges.csv", sep=',', index=False)
    df_rangeAge.to_csv(output + "_rangeAges.csv", sep=',', index=False)
    
    #Either Create file for each MSOA
    #OR create a file from groupby MSOA, EThicity, Age, Sex
    df_MSOA = df.groupby(["MSOA","Ethnicity","Age","Sex"]).size().unstack().reset_index()
    
    df_rangeAge_MSOA = df[["MSOA","Ethnicity","Age","Sex"]].copy()
    df_rangeAge_MSOA['Age_Range'] = pd.cut(df_rangeAge_MSOA['Age'], 
                         include_lowest=True,
                         right = False,
                         bins=bins,
                         labels=labels
                        )
    
    #Count for Ethnicity and age range
    df_rangeAge_MSOA = df_rangeAge_MSOA.groupby(["MSOA","Ethnicity","Age_Range","Sex"]).size().unstack().reset_index()
    #This has null values
    
    #Save population to csv files
    df_MSOA.to_csv(output + "_allAges_MSOA.csv", sep=',', index=False)
    df_rangeAge_MSOA.to_csv(output + "_rangeAges_MSOA.csv", sep=',', index=False)