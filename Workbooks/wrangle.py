import openpyxl
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

import sklearn.preprocessing
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.datasets.samples_generator import make_blobs

#-----------------------------------------------------------------------------

def acquire_heart():
    '''Takes the heart disease CSV
    turns csv into a readable pandas dataframe'''
    # Get the csv
    df = pd.read_csv('heart.csv')
    # return pandas df
    return df

#-----------------------------------------------------------------------------
# Official clean functions
def clean_heart_data(df):
    '''Takes in pandas dataframe
    renames columns using renaming function
    creates new columns using create features function
    bin large features using binning function
    returnn cleaned pandas dataframe'''
    # rename
    df = rename_features(df)
    # create
    df = create_features(df)
    # bin
    df = bin_large_data(df)
    # return cleaned dataframe
    return df
#-----------------------------------------------------------------------------
# sub clean funcitons that feed into official cleaning function

def rename_features(df):
    '''Takes pandas dataframe and renames columns'''
    # rename all columns
    df.columns = ['age','is_male', 'chest_pain', 'resting_bp', 'cholestoral', 'blood_sugar_above_120', 
              'resting_electocardio', 'max_heart_rate', 'exercise_angina', 'rest_angina', 'slope',
              'count_major_vessels', 'defect_type', 'has_heart_disease']
    return df

def create_features(df):
    '''Takes in df
    creates new features for ratios 
    between chest pain, age, heart rate, and cholestoral'''
    # chest pain and age ratio
    df['chest_age_ratio'] = df.chest_pain/df.age
    # age and cholestoral ratio
    df['age_chol_ratio'] = df.age/df.cholestoral
    # age and heart rate ratio
    df['age_heart_ratio'] = df.age/df.max_heart_rate
    # chest pain and cholestoral ratio
    df['chest_chol_ratio'] = df.chest_pain/df.cholestoral
    # cholestoral and heart rate ratio
    df['chol_heart_ratio'] = df.cholestoral/df.max_heart_rate
    # chest pain and heart rate ratio
    df['chest_heart_ratio'] = df.chest_pain/df.max_heart_rate
    # return pandas datafram
    return df

def bin_large_data(df):
    '''Takes in df and bins larger datas
    set labels for each bin
    returns pandas dataframe'''
    # create age bins
    df['age_groups'] = pd.cut(df.age, 
                            bins = [29,35,40,45,50,55,60,65,70,75,80],
                            labels = ['Below 35', '35-40', '40-45', "45-50", 
                                      "50-55", '55-60', '60-65', '65-70', 
                                      '70-75', '75-80'])
    # bin cholestoral
    df['levels_of_chol'] = pd.cut(df.cholestoral, 
                            bins = [0,200,240,800],
                            labels = ['Desireable', 'Boarderline High', 'High'])
    # bin heart rate
    df['heart_rate_levels'] = pd.cut(df.max_heart_rate, 
                            bins = [0,120,130,140,180,250],
                            labels = ['Normal', 'Elevated', 'High Stage 1', 
                                      'High Stage 2', 'Hypertensive Crisis'])
    # return pandas data frame
    return df


#-----------------------------------------------------------------------------

# Split the Data into Tain, Test, and Validate.

def split_heart_disease(df):
    '''This fuction takes in a df 
    splits into train, test, validate
    return: three pandas dataframes: train, validate, test
    '''
    # split the focused zillow data
    train_validate, test = train_test_split(df, test_size=.2, random_state=1234)
    train, validate = train_test_split(train_validate, test_size=.3, 
                                       random_state=1234)
    return train, validate, test

# Split the data into X_train, y_train, X_vlaidate, y_validate, X_train, and y_train

def split_train_validate_test(train, validate, test):
    ''' This function takes in train, validate and test
    splits them into X and y versions
    returns X_train, X_validate, X_test, y_train, y_validate, y_test'''
    X_train = train.drop(columns = ['has_heart_disease'])
    y_train = pd.DataFrame(train.has_heart_disease)
    X_validate = validate.drop(columns=['has_heart_disease'])
    y_validate = pd.DataFrame(validate.has_heart_disease)
    X_test = test.drop(columns=['has_heart_disease'])
    y_test = pd.DataFrame(test.has_heart_disease)
    return X_train, X_validate, X_test, y_train, y_validate, y_test

#-----------------------------------------------------------------------------

# Scale the Data

def scale_my_data(train, validate, test):
    scale_columns = ['Global_Sales', 'Year']
    scaler = MinMaxScaler()
    scaler.fit(train[scale_columns])

    train_scaled = scaler.transform(train[scale_columns])
    validate_scaled = scaler.transform(validate[scale_columns])
    test_scaled = scaler.transform(test[scale_columns])
    #turn into dataframe
    train_scaled = pd.DataFrame(train_scaled)
    validate_scaled = pd.DataFrame(validate_scaled)
    test_scaled = pd.DataFrame(test_scaled)
    
    return train_scaled, validate_scaled, test_scaled

#-----------------------------------------------------------------------------

# Focused Data

def focused_game_sales(df):
    '''
    takes in train
    sets sepecific features to focus on
    returns a focused data frame in a pandas dataframe
    '''
    # choose features to focus on
    features = [
    'Year','level_of_success',
    'Nintendo','Playstation','Xbox','Computer','Sega','Other',
    'Action_Adventure', 'Simulation', 'Sports', 'Misc', 'Role_Playing', 'Shooter',
    'Strategy'] 
    # the target is level of success
    # return a df based only on these features
    df2 = df[features]
    return df2

def split_focused_game_sales(df2):
    '''This fuction takes in a df 
    splits into train, test, validate
    return: three pandas dataframes: train, validate, test
    '''
    # split the focused zillow data
    train_validate2, test2 = train_test_split(df2, test_size=.2, random_state=1234)
    train2, validate2 = train_test_split(train_validate2, test_size=.3, 
                                       random_state=1234)
    return train2, validate2, test2

def split_train2_validate2_test2(train2, validate2, test2):
    ''' This function takes in train, validate and test
    splits them into X and y versions
    returns X_train, X_validate, X_test, y_train, y_validate, y_test'''
    X_train = train.drop(columns = ['level_of_success'])
    y_train = pd.DataFrame(train.level_of_success)
    X_validate = validate.drop(columns=['level_of_success'])
    y_validate = pd.DataFrame(validate.level_of_success)
    X_test = test.drop(columns=['level_of_success'])
    y_test = pd.DataFrame(test.level_of_success)
    return X_train2, X_validate2, X_test2, y_train2, y_validate2, y_test2