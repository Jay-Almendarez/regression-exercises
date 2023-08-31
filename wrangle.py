import pandas as pd
import numpy as np
import env
import os

def get_connection(db, user=env.user, host=env.host, password=env.password):
    '''
    get_connection will determine the database we are wanting to access, and load the database along with env stored values like username, password, and host
    to create the url needed for SQL to read the correct database.
    '''
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'

def acquire():
    '''
    get_zillow_data will determine if 'zillow.csv' exists, if it does, it will load the dataframe zillow,
    if it does not exist, it will write the dataframe zillow into a .csv
    '''
    file_name = 'zillow.csv'
    
    
    if os.path.isfile(file_name):
        return pd.read_csv(file_name)
    else:
        query = 'SELECT bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet, taxvaluedollarcnt, yearbuilt, taxamount, fips FROM properties_2017 JOIN propertylandusetype USING (propertylandusetypeid) WHERE propertylandusedesc = "Single Family Residential"'
        connection = get_connection('zillow')
        df = pd.read_sql(query, connection)
        df.to_csv('zillow.csv', index=False)
        return df
    
def prep():
    '''
    prep will take the acquired df and rename columns to be more legible, fill in  null values, change data types, and return the newly cleaned dataframe.
    '''
    df = acquire()
    # I need to rename these columns so they're easier to type out
    df = df.rename(columns={'bedroomcnt': 'beds', 'bathroomcnt': 'baths', 'calculatedfinishedsquarefeet': 'square_feet', 'taxvaluedollarcnt' : 'value', 'yearbuilt': 'year_built', 'taxamount':'tax_amount', 'fips': 'federal_processing_code'})
    # Then replace the values of 0 in beds and bath will null since it's the same thing
    df['beds'] = df['beds'].replace(0.0, np.nan)
    df['baths'] = df['baths'].replace(0.0, np.nan)
    # the federal processing code is presented with no null values and does not need to have a decimal
    df['federal_processing_code'] = df['federal_processing_code'].astype(int)
    # I'm gonna go ahead and drop the null values since collectively they make up about 1/8th of a percent and I don't feel that's significant enough.
    df = df.dropna()
    # I'm also going to convert most of my columns into integers since I don't need to know there's 4.0 bedrooms.
    df['beds'] = df['beds'].astype(int)
    df['year_built'] = df['year_built'].astype(int)
    df['square_feet'] = df['square_feet'].astype(int)
    df['value'] = df['value'].astype(int)
    return df
    
def wrangle_zillow():
    '''
    wrangle_zillow will both perform the data acquisition for our zillow database from SQL and then prep the data by renaming columns, filling in null values, changing data types, and returning the acquired and cleaned data.
    '''
    df = acquire()
    df = prep()
    return df