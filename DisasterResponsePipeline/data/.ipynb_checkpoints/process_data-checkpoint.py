import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    input:
        messages_filepath: The path of messages dataset.
        categories_filepath: The path of categories dataset.
    output:
        messages (dataframe): messages dataframe
        categories (dataframe)= categories dataframe
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    return messages,categories


def clean_data(messages,categories):
    '''
    input:
        messages (dataframe): messages dataframe
        categories (dataframe)= categories dataframe
    output:
        df: merged of messages and categories after cleaning the categories dataframe
    '''
    
    ##Cleaning the categories dataframe
    # Step1: expand the categories into multiple columns
    categories_expanded = categories.categories.str.split(";",expand=True)
    # Step 2 : concat the original dataframe with the one having the categories expanded
    categories_n = pd.concat([categories,categories_expanded],axis=1)
    categories_n.drop(["categories"],axis=1,inplace = True)
    categories = categories_n.copy()
    # Step 3 : rename the columns
    row = categories.iloc[0]
    category_colnames = []
    for cat in row[1:]:
        category_colnames.append(cat.split("-")[0])
    categories.columns = ["id"]+category_colnames
    for column in categories:
        if column=="id":
            continue
        # set each value to be the last character of the string
        #categories[column] = categories[column].str.split("-",expand=True).loc[:,1]
        categories[column] = categories[column].apply(lambda x: x.split("-")[1])

        # convert column from string to numeric
        categories[column] = categories[column].astype("int")
    
    ## Merging the two dataframe
    df = pd.merge(messages, categories, how='left', left_on='id', right_on='id')
    
    ##Removing duplicates values
    df.drop_duplicates(inplace=True)

    return df


def save_data(df, database_filename):
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('messages', engine, index=False)
   

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        messages,categories = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(messages,categories)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()