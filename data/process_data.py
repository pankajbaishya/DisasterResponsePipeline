import sys
# import libraries
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    Load datasets:
        - Load messages_filepath into a dataframe and inspect the first few lines.
        - Load categories_filepath into a dataframe and inspect the first few lines.
    Merge datasets:
        - Merge the messages and categories datasets using the common id.
        - Assign this combined dataset to `df`, which will be cleaned in the following steps.
    '''
    #print(messages_filepath)
    #print(categories_filepath)
    
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    messages.head()
    
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    categories.head()
    
    # merge datasets
    df = pd.merge(messages, categories, how='inner', on='id')
    df.head()
    
    return df


def clean_data(df):
    '''
    Split `categories` into separate category columns:
        - Split the values in the `categories` column on the `;` character so that each value becomes a separate column. 
        - Use the first row of categories dataframe to create column names for the categories data.
        - Rename columns of `categories` with new column names.
    Convert category values to just numbers 0 or 1:
        - Iterate through the category columns in df to keep only the last character of each string (the 1 or 0).     
    Replace `categories` column in `df` with new category columns.
        - Drop the categories column from the df dataframe since it is no longer needed.
        - Concatenate df and categories data frames.
    Further data cleaning:
        - Check and drop duplicates.
        - Check and remove category rows which are not 1 or 0 i.e. converting all the category values to binary.
    '''
    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(';',expand=True)
    categories.head()
    # select the first row of the categories dataframe
    row = categories.loc[0]
    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = list(row.apply(lambda x: x[:-2])) #[x[:-2] for x in row]
    #print(category_colnames)
    # rename the columns of `categories`
    categories.columns = category_colnames
    categories.head()
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
    
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    categories.head()

    # drop the original categories column from `df`
    df.drop(columns='categories', inplace=True)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories], axis=1)
    df.head()
    
    # check number of duplicates
    print(df.duplicated().sum())

    # drop duplicates
    df.drop_duplicates(inplace=True)
    
    # check number of duplicates
    print(df.duplicated().sum())
    
    #Checking category rows which are not 1 or 0 
    print('Categories with non binary values:')
    for col in categories.columns:
        if df[(df[col]!=1) & (df[col]!=0)].shape[0]>0:
            print(col, ":", df[(df[col]!=1) & (df[col]!=0)].shape)
    
    #Removing category rows which are not 1 or 0 i.e. converting all the category values to binary
    for col in categories.columns:
        #print(col, ":", df[(df[col]==1) | (df[col]==0)].shape)
        df = df[(df[col]==1) | (df[col]==0)]
    
    #Checking if there are still category rows which are not 1 or 0 
    print('Categories with non binary values after cleaning:')
    for col in categories.columns:
        if df[(df[col]!=1) & (df[col]!=0)].shape[0]>0:
            print(col, ":", df[(df[col]!=1) & (df[col]!=0)].shape)
        
    return df


def save_data(df, database_filename):
    '''
    Save the cleaned dataset into an sqlite database.
    Also, drop the table if already exists.
    '''
    engine = create_engine('sqlite:///' + database_filename)
    #conn = engine.connect()
    
    #Dropping the table if already exists
    #print("Dropping the table if exists")
    #dropTableStatement = "DROP TABLE IF EXISTS Categories_Table"
    
    #conn.execute(dropTableStatement) 
    #conn.commit()
    #conn.close()
    
    df.to_sql('Categories_Table', engine, index=False, if_exists = 'replace')  
    
    
def main():
    '''
    main function for the ETL script, process_data.py. 
    The script takes the file paths of the two datasets and database, cleans the datasets, and stores the clean data into a SQLite database in the specified database file path.    
    '''
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        print('Shape of cleaned dataframe:', df.shape)
        
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