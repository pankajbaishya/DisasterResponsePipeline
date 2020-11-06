#ETL pipeline that cleans data and stores in database
import sys
# import libraries
import pandas as pd
from sqlalchemy import create_engine

# ### 1. load datasets.
    # - Load `messages.csv` into a dataframe and inspect the first few lines.
    # - Load `categories.csv` into a dataframe and inspect the first few lines.
# ### 2. Merge datasets
    # - Merge the messages and categories datasets using the common id
    # - Assign this combined dataset to `df`, which will be cleaned in the following steps

def load_data(messages_filepath, categories_filepath):
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

# ### 3. Split `categories` into separate category columns.
    # - Split the values in the `categories` column on the `;` character so that each value becomes a separate column. 
    # - Use the first row of categories dataframe to create column names for the categories data.
# - Rename columns of `categories` with new column names.
# ### 4. Convert category values to just numbers 0 or 1.
    # - Iterate through the category columns in df to keep only the last character of each string (the 1 or 0).     
# ### 5. Replace `categories` column in `df` with new category columns.
    # - Drop the categories column from the df dataframe since it is no longer needed.
    # - Concatenate df and categories data frames.
def clean_data(df):
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
    
    return df

# ### 7. Save the clean dataset into an sqlite database.
def save_data(df, database_filename):
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('Categories_Table', engine, index=False)  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
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