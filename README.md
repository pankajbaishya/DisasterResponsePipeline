### Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Running the Scripts and Results](#results)
5. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>

There should be Anaconda distribution of Python.  The code should run with no issues using Python versions 3.*. In addition following libraries are used in the code.
1. pandas
2. sqlalchemy
3. numpy
4. sklearn
5. nltk
6. pickle
7. json
8. plotly
9. flask
10. sys

## Project Motivation<a name="motivation"></a>

Purpose of this project is to apply data engineering skills to analyze disaster data from Figure Eight and build a model for an API that classifies disaster messages.  Project data set contains real messages that were sent during disaster events. As part of this, created a machine learning pipeline to categorize these events so that you can send the messages to an appropriate disaster relief agency.  This project also include a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data.


## File Descriptions <a name="files"></a>

There are two jupyter notebook files where the initial code for ETL and ML pipeline were created. Later this code were put together in modular format in python (.py) files. The files are self-exploratory with comments for each sections.  Below are the list of files under different folders.
### Jupyter Notebooks
1. ETL Pipeline Preparation.ipynb - This file consists the first part of data pipeline i.e. the Extract, Transform, and Load process. Here, it is reading the dataset, cleaning the data, and then store it in a SQLite database. Data cleaning was done with pandas. To load the data into an SQLite database, it uses the pandas dataframe .to_sql() method, which we can use with an SQLAlchemy engine.
2. ML Pipeline Preparation.ipynb - This file consists of the machine learning portion, which splits the data into a training set and a test set. Then, it creates a machine learning pipeline that uses NLTK, as well as scikit-learn's Pipeline and GridSearchCV to output a final model that uses the message column to predict classifications for 36 categories (multi-output classification). Finally, it exports the model to a pickle file. 

### data
1. process_data.py - Code from "ETL Pipeline Preparation.ipynb" is put in a modular format under this file.
2. disaster_messages.csv - Consists of the real messages that were sent during disaster events.
3. disaster_categories.csv - Consists of list of categories which the machine learning pipeline needs to categorize the events.
4. DisasterResponse.db - This is the output database which was created by the code in process_data.py.

### models
1. train_classifier.py - Code from "ML Pipeline Preparation.ipynb" is put in a modular format under this file.
Note: Unable to upload the saved model file (classifier.pkl) due to file size limitation of github.

### app
1. run.py - This file is used to display the results in a Flask web app. Code framework was provided as part of the project from Udacity. 
2. templates\master.html - master html file used in the Flask app.
3. templates\go.html



## Running the Scripts and Results<a name="results"></a>

Follow below instructions to run the scripts.
1. Run the following commands in the project's root directory to set up your database and model.
    - To run ETL pipeline that cleans data and stores in database
        'python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db'
    - To run ML pipeline that trains classifier and saves
        'python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl'

2. Run the following command in the app's directory to run your web app.
    'python run.py'

3. Go to http://0.0.0.0:3001/ or http://127.0.0.1:3001/ (in case the first one do not work).

Once the model is ready, saved and app is running, the Flask app html page is able to categorize any message to the corresponding category.

### Web App Screenshots
![Visualization of Data](https://github.com/pankajbaishya/DisasterResponsePipeline/blob/main/WebApp_Screenshot1.png)

![Message Categorization](https://github.com/pankajbaishya/DisasterResponsePipeline/blob/main/WebApp_Screenshot2.png)

## Licensing, Authors, Acknowledgements<a name="licensing"></a>

This app is created as part of the Udacity Data Scientist Nanodegree. I would like to give credit to udacity for the basic framework of the solution. The data used was originally sourced by Udacity from Figure Eight.