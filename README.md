# Disaster Response Pipeline Project
---
This is a Data Science project for the Udacity's Data Scientist Nanodegree. In this project, we build a complete NLP pipeline, composed by an ETL and ML pipelines, along with a model deployment into a web app.

Our model was built to classify emergency messages into well-defined categories so these messages may be sent to an appropriate disaster relief agency. The messages were extracted from real disaster data, provided by [Figure Eight](https://appen.com/). The app also provides some visuals and insights on the dataset.

![Web app - Message classification.](https://raw.githubusercontent.com/lbaiao/udacity-project2/main/pictures/pic1.png)

![Web app - Visuals.](https://raw.githubusercontent.com/lbaiao/udacity-project2/main/pictures/pic2.png)

# Libraries
---
* flask 2.0.1
* matplotlib 3.4.2
* numpy 1.20.3
* nltk 3.6.2
* plotly 5.1.0
* scikit-learn 0.24.2
* seaborn 0.11.1
* sqlalchemy 1.4.22
* stopwords 1.0.0

# Files
---
asdasd


# Instructions:
---
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
