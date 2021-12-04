# DASC-7008-Project


#### Program Overview

* We divided our program into four parts: Data Crawling, Text Analysis, Model Building, Model Evaluation;
* We firstly crawled objective data and text data from different websites, then we did some data pre-processing, to generate formed data;
* As for the text data, we used some NLP methods to deal with them. Then we get the TF - IDF value of words in the text and the sentiment score of single sentence(word series), which is considered for the prediction models next;
* Then we build different models to predict the trend of the property market in Hong Kong, and we add the sentiment score of our text data as the parameter to optimize the model;
* Finally, we use some statistics methods to evaluate our model, and do some visualization.



#### Program flow


<img width="914" alt="project_flow" src="https://user-images.githubusercontent.com/60178386/144702951-7572e2f7-8025-4514-942a-0d46d5602394.png">


#### Data Crawling

##### Data Searching and extraction

- After scanning many data sources, we finally select five data sources from different website, including forums, websites of property company and some open databases.  
- Our goal is to crawl time, titles of news, contents of news and articles from websites. 、
- libraries
  - pandas: Storing data. For writing dataframe to csv;
  - numpy: Generate arrays and matrix; 
  - requests: Get pages' text from websites
  - bs4: We use beautifulsoup to extract information stored in tags. 

- Variables:
  - headers: For request;
  - soup: All the tags information from websites;
  - urls/urllist: Targeted websites;
  - titles: Titles of news/comments;
  - contents/brief: Contents of news/comments;
  - time/times: Store the time information.

##### Data processing

- We adjust all the time data into the same format, and separate year and month from time.
- Concatenating all the data into a dataframe.
- Sort the data by the date. 

- Librarys: Pandas and numpy (the same as the previous process)
- Variables:
  - forum/midland...: Dataframes storing the data crawled from websites.
  - index: The index of each news.
  - year,month,date: When each news was published. 



#### Text Analysis

* libraries:

  * pandas, os.path, re: basic lib;
  * urllib.request: used to request for our dictionary;
  * jieba: the segmentation lib we used;
  * pycantonese: the segmentation lib we used for Cantonese
  * collections.Counter: calculate the number of each element in the iterable sequence;
  * snownlp.SnowNLP: the sentiment analysis tool we used to get the sentiment score of our data.

  

##### Data Preparation

* We choose news data as social media data: Acquire the property market news data in the past five years as social media data that needs to be processed, and obtain consumer preferences through text processing and analysis.



* varibles: 

  * news - text data we will use;
  * news_content - the text part of our data.

  

##### Segmentation

###### Simplified Chinese

* For Simplified Chinese, the word segmentation effect of jieba's default corpus is good enough.

###### Cantonese

* Cantonese (Hong Kong) currently has fewer NLP tools. The only mainstream ones are jieba's traditional Chinese processing tools and PyCantonese;

* We use two tools to process our text separately to compare effects. After comparison, we try to merge the two effective dictionaries by adjusting the word frequency and use them as the prefix dictionary for our text segmentation.



* varibles:

  * words - the segmentation results of our data;
  * news_cantonese - the Cantonese text we will use;
  * cantonese_content - the text part of our Cantonese text;
  * words_cantonese - the segmentation results of our data(Cantonese);

  * dir, url, path_ - used for crawling dictionaries;
  * corpus - the corpus of PyCantonese;
  * c - counter we used to get the most common words in the corpus;
  * tokens - the tokens we get from our data using PyCantonese;
  * counter_jieba, counter_Pycantonese - the counter we used to get the word frequency of the two dictionaries;
  * de_freq, ge_freq - the frequency of the key word '的' and '嘅';
  * w: the weight of the key word we used to merge the two dictionaries.



##### Text preprocessing

* After word segmentation, remove useless tags and stop words.

###### Part-of-speech tagging

* For sentiment analysis, we mainly focus on words such as adjectives and verbs that can express market trends and consumer preferences;

* So we re-segment the text and perform part-of-speech tagging to filter out the word types we need.

###### Standardization

* Stemming and lemmatization: Not currently supported, no formed corpus.



* varibles:
  * words_new - re-segment result with the POS tagging;
  * words_key - the result after key(emotional) word extraction;
  * key_set - the key POS we forcus;



* functions:
  * text_process: used to remove English, numbers and special symbols, useless tags and stop words



##### Text Analysis

* We use jieba's own TF-IDF keyword extraction, and also remove the stop words (IDF)， then we can get the TF-IDF sorting list, and generating some word-clouds.



* varibles:

  * vectorizer, X: the word frequency/TF-IDF matrix of our words data;
  * data, df: the dataframe we used to store the <word, tf-idf value> pairs;
  * df_new: the sorted dataframe by tf-idf value;



##### Sentiment Analysis

###### SnowNLP

* Take a month as a time slice to count the emotional average index within a single month.

* Use SnowNLP to quickly analyze news sentiment based on news headline content.

* The title is the simplification of the content and the extraction of key information. We can get a basic reference value by performing sentiment analysis on the title (baseline).

###### BosonNLP

* Through the data set that comes with SnowNLP, the official recommendation is that the data calculation accuracy of e-commerce reviews is relatively high.

* So we try to calculate the sentiment value based on the labeled sentiment dictionary, and use your own data to train the model.

* There are a Chinese sentiment dictionary that are widely used at present: BosonNLP sentiment dictionary. We use this industry standard emotional dictionaries to define the emotional value of our data.



* varibles:

  * senti_score: the sentiment score of single document(sentence) of our text;

  * time_slice: the time slice we considered;

  * senti_df: the dataframe we used to store the <time slice, sentiment score> pairs;

  * mean_senti: the average sentiment score in each month;

  * bosonNLP_sentiment: the BonsonNLP dictionary;
  * key. score: the words and the corresponding sentiment score of BonsonNLP dictionary.



* functions:
  * getscore: function used to calculate sentiment score.



#### Model Building

* libraries:

  * pandas, numpy, matplotlib, seaborn: basic lib;
  * sklearn.preprocessing: used to standardize data;
  * sklearn.model_selection: used to split data and adjust hyper-parameters;
  * sklearn.linear_model: used to fit ridge model;
  * regressors.stats: used to show statistic results of ridge model;
  * sklearn.tree: used to fit decision tree regression model;
  * sklearn.ensemble: used to fit random forest regression model.
  * sklearn.svm: used to fit SVR model.
  * xgboost: used to fit xgboost regression model.
  * sklearn.metrics: used to show MSE, MAE, R2_score of each model.
  * warnings: used to ignore warning messages.

##### Data preprocessing

* After reading data, fill missing data with its previous non-missing value. And create two new features as needed. 


* varibles:

  * df: original dataframe collected from Hong Kong Census and Statistics Department;


##### Feature selecting

* Draw heatmap to check correlations between variables.Select target value and variables from processed data into two dataframes. The first dataframe does not include the variable sentiment, but the second does.Then standardize all continuous features. Finally, split the data into 80% training set and 20% test set.


* varibles:

  * df1: the processed dataframe;

  * y: target value(Domestic Price Indices) of model;

  * X: variables of the first model which doesn't include sentiment index;

  * X1: variables of the first model which includes sentiment index;

  * x_train: features of train data set of the model without sentiment index

  * x_test: features' test data set of the model without sentiment index

  * y_train: response value's train data set of the model without sentiment index

  * y_test: features' test data set of the model without sentiment index

  * x_train1: features of train data set of the model with sentiment index

  * x_test1: features' test data set of the model with sentiment index

  * y_train1: response value's train data set of the model with sentiment index

  * y_test1: features' test data set of the model with sentiment index


##### Data training

* Train data by different models such as Ridge, Decision Tree, Random Forest, SVM, and XGBoost respectively. Then adjust hyper-parameters of each model. And show MSE, MAE, R2_score of each model.


* varibles:

  * y_predict: predicted value by model without sentiment index

  * y_predict1: predicted value by model with sentiment index



#### Model Evaluation

##### record_metrics:

* function to record model's metrics into dict for evaluation

* varibles: 

  * models_metrics - dict


  * mse - numerical value


  * mae - numerical value


  * r2_score - numerical value


  * type - 0 for origin model, 1 for model with sentiment


* return: none



##### residuals_compare:

* Function to draw residuals scatter of different models

* varibles: 

  * models - list stored pre trained models


  * models_name - models' name


  * x_train - features' train data set


  * x_test - features' test data set


  * y_train - response value's train data set


  * y_test - features' test data set


* return: none



##### metrics_compare:

* Function for models compare(based on metrics)

* varibles: 

  * models_metrics - dict stored metrics we need


  * models_name - list stored models' name


* return: none



##### overfit_vali:

* Check models with sentiment factor are overfitted or not(based on r2_score)

* varibles: 

  * models - list stored models pre trained


  * models_name - list stored models' name


  * test_r2_score - list stored r2_score for test data set


  * x_train - features' train data set


  * y_train - response's train data set


* return: none



