
import pandas as pd

# Replace 'tweet_dataset.csv' with the actual path to your CSV file
file_path = 'tweet_dataset.csv'

# Try reading the file with different encodings
# encodings_to_try = ['utf-8', 'latin1', 'ISO-8859-1', 'cp1252']
# for encoding in encodings_to_try:
try:
    df = pd.read_csv(file_path)
   
except UnicodeDecodeError:
    print(f"Failed to read with encoding {encoding}")

# Display the first few rows of the DataFrame
df.head()


import pandas as pd 

setiment_analysis = pd.read_csv('tweet_dataset.csv',encoding='unicode_escape')

setiment_analysis.head()

setiment_analysis.head()

setiment_analysis.info()

setiment_analysis.isna().sum()

input_data = setiment_analysis['selected_text']
output_data = setiment_analysis['sentiment']
print(input_data.shape)
print(output_data.shape)

print(output_data)
from sklearn.model_selection import train_test_split
input_data_train,input_data_test,output_data_train,output_data_test = train_test_split(input_data,output_data,test_size = 0.2)


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
model = make_pipeline(TfidfVectorizer(), MultinomialNB())

model.fit(input_data_train,output_data_train)

predicted_sentiment = model.predict(input_data_test)

pd.DataFrame(confusion_matrix(output_data_test,predicted_sentiment), columns = ['predicted nagtive','predicted neutral','predicted positive'], index=['actual negative','actual neutral','actual positive'])

from sklearn.metrics import accuracy_score
accuracy_info = accuracy_score(output_data_test,predicted_sentiment)


def predicted_sentiment(txt ,train = input_data_train ,model = model):
    pred = model.predict([txt])
    return pred


predicted_sentiment('i am using  python')




