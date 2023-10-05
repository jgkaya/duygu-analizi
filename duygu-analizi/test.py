import sklearn
import numpy as np
import pandas as pd

# training data
train = pd.read_csv("train.csv")
# test data
test = pd.read_csv("test.csv")
train.head()

import seaborn as sns
import re
import matplotlib.pyplot as plt
import string
import nltk
import warnings
from wordcloud import WordCloud
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings('ignore')


#test için dataset örneğimiz daha sonra özelleştirilecek
def clean_data(data):
    # datasetteki kirlilik yaratan bahsetmeleri temizlemek için
    def remove_pattern(input_txt, pattern):
        r = re.findall(pattern, input_txt)
        for word in r:
            input_txt = re.sub(word, "", input_txt)
        return input_txt

    data['temiz_tweet'] = np.vectorize(remove_pattern)(data['tweet'], "@[\w]*")
    data.head()

    # datasetteki kirlilik yaratan özel karakterleri ve sayıları temizlemek için
    data['temiz_tweet'] = data['temiz_tweet'].str.replace("[^a-zA-Z#]", " ")
    data.head()

    # tweetteki kelimeleri ayırıyoruz
    kelimeler = data['temiz_tweet'].apply(lambda x: x.split())
    kelimeler.head()

clean_data(train)
clean_data(test)
# remove special characters using the regular expression library


# clean the test data and append the cleaned tweets to the test data
test_tweet =test['temiz_tweet']
test_tweet = pd.DataFrame(test_tweet)
# append cleaned tweets to the training data
test["clean_tweet"] = test_tweet

from sklearn.model_selection import train_test_split

# extract the labels from the train data
y = train.label.values

# use 70% for the training and 30% for the test
x_train, x_test, y_train, y_test = train_test_split(train.temiz_tweet.values, y,
                                                    stratify=y,
                                                    random_state=1,
                                                    test_size=0.3, shuffle=True)
from sklearn.feature_extraction.text import CountVectorizer



# check the result

# vectorize tweets for model building
vectorizer = CountVectorizer(binary=True, stop_words='english')

# learn a vocabulary dictionary of all tokens in the raw documents
vectorizer.fit(list(x_train) + list(x_test))

# transform documents to document-term matrix
x_train_vec = vectorizer.transform(x_train)
x_test_vec = vectorizer.transform(x_test)


from sklearn import svm
# classify using support vector classifier
svm = svm.SVC(kernel = 'linear', probability=True)

# fit the SVC model based on the given training data
prob = svm.fit(x_train_vec, y_train).predict_proba(x_test_vec)

# perform classification and prediction on samples in x_test
y_pred_svm = svm.predict(x_test_vec)

from sklearn.metrics import accuracy_score
print("SVC için doğruluk skoru: ", accuracy_score(y_test, y_pred_svm) * 100, '%')

from sklearn.naive_bayes import MultinomialNB

classifier = MultinomialNB()
classifier.fit(x_train_vec, y_train)
predictions = classifier.predict(x_test_vec)
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Multinomial Naive Bayes için doğruluk skoru: {:.2}".format(metrics.accuracy_score(y_test, predictions)))

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, random_state=0)
rf.fit(x_train_vec, y_train) # Eğitim Verisi ile eğitim gerçekleşiyor
pred_knn = rf.predict(x_test_vec)
print("Random Forest için doğruluk skoru: ",accuracy_score(y_test, pred_knn)* 100, '%')

from sklearn.metrics import confusion_matrix
#matriksi görselleştirmek için (hata matriksi)
cf_matrix=confusion_matrix(y_test, y_pred_svm)
sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True,fmt='.2%', cmap='Blues')
plt.show()

#train, test tweet sayıları grafiği
length_train = train['tweet'].str.len()
length_test = test['tweet'].str.len()
plt.hist(length_train, bins=20, label="train_tweets")
plt.hist(length_test, bins=20, label="test_tweets")
plt.legend()
plt.show()

#pozitif kelimeler
normal_words = ' '.join([word for word in train['temiz_tweet'][train['label'] == 0]])
wordcloud = WordCloud(width = 800, height = 500, max_font_size = 110).generate(normal_words)
plt.figure(figsize= (12,8))
plt.imshow(wordcloud, interpolation = 'bilinear')
plt.axis('off')
plt.show()

#negatif kelimeler
negative_words = ' '.join([word for word in train['temiz_tweet'][train['label'] == 1]])
wordcloud = WordCloud(width = 800, height = 500, max_font_size = 110).generate(negative_words)
plt.figure(figsize= (12,8))
plt.imshow(wordcloud, interpolation = 'bilinear')
plt.axis('off')
plt.show()

#Select all words from normal tweet (grafik)
normal_words = ' '.join([word for word in train['temiz_tweet'][train['label'] == 0]])
#Collect all hashtags
pos_htag = [htag for htag in normal_words.split() if htag.startswith('#')]
#Remove hashtag symbol (#)
pos_htag = [pos_htag[i][1:] for i in range(len(pos_htag))]
#Count frequency of each word
pos_htag_freqcount = nltk.FreqDist(pos_htag)
pos_htag_df = pd.DataFrame({'Hashtag' : list(pos_htag_freqcount.keys()),
                            'Count' : list(pos_htag_freqcount.values())})
#Select top 20 most frequent hashtags and plot them
most_frequent = pos_htag_df.nlargest(columns="Count", n = 16)
plt.figure(figsize=(20,8))
ax = sns.barplot(data=most_frequent, x= "Hashtag", y = "Count")
ax.set(ylabel = 'Count')
plt.show()

#Repeat same steps for negative tweets (grafik)
negative_words = ' '.join([word for word in train['temiz_tweet'][train['label'] == 1]])
neg_htag = [htag for htag in negative_words.split() if htag.startswith('#')]
neg_htag = [neg_htag[i][1:] for i in range(len(neg_htag))]
neg_htag_freqcount = nltk.FreqDist(neg_htag)
neg_htag_df = pd.DataFrame({'Hashtag' : list(neg_htag_freqcount.keys()),
                            'Count' : list(neg_htag_freqcount.values())})
most_frequent = neg_htag_df.nlargest(columns="Count", n = 16)
plt.figure(figsize=(15,5))
ax = sns.barplot(data=most_frequent, x= "Hashtag", y = "Count")
plt.show()