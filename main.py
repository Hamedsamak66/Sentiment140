import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import lightgbm as lgb
from sklearn.metrics import accuracy_score
import nltk
from nltk.corpus import stopwords
import re
import kagglehub

#nltk.download('stopwords')
#path = kagglehub.dataset_download("kazanova/sentiment140")
#print("Path to dataset files:", path)
# بارگذاری و پیش‌پردازش داده‌ها
def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path, encoding='latin-1', header=None)
    data.columns = ['target', 'ids', 'date', 'flag', 'user', 'text']

    # تمیز کردن متن
    stop_words = set(stopwords.words('english'))

    def clean_text(text):
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'[^A-Za-z]+', ' ', text)
        text = text.lower()
        text = ' '.join([word for word in text.split() if word not in stop_words])
        return text

    data['clean_text'] = data['text'].apply(clean_text)

    X = data['clean_text']
    y = data['target'].apply(lambda x: 0 if x == 0 else 1)  # تبدیل 4 به 1 برای احساسات مثبت

    return X, y

# تقسیم داده‌ها
def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)

# اجرای مدل‌ها
def run_models(X_train, X_test, y_train, y_test):
    # تبدیل متن به ویژگی‌های عددی
    vectorizer = TfidfVectorizer(max_features=1000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # مدل SVM
    svm = SVC()
    svm.fit(X_train_tfidf, y_train)
    y_pred_svm = svm.predict(X_test_tfidf)
    print(f"SVM Accuracy: {accuracy_score(y_test, y_pred_svm):.4f}")

    # مدل درخت تصمیم
    tree = DecisionTreeClassifier()
    tree.fit(X_train_tfidf, y_train)
    y_pred_tree = tree.predict(X_test_tfidf)
    print(f"Decision Tree Accuracy: {accuracy_score(y_test, y_pred_tree):.4f}")

    # مدل LightGBM
    lgb_train = lgb.Dataset(X_train_tfidf, label=y_train)
    params = {'objective': 'binary', 'verbose': -1}
    lgb_model = lgb.train(params, lgb_train, num_boost_round=100)
    y_pred_lgb = np.round(lgb_model.predict(X_test_tfidf)).astype(int)
    print(f"LightGBM Accuracy: {accuracy_score(y_test, y_pred_lgb):.4f}")

# مسیر فایل داده‌های Sentiment140 را قرار دهید
file_path = 'training.1600000.processed.noemoticon.csv'
X, y = load_and_preprocess_data(file_path)
X_train, X_test, y_train, y_test = split_data(X, y)
run_models(X_train, X_test, y_train, y_test)
