# nlp_en_for_blog

### 英語のNLP
#### 実施した項目

‐ 言語の判定  
```python
import langdetect 
langdetect.detect(x)
```
‐ 正規表現を使った前処理  
```python
re.sub(r'[^\w\s]', '', x)
```
‐ トークン化  
```python
x.split
```
‐ ストップワード  
```python
nltk.download('stopwords')
lst_stopwords = nltk.corpus.stopwords.words("english")

# ストップワードの削除
for i in range(len(txt_train)):
    txt_train[i] = [word for word in txt_train[i] if word not in lst_stopwords]
```
‐ 語幹の抽出  
```python
ps = nltk.stem.porter.PorterStemmer()
ps.stem(x)
```
‐ 見出し語化  
```python
lem = nltk.stem.wordnet.WordNetLemmatizer()
nltk.download('wordnet')

lem.lemmatize(x)
```
‐ 単語・文字・文章のカウントによる特徴量の作成  
```python
# 単語カウント数
train["text"].apply(lambda x: len(str(x).split(" ")))

# 文字カウント数
train["text"].apply(lambda x: sum(len(word) for word in str(x).split(" ")))

# 文章カウント数
train["text"].apply(lambda x: len(str(x).split(".")))
```
‐ 感情分析  
```python
TextBlob(x).sentiment.polarity
```
‐ アノテーション  
```python
nlp = spacy.load("en_core_web_lg")
nlp(x).ents.text
nlp(x).ents.label_ 
```
‐ ワードクラウド  
```python
corpus = train["xxx"]
wc = wordcloud.WordCloud(background_color='black', max_words=100, 
                         max_font_size=35)
wc = wc.generate(str(corpus))
```
‐ word embeddings  
```python
nlp = gensim_api.load("glove-wiki-gigaword-300")
```
‐ LDA  
```python
lda_model = gensim.models.ldamodel.LdaModel()
```

参考記事 Text Analysis & Feature Engineering with NLP  
https://towardsdatascience.com/text-analysis-feature-engineering-with-nlp-502d6ea9225d

データセット Real or Not? NLP with Disaster Tweets  
https://www.kaggle.com/c/nlp-getting-started/data
