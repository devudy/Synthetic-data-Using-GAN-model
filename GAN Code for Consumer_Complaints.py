{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "567fe819",
   "metadata": {},
   "source": [
    "# Importing the Necessary Libraries"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "195b94cf",
   "metadata": {},
   "outputs": [],
   "source": [
    "!pip install ctgan \n",
    "!pip install table_evaluator\n",
    "!pip install tensorflow\n",
    "!pip install sdv\n",
    "!pip install wordcloud\n",
    "!pip install textwrap\n",
    "!pip install spacy\n",
    "!pip install textblob"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e77bc8a6",
   "metadata": {},
   "outputs": [],
   "source": [
    "from wordcloud import WordCloud\n",
    "from textwrap import wrap\n",
    "from nltk.corpus import stopwords\n",
    "from collections import  Counter\n",
    "from sklearn.feature_extraction.text import CountVectorizer\n",
    "from matplotlib.ticker import StrMethodFormatter\n",
    "from nltk.stem import WordNetLemmatizer,PorterStemmer\n",
    "from statsmodels.graphics.mosaicplot import mosaic\n",
    "from sdv.evaluation import evaluate\n",
    "from ctgan import CTGANSynthesizer\n",
    "from textblob import TextBlob\n",
    "from nltk.stem import WordNetLemmatizer,PorterStemmer\n",
    "from nltk.tokenize import word_tokenize\n",
    "from wordcloud import WordCloud, STOPWORDS"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a35c30bd",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import re\n",
    "import string\n",
    "import matplotlib.pyplot as plt\n",
    "import nltk\n",
    "import seaborn as sns\n",
    "import en_core_web_sm\n",
    "import plotly.graph_objects as go\n",
    "import spacy"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f5707684",
   "metadata": {},
   "outputs": [],
   "source": [
    "nltk.download('punkt')\n",
    "nltk.download('wordnet')\n",
    "nltk.download('omw-1.4')\n",
    "nltk.download('stopwords')"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f10825e2",
   "metadata": {},
   "source": [
    "# Importing our Data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7b54666c",
   "metadata": {
    "scrolled": false
   },
   "outputs": [],
   "source": [
    "complaints_dt = pd.read_csv('C:/Users/ryan/Downloads/consumer_complaints (1).csv')"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "da3b327c",
   "metadata": {},
   "source": [
    "#Data examination"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7dc1f46c",
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "complaints_dt.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "907d3730",
   "metadata": {},
   "outputs": [],
   "source": [
    "complaints_dt.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "1962dad8",
   "metadata": {},
   "outputs": [],
   "source": [
    "complaints_dt.drop(['zipcode','date_received','consumer_complaint_narrative', 'company_public_response', 'sub_issue', \n",
    "                    'sub_product', 'company', 'tags', 'consumer_consent_provided', 'date_sent_to_company','complaint_id',\n",
    "                    'timely_response', 'consumer_disputed?'], axis=1, inplace=True)\n",
    "print(complaints_dt.columns)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "3ecbeae6",
   "metadata": {},
   "outputs": [],
   "source": [
    "complaints_dt.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d73a143b",
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "complaints_dt.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "0c018031",
   "metadata": {},
   "source": [
    "# Exploratory data analysis"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "1e55adf3",
   "metadata": {},
   "outputs": [],
   "source": [
    "fig1 = complaints_dt.groupby('product')['product'].count().sort_values()\n",
    "plt.figure(figsize=(15, 8))\n",
    "fig1.plot(kind='barh')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "67cdeab0",
   "metadata": {},
   "outputs": [],
   "source": [
    "response_values = complaints_dt['company_response_to_consumer'].value_counts()\n",
    "response_labels =  complaints_dt['company_response_to_consumer'].unique().tolist()\n",
    "fig2= go.Figure(data=[go.Pie(values=response_values, labels=response_labels, hole=.3)])\n",
    "fig2.show()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d6148fed",
   "metadata": {},
   "outputs": [],
   "source": [
    "complaints_dt1 = complaints_dt.dropna()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "272abf43",
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "complaints_dt1.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "92dc6054",
   "metadata": {},
   "outputs": [],
   "source": [
    "complaints_dt1.shape"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "cc8a7180",
   "metadata": {},
   "source": [
    "# Expand Contractions"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ba45234f",
   "metadata": {},
   "outputs": [],
   "source": [
    "contractions_dict = { \"ain't\": \"are not\",\"'s\":\" is\",\"aren't\": \"are not\",\n",
    "                     \"can't\": \"cannot\",\"can't've\": \"cannot have\",\n",
    "                     \"'cause\": \"because\",\"could've\": \"could have\",\"couldn't\": \"could not\",\n",
    "                     \"couldn't've\": \"could not have\", \"didn't\": \"did not\",\"doesn't\": \"does not\",\n",
    "                     \"don't\": \"do not\",\"hadn't\": \"had not\",\"hadn't've\": \"had not have\",\n",
    "                     \"hasn't\": \"has not\",\"haven't\": \"have not\",\"he'd\": \"he would\",\n",
    "                     \"he'd've\": \"he would have\",\"he'll\": \"he will\", \"he'll've\": \"he will have\",\n",
    "                     \"how'd\": \"how did\",\"how'd'y\": \"how do you\",\"how'll\": \"how will\",\n",
    "                     \"I'd\": \"I would\", \"I'd've\": \"I would have\",\"I'll\": \"I will\",\n",
    "                     \"I'll've\": \"I will have\",\"I'm\": \"I am\",\"I've\": \"I have\", \"isn't\": \"is not\",\n",
    "                     \"it'd\": \"it would\",\"it'd've\": \"it would have\",\"it'll\": \"it will\",\n",
    "                     \"it'll've\": \"it will have\", \"let's\": \"let us\",\"ma'am\": \"madam\",\n",
    "                     \"mayn't\": \"may not\",\"might've\": \"might have\",\"mightn't\": \"might not\", \n",
    "                     \"mightn't've\": \"might not have\",\"must've\": \"must have\",\"mustn't\": \"must not\",\n",
    "                     \"mustn't've\": \"must not have\", \"needn't\": \"need not\",\n",
    "                     \"needn't've\": \"need not have\",\"o'clock\": \"of the clock\",\"oughtn't\": \"ought not\",\n",
    "                     \"oughtn't've\": \"ought not have\",\"shan't\": \"shall not\",\"sha'n't\": \"shall not\",\n",
    "                     \"shan't've\": \"shall not have\",\"she'd\": \"she would\",\"she'd've\": \"she would have\",\n",
    "                     \"she'll\": \"she will\", \"she'll've\": \"she will have\",\"should've\": \"should have\",\n",
    "                     \"shouldn't\": \"should not\", \"shouldn't've\": \"should not have\",\"so've\": \"so have\",\n",
    "                     \"that'd\": \"that would\",\"that'd've\": \"that would have\", \"there'd\": \"there would\",\n",
    "                     \"there'd've\": \"there would have\", \"they'd\": \"they would\",\n",
    "                     \"they'd've\": \"they would have\",\"they'll\": \"they will\",\n",
    "                     \"they'll've\": \"they will have\", \"they're\": \"they are\",\"they've\": \"they have\",\n",
    "                     \"to've\": \"to have\",\"wasn't\": \"was not\",\"we'd\": \"we would\",\n",
    "                     \"we'd've\": \"we would have\",\"we'll\": \"we will\",\"we'll've\": \"we will have\",\n",
    "                     \"we're\": \"we are\",\"we've\": \"we have\", \"weren't\": \"were not\",\"what'll\": \"what will\",\n",
    "                     \"what'll've\": \"what will have\",\"what're\": \"what are\", \"what've\": \"what have\",\n",
    "                     \"when've\": \"when have\",\"where'd\": \"where did\", \"where've\": \"where have\",\n",
    "                     \"who'll\": \"who will\",\"who'll've\": \"who will have\",\"who've\": \"who have\",\n",
    "                     \"why've\": \"why have\",\"will've\": \"will have\",\"won't\": \"will not\",\n",
    "                     \"won't've\": \"will not have\", \"would've\": \"would have\",\"wouldn't\": \"would not\",\n",
    "                     \"wouldn't've\": \"would not have\",\"y'all\": \"you all\", \"y'all'd\": \"you all would\",\n",
    "                     \"y'all'd've\": \"you all would have\",\"y'all're\": \"you all are\",\n",
    "                     \"y'all've\": \"you all have\", \"you'd\": \"you would\",\"you'd've\": \"you would have\",\n",
    "                     \"you'll\": \"you will\",\"you'll've\": \"you will have\", \"you're\": \"you are\",\n",
    "                     \"you've\": \"you have\"}\n",
    "\n",
    "\n",
    "contractions_re=re.compile('(%s)' % '|'.join(contractions_dict.keys()))\n",
    "\n",
    "def contractions_expansion(text,contractions_dict=contractions_dict):\n",
    "  def replace(match):\n",
    "    return contractions_dict[match.group(0)]\n",
    "  return contractions_re.sub(replace, text)\n",
    "\n",
    "complaints_dt1['issue']=complaints_dt1['issue'].apply(lambda x:contractions_expansion(x))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "99ad57da",
   "metadata": {},
   "source": [
    "# Lowercase the issues"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "14512fae",
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "complaints_dt1['cleaned']=complaints_dt1['issue'].apply(lambda x: x.lower())"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f5cd5790",
   "metadata": {},
   "source": [
    "# Remove Punctuations"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ce9160b8",
   "metadata": {},
   "outputs": [],
   "source": [
    "complaints_dt1['cleaned']=complaints_dt1['cleaned'].apply(lambda x: re.sub('[%s]' % re.escape(string.punctuation), '', x))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "14e1c7e4",
   "metadata": {},
   "outputs": [],
   "source": [
    "complaints_dt1.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "13a09fc5",
   "metadata": {},
   "source": [
    "# Text looks after cleaning"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "96b6c9cd",
   "metadata": {},
   "outputs": [],
   "source": [
    "for index,text in enumerate(complaints_dt1['cleaned'][35:40]):\n",
    "  print('Review %d:\\n'%(index+1),text)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "57d0cc0f",
   "metadata": {},
   "outputs": [],
   "source": [
    "stop=set(stopwords.words('english'))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "4971be3b",
   "metadata": {},
   "outputs": [],
   "source": [
    "def top_stopwords_barchart(text):\n",
    "    stop=set(stopwords.words('english'))\n",
    "    \n",
    "    new= text.str.split()\n",
    "    new=new.values.tolist()\n",
    "    corpus=[word for i in new for word in i]\n",
    "    from collections import defaultdict\n",
    "    dic=defaultdict(int)\n",
    "    for word in corpus:\n",
    "        if word in stop:\n",
    "            dic[word]+=1\n",
    "            \n",
    "    top=sorted(dic.items(), key=lambda x:x[1],reverse=True)[:10] \n",
    "    x,y=zip(*top)\n",
    "    plt.bar(x,y)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "32302f59",
   "metadata": {},
   "outputs": [],
   "source": [
    "top_stopwords_barchart(complaints_dt1['cleaned'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9d03b4bf",
   "metadata": {},
   "outputs": [],
   "source": [
    "def top_non_stopwords_barchart(text):\n",
    "    stop=set(stopwords.words('english'))\n",
    "    \n",
    "    new= text.str.split()\n",
    "    new=new.values.tolist()\n",
    "    corpus=[word for i in new for word in i]\n",
    "\n",
    "    counter=Counter(corpus)\n",
    "    most=counter.most_common()\n",
    "    x, y=[], []\n",
    "    for word,count in most[:40]:\n",
    "        if (word not in stop):\n",
    "            x.append(word)\n",
    "            y.append(count)\n",
    "            \n",
    "    sns.barplot(x=y,y=x)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ce7c645c",
   "metadata": {},
   "outputs": [],
   "source": [
    "top_non_stopwords_barchart(complaints_dt1['cleaned'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "86ef58c3",
   "metadata": {},
   "outputs": [],
   "source": [
    "def top_ngrams_barchart(text, n=2):\n",
    "    stop=set(stopwords.words('english'))\n",
    "\n",
    "    new= text.str.split()\n",
    "    new=new.values.tolist()\n",
    "    corpus=[word for i in new for word in i]\n",
    "\n",
    "    def _get_top_ngram(corpus, n=None):\n",
    "        vec = CountVectorizer(ngram_range=(n, n)).fit(corpus)\n",
    "        bag_of_words = vec.transform(corpus)\n",
    "        sum_words = bag_of_words.sum(axis=0) \n",
    "        words_freq = [(word, sum_words[0, idx]) \n",
    "                      for word, idx in vec.vocabulary_.items()]\n",
    "        words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)\n",
    "        return words_freq[:10]\n",
    "\n",
    "    top_n_bigrams=_get_top_ngram(text,n)[:10]\n",
    "    x,y=map(list,zip(*top_n_bigrams))\n",
    "    sns.barplot(x=y,y=x)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "1ffd64ce",
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "top_ngrams_barchart(complaints_dt1['cleaned'],2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "08861766",
   "metadata": {},
   "outputs": [],
   "source": [
    "top_ngrams_barchart(complaints_dt1['cleaned'],3)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "2d2b5261",
   "metadata": {},
   "source": [
    "# Preparing Text Data for sentiment analysis"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f08fdf57",
   "metadata": {},
   "source": [
    "# Stopwords Removal & Lemmatization"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "606ce5ee",
   "metadata": {},
   "outputs": [],
   "source": [
    "nlp = spacy.load('en_core_web_sm',disable=['parser', 'ner'])\n",
    "\n",
    "complaints_dt1['lemmatized']=complaints_dt1['cleaned'].apply(lambda x: ' '.join([token.lemma_ for token in list(nlp(x)) \n",
    "                                                                                 if (token.is_stop==False)]))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "72f6bc91",
   "metadata": {},
   "source": [
    "# Creating Document Term Matrix"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "279c686c",
   "metadata": {},
   "outputs": [],
   "source": [
    "complaints_dt1_grouped=complaints_dt1[['product','lemmatized']].groupby(by='product').agg(lambda x:' '.join(x))\n",
    "complaints_dt1_grouped.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "38a80928",
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "from sklearn.feature_extraction.text import CountVectorizer\n",
    "cv=CountVectorizer(analyzer='word')\n",
    "data=cv.fit_transform(complaints_dt1_grouped['lemmatized'])\n",
    "complaints_dt1_dtm = pd.DataFrame(data.toarray(), columns=cv.get_feature_names())\n",
    "complaints_dt1_dtm.index=complaints_dt1_grouped.index\n",
    "complaints_dt1_dtm.head(3)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f649b868",
   "metadata": {},
   "source": [
    "# Word clouds for each product"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7f6030f4",
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "def wordcloud_generation(data,title):\n",
    "  wc = WordCloud(width=400, height=330).generate_from_frequencies(data)\n",
    "  plt.figure(figsize=(8,8))\n",
    "  plt.imshow(wc, interpolation='bilinear')\n",
    "  plt.axis(\"off\")\n",
    "  plt.title('\\n'.join(wrap(title,60)),fontsize=15)\n",
    "  plt.show()\n",
    "\n",
    "complaints_dt1_dtm=complaints_dt1_dtm.transpose()    \n",
    "\n",
    "for index,product in enumerate(complaints_dt1_dtm.columns):\n",
    "  wordcloud_generation(complaints_dt1_dtm[product].sort_values(ascending=False),product)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "4c26a300",
   "metadata": {},
   "outputs": [],
   "source": [
    "complaints_dt1['polarity']=complaints_dt1['lemmatized'].apply(lambda x:TextBlob(x).sentiment.polarity)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f1774d5e",
   "metadata": {},
   "outputs": [],
   "source": [
    "product_polarity_sorted=pd.DataFrame(complaints_dt1.groupby('product')['polarity'].mean().sort_values(ascending=True))\n",
    "\n",
    "plt.figure(figsize=(15,8))\n",
    "plt.xlabel('Polarity')\n",
    "plt.ylabel('Products')\n",
    "plt.title('Polarity of Different Product issues')\n",
    "polarity_graph=plt.barh(np.arange(len(product_polarity_sorted.index)),product_polarity_sorted['polarity'],color='red')\n",
    "\n",
    "\n",
    "for bar,product in zip(polarity_graph,product_polarity_sorted.index):\n",
    "  plt.text(0.01,bar.get_y()+bar.get_width(),'{}'.format(product),fontsize=14,color='black')\n",
    "\n",
    "for bar,polarity in zip(polarity_graph,product_polarity_sorted['polarity']):\n",
    "  plt.text(bar.get_width(),bar.get_y()+bar.get_width(),'%.3f'%polarity,va='center',fontsize=12,color='black')\n",
    "  \n",
    "plt.yticks([])\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "145c0b38",
   "metadata": {},
   "outputs": [],
   "source": [
    "complaints_cat = ['product', 'issue', 'state', 'submitted_via','company_response_to_consumer']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "1808c04f",
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "complaints_dt1.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "52284cf9",
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "complaints_dt1.drop(['cleaned','lemmatized','polarity'], axis=1, inplace=True)\n",
    "print(complaints_dt1.columns)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "bdf14aaf",
   "metadata": {},
   "source": [
    "# Original data subset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "175f6649",
   "metadata": {},
   "outputs": [],
   "source": [
    "complaints_sample= complaints_dt1.sample(frac=0.1).reset_index(drop=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b645b2c4",
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "complaints_sample.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "071015ad",
   "metadata": {},
   "source": [
    "# Original data sample visualization"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "6a08bc31",
   "metadata": {},
   "outputs": [],
   "source": [
    "x1 = complaints_sample.groupby('product')['product'].count().sort_values()\n",
    "plt.figure(figsize=(15, 8))\n",
    "plt.title('product values frequencies', fontsize=20)\n",
    "x1.plot(kind='bar')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0612ec3d",
   "metadata": {},
   "outputs": [],
   "source": [
    "def Wordcloud_plot(text):\n",
    "    nltk.download('stopwords')\n",
    "    stop=set(stopwords.words('english'))\n",
    "\n",
    "    def _preprocess_text(text):\n",
    "        corpus=[]\n",
    "        stem=PorterStemmer()\n",
    "        lem=WordNetLemmatizer()\n",
    "        for news in text:\n",
    "            words=[w for w in word_tokenize(news) if (w not in stop)]\n",
    "\n",
    "            words=[lem.lemmatize(w) for w in words if len(w)>2]\n",
    "\n",
    "            corpus.append(words)\n",
    "        return corpus\n",
    "    \n",
    "    corpus=_preprocess_text(text)\n",
    "    \n",
    "    wordcloud = WordCloud(\n",
    "        background_color='white',\n",
    "        stopwords=set(STOPWORDS),\n",
    "        max_words=100,\n",
    "        max_font_size=30, \n",
    "        scale=3,\n",
    "        random_state=1)\n",
    "    \n",
    "    wordcloud=wordcloud.generate(str(corpus))\n",
    "\n",
    "    fig = plt.figure(1, figsize=(12, 12))\n",
    "    plt.axis('off')\n",
    " \n",
    "    plt.imshow(wordcloud)\n",
    "    plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f261b9d3",
   "metadata": {},
   "outputs": [],
   "source": [
    "Wordcloud_plot(complaints_sample['issue'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0f981d3f",
   "metadata": {},
   "outputs": [],
   "source": [
    "values = complaints_sample['submitted_via'].value_counts()\n",
    "labels =  complaints_sample['submitted_via'].unique().tolist()\n",
    "fig= go.Figure(data=[go.Pie(values=values, labels=labels, hole=.3)])\n",
    "fig.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "3c7a4450",
   "metadata": {},
   "outputs": [],
   "source": [
    "x = complaints_sample.groupby('company_response_to_consumer')['company_response_to_consumer'].count().sort_values()\n",
    "plt.figure(figsize=(15, 8))\n",
    "plt.title('company_response_to_consumer values frequencies', fontsize=20)\n",
    "x.plot(kind='barh')"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "d109e5bd",
   "metadata": {},
   "source": [
    "# Model training"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f7a68de1",
   "metadata": {},
   "outputs": [],
   "source": [
    "ctgan = CTGANSynthesizer(verbose=True)\n",
    "ctgan.fit(complaints_sample, complaints_cat, epochs = 100)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "10424924",
   "metadata": {},
   "source": [
    "# Synthetic data sample"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "88c0ca8e",
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "samples = ctgan.sample(55000)\n",
    "\n",
    "print(samples.head())"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "2562b4b9",
   "metadata": {},
   "source": [
    "# Synthetic data sample visualization"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a4695491",
   "metadata": {},
   "outputs": [],
   "source": [
    "x1 = samples.groupby('product')['product'].count().sort_values()\n",
    "plt.figure(figsize=(15, 8))\n",
    "plt.title('product values frequencies', fontsize=20)\n",
    "x1.plot(kind='bar')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "4504b686",
   "metadata": {},
   "outputs": [],
   "source": [
    "def Wordcloud_plot(text):\n",
    "    nltk.download('stopwords')\n",
    "    stop=set(stopwords.words('english'))\n",
    "\n",
    "    def _preprocess_text(text):\n",
    "        corpus=[]\n",
    "        stem=PorterStemmer()\n",
    "        lem=WordNetLemmatizer()\n",
    "        for news in text:\n",
    "            words=[w for w in word_tokenize(news) if (w not in stop)]\n",
    "\n",
    "            words=[lem.lemmatize(w) for w in words if len(w)>2]\n",
    "\n",
    "            corpus.append(words)\n",
    "        return corpus\n",
    "    \n",
    "    corpus=_preprocess_text(text)\n",
    "    \n",
    "    wordcloud = WordCloud(\n",
    "        background_color='white',\n",
    "        stopwords=set(STOPWORDS),\n",
    "        max_words=100,\n",
    "        max_font_size=30, \n",
    "        scale=3,\n",
    "        random_state=1)\n",
    "    \n",
    "    wordcloud=wordcloud.generate(str(corpus))\n",
    "\n",
    "    fig = plt.figure(1, figsize=(12, 12))\n",
    "    plt.axis('off')\n",
    " \n",
    "    plt.imshow(wordcloud)\n",
    "    plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "cfe3dd19",
   "metadata": {},
   "outputs": [],
   "source": [
    "Wordcloud_plot(samples['issue'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9e561a0b",
   "metadata": {},
   "outputs": [],
   "source": [
    "values = samples['submitted_via'].value_counts()\n",
    "labels =  samples['submitted_via'].unique().tolist()\n",
    "fig= go.Figure(data=[go.Pie(values=values, labels=labels, hole=.3)])\n",
    "fig.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "72c67b44",
   "metadata": {},
   "outputs": [],
   "source": [
    "x = samples.groupby('company_response_to_consumer')['company_response_to_consumer'].count().sort_values()\n",
    "plt.figure(figsize=(15, 8))\n",
    "plt.title('company_response_to_consumer values frequencies', fontsize=20)\n",
    "x.plot(kind='barh')"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "3b1f993a",
   "metadata": {},
   "source": [
    "# Orignal and sythetic data sample evaluation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5fef1574",
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "evaluate(samples, complaints_sample)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b70375de",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "4de1b2e2",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ab76f857",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
