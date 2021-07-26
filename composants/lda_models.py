#basic
import pandas as pd
import numpy as np

# Visualisation
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='whitegrid', context='talk')

from wordcloud import WordCloud

#db
import requests

#nltk
import nltk
nltk.download('stopwords') 
nltk.download('wordnet')

#!pip install wordcloud

# Text preprocessing and modelling
from nltk.tokenize import RegexpTokenizer
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.pipeline import Pipeline

# Warnings
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Stopwords
stop_words = set(ENGLISH_STOP_WORDS).union(stopwords.words('english'))
stop_words = stop_words.union(['let', 'mayn', 'ought', 'oughtn','shall'])

class Lda_model:
    
    def __init__(self):
        #-------------------------------------------------------
        """ Init object, load df, load stop_words
        Return None """
        #-------------------------------------------------------       
        self.df = pd.DataFrame(requests.get('https://safe-sierra-02925.herokuapp.com/data/').json())
        self.stop_words = stop_words
    

    def delete_words(self):
        #-------------------------------------------------------
        """ delete some keywords, modify self.df
        return None """
        #-------------------------------------------------------
        #Delete written
        self.df['synopsis'] = self.df['synopsis'].apply(lambda x: x.replace('Written',''))
        #delete family
        self.df['synopsis'] = self.df['synopsis'].apply(lambda x: x.replace('family',''))
        #delete life
        self.df['synopsis'] = self.df['synopsis'].apply(lambda x: x.replace('life',''))
        #delete year
        self.df['synopsis'] = self.df['synopsis'].apply(lambda x: x.replace('year',''))
        #delete man
        self.df['synopsis'] = self.df['synopsis'].apply(lambda x: x.replace('man',''))
        #delete story
        self.df['synopsis'] = self.df['synopsis'].apply(lambda x: x.replace('story',''))
        #delete time
        self.df['synopsis'] = self.df['synopsis'].apply(lambda x: x.replace('time',''))
        #delete movie
        self.df['synopsis'] = self.df['synopsis'].apply(lambda x: x.replace('movie',''))
        #delete film
        self.df['synopsis'] = self.df['synopsis'].apply(lambda x: x.replace('film',''))
        #delete Tamil
        self.df['synopsis'] = self.df['synopsis'].apply(lambda x: x.replace('Tamil',''))
        #delete tamil
        self.df['synopsis'] = self.df['synopsis'].apply(lambda x: x.replace('tamil',''))
        #delete come
        self.df['synopsis'] = self.df['synopsis'].apply(lambda x: x.replace('come',''))

        return None
    
    def preprocess_text(self, document):
        #-------------------------------------------------------
        """ Keep only alpha character group by 3, select somes tag word, remove stop-words, lemmatize words, Preprocess document into normalised tokens.
        Return array of final words """
        #-------------------------------------------------------
        # Tokenise words into alphabetic tokens with minimum length of 3
        tokeniser = RegexpTokenizer(r'[A-Za-z]{3,}')
        tokens = tokeniser.tokenize(document)
        
        # Tag words with POS tag
        #pos_map = {'J': 'a', 'N': 'n', 'R': 'r', 'V': 'v'}
        pos_map = {'N': 'n'}
        
        pos_tags = pos_tag(tokens)
        
        # Lowercase and lemmatise 
        lemmatiser = WordNetLemmatizer()
        lemmas = [lemmatiser.lemmatize(t.lower(), pos=pos_map.get(p[0], 'v')) for t, p in pos_tags]
        
        # Remove stopwords
        keywords= [lemma for lemma in lemmas if lemma not in self.stop_words]
        return keywords
    
    def split(self):
        #-------------------------------------------------------
        """ Split Dataset
        Return shape of X_train and shape of X_test """
        #-------------------------------------------------------         
        self.X_train, self.X_test = train_test_split(self.df, test_size=0.2, 
                                   random_state=1)

        return self.X_train.shape, self.X_test.shape

    def describe_topics(self, lda, feature_names, top_n_words=5, show_weight=False):
        #-------------------------------------------------------
        """ Show main words of each topics from lda model
        Return array of topics descriptions """
        #-------------------------------------------------------
        normalised_weights = lda.components_ / lda.components_.sum(axis=1)[:, np.newaxis]
        topics_descriptions = []
    
        for i, weights in enumerate(normalised_weights):  
            a = f"Topic {i+1} : "
                    
            if show_weight:
                feature_weights = [*zip(np.round(weights, 4), feature_names)]
                feature_weights.sort(reverse=True)
                b = ', '.join(feature_weights[:top_n_words])
                         
            else:
                top_words = [feature_names[i] for i in weights.argsort()[:-top_n_words-1:-1]]
                b = ', '.join(top_words)
                         
            topics_descriptions.append(a+b)          
        
        return topics_descriptions
    
    def model_creation(self):
        #-------------------------------------------------------
        """ create models and inspects topics words
        Return none """
        #-------------------------------------------------------
        #Define number of Contents
        #Topics creation fon n_components = n
        #**************************************
        self.n_components = 3
        #**************************************
        self.pipe = Pipeline([('vectoriser', CountVectorizer(analyzer=self.preprocess_text, min_df=5)),
                        ('lda', LatentDirichletAllocation(n_components=self.n_components, topic_word_prior = 0.9, doc_topic_prior = 0.1, learning_method='batch', random_state=0))])

        self.pipe.fit(self.X_train['synopsis'])
        self.pipe_vectoriser = self.pipe['vectoriser']

        # Inspect topics
        self.feature_names = self.pipe['vectoriser'].get_feature_names()
        topics_description = self.describe_topics(self.pipe['lda'], self.feature_names, top_n_words=20)

        #Transform Train & Viz
        self.train = pd.DataFrame(self.X_train)
        columns = ['topic'+str(i+1) for i in range(self.n_components)]
        self.train[columns] = self.pipe.transform(self.X_train['synopsis'])

        self.train = self.train.assign(top1=np.nan, prob1=np.nan, top2=np.nan, 
                     prob2=np.nan, top3=np.nan, prob3=np.nan)

        top_liste = [f'top{i}' for i in range(1,self.n_components+1)]
        top_prob = [f'prob{i}' for i in range(1,self.n_components+1)]
        last_topic = f'topic{self.n_components}'

        for record in self.train.index:
            top = self.train.loc[record, 'topic1':last_topic].astype(float).nlargest(self.n_components)
            self.train.loc[record, top_liste] = top.index
            self.train.loc[record, top_prob] = top.values

        return self.train, topics_description
   
    def the_plot_proba(self, train):
        #-------------------------------------------------------
        """ Plot probability of dominant topic colour coded by topics
        Return plt.figure """
        #-------------------------------------------------------
        fig = plt.figure(figsize=(12,5))
        sns.kdeplot(data=train, x='prob1', hue='top1', shade=True, 
                    common_norm=False)
        plt.title("Probability of dominant topic colour coded by topics")
        return fig

    def viztopics(self, train):
        #-------------------------------------------------------
        """ create sample of movies title for each topic
        Return dataframe of comparaison """
        #-------------------------------------------------------
        df0 = train[['titres','top1']]
        df_compare=pd.DataFrame()
        df1 = df0.sample(len(df0))

        for i in range(1,self.n_components+1):
            df_compare[f'topic{i}'] = list(df1['titres'][df1['top1'] == f'topic{i}'].head(10))

        return df_compare

    def inspect_term_frequency(self, df, vectoriser, n=30):
        #-------------------------------------------------------
        """ Show main words sorted for each topics
        Return none """
        #-------------------------------------------------------
        document_term_matrix = vectoriser.transform(df)
        document_term_matrix_df = pd.DataFrame(document_term_matrix.toarray(), 
                                            columns = self.feature_names)
        term_frequency = pd.DataFrame(document_term_matrix_df.sum(axis=0), 
                                    columns=['frequency'])
        return term_frequency.nlargest(n, 'frequency')

    def the_plot_main_word(self, train):
        #-------------------------------------------------------
        """ Plot list of Main Words sorted
        Return plt.figure """
        #-------------------------------------------------------         
        fig, ax = plt.subplots(1, self.n_components, figsize=(16,12))
        
        for i in range(self.n_components):
            topic = 'topic' + str(i+1)
            topic_df = train.loc[train['top1']==topic, 'synopsis']
            freqs = self.inspect_term_frequency(topic_df, self.pipe_vectoriser)
            sns.barplot(data=freqs, x='frequency', y=freqs.index, ax=ax[i])
            ax[i].set_title(f"Top words for {topic}")
        plt.tight_layout()

        return fig
    
    def the_plot_wordcloud(self, train):
        #-------------------------------------------------------
        """ Plot with wordcloud
        Return plt.figure """
        #-------------------------------------------------------
        fig, ax = plt.subplots(1, self.n_components, figsize=(20, 8))
        for i in range(self.n_components):
            topic = 'topic' + str(i+1)
            text = ' '.join(train.loc[train['top1']==topic, 'synopsis'].values)    
            wordcloud = WordCloud(width=1000, height=1000, random_state=1, background_color='Black', 
                                colormap='Set2', collocations=False, stopwords=stop_words).generate(text)
            ax[i].imshow(wordcloud) 
            ax[i].set_title(topic)
            ax[i].axis("off")

        return fig 

    def topic_mapping(self, dico):
        #-------------------------------------------------------
        """ Save User Preference and change topics names in self.train
        Return plt.figure """
        #-------------------------------------------------------
        self.mapping = dico
        self.train['topic'] = self.train['top1'].map(self.mapping)
        self.train[['synopsis', 'topic']].head() 

        return None

    def assign_topic(self, document):
        #-------------------------------------------------------
        """ Assign a topic from data send using lda model prediction
        Return string with topic name """
        #-------------------------------------------------------
        probabilities = self.pipe.transform(document)
        topic = probabilities.argmax()
        topic_name = self.mapping['topic'+str(topic+1)]
        return topic_name

    def viz_topics_user_choice(self):
        #-------------------------------------------------------
        """ Viz some topics with real user choice topic name
        Return dataset """
        #-------------------------------------------------------
        user_topic_titres = []
        user_topic_resume = []
        user_topic_assigned = []

        for i, document in enumerate(self.X_train.sample(3).values):
            user_topic_titres.append(document[0])
            user_topic_resume.append(document[1])
            user_topic_assigned.append(self.assign_topic(np.atleast_1d(document[1])))
            
        return pd.DataFrame({'titres': user_topic_titres, 'resumes': user_topic_resume, 'assigned topic': user_topic_assigned})

    def predict_topics_user_choice(self):
        #-------------------------------------------------------
        """ Predict some topics with real user choice topic name
        Return dataset """
        #-------------------------------------------------------
        user_topic_titres = []
        user_topic_resume = []
        user_topic_assigned = []

        for i, document in enumerate(self.X_test.values):
            user_topic_titres.append(document[0])
            user_topic_resume.append(document[1])
            user_topic_assigned.append(self.assign_topic(np.atleast_1d(document[1])))
            
        return pd.DataFrame({'titres': user_topic_titres, 'resumes': user_topic_resume, 'assigned topic': user_topic_assigned})

      
