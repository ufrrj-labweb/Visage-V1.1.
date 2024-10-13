import pandas as pd
from itertools import islice
from tqdm.auto import tqdm
import spacy
from nltk.corpus import stopwords as nltk_stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from enelvo.normaliser import Normaliser
class DataProcessing:
    # Junta os dataframes dentro do vetor de dataframes
    def append_data(self,df_vector):
        df_final=df_vector[0]
        df_final=df_final[['text','Total(SUM)','Classe de Violência']]
        for df in islice(df_vector, 1, None) :
            df=df[['text','Total(SUM)','Classe de Violência']]
            df_final=pd.concat([df_final]+[df])
        df_final.reset_index()
        df_final['text']=df_final['text'].astype('str')
        return df_final
    
    #Normalização usando spacy
    def text_normalizer_spacy(self,corpus):
        nlp = spacy.load('pt_core_news_sm', disable=['parser', 'ner'])
        lemm=[]
        for text in tqdm(corpus):
            doc = nlp(text)
            #tokens = [token.lemma_ for token in doc if not token.is_stop]
            tokens = [token.lemma_ for token in doc]
            text= ' '.join(tokens)
            
            lemm.append(text)
        return lemm
    # Normalização usando enelvo e vetorização usando nltk 
    def text_preprocessing_nltk(self,corpus):
        stop_words=list(nltk_stopwords.words('portuguese'))
        norm = Normaliser(tokenizer='readable',sanitize=True)
        lemm=[]
        for texts in corpus:
            lemm.append(norm.normalise(texts))
        print(lemm)
        vect=TfidfVectorizer(stop_words=stop_words)
        processed=vect.fit_transform(lemm)
        return processed
    #Vetorização usando nltk
    def text_preprocessing_nltk_no_norm(self,corpus):
        stop_words=list(nltk_stopwords.words('portuguese'))
        vect=TfidfVectorizer(stop_words=stop_words)
        processed=vect.fit_transform(corpus)
        return processed
    #Mudar target para valor numerico
    def numerical_target(target):
        target.replace('Not Violence',0,inplace=True)
        target.replace('Low',1,inplace=True)
        target.replace('Medium',2,inplace=True)
        target.replace('High',3,inplace=True)
        target.replace('VeryHight',4,inplace=True)
        return target
    #fraction é a fração que vai sobrar do original, deve ser colocado um valor entre 0 e 1
    # Se usado 0.3 por exemplo, perderemos 60% dos registros daquele target, sobrando 30 porcento
    def downsample(self,features, target, fraction,value):
        features_true = features[target == value]
        features_false = features[target != value]
        target_true = target[target == value]
        target_false = target[target != value]

        features_downsampled = pd.concat(
            [features_true.sample(frac=fraction, random_state=12345)]
            + [features_false]
        )
        target_downsampled = pd.concat(
            [target_true.sample(frac=fraction, random_state=12345)]
            + [target_false]
        )

        return features_downsampled, target_downsampled
    # repeat é o numero de vezes que aquele target sera clonado, deve ser um int maior que 1
    def upsample(self,features, target, repeat,value):
        features_true = features[target == value]
        features_false = features[target != value]
        target_true = target[target == value]
        target_false = target[target != value]

        features_upsampled = pd.concat([features_false] + [features_true] * repeat)
        target_upsampled = pd.concat([target_false] + [target_true] * repeat)

        return features_upsampled, target_upsampled