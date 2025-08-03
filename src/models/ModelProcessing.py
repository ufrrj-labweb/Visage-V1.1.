from tqdm.auto import tqdm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
from nltk.corpus import stopwords as nltk_stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from enelvo.normaliser import Normaliser
import pandas as pd
import numpy as np


# Classe de treinamento dos modelos e metricas não otimizadas/todas as iterações
class ModelProcessing:
    def dummy_model(data,target):
        kf=StratifiedKFold(n_splits=10,shuffle=True,random_state=12345)
        model=DummyClassifier(strategy="most_frequent")
        scoring = ['accuracy','precision_weighted', 'recall_weighted','f1_weighted']
        scores = cross_validate(model, data, target,cv=kf, scoring=scoring)
        score_values=[score.mean() for score in scores.values()]
        print(sorted(scores.keys()))
        print([score.mean() for score in scores.values()])
        print('best recall:',score_values[4])
        print('best accuracy:',score_values[2])
        print('best f1:',score_values[5])
        print('best precision:',score_values[3])
        model.fit(data,target)
        return model
    def adaboost_model(data,target):
        kf=StratifiedKFold(n_splits=10,shuffle=True,random_state=12345)
        best_score_values=[0,0,0,0,0,0]
        #Talvez não tenha parametro  ou random_state
        #Talvez tenha parametro learning rate, talvez vale a pena testar
        for estimators in tqdm(range(1,100)):
            model = AdaBoostClassifier(n_estimators=estimators, random_state=12345)
            scoring = ['accuracy','precision_weighted', 'recall_weighted','f1_weighted']
            scores = cross_validate(model, data, target,cv=kf, scoring=scoring)
            scores_values=[score.mean() for score in scores.values()]
            if best_score_values[5]<scores_values[5]:
                best_model=model
                best_estimators=estimators
                best_score=scores
                best_score_values=scores_values
        print('best number of estimators:',best_estimators)
        print('best recall:',best_score_values[4])
        print('best accuracy:',best_score_values[2])
        print('best f1:',best_score_values[5])
        print('best precision:',best_score_values[3])
        best_model.fit(data,target)
        return best_model
    def florest_model(data,target):
        kf=StratifiedKFold(n_splits=10,shuffle=True,random_state=12345)
        best_score_values=[0,0,0,0,0,0]
        for size in tqdm(range(1,50)):
            for depth in range(1,30):
                model=RandomForestClassifier(random_state=123456789,max_depth=depth,n_estimators=size)
                scoring = ['accuracy','precision_weighted', 'recall_weighted','f1_weighted']
                scores = cross_validate(model, data, target,cv=kf, scoring=scoring)
                scores_values=[score.mean() for score in scores.values()]
                if best_score_values[5]<scores_values[5]:
                    best_model=model
                    best_size=size
                    best_depth=depth
                    best_score=scores
                    best_score_values=scores_values
        print('best depth:',best_depth)
        print('best size:',best_size)
        print('best recall:',best_score_values[4])
        print('best accuracy:',best_score_values[2])
        print('best f1:',best_score_values[5])
        print('best precision:',best_score_values[3])
        best_model.fit(data,target)
        return best_model
    def tree_model(data,target):
        kf=StratifiedKFold(n_splits=10,shuffle=True,random_state=12345)
        best_score_values=[0,0,0,0,0,0]
        for depth in tqdm(range(1,100)):
            model=DecisionTreeClassifier(random_state=123456789,max_depth=depth)
            scoring = ['accuracy','precision_weighted', 'recall_weighted','f1_weighted']
            scores = cross_validate(model, data, target,cv=kf, scoring=scoring)
            scores_values=[score.mean() for score in scores.values()]
            if best_score_values[5]<scores_values[5]:
                best_model=model
                best_depth=depth
                best_score=scores
                best_score_values=scores_values
        print('best depth:',best_depth)
        print('best recall:',best_score_values[4])
        print('best accuracy:',best_score_values[2])
        print('best f1:',best_score_values[5])
        print('best precision:',best_score_values[3])
        best_model.fit(data,target)
        return best_model
    def naivebayes_model(data,target):
        kf=StratifiedKFold(n_splits=10,shuffle=True,random_state=12345)
        model = GaussianNB()
        scoring = ['accuracy','precision_weighted', 'recall_weighted','f1_weighted']
        scores = cross_validate(model, data, target,cv=kf, scoring=scoring)
        score_values=[score.mean() for score in scores.values()]
        print(sorted(scores.keys()))
        print([score.mean() for score in scores.values()])
        print('best recall:',score_values[4])
        print('best accuracy:',score_values[2])
        print('best f1:',score_values[5])
        print('best precision:',score_values[3])
        model.fit(data,target)
        return model
    
    def naivebayes_model_upsampled(data,target):
        vect = None
        def text_preprocessing_nltk(corpus,vect):
            norm = Normaliser(tokenizer="readable", sanitize=True)
            lemm = []
            for texts in corpus:
                lemm.append(norm.normalise(texts))
            processed = vect.transform(lemm)
            return processed

        def upsample(features, target, repeat, value):
            features_true = features[target == value]
            features_false = features[target != value]
            target_true = target[target == value]
            target_false = target[target != value]

            features_upsampled = pd.concat([features_false] + [features_true] * repeat)
            target_upsampled = pd.concat([target_false] + [target_true] * repeat)

            return features_upsampled, target_upsampled
        
        cv=StratifiedKFold(n_splits=10,shuffle=True,random_state=12345)
        best_score_values=[0,0,0,0,0,0]

        norm = Normaliser(tokenizer="readable", sanitize=True)
        lemm = []
        for texts in data:
            lemm.append(norm.normalise(texts))
        stop_words = list(nltk_stopwords.words("portuguese"))
        vect = TfidfVectorizer(stop_words=stop_words)
        vect.fit(lemm)

        #Talvez não tenha parametro  ou random_state
        #Talvez tenha parametro learning rate, talvez vale a pena testar
        
        f1=[]
        recall=[]
        precision=[]
        accuracy=[]
        for fold, (train, test) in enumerate(cv.split(data, target)):
                model = GaussianNB()
                test_data=data[test]
                test_target=target[test]

                train_data=data[train]
                train_target=target[train]
                train_data,train_target=upsample(train_data,train_target,71, "Medium")
                train_data,train_target=upsample(train_data,train_target,13, "High")
                train_data,train_target=upsample(train_data,train_target,3, "VeryHigh")
                train_data,train_target=upsample(train_data,train_target,240, "Low")

                train_data=text_preprocessing_nltk(train_data,vect)
                test_data=text_preprocessing_nltk(test_data,vect)

                model.fit(train_data.toarray(),train_target)
                prediction=model.predict(test_data.toarray())

                f1.append(f1_score(test_target,prediction,average='weighted'))
                recall.append(recall_score(test_target,prediction,average='weighted'))
                precision.append(precision_score(test_target,prediction,average='weighted'))
                accuracy.append(accuracy_score(test_target,prediction))
        acc_mean=np.mean(accuracy)
        recall_mean=np.mean(recall)
        precision_mean=np.mean(precision)
        f1_mean=np.mean(f1)
        print('best recall:',recall_mean)
        print('best accuracy:', acc_mean)
        print('best f1:',f1_mean)
        print('best precision:',precision_mean)
        return model
    
    def tree_model_upsampled(data,target):
        vect = None
        def text_preprocessing_nltk(corpus,vect):
            norm = Normaliser(tokenizer="readable", sanitize=True)
            lemm = []
            for texts in corpus:
                lemm.append(norm.normalise(texts))
            processed = vect.transform(lemm)
            return processed

        def upsample(features, target, repeat, value):
            features_true = features[target == value]
            features_false = features[target != value]
            target_true = target[target == value]
            target_false = target[target != value]

            features_upsampled = pd.concat([features_false] + [features_true] * repeat)
            target_upsampled = pd.concat([target_false] + [target_true] * repeat)

            return features_upsampled, target_upsampled
        
        cv=StratifiedKFold(n_splits=10,shuffle=True,random_state=12345)
        best_score_values=[0,0,0,0,0,0]

        norm = Normaliser(tokenizer="readable", sanitize=True)
        lemm = []
        for texts in data:
            lemm.append(norm.normalise(texts))
        stop_words = list(nltk_stopwords.words("portuguese"))
        vect = TfidfVectorizer(stop_words=stop_words)
        vect.fit(lemm)

        #Talvez não tenha parametro  ou random_state
        #Talvez tenha parametro learning rate, talvez vale a pena testar
        
        f1=[]
        recall=[]
        precision=[]
        accuracy=[]
        for fold, (train, test) in enumerate(cv.split(data, target)):
                model = DecisionTreeClassifier(random_state=123456789,max_depth=28)
                test_data=data[test]
                test_target=target[test]

                train_data=data[train]
                train_target=target[train]
                train_data,train_target=upsample(train_data,train_target,71, "Medium")
                train_data,train_target=upsample(train_data,train_target,13, "High")
                train_data,train_target=upsample(train_data,train_target,3, "VeryHigh")
                train_data,train_target=upsample(train_data,train_target,240, "Low")

                train_data=text_preprocessing_nltk(train_data,vect)
                test_data=text_preprocessing_nltk(test_data,vect)

                model.fit(train_data.toarray(),train_target)
                prediction=model.predict(test_data.toarray())

                f1.append(f1_score(test_target,prediction,average='weighted'))
                recall.append(recall_score(test_target,prediction,average='weighted'))
                precision.append(precision_score(test_target,prediction,average='weighted'))
                accuracy.append(accuracy_score(test_target,prediction))
        acc_mean=np.mean(accuracy)
        recall_mean=np.mean(recall)
        precision_mean=np.mean(precision)
        f1_mean=np.mean(f1)
        print('best recall:',recall_mean)
        print('best accuracy:', acc_mean)
        print('best f1:',f1_mean)
        print('best precision:',precision_mean)
        return model
    
    def forest_model_upsampled(data,target):
        vect = None
        def text_preprocessing_nltk(corpus,vect):
            norm = Normaliser(tokenizer="readable", sanitize=True)
            lemm = []
            for texts in corpus:
                lemm.append(norm.normalise(texts))
            processed = vect.transform(lemm)
            return processed

        def upsample(features, target, repeat, value):
            features_true = features[target == value]
            features_false = features[target != value]
            target_true = target[target == value]
            target_false = target[target != value]

            features_upsampled = pd.concat([features_false] + [features_true] * repeat)
            target_upsampled = pd.concat([target_false] + [target_true] * repeat)

            return features_upsampled, target_upsampled
        
        cv=StratifiedKFold(n_splits=10,shuffle=True,random_state=12345)
        best_score_values=[0,0,0,0,0,0]

        norm = Normaliser(tokenizer="readable", sanitize=True)
        lemm = []
        for texts in data:
            lemm.append(norm.normalise(texts))
        stop_words = list(nltk_stopwords.words("portuguese"))
        vect = TfidfVectorizer(stop_words=stop_words)
        vect.fit(lemm)

        #Talvez não tenha parametro  ou random_state
        #Talvez tenha parametro learning rate, talvez vale a pena testar
        
        f1=[]
        recall=[]
        precision=[]
        accuracy=[]
        for fold, (train, test) in enumerate(cv.split(data, target)):
                model = RandomForestClassifier(random_state=123456789,max_depth=28,n_estimators=12)
                test_data=data[test]
                test_target=target[test]

                train_data=data[train]
                train_target=target[train]
                train_data,train_target=upsample(train_data,train_target,71, "Medium")
                train_data,train_target=upsample(train_data,train_target,13, "High")
                train_data,train_target=upsample(train_data,train_target,3, "VeryHigh")
                train_data,train_target=upsample(train_data,train_target,240, "Low")

                train_data=text_preprocessing_nltk(train_data,vect)
                test_data=text_preprocessing_nltk(test_data,vect)

                model.fit(train_data.toarray(),train_target)
                prediction=model.predict(test_data.toarray())

                f1.append(f1_score(test_target,prediction,average='weighted'))
                recall.append(recall_score(test_target,prediction,average='weighted'))
                precision.append(precision_score(test_target,prediction,average='weighted'))
                accuracy.append(accuracy_score(test_target,prediction))
        acc_mean=np.mean(accuracy)
        recall_mean=np.mean(recall)
        precision_mean=np.mean(precision)
        f1_mean=np.mean(f1)
        print('best recall:',recall_mean)
        print('best accuracy:', acc_mean)
        print('best f1:',f1_mean)
        print('best precision:',precision_mean)
        return model
    
    def adaboost_model_upsampled(data,target):
        vect = None
        def text_preprocessing_nltk(corpus,vect):
            norm = Normaliser(tokenizer="readable", sanitize=True)
            lemm = []
            for texts in corpus:
                lemm.append(norm.normalise(texts))
            processed = vect.transform(lemm)
            return processed

        def upsample(features, target, repeat, value):
            features_true = features[target == value]
            features_false = features[target != value]
            target_true = target[target == value]
            target_false = target[target != value]

            features_upsampled = pd.concat([features_false] + [features_true] * repeat)
            target_upsampled = pd.concat([target_false] + [target_true] * repeat)

            return features_upsampled, target_upsampled
        
        cv=StratifiedKFold(n_splits=10,shuffle=True,random_state=12345)
        best_score_values=[0,0,0,0,0,0]

        norm = Normaliser(tokenizer="readable", sanitize=True)
        lemm = []
        for texts in data:
            lemm.append(norm.normalise(texts))
        stop_words = list(nltk_stopwords.words("portuguese"))
        vect = TfidfVectorizer(stop_words=stop_words)
        vect.fit(lemm)

        #Talvez não tenha parametro  ou random_state
        #Talvez tenha parametro learning rate, talvez vale a pena testar
        
        f1=[]
        recall=[]
        precision=[]
        accuracy=[]
        for fold, (train, test) in enumerate(cv.split(data, target)):
                model = AdaBoostClassifier(n_estimators=6, random_state=12345)
                test_data=data[test]
                test_target=target[test]

                train_data=data[train]
                train_target=target[train]
                train_data,train_target=upsample(train_data,train_target,71, "Medium")
                train_data,train_target=upsample(train_data,train_target,13, "High")
                train_data,train_target=upsample(train_data,train_target,3, "VeryHigh")
                train_data,train_target=upsample(train_data,train_target,240, "Low")

                train_data=text_preprocessing_nltk(train_data,vect)
                test_data=text_preprocessing_nltk(test_data,vect)

                model.fit(train_data.toarray(),train_target)
                prediction=model.predict(test_data.toarray())

                f1.append(f1_score(test_target,prediction,average='weighted'))
                recall.append(recall_score(test_target,prediction,average='weighted'))
                precision.append(precision_score(test_target,prediction,average='weighted'))
                accuracy.append(accuracy_score(test_target,prediction))
        acc_mean=np.mean(accuracy)
        recall_mean=np.mean(recall)
        precision_mean=np.mean(precision)
        f1_mean=np.mean(f1)
        print('best recall:',recall_mean)
        print('best accuracy:', acc_mean)
        print('best f1:',f1_mean)
        print('best precision:',precision_mean)
        return model