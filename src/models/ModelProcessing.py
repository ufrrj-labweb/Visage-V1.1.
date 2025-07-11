from tqdm.auto import tqdm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.dummy import DummyClassifier


# Classe de treinamento dos modelos e metricas não otimizadas/todas as iterações
class ModelProcessing:
    def dummy_model(data,target):
        kf=StratifiedKFold(n_splits=5,shuffle=True,random_state=12345)
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
        return model
    def adaboost_model(data,target):
        kf=StratifiedKFold(n_splits=5,shuffle=True,random_state=12345)
        best_score_values=[0,0,0,0,0,0]
        #Talvez não tenha parametro  ou random_state
        #Talvez tenha parametro learning rate, talvez vale a pena testar
        for estimators in tqdm(range(1,100)):
            model = AdaBoostClassifier(n_estimators=estimators, random_state=12345)
            scoring = ['accuracy','precision_weighted', 'recall_weighted','f1_weighted']
            scores = cross_validate(model, data, target,cv=kf, scoring=scoring)
            scores_values=[score.mean() for score in scores.values()]
            if best_score_values[4]<scores_values[4]:
                best_model=model
                best_estimators=estimators
                best_score=scores
                best_score_values=scores_values
        print('best number of estimators:',best_estimators)
        print('best recall:',best_score_values[4])
        print('best accuracy:',best_score_values[2])
        print('best f1:',best_score_values[5])
        print('best precision:',best_score_values[3])
        return best_model
    def florest_model(data,target):
        kf=StratifiedKFold(n_splits=5,shuffle=True,random_state=12345)
        best_score_values=[0,0,0,0,0,0]
        for size in tqdm(range(1,50)):
            for depth in range(1,30):
                model=RandomForestClassifier(random_state=123456789,max_depth=depth,n_estimators=size)
                scoring = ['accuracy','precision_weighted', 'recall_weighted','f1_weighted']
                scores = cross_validate(model, data, target,cv=kf, scoring=scoring)
                scores_values=[score.mean() for score in scores.values()]
                if best_score_values[4]<scores_values[4]:
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
        return best_model
    def tree_model(data,target):
        kf=StratifiedKFold(n_splits=5,shuffle=True,random_state=12345)
        best_score_values=[0,0,0,0,0,0]
        for depth in tqdm(range(1,100)):
            model=DecisionTreeClassifier(random_state=123456789,max_depth=depth)
            scoring = ['accuracy','precision_weighted', 'recall_weighted','f1_weighted']
            scores = cross_validate(model, data, target,cv=kf, scoring=scoring)
            scores_values=[score.mean() for score in scores.values()]
            if best_score_values[4]<scores_values[4]:
                best_model=model
                best_depth=depth
                best_score=scores
                best_score_values=scores_values
        print('best depth:',best_depth)
        print('best recall:',best_score_values[4])
        print('best accuracy:',best_score_values[2])
        print('best f1:',best_score_values[5])
        print('best precision:',best_score_values[3])
        return best_model
    def naivebayes_model(data,target):
        kf=StratifiedKFold(n_splits=5,shuffle=True,random_state=12345)
        model = GaussianNB()
        scoring = ['accuracy','precision_weighted', 'recall_weighted','f1_weighted']
        scores = cross_validate(model, data.toarray(), target,cv=kf, scoring=scoring)
        score_values=[score.mean() for score in scores.values()]
        print(sorted(scores.keys()))
        print([score.mean() for score in scores.values()])
        print('best recall:',score_values[4])
        print('best accuracy:',score_values[2])
        print('best f1:',score_values[5])
        print('best precision:',score_values[3])
        return model