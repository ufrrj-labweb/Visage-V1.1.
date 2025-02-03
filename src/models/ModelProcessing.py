from tqdm.auto import tqdm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.dummy import DummyClassifier
#Classe de treinamento dos modelos e metricas não otimizadas/todas as iterações
class ModelProcessing:
    def dummy_model(train_data,train_target,test_data,test_target):
        model=DummyClassifier(strategy="most_frequent")
        model.fit(train_data.toarray(),train_target)
        prediction=model.predict(test_data)
        recall=recall_score(test_target,prediction,average='weighted')
        acc=accuracy_score(test_target,prediction)
        f1=f1_score(test_target,prediction,average='weighted')
        precision=precision_score(test_target,prediction,average='weighted')
        print('best recall:',recall)
        print('best accuracy:',acc)
        print('best f1:',f1)
        print('best precision:',precision)
        return model
    def adaboost_model(train_data,train_target,test_data,test_target):
        best_recall=0
        #Talvez não tenha parametro  ou random_state
        #Talvez tenha parametro learning rate, talvez vale a pena testar
        for estimators in tqdm(range(1,100)):
            for depth in range(1,10):
                model = AdaBoostClassifier(n_estimators=estimators, random_state=12345)
                model.fit(train_data,train_target)
                prediction=model.predict(test_data)
                recall=recall_score(test_target,prediction,average='weighted')
                if best_recall<recall:
                    best_model=model
                    best_depth=depth
                    best_recall=recall
                    best_estimators=estimators
                    best_acc=accuracy_score(test_target,prediction)
                    best_f1=f1_score(test_target,prediction,average='weighted')
                    best_precision=precision_score(test_target,prediction,average='weighted',zero_division=0)
        print('best depth:',best_depth)
        print('best number of estimators:',best_estimators)
        print('best recall:',best_recall)
        print('best accuracy:',best_acc)
        print('best f1:',best_f1)
        print('best precision:',best_precision)
        return best_model
    def florest_model(train_data,train_target,test_data,test_target):
        best_recall=0
        for size in tqdm(range(1,50)):
            for depth in range(1,30):
                model=RandomForestClassifier(random_state=123456789,max_depth=depth,n_estimators=size)
                model.fit(train_data,train_target)
                prediction=model.predict(test_data)
                recall=recall_score(test_target,prediction,average='weighted')
                if best_recall<recall:
                    best_model=model
                    best_depth=depth
                    best_recall=recall
                    best_size=size
                    best_acc=accuracy_score(test_target,prediction)
                    best_f1=f1_score(test_target,prediction,average='weighted')
                    best_precision=precision_score(test_target,prediction,average='weighted',zero_division=0)
        print('best depth:',best_depth)
        print('best size:',best_size)
        print('best recall:',best_recall)
        print('best accuracy:',best_acc)
        print('best f1:',best_f1)
        print('best precision:',best_precision)
        return best_model
    def tree_model(train_data,train_target,test_data,test_target):
        best_recall=0
        for depth in tqdm(range(1,100)):
            model=DecisionTreeClassifier(random_state=123456789,max_depth=depth)
            model.fit(train_data,train_target)
            prediction=model.predict(test_data)
            recall=recall_score(test_target,prediction,average='weighted')
            if best_recall<recall:
                best_model=model
                best_depth=depth
                best_recall=recall
                best_acc=accuracy_score(test_target,prediction)
                best_f1=f1_score(test_target,prediction,average='weighted')
                best_precision=precision_score(test_target,prediction,average='weighted',zero_division=0)
        print('best depth:',best_depth)
        print('best recall:',best_recall)
        print('best accuracy:',best_acc)
        print('best f1:',best_f1)
        print('best precision:',best_precision)
        return best_model
    def naivebayes_model(train_data,train_target,test_data,test_target):
        model = GaussianNB()
        model.fit(train_data.toarray(),train_target)
        prediction=model.predict(test_data.toarray())
        recall=recall_score(test_target,prediction,average='weighted')
        acc=accuracy_score(test_target,prediction)
        f1=f1_score(test_target,prediction,average='weighted')
        precision=precision_score(test_target,prediction,average='weighted')
        print('best recall:',recall)
        print('best accuracy:',acc)
        print('best f1:',f1)
        print('best precision:',precision)
        return model