from tqdm.auto import tqdm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
#Classe de treinamento dos modelos e metricas não otimizadas/todas as iterações
class ModelProcessing:
    def RandomForestClassifier(train_data,train_target,test_data,test_target,size,depth):
        best_recall=0
        recall_list_florest=[]
        precision_list_florest=[]
        for size in tqdm(range(1,50)):
            for depth in range(1,30):
                model=RandomForestClassifier(random_state=123456789,max_depth=depth,n_estimators=size)
                model.fit(train_data,train_target)
                prediction=model.predict(test_data)
                recall=recall_score(test_target,prediction,average='weighted')
                precision=precision_score(test_target,prediction,average='weighted',zero_division=0)
                recall_list_florest.append(recall)
                precision_list_florest.append(precision)
                if best_recall<recall:
                    best_recall=recall
                    best_model=model
                    best_depth=depth
                    best_size=size
        return best_model,[best_size,best_depth],recall_list_florest,precision_list_florest
    
    def DecisionTreeClassifier(train_data,train_target,test_data,test_target,depth):
        best_recall=0
        recall_list_tree=[]
        precision_list_tree=[]
        for depth in tqdm(range(1,100)):
            model=DecisionTreeClassifier(random_state=123456789,max_depth=depth)
            model.fit(train_data,train_target)
            prediction=model.predict(test_data)
            recall=recall_score(test_target,prediction,average='weighted')
            precision=precision_score(test_target,prediction,average='weighted',zero_division=0)
            recall_list_tree.append(recall)
            precision_list_tree.append(precision)
            if best_recall<recall:
                best_recall=recall
                best_depth=depth
                best_model=model
        return best_model,best_depth,recall_list_tree,precision_list_tree
    
    def AdaBoostClassifier(train_data,train_target,test_data,test_target,estimators,depth):
        best_recall=0
        recall_list_adaboost=[]
        precision_list_adaboost=[]
        for estimators in tqdm(range(1,100)):
            for depth in range(1,10):
                model = AdaBoostClassifier(n_estimators=estimators, random_state=12345)
                model.fit(train_data,train_target)
                prediction=model.predict(test_data)
                recall=recall_score(test_target,prediction,average='weighted')
                precision=precision_score(test_target,prediction,average='weighted',zero_division=0)
                recall_list_adaboost.append(recall)
                precision_list_adaboost.append(precision)
                if best_recall<recall:
                    best_recall=recall
                    best_model=model
                    best_depth=depth
                    best_estimators=estimators
        return best_model,[best_estimators,best_depth],recall_list_adaboost,precision_list_adaboost
    
    def XGBoostClassifier():
        #Implementar quando consertar erro de kernel
        print("Ainda não foi implementado")
        return 0
    
    def NaiveBayesClassifier(train_data,train_target,test_data,test_target):
        model = GaussianNB()
        model.fit(train_data.toarray(),train_target)
        prediction=model.predict(test_data.toarray())
        recall=recall_score(test_target,prediction,average='weighted')
        precision=precision_score(test_target,prediction)
        return model,recall,precision