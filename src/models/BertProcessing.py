from enelvo.normaliser import Normaliser
from transformers import (
    TrainingArguments,
    Trainer,
    AutoTokenizer,
    BertForSequenceClassification,
    pipeline,
)
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from datasets import load_dataset
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, RocCurveDisplay, auc, roc_curve ,precision_recall_curve, PrecisionRecallDisplay
import sklearn.metrics as metrics
from sklearn.model_selection import cross_validate, StratifiedKFold
import os
from accelerate import Accelerator
import torch


class BertProcessing:
    tokenizer = None
    norm = None
    pipe = None
    max_length = 512

    # repeat é o numero de vezes que aquele target sera clonado, deve ser um int maior que 1
    def upsample(self, features, target, repeat, value):
        features_true = features[target == value]
        features_false = features[target != value]
        target_true = target[target == value]
        target_false = target[target != value]

        features_upsampled = pd.concat([features_false] + [features_true] * repeat)
        target_upsampled = pd.concat([target_false] + [target_true] * repeat)

        return features_upsampled, target_upsampled

    def convert_label(self, labels):
        new_labels = []
        for label in labels:
            if label == "LABEL_0":
                new_labels.append(0)
            if label == "LABEL_3":
                new_labels.append(3)
            if label == "LABEL_4":
                new_labels.append(4)
        return new_labels

    # Mudar target para valor numerico
    def numerical_target(self, target):
        # Acts as a pointer, be careful
        target.replace("Not Violence", 0, inplace=True)
        target.replace("Low", 1, inplace=True)
        target.replace("Medium", 2, inplace=True)
        target.replace("High", 3, inplace=True)
        target.replace("VeryHight", 4, inplace=True)
        return target

    def text_preprocessing(self, corpus):
        if self.norm is None:
            self.norm = Normaliser(tokenizer="readable", sanitize=True)
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                "neuralmind/bert-base-portuguese-cased",
                do_lower_case=False,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
            )

        lemm = []
        for texts in corpus["text"]:
            lemm.append(self.norm.normalise(texts))

        # encodings = tokenizer.batch_encode_plus(lemm, return_tensors='pt',padding=True)
        # batch_token_ids, attention_masks=encodings['input_ids'],encodings['attention_mask']

        # return batch_token_ids, attention_masks

        encodings = self.tokenizer.batch_encode_plus(
            lemm, return_tensors="pt", padding=True
        )
        # encodings=dict(encodings)

        return encodings

    def division(self, df_final):
        # Talvez precise de Drop=True, checar integridade se função for reinstatada
        df_final = df_final.drop_duplicates().reset_index()
        df_final["label"] = df_final["Classe de Violência"]
        df_train, df_test = train_test_split(
            df_final,
            test_size=0.3,
            random_state=12345,
            shuffle=True,
            stratify=df_final["Classe de Violência"],
        )

        # features_train,train_attention_mask=self.text_preprocessing(df_train['text'])
        # target_train=df_train['Classe de Violência']
        # features_test,test_attention_mask=self.text_preprocessing(df_test['text'])
        # target_test=df_test['Classe de Violência']

        # return features_train, train_attention_mask, target_train, features_test, test_attention_mask, target_test

        features_train = self.text_preprocessing(df_train["text"])
        target_train = df_train["Classe de Violência"]
        features_test = self.text_preprocessing(df_test["text"])
        target_test = df_test["Classe de Violência"]

        return features_train, target_train, features_test, target_test

    def loader(self, df_final):
        # df_final['Classe de Violência']=DataProcessing().numerical_target(df_final['Classe de Violência'])
        df_train, df_test = train_test_split(
            df_final.drop_duplicates().reset_index(),
            test_size=0.3,
            random_state=12345,
            shuffle=True,
            stratify=df_final["Classe de Violência"],
        )

        df_train["Classe de Violência"] = self.numerical_target(
            df_train["Classe de Violência"]
        )
        df_test["Classe de Violência"] = self.numerical_target(
            df_test["Classe de Violência"]
        )

        train_data = pd.DataFrame()
        train_data["label"] = df_train["Classe de Violência"]
        train_data["text"] = df_train["text"]
        train_data.to_csv("train.csv", index=False)

        test_data = pd.DataFrame()
        test_data["label"] = df_test["Classe de Violência"]
        test_data["text"] = df_test["text"]
        test_data.to_csv("test.csv", index=False)

        dataset = load_dataset(
            "csv", data_files={"train": "train.csv", "test": "test.csv"}
        )
        # Trocar batch_size por len(df), evitar erro de eval_stategy
        tokenized_datasets = dataset.map(
            self.text_preprocessing, batched=True, batch_size=2000
        )

        train_dataset = tokenized_datasets["train"]
        test_dataset = tokenized_datasets["test"]

        return train_dataset, test_dataset

    def loader_upsample(self, df_final):
        # df_final['Classe de Violência']=DataProcessing().numerical_target(df_final['Classe de Violência'])
        df_train, df_test = train_test_split(
            df_final.drop_duplicates().reset_index(),
            test_size=0.3,
            random_state=12345,
            shuffle=True,
            stratify=df_final["Classe de Violência"],
        )
        features_sampled, target_sampled = self.upsample(
            df_train["text"], df_train["Classe de Violência"], 240, "Low"
        )
        features_sampled, target_sampled = self.upsample(
            features_sampled, target_sampled, 71, "Medium"
        )
        features_sampled, target_sampled = self.upsample(
            features_sampled, target_sampled, 13, "High"
        )
        features_sampled, target_sampled = self.upsample(
            features_sampled, target_sampled, 3, "VeryHigh"
        )

        df_train["Classe de Violência"] = self.numerical_target(
            df_train["Classe de Violência"]
        )
        df_test["Classe de Violência"] = self.numerical_target(
            df_test["Classe de Violência"]
        )

        train_data = pd.DataFrame()
        train_data["label"] = self.numerical_target(target_sampled)
        train_data["text"] = features_sampled
        train_data.to_csv("train.csv", index=False)

        test_data = pd.DataFrame()
        test_data["label"] = df_test["Classe de Violência"]
        test_data["text"] = df_test["text"]
        test_data.to_csv("test.csv", index=False)

        dataset = load_dataset(
            "csv", data_files={"train": "train.csv", "test": "test.csv"}
        )
        tokenized_datasets = dataset.map(
            self.text_preprocessing, batched=True, batch_size=2000
        )

        train_dataset = tokenized_datasets["train"]
        test_dataset = tokenized_datasets["test"]

        return train_dataset, test_dataset

    def compute_metrics(self, p):
        pred, labels = p
        pred = np.argmax(pred, axis=1)
        accuracy = accuracy_score(y_true=labels, y_pred=pred)
        recall = recall_score(y_true=labels, y_pred=pred, average="weighted")
        precision = precision_score(
            y_true=labels, y_pred=pred, average="weighted", zero_division=0
        )
        f1 = f1_score(y_true=labels, y_pred=pred, average="weighted")
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    def model(self, train_dataset, test_dataset):
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # model = AutoModel.from_pretrained('neuralmind/bert-base-portuguese-cased', num_labels=5,  torch_dtype="auto")
        model = BertForSequenceClassification.from_pretrained(
            "neuralmind/bert-base-portuguese-cased", num_labels=5, torch_dtype="auto"
        )
        training_args = TrainingArguments(
            output_dir="test_trainer", evaluation_strategy="epoch"
        )

        """tensor_x = torch.Tensor(features_train) 
        tensor_y = torch.Tensor(target_train.rename(columns={"Classe de Violência": "label"}, inplace=True))
        train_dataset = TensorDataset(tensor_x,tensor_y)
        
        tensor_x = torch.Tensor(features_test) 
        tensor_y = torch.Tensor(target_test.rename(columns={"Classe de Violência": "label"}, inplace=True))
        test_dataset = TensorDataset(tensor_x,tensor_y) """

        """dataloader_config = DataLoaderConfiguration(
            dispatch_batches=False,  # Each process fetches its own batch
            split_batches=True       # Split fetched batches across processes
        )"""
        # accelerator = Accelerator(dataloader_config=dataloader_config)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            compute_metrics=self.compute_metrics,
        )
        trainer.train()

        self.tokenizer = AutoTokenizer.from_pretrained(
            "neuralmind/bert-base-portuguese-cased",
            do_lower_case=False,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )
        os.environ["TOKENIZERS_PARALLELISM"] = "true"

        return model
    
    def train_model(self,model,X_train,y_train,X_test,y_test):
            
        df_train=pd.DataFrame()
        df_train["Classe de Violência"]=y_train
        df_train["text"]=X_train
        df_test=pd.DataFrame()
        df_test["Classe de Violência"]=y_test
        df_test["text"]=X_test
        df_train['Classe de Violência']=self.numerical_target(df_train['Classe de Violência'])
        df_test['Classe de Violência']=self.numerical_target(df_test['Classe de Violência'])
            
        train_data=pd.DataFrame()
        train_data['label']=df_train['Classe de Violência']
        train_data['text']=df_train['text']
        train_data.to_csv('./data/train.csv',index=False)
            
        test_data=pd.DataFrame()
        test_data['label']=df_test['Classe de Violência']
        test_data['text']=df_test['text']
        test_data.to_csv('./data/test.csv',index=False)

        device=torch.device("mps")
        dataset = load_dataset('csv', data_files={'train': "./data/train.csv",'test': "./data/test.csv"}).with_format("torch", device=device)
        tokenized_datasets = dataset.map(self.text_preprocessing, batched=True,batch_size=len(y_train)+len(y_test))
            
        train_dataset = tokenized_datasets["train"]
        test_dataset = tokenized_datasets["test"]

        training_args = TrainingArguments(output_dir="test_trainer",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=False,
        learning_rate=2e-6,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5,
        weight_decay=0.05,
        )
        accelerator = Accelerator()
        model=accelerator.prepare(model)
        trainer = accelerator.prepare(Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            compute_metrics=self.compute_metrics,
        ))
        trainer.train()

        return model

    def predict(self, model, corpus):
        max_length = 512
        if self.pipe is None:
            self.pipe = pipeline(
                "text-classification",
                model=model,
                tokenizer=self.tokenizer,
                device_map="auto",
            )

        lemm = []
        for texts in corpus:
            lemm.append(self.norm.normalise(texts))

        answer = pd.DataFrame(self.pipe(lemm))

        return answer

    def probability_bert(self, probability, test_target, target):
        target_proba = []
        probability = probability.values
        test_target = test_target
        target = target
        for i in range(len(test_target)):
            if test_target[i] == target[i]:
                target_proba.append(probability[i])
            else:
                target_proba.append(1 - probability[i])
        return list(target_proba)

    def prediction_matriz_by_class_bert(self, data):
        matriz = pd.DataFrame()
        vector = []
        classes = [0, 1, 2, 3, 4]
        for column in classes:
            for entry in data:
                if entry == column:
                    vector.append(1)
                else:
                    vector.append(0)
            matriz[column] = vector
            vector.clear()
        return matriz

    def prediction_matriz_by_class_bert_alt(self, data, proba):
        matriz = pd.DataFrame()
        vector = []
        classes = [0, 1, 2, 3, 4]
        for column in classes:
            for i in range(len(data)):
                if data[i] == column:
                    vector.append(proba[i])
                else:
                    vector.append((1 - proba[i]) / 4)
            matriz[column] = vector
            vector.clear()
        return matriz

    def binary_goal(self, prediction, target):
        # safe
        vector = []
        for i in range(len(target)):
            if prediction[i] == target[i]:
                vector.append(1)
            else:
                vector.append(0)
        return vector

    def evaluate(self, model, train_dataset, test_dataset):
        eval_stats = {}

        fig, axs = plt.subplots(1, 3, figsize=(20, 6))

        train_features = train_dataset["text"]
        test_features = test_dataset["text"]

        train_target = train_dataset["label"]
        test_target = test_dataset["label"]

        if self.pipe is None:
            self.pipe = pipeline(
                "text-classification",
                model=model,
                tokenizer=self.tokenizer,
                device_map="auto",
            )

        for type, features, target in (
            ("train", train_features, train_target),
            ("test", test_features, test_target),
        ):
            eval_stats[type] = {}

            output = pd.DataFrame(self.pipe.predict(features))
            pred_target = output[:]["label"]
            pred_proba = output[:]["score"]
            pred_target = self.convert_label(pred_target)
            test_matriz = self.prediction_matriz_by_class_bert(data=target)
            pred_proba_matriz = self.prediction_matriz_by_class_bert(data=pred_target)

            # F1
            f1_thresholds = np.arange(0, 1.01, 0.05)
            f1_scores = [
                metrics.f1_score(
                    self.binary_goal(pred_target, target),
                    self.probability_bert(pred_proba, pred_target, target) >= threshold,
                    average="micro",
                )
                for threshold in f1_thresholds
            ]

            # ROC over all micro-media
            fpr, tpr, roc_thresholds = metrics.roc_curve(
                test_matriz.values.ravel(), pred_proba_matriz.values.ravel()
            )
            roc_auc = metrics.roc_auc_score(
                test_matriz.values.ravel(),
                pred_proba_matriz.values.ravel(),
                average="micro",
            )
            eval_stats[type]["ROC AUC"] = roc_auc

            # Curva de precisão-revocação over all micro-media
            precision, recall, pr_thresholds = metrics.precision_recall_curve(
                test_matriz.values.ravel(), pred_proba_matriz.values.ravel()
            )
            aps = metrics.average_precision_score(
                test_matriz.values.ravel(),
                pred_proba_matriz.values.ravel(),
                average="micro",
            )
            eval_stats[type]["APS"] = aps

            if type == "train":
                color = "blue"
            else:
                color = "green"

            # Valor F1
            ax = axs[0]
            max_f1_score_idx = np.argmax(f1_scores)
            ax.plot(
                f1_thresholds,
                f1_scores,
                color=color,
                label=f"{type}, max={f1_scores[max_f1_score_idx]:.2f} @ {f1_thresholds[max_f1_score_idx]:.2f}",
            )
            # definindo cruzamentos para alguns limiares
            for threshold in (0.2, 0.4, 0.5, 0.6, 0.8):
                closest_value_idx = np.argmin(np.abs(f1_thresholds - threshold))
                marker_color = "orange" if threshold != 0.5 else "red"
                ax.plot(
                    f1_thresholds[closest_value_idx],
                    f1_scores[closest_value_idx],
                    color=marker_color,
                    marker="X",
                    markersize=7,
                )
            ax.set_xlim([-0.02, 1.02])
            ax.set_ylim([-0.02, 1.02])
            ax.set_xlabel("threshold")
            ax.set_ylabel("F1")
            ax.legend(loc="lower center")
            ax.set_title(f"Valor F1")

            # ROC
            ax = axs[1]
            ax.plot(fpr, tpr, color=color, label=f"{type}, ROC AUC={roc_auc:.2f}")
            # setting crosses for some thresholds
            for threshold in (0.2, 0.4, 0.5, 0.6, 0.8):
                closest_value_idx = np.argmin(np.abs(roc_thresholds - threshold))
                marker_color = "orange" if threshold != 0.5 else "red"
                ax.plot(
                    fpr[closest_value_idx],
                    tpr[closest_value_idx],
                    color=marker_color,
                    marker="X",
                    markersize=7,
                )
            ax.plot([0, 1], [0, 1], color="grey", linestyle="--")
            ax.set_xlim([-0.02, 1.02])
            ax.set_ylim([-0.02, 1.02])
            ax.set_xlabel("FPR")
            ax.set_ylabel("TPR")
            ax.legend(loc="lower center")
            ax.set_title(f"Curva ROC Media")

            # Curva de precisão-revocação
            ax = axs[2]
            ax.plot(recall, precision, color=color, label=f"{type}, AP={aps:.2f}")
            # definindo cruzamentos para alguns limiares
            for threshold in (0.2, 0.4, 0.5, 0.6, 0.8):
                closest_value_idx = np.argmin(np.abs(pr_thresholds - threshold))
                marker_color = "orange" if threshold != 0.5 else "red"
                ax.plot(
                    recall[closest_value_idx],
                    precision[closest_value_idx],
                    color=marker_color,
                    marker="X",
                    markersize=7,
                )
            ax.set_xlim([-0.02, 1.02])
            ax.set_ylim([-0.02, 1.02])
            ax.set_xlabel("recall")
            ax.set_ylabel("precision")
            ax.legend(loc="lower center")
            ax.set_title(f"PRC Media")

            eval_stats[type]["Accuracy"] = metrics.accuracy_score(target, pred_target)

        df_eval_stats = pd.DataFrame(eval_stats)
        df_eval_stats = df_eval_stats.round(2)
        df_eval_stats = df_eval_stats.reindex(
            index=("Acurácia", "F1", "APS", "ROC AUC")
        )

        print(df_eval_stats)

        return
    
    def evaluate_alt(self,X,y,n_splits=5):
        cv = StratifiedKFold(n_splits=n_splits)
        y=y.reset_index(drop=True)
        tprs = []
        aucs = []
        precision_vec=[]
        precision_vec_alt=[]
        recall_vec=[]
        aps_vec=[]
        mean_fpr = np.linspace(0, 1, 100)
        mean_test = np.linspace(0, 1, 100)

        fig, axs = plt.subplots(1,2,figsize=(15, 6))
        for fold, (train, test) in enumerate(cv.split(X, y)):
            #Train BERT with data
            classifier = BertForSequenceClassification.from_pretrained(
            "neuralmind/bert-base-portuguese-cased", num_labels=5, torch_dtype="auto"
            )

            model=self.train_model(classifier,X[train],y[train],X[test],y[test])

            tokenizer=AutoTokenizer.from_pretrained(
                "neuralmind/bert-base-portuguese-cased",
                do_lower_case=False,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
            )

            local_pipe=pipeline(
                "text-classification",
                model=model,
                tokenizer=tokenizer,
                device_map="auto",
            )

            #Probability and target matrix
            output = pd.DataFrame(local_pipe.predict(list(X[test])))
            pred_target = output[:]["label"]
            prob_test_vec = output[:]["score"]
            pred_target = self.convert_label(pred_target)
            test_matriz = self.prediction_matriz_by_class_bert(data=y[test])
            pred_proba_matriz = self.prediction_matriz_by_class_bert(data=pred_target)

            #ROC-AUC score
            fpr, tpr, thresholds = roc_curve(test_matriz.values.ravel(),pred_proba_matriz.values.ravel())
            auc_score = auc(fpr, tpr)

            viz = RocCurveDisplay.from_predictions(
                test_matriz.values.ravel(),
                pred_proba_matriz.values.ravel(),
                name=f"ROC fold {fold}",
                ax=axs[0],
                plot_chance_level=(fold == n_splits - 1),
            )
            interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(viz.roc_auc)

            #PRC-APS score
            precision, recall, thresholds = precision_recall_curve(test_matriz.values.ravel(),pred_proba_matriz.values.ravel())
            aps = metrics.average_precision_score(test_matriz.values.ravel(),pred_proba_matriz.values.ravel(), average="micro")

            dis = PrecisionRecallDisplay.from_predictions(test_matriz.values.ravel(),pred_proba_matriz.values.ravel(),name=f"PRC fold {fold}",ax=axs[1],plot_chance_level=(fold == n_splits - 1))
            #interp_tpr_aps = np.interp(mean_fpr, dis.fpr, dis.tpr)
            #interp_tpr_aps[0] = 0.0
            precision_vec.append(precision)
            recall_vec.append(recall)
            aps_vec.append(aps)

        #ROC
        ax = axs[0]
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        ax.plot(
            mean_fpr,
            mean_tpr,
            color="b",
            label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
            lw=2,
            alpha=0.8,
        )

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax.fill_between(
            mean_fpr,
            tprs_lower,
            tprs_upper,
            color="grey",
            alpha=0.2,
            label=r"$\pm$ 1 std. dev.",
        )

        ax.set(
            xlabel="False Positive Rate",
            ylabel="True Positive Rate",
            title=f"Mean ROC curve with variability\n(Positive label '')",
        )
        ax.legend(loc="lower right")

        #PRC
        ax = axs[1]
        #mean_precision = np.mean(precision_vec, axis=0)
        #mean_recall=np.mean(recall_vec, axis=0)
        mean_aps = np.mean(aps_vec)
        std_aps = np.std(aps_vec)
        #ax.plot(
        #    mean_recall,
        #    mean_precision,
        #    color="b",
        #    label=r"Mean PRC (APS = %0.2f $\pm$ %0.2f)" % (mean_aps, std_aps),
        #    lw=2,
        #    alpha=0.8,
        #)
        print("Mean APS:",mean_aps)
        print("stardard deviation APS:",std_aps)
        ax.set(
            xlabel="Recall",
            ylabel="Precision",
            title=f"Mean PRC curve with variability\n(Positive label '')",
        )
        ax.legend(loc="lower right")

        plt.show()

        model_final = BertForSequenceClassification.from_pretrained(
            "neuralmind/bert-base-portuguese-cased", num_labels=5, torch_dtype="auto"
        )
        model_final=self.train_model(classifier,X,y,X[test],y[test])

        return model_final
