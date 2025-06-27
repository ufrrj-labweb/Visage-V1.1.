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
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
import sklearn.metrics as metrics
import os


class BertProcessing:
    tokenizer = None
    norm = None
    pipe = None
    max_length = 512

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

        df_train["Classe de Violência"] = DataProcessing().numerical_target(
            df_train["Classe de Violência"]
        )
        df_test["Classe de Violência"] = DataProcessing().numerical_target(
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
        DataProcess = DataProcessing()
        df_train, df_test = train_test_split(
            df_final.drop_duplicates().reset_index(),
            test_size=0.3,
            random_state=12345,
            shuffle=True,
            stratify=df_final["Classe de Violência"],
        )
        features_sampled, target_sampled = DataProcess.upsample(
            df_train["text"], df_train["Classe de Violência"], 240, "Low"
        )
        features_sampled, target_sampled = DataProcess.upsample(
            features_sampled, target_sampled, 71, "Medium"
        )
        features_sampled, target_sampled = DataProcess.upsample(
            features_sampled, target_sampled, 13, "High"
        )
        features_sampled, target_sampled = DataProcess.upsample(
            features_sampled, target_sampled, 3, "VeryHigh"
        )

        df_train["Classe de Violência"] = DataProcessing().numerical_target(
            df_train["Classe de Violência"]
        )
        df_test["Classe de Violência"] = DataProcessing().numerical_target(
            df_test["Classe de Violência"]
        )

        train_data = pd.DataFrame()
        train_data["label"] = DataProcessing().numerical_target(target_sampled)
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
            pred_target = DataProcessing().convert_label(pred_target)
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
