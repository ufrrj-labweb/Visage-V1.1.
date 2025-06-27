import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics as metrics
from sklearn.metrics import recall_score
from sklearn.decomposition import TruncatedSVD


class MetricsProcessing:
    def probability(self, probability, test_target):
        target_proba = []
        test_target = test_target.values
        for i in range(len(test_target)):
            if test_target[i] == "High":
                target_proba.append(probability[i, 0])
            if test_target[i] == "Low":
                target_proba.append(probability[i, 1])
            if test_target[i] == "Medium":
                target_proba.append(probability[i, 2])
            if test_target[i] == "Not Violence":
                target_proba.append(probability[i, 3])
            if test_target[i] == "VeryHight":
                target_proba.append(probability[i, 4])
        return target_proba

    def probability_class(self, type_class, model, test_data):
        probability = model.predict_proba(test_data)
        # High:0, low:1, medium:2, not violence:3, VeryHigh:4
        return probability[:, type_class]

    def prediction_matriz_by_class(self, data):
        matriz = pd.DataFrame()
        classes = ["High", "Low", "Medium", "Not Violence", "VeryHight"]
        for column in classes:
            matriz[column] = np.where(data == column, 1, 0)
        return matriz

    def binary_goal(self, prediction, target):
        # safe
        vector = np.where(prediction == target, 1, 0)
        return vector

    def evaluate_model(
        self, model, train_features, train_target, test_features, test_target
    ):
        eval_stats = {}

        fig, axs = plt.subplots(1, 3, figsize=(20, 6))

        for type, features, target in (
            ("train", train_features, train_target),
            ("test", test_features, test_target),
        ):
            eval_stats[type] = {}

            pred_target = model.predict(features)
            pred_proba = model.predict_proba(features)
            test_matriz = self.prediction_matriz_by_class(data=target)

            # F1
            f1_thresholds = np.arange(0, 1.01, 0.05)
            f1_scores = [
                metrics.f1_score(
                    self.binary_goal(pred_target, target),
                    self.probability(pred_proba, target) >= threshold,
                    average="micro",
                )
                for threshold in f1_thresholds
            ]

            # ROC over all micro-media
            fpr, tpr, roc_thresholds = metrics.roc_curve(
                test_matriz.values.ravel(), pred_proba.ravel()
            )
            roc_auc = metrics.roc_auc_score(
                test_matriz.values.ravel(), pred_proba.ravel(), average="micro"
            )
            eval_stats[type]["ROC AUC"] = roc_auc

            # Curva de precisão-revocação over all micro-media
            precision, recall, pr_thresholds = metrics.precision_recall_curve(
                test_matriz.values.ravel(), pred_proba.ravel()
            )
            aps = metrics.average_precision_score(
                test_matriz.values.ravel(), pred_proba.ravel(), average="micro"
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

    def pca_evaluator(model, model2, test_data, test_target, n_components):
        recall_list = []
        components_list = []
        for components in range(1, n_components):
            original = recall_score(
                test_target, model.predict(test_data), average="weighted"
            )
            pca = TruncatedSVD(n_components=components, random_state=12345)
            new_data = pca.fit_transform(test_data)
            new_data = pca.inverse_transform(new_data)
            prediction = model.predict(new_data)
            recall_list.append(
                recall_score(test_target, prediction, average="weighted") / original
            )
            components_list.append(components)
        recall_list2 = []
        components_list2 = []
        for components in range(1, n_components):
            original = recall_score(
                test_target, model2.predict(test_data), average="weighted"
            )
            pca = TruncatedSVD(n_components=components, random_state=12345)
            new_data = pca.fit_transform(test_data)
            new_data = pca.inverse_transform(new_data)
            prediction = model2.predict(new_data)
            recall_list2.append(
                recall_score(test_target, prediction, average="weighted") / original
            )
            components_list2.append(components)
        plt.xlabel("Number of dimensions")
        plt.ylabel("Percentage of the original recall")
        plt.plot(components_list, recall_list, color="blue", marker="X", markersize=7)
        plt.xlabel("Number of dimensions")
        plt.ylabel("Percentage of the original recall")
        plt.plot(components_list2, recall_list2, color="red", marker="X", markersize=7)
        return

    def compute_metrics(p):
        pred, labels = p
        pred = np.argmax(pred, axis=1)
        accuracy = accuracy_score(y_true=labels, y_pred=pred)
        recall = recall_score(y_true=labels, y_pred=pred, average="weighted")
        precision = precision_score(y_true=labels, y_pred=pred, average="weighted")
        f1 = f1_score(y_true=labels, y_pred=pred, average="weighted")
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
