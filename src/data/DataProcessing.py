import pandas as pd
from itertools import islice
from tqdm.auto import tqdm
import spacy
from nltk.corpus import stopwords as nltk_stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from enelvo.normaliser import Normaliser
from sklearn.model_selection import train_test_split
import pickle


class DataProcessing:
    vect = None

    def import_vect(self, vectorizer):
        self.vect = vectorizer

    def export_vect(self):
        pickle.dump(self.vect, open("tfidf.pickle", "wb"))

    # Junta os dataframes dentro do vetor de dataframes
    def append_data(self, df_vector):
        df_final = df_vector[0]
        df_final = df_final[["text", "Total(SUM)", "Classe de Violência"]]
        for df in islice(df_vector, 1, None):
            df = df[["text", "Total(SUM)", "Classe de Violência"]]
            df_final = pd.concat([df_final] + [df])
        df_final.reset_index()
        df_final["text"] = df_final["text"].astype("str")
        return df_final

    # Normalização usando spacy
    def text_normalizer_spacy(self, corpus):
        nlp = spacy.load("pt_core_news_sm", disable=["parser", "ner"])
        lemm = []
        for text in tqdm(corpus):
            doc = nlp(text)
            # tokens = [token.lemma_ for token in doc if not token.is_stop]
            tokens = [token.lemma_ for token in doc]
            text = " ".join(tokens)

            lemm.append(text)
        return lemm

    # Normalização usando enelvo e vetorização usando nltk
    def text_preprocessing_nltk(self, corpus):
        stop_words = list(nltk_stopwords.words("portuguese"))
        norm = Normaliser(tokenizer="readable", sanitize=True)
        lemm = []
        for texts in corpus:
            lemm.append(norm.normalise(texts))
        if self.vect is None:
            self.vect = TfidfVectorizer(stop_words=stop_words)
            self.vect.fit(corpus)
        processed = self.vect.transform(lemm)
        return processed

    # Vetorização usando nltk
    def text_preprocessing_nltk_no_norm(self, corpus):
        stop_words = list(nltk_stopwords.words("portuguese"))
        if self.vect is None:
            self.vect = TfidfVectorizer(stop_words=stop_words)
            self.vect.fit(corpus)
        processed = self.vect.transform(corpus)
        return processed

    # Mudar target para valor numerico
    def numerical_target(self, target):
        # Acts as a pointer, be careful
        target.replace("Not Violence", 0, inplace=True)
        target.replace("Low", 1, inplace=True)
        target.replace("Medium", 2, inplace=True)
        target.replace("High", 3, inplace=True)
        target.replace("VeryHight", 4, inplace=True)
        return target

    # fraction é a fração que vai sobrar do original, deve ser colocado um valor entre 0 e 1
    # Se usado 0.3 por exemplo, perderemos 60% dos registros daquele target, sobrando 30 porcento
    def downsample(self, features, target, fraction, value):
        features_true = features[target == value]
        features_false = features[target != value]
        target_true = target[target == value]
        target_false = target[target != value]

        features_downsampled = pd.concat(
            [features_true.sample(frac=fraction, random_state=12345)] + [features_false]
        )
        target_downsampled = pd.concat(
            [target_true.sample(frac=fraction, random_state=12345)] + [target_false]
        )

        return features_downsampled, target_downsampled

    # repeat é o numero de vezes que aquele target sera clonado, deve ser um int maior que 1
    def upsample(self, features, target, repeat, value):
        features_true = features[target == value]
        features_false = features[target != value]
        target_true = target[target == value]
        target_false = target[target != value]

        features_upsampled = pd.concat([features_false] + [features_true] * repeat)
        target_upsampled = pd.concat([target_false] + [target_true] * repeat)

        return features_upsampled, target_upsample

    def division(self, df_final):
        df_final = df_final.drop_duplicates().reset_index()
        features = self.text_preprocessing_nltk(df_final["text"])
        target = df_final["Classe de Violência"]
        train_data, test_data, train_target, test_target = train_test_split(
            features,
            target,
            test_size=0.3,
            random_state=12345,
            shuffle=True,
            stratify=target,
        )

        class_size = [
            len(target[target == "Not Violence"]) / len(target),
            len(target[target == "Low"]) / len(target),
            len(target[target == "Medium"]) / len(target),
            len(target[target == "High"]) / len(target),
            len(target[target == "VeryHight"]) / len(target),
        ]
        class_name = ["Not Violence", "Low", "Medium", "High", "VeryHight"]
        print(target.unique())
        print(len(target[target == "Not Violence"]))
        print(len(target[target == "Low"]))
        print(len(target[target == "Medium"]))
        print(len(target[target == "High"]))
        print(len(target[target == "VeryHight"]))
        print(len(target))

        plt.bar(class_name, class_size)
        plt.title("Distribution of classes in the dataset")
        plt.xlabel("Name of the class")
        plt.ylabel("Percentage of the class in the dataset")
        plt.figtext(0.6, 0.8, "Size of the dataset is 1758 entries")
        plt.grid(axis="y")
        plt.tight_layout()
        plt.show()

        return df_final, train_data, test_data, train_target, test_target

    def upsampled_division(self, df_final):
        df_sampled, garbage = train_test_split(
            df_final.drop_duplicates().reset_index(),
            test_size=0.3,
            random_state=12345,
            shuffle=True,
            stratify=df_final["Classe de Violência"],
        )
        features_sampled, target_sampled = self.upsample(
            df_sampled["text"], df_sampled["Classe de Violência"], 240, "Low"
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
        features_sampled = self.text_preprocessing_nltk(features_sampled)
        return features_sampled, target_sampled

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
