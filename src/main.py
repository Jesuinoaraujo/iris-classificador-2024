import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


class Modelo:
    def __init__(self, path):
        self.path = path
        self.df = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.modelo_svm = None
        self.modelo_lr = None

    def CarregarDataset(self):
        if os.path.exists(self.path):
            print("Arquivo encontrado:", self.path)
            try:
                self.df = pd.read_csv(self.path, names=[
                                      'SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm', 'Species'])
                print("Dataset carregado com sucesso!")
            except Exception as e:  # Lidando com possíveis erros na leitura do CSV
                print(f"Erro ao carregar o dataset: {e}")
                return
        else:
            print("Arquivo não encontrado. Verifique o caminho:", self.path)
            return

    def TratamentoDeDados(self):
        if self.df is not None:
            print("Informações sobre o dataset:")
            print(self.df.info())
            print(self.df.describe())
            print(self.df.head())

            # Converter a coluna 'Species' para numérica (se necessário)
            # Verificar se a coluna é do tipo object (string)
            if self.df['Species'].dtype == object:
                species_mapping = {'Iris-setosa': 0,
                                   'Iris-versicolor': 1, 'Iris-virginica': 2}
                self.df['Species'] = self.df['Species'].map(species_mapping)

            self.X = self.df.drop('Species', axis=1)
            self.y = self.df['Species']

            scaler = StandardScaler()
            self.X = scaler.fit_transform(self.X)

    def Treinamento(self):
        if self.X is not None and self.y is not None:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X, self.y, test_size=0.3, random_state=42)

            self.modelo_svm = SVC(kernel='linear', random_state=0)
            self.modelo_svm.fit(self.X_train, self.y_train)

            self.modelo_lr = LogisticRegression(random_state=0)
            self.modelo_lr.fit(self.X_train, self.y_train)

    def Teste(self):
        if self.modelo_svm is not None and self.modelo_lr is not None:

            # Avaliar o modelo SVM
            y_pred_svm = self.modelo_svm.predict(self.X_test)
            accuracy_svm = accuracy_score(self.y_test, y_pred_svm)
            cm_svm = confusion_matrix(self.y_test, y_pred_svm)

            print("Resultados do SVM:")
            print(f"Acurácia: {accuracy_svm}")
            print("Matriz de Confusão (SVM):\n", cm_svm)

            # ... (opcional) Display da matriz com ConfusionMatrixDisplay

            print("\nRelatório de Classificação (SVM):")
            print(classification_report(self.y_test, y_pred_svm))

            # Avaliar o modelo de Regressão Logística
            y_pred_lr = self.modelo_lr.predict(self.X_test)
            accuracy_lr = accuracy_score(self.y_test, y_pred_lr)
            cm_lr = confusion_matrix(self.y_test, y_pred_lr)

            print("\nResultados da Regressão Logística:")
            print(f"Acurácia: {accuracy_lr}")
            print("Matriz de Confusão (Regressão Logística):\n", cm_lr)
            # ... (opcional) Display da matriz com ConfusionMatrixDisplay

            print("\nRelatório de Classificação (Regressão Logística):")
            print(classification_report(self.y_test, y_pred_lr))


# Criar uma instância da classe Modelo (ajuste o caminho se necessário)
modelo = Modelo(
    r"C:\Users\Jesuino\Documents\iris-classificador-2024\data\iris.csv")

# Executar as etapas
modelo.CarregarDataset()
modelo.TratamentoDeDados()
modelo.Treinamento()
modelo.Teste()
