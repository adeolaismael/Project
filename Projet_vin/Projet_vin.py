
#Création d'un modèle de ML pour prédiire le score de la qualité d'un vin

"""
        Compréhension de la problématique :
    La problématique concerne la production du vin plus précisement la qualité du vin.
    En tant producteur de vin, si vous ne voulez pas connaître une perte, il faut que le vin que vous produisez soit de bonne qualité. 
    Ainsi, en tant que producteur de vin votre vin doit être certifié et on vous attribue un score de bonne qualité.

    Il s'agit donc de savoir :  Comment améliorer d'une part la certification du vin en utilisant une approche beaucoup plus scientifique 
    et d'autre part comment aider les producteurs à anticiper sur la qualité de leur vin et ainsi avoir un bon chiffre d'affaire ?
"""

"""
    Dans ce projet, nous allons construire un modèle de machine learning pour prédire le score de qualité d'un vin. L'objectif étant de trouver les attributs nécessaires pour fabriquer 
    un vin de qualité afin d'aider les producteurs à optimiser leurs efforts.
"""

#Etape de la mise en place du modèle
"""
    1. Compréhension de la problématique
    2. Collecte des données
    3. Analyse exploratoire
    4. Pré traitement des données
    5. Modélisation
    6. Résultats et conclusion
    
"""

#Importation des librairies
import os
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# Pour faire la mise en place de ce modèle nous allons utiliser deux datasets de deux vins différents qu'on va mutaliser pour avoir un seul dataset

red=pd.read_csv("winequality-red.csv", sep=";")
print(red.head())

white=pd.read_csv("winequality-white.csv", sep=";")
print(white.head())