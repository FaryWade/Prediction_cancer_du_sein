#  Prediction de survie des patients atteint du cancer du sein 

Ce projet vise à construire un modèle de machine learning capable de prédire le statut d'un patient atteint de cancer du sein (vivant ou décédé) à partir de caractéristiques cliniques.  

L'application Streamlit permet une saisie simplifiée des données et fournit une prédiction instantanée.

---

## Contenu du projet

- `prediction_cancer_du_sein.py` : code principal de l'application Streamlit  
- `Breast_Cancer.csv` : jeu de données utilisé pour entraîner le modèle  
- `requirements.txt` : liste des bibliothèques nécessaires

---

## Jeu de données

Le jeu de données contient des informations sur 4 024 patients atteints de cancer du sein, dont :  

- Âge  
- Taille de la tumeur  
- Nombre de ganglions examinés  
- Nombre de ganglions positifs  
- Stade T, N, 6th, A  
- Différenciation  
- Grade  
- Statut des récepteurs hormonaux  
- Mois de survie  

La variable cible est :  

- `Status` : 0 = vivant, 1 = décédé  

---

## Modèles utilisés

- **Régression logistique**
- **K-Nearest Neighbors (KNN)**

 **Résultat final :**  
Le modèle de régression logistique a été retenu car il présente :  
- une précision générale de **89 %**  
- un meilleur rappel pour la classe décès (43 %) comparé au KNN (18 %)

---
## Auteur

FARIMATA WADE

## Déploiement

Cette application est déployée sur Streamlit Cloud.
Lien de l'application: https://predictioncancerdusein-932owh97axz7lyybkri7kz.streamlit.app/
