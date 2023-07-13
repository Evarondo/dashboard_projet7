# **Projet 7 : Implémentez un modèle de scoring/dashboard**
## <u>Mission</u>
En tant que Data Scientist au sein de l'entreprise "Prêt à dépenser", proposant des crédits à la consommation pour des personnes ayant peu ou pas du tout d'historique de prêt, l'idée est de mettre en oeuvre un outil de "**scoring crédit**" pour calculer la probabilité qu'un client rembourse son crédit, et ainsi, de classifier la demande en crédit *accordé* ou *refusé*. Pour cela, il est nécessaire de développer un algorithme de classification en s'appuyant sur différentes sources de données (comportement, provenant d'institutions financières, ...).
La création d'un **dashboard interactif** est développé pour plus de transparence lors de l'octroi de crédit et pour que les clients aient accès à leurs données personnelles et puissent les explorer plus facilement. 

Pour cela, le modèle de scoring de prédiction est mis en production à l'aide d'une API, puis le dashboard interactif appelle l'API pour les prédictions. 

## <u>Données</U>
Pour mener à bien le projet, différents fichiers .csv contenant les informations nécessaires sont téléchargés [ici](https://www.kaggle.com/c/home-credit-default-risk/data). Les fichiers sont les suivants:
- Fichier HomeCredit_columns_description.csv
- Fichier application_train.csv
- Fichier application_test.csv
- Fichier bureau.csv
- Fichier bureau_balance.csv
- Fichier credit_card_balance.csv 
- Fichier installments_payments.csv
- Fichier POS_CASH_balance.csv
- Fichier previous_application.csv
- Fichier sample_submission.csv

## <u>Description du répertoire</u>
Le répertoire contient d'abord le notebook jupyter de nettoyage et modélisation : `Projet7_nettoyage_modelisation.ipynb`, ainsi que la ``Note_methodologique.pdf`` et d'un fichier `data_processing.py`.

Le dossier "dashboard" est constitué d'un fichier `dashbaord_streamlit.py` permettant l'affichage du tableau de bord interactif contenant les informations clients avec un menu déroulant permettant de sélectionner l'identifiant client voulu et affichant les informations relatives à ce client ainsi qu'une jauge montrant la position du score du client par rapport à un seuil optimal, et ainsi, affichant si le prêt du client est accepté ou refusé. Des graphiques complémentaires sont affichés (feature importance globale et locale, distribution selon la feature sélectionnée et graphique bi-varié entre 2 features). 

Un fichier `setup.sh` contient les instructions de configuration Streamlit.

## <u>Lancement du dashboard en local</u>
Une fois le fichier `dashboard_streamlit.py` enregistré et l'url spécifiée pour effectuer une requête http GET et récupérer les informations d'un client à partir de l'api, pour effectuer le lancement en local du dashboard, dans un terminal, taper la commande : `streamlit run dashboard_streamlit.py`.

## <u>Déploiement du dashboard sur Heroku</u>
Pour le déploiement sur Heroku, les étapes sont les mêmes que pour le déploiement de l'API, à savoir :
* Dans "dashboard", cliquer sur `New` > `Create new app`
* Donner un nom à l'api
* Préciser la méthode de déploiement (GitHub dans notre cas)
* Connecter à GitHub
* Choisir le déploiement automatique
* Déployer la branche `Deploy Branch`
* Lorsque le déploiement est fait, cliquez sur `Open app` dirigeant directement sur l'url de l'application