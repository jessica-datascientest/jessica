import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#importer la base d'étude
df = pd.read_csv("reviews_trust (2).csv")



st.title("SUPPLY CHAIN")
st.sidebar.title("Sommaire")
pages=["Présentation du projet","Exploration", "DataVizualization", "Text Mining","Modélisation", "Interprétabilité", "Demo","Conclusion"]

page=st.sidebar.radio("Aller vers", pages)


###############################################################  INTRODUCTION ###############################################################

if page == pages[0] : 
  st.write("### Objectif du projet")
  st.write("L’objectif de ce projet est d’extraire de l’information des commentaires afin de :")
  st.write("    -	Prédire la satisfaction d’un client : problème de régression (prédire le nombre d'étoiles).")
  st.write("    -	Identifier les entités importantes d’un message : localisation, nom d’entreprise...")
  st.write("    -	Extraire les éléments pertinents à partir des verbatims clients les problèmes rencontrées par les clients  (problème de livraison, article défectueux...) ")
  st.write("    -	Détecter les axes d’amélioration et les anomalies au niveau de la chaine d’approvisionnement du produit ")
  st.write("    -	Se positionner par rapport à la concurrence sur le marché")
  st.write("    -	Aider les entreprises à comprendre les besoins et les frustrations des clients afin d’améliorer la satisfaction client et réduire le taux de perte.")
  
  st.write("### Démarche")
  st.write("Nous procéderons de la manière suivante :")
  st.write("    -	Nettoyage de données ")
  st.write("    -	Analyse de données ")
  st.write("    -	Text Mining ")
  st.write("    -	Modélisation ")
  st.write("    -	Interprétabilité ")
  st.write("    -	Prompt Engineering ")


###############################################################  EXPLORATION ###############################################################

if page == pages[1] : 
  st.write("### Introduction")
  st.dataframe(df.head(10))
  st.write("Dimension de la base 'Review Trust'")
  st.write(df.shape)
  st.dataframe(df.describe()) 
  if st.checkbox("Afficher les NA") :
    st.dataframe(df.isna().sum())
  df = df.dropna(axis = 0 , how="all", subset = ["Commentaire"])
  st.write("Nombre de commentaire après suppression des valeurs manquantes")
  st.write(df["Commentaire"].count())
  st.write("Nombre de doublons dans la base")
  st.write(df.duplicated().sum())

  #supprimer le dernier doubon s'il y en a
  df.drop_duplicates(keep= "first")


###############################################################  DataVizualization ############################################################### 

if page == pages[2] :
   st.write("### DataVizualization")
   fig = (plt.figure(figsize =(10,10)))
   plt.pie(df["star"].value_counts(),labels = ["5","1","4","3","2"], colors = ["red","orange","yellow","green","blue"],
            explode =[0.2,0,0,0,0],autopct = '%0.2f%%' , pctdistance =0.5,labeldistance = 1.2, shadow =True)
   plt.title("Répartition des notes")
   st.pyplot(fig)
   fig = plt.figure()

   import seaborn as sns 
   sns.countplot(x="company", hue ="source", data = df)
   plt.title("Répartition des commentaires selon la source")
   st.pyplot(fig)
   
   st.write("Comparer la distribution de la note en fonction de la compagnie Veepee ou ShowRoom")
   fig =plt.figure(figsize=(10,5))
   plt.subplot(121)
   colors = sns.color_palette('coolwarm', n_colors=5)
   data=df[df['company']=='ShowRoom'].star.value_counts(True, False)
   plt.pie(x=data, autopct="%.1f%%", labels=data.keys(), pctdistance=0.5, colors=colors)
   plt.title("ShowRoom")
   plt.subplot(122)
   colors = sns.color_palette('coolwarm', n_colors=5)
   data=df[df['company']=='VeePee'].star.value_counts(True, False)
   plt.pie(x=data, autopct="%.1f%%", labels=data.keys(), pctdistance=0.5, colors=colors)
   plt.title("VeePee")
   st.pyplot(fig)  

   st.write("Comparer la distribution de la note en fonction de la source TrustedShop ou TrustPilot")
   fig=plt.figure(figsize=(10,5))
   plt.subplot(121)
   colors = sns.color_palette('coolwarm', n_colors=5)
   data=df[df['source']=='TrustedShop'].star.value_counts(True, False)
   plt.pie(x=data, autopct="%.1f%%", labels=data.keys(), pctdistance=0.5, colors=colors)
   plt.title("TrustedShop")
   plt.subplot(122)
   data=df[df['source']=='TrustPilot'].star.value_counts(True, False)
   plt.pie(x=data, autopct="%.1f%%",labels=data.keys(), pctdistance=0.5, colors=colors)
   plt.title("TrustPilot")
   st.pyplot(fig)


   df['len_commentaire']=df['Commentaire'].apply(lambda x:len(str(x)))
   fig=plt.figure(figsize=(4,4))
   colors = sns.color_palette('coolwarm', n_colors=5)
   data=df.groupby(['star'])['len_commentaire'].mean()
   plt.pie(x=data, autopct="%.1f%%", labels=data.keys(), pctdistance=0.5, colors=colors)
   plt.title("Longueur moyenne des commentaires selon la note")
   st.pyplot(fig)
   st.write("Plus la note est négative plus les commentaires sont longs ")
   
   fig=plt.figure(figsize=(5,5))
   colors = sns.color_palette('coolwarm', n_colors=5)
   data=df.groupby(['star'])['ecart'].mean()
   plt.pie(x=data, autopct="%.1f%%", labels=data.keys(), pctdistance=0.5, colors=colors)
   plt.title("Ecart moyen entre les date de commande et du commentaire en fonction de la note")
   st.pyplot(fig)
   st.write("L'écart n'est pas proportionnel à la note ")


###############################################################  TEXT MINING ############################################################### 

def return_ngram(texts, ngram_range):
        #vectorizer = CountVectorizer(stop_words=list(stop_words), ngram_range=ngram_range)
        vectorizer = joblib.load("vectorizer_test.sav")
        vectorizer.fit(texts)
        count_list = np.array(vectorizer.transform(texts).sum(0))[0]
        count_words = list(zip(vectorizer.get_feature_names_out(), count_list))

        count_words = sorted(count_words, key=lambda x: x[1], reverse=True)

        count_words = pd.DataFrame(count_words, columns=['word', 'count'])
        return count_words


if page==pages[3] :
    st.cache_data
    st.write("### Text Mining")

   #remplacer les commertaires vides par ""
    #df.Commentaire= df.Commentaire.fillna(' ')

    from sklearn.feature_extraction.text import CountVectorizer
    from nltk.corpus import stopwords
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt 
    from wordcloud import WordCloud
    sns.set()
  
    # Définir la variable text
   
    

    # Importer stopwords de la classe nltk.corpus
    from nltk.corpus import stopwords


    from wordcloud import WordCloud
    #text = ""
  
    #for comment in df.Commentaire: 
    #   text += str(comment)
    # Définir le calque du nuage des mots

    #remplacer les commertaires vides par ""
    #df.Commentaire= df.Commentaire.fillna(' ')

    import nltk
    

    # Initialiser la variable des mots vides
    #stop_words = set(stopwords.words('french'))
    #print(stop_words)
    #stop_words.update(["commande","livraison", "article", "service client","a"," service client","service","client"])
    #print(stop_words)
    #stop_words.update([",", "."])
    #wc = WordCloud(background_color="black", max_words=10, stopwords=stop_words, max_font_size=50, random_state=42)

    import matplotlib.pyplot as plt 

  # Générer et 
    st.write("Affichage du nuage des mots les plus fréquents")
    fig =(plt.figure(figsize= (10,6)))# Initialisation d'une figure
    #wc.generate(text)           # "Calcul" du wordcloud
    import joblib
    nuage_de_mots = joblib.load("nuages_de_mots_plus_frequents_avant_nettoyage.sav")
    plt.imshow(nuage_de_mots) # Affichage
    plt.show()
    st.pyplot(fig)


    st.write("Après suppression des mots suivants : 'commande','livraison','article','service client','a','service client','service','client'")

    nuage_de_mots = joblib.load("nuage_de_mots_après_suppression_mots.sav")
    plt.imshow(nuage_de_mots) # Affichage
    plt.show()
    st.pyplot(fig)



    df_pos=df[df.star>=4]
    df_neg=df[df.star <=3]

    st.write("Nb de commentaires négatifs")
    st.write(len(df_neg))

    st.write("Nb de commentaires positifs")
    st.write(len(df_pos))



    #text_pos = "" 
    #for e in df_pos['Commentaire'] : text_pos  += str(e)
 

    #text_neg = "" 
    #for e in df_neg['Commentaire'] : text_neg  += str(e)







    # This allows to create individual objects from a bog of words
    from nltk.tokenize import word_tokenize
    # Lemmatizer helps to reduce words to the base form
    from nltk.stem import WordNetLemmatizer
    # Ngrams allows to group words in common pairs or trigrams..etc
    from nltk import ngrams
    # We can use counter to count the objects
    from collections import Counter
    # This is our visual library
    import seaborn as sns

    from sklearn.feature_extraction.text import CountVectorizer
    from nltk.corpus import stopwords
    import numpy as np
    sns.set()
    import joblib



    #Les combinaisons de mots les plus fréquentes
    display = st.radio('Que souhaitez-vous montrer  ?', ('Répartition des mots les plus fréquents', 'Répartition par groupe de 2 mots','Répartition par groupe de 3 mots'))
    if display == 'Répartition des mots les plus fréquents':


       #count_words = return_ngram(df.Commentaire, ngram_range=(1,1))
       count_words = joblib.load("count_words_1.sav")
       fig = (plt.figure(figsize=(15,10)))
       sns.barplot(x='count', y='word', data=count_words.head(10))
       plt.title("Répartition des mots les plus fréquents")
       st.pyplot(fig,display)
    elif display == 'Répartition par groupe de 2 mots':

       

       #count_words = return_ngram(df.Commentaire, ngram_range=(2,2))
       count_words = joblib.load("count_words_2.sav")
       fig =(plt.figure(figsize=(15,15)))
       sns.barplot(x='count', y='word', data=count_words.head(10))
       plt.title("Répartition par groupe de 2 mots")
       st.pyplot(fig,display)
    elif display == 'Répartition par groupe de 3 mots':
       count_words = joblib.load("count_words_3.sav")


       #count_words = return_ngram(df.Commentaire, ngram_range=(3,3))
       fig =(plt.figure(figsize=(15,15)))
       sns.barplot(x='count', y='word', data=count_words.head(10))
       plt.title("Répartition par groupe de 3 mots")
       st.pyplot(fig,display)

      

    display = st.radio('Que souhaitez-vous montrer  ?', ('Répartition des mots classés dans le sentiment positif', 'Répartition des mots classés dans le sentiment négatif'))
    if display == 'Répartition des mots classés dans le sentiment positif':
       st.write("#### Nuage de mots positifs")
       fig = (plt.figure(figsize= (10,6)))# Initialisation d'une figure
       #wc.generate(text_pos)           # "Calcul" du wordcloud
       nuage_de_mots = joblib.load("nuage_de_mots_positifs.sav")
       plt.imshow(nuage_de_mots) # Affichage
       plt.show()
       st.pyplot(fig)
    elif display == 'Répartition des mots classés dans le sentiment négatif':

       st.write("#### Nuage de mots négatifs")
       fig = (plt.figure(figsize= (10,6)))# Initialisation d'une figure
       #wc.generate(text_neg)           # "Calcul" du wordcloud
       nuage_de_mots = joblib.load("nuage_de_mots_négatifs.sav")
       plt.imshow(nuage_de_mots) # Affichage
       plt.show()
       st.pyplot(fig)



if page == pages[4] :
    st.write("### Modélisation")
    st.write("Les taux de précision (accuracy) et les matrices de confusion ci-dessous représentent les résultats pour chacun des modèles utilisés.")
  #Modélisation 
  #Classification à 2 classes 
    

  #importer le vectorizer
    import joblib

    radio_choice = st.radio("Faites votre choix :", ("Accuracy" , "Matrice de confusion"))
    if  radio_choice == "Accuracy" :
        choix =["Arbre de décision", "Gradient Boosting Classifier","Logistic Regression","KNeighbors Classifier", 'RandomForest']
        option = st.selectbox('Choix du modèle', choix)
        if option == "Logistic Regression" : 
           st.text(joblib.load("scores_reg.sav"))
        elif option == "Arbre de décision" :
           st.text(joblib.load("scores_dt.sav"))
        elif option == "Gradient Boosting Classifier":
           st.text(joblib.load("scores_gb.sav"))
        elif option == "KNeighbors Classifier":
           st.text(joblib.load("scores_knn.sav"))      
        elif option == "RandomForest": 
           st.text(joblib.load("scores_rf.sav"))        
           
    elif radio_choice == "Matrice de confusion" :
        choix =["Arbre de décision", "Gradient Boosting Classifier","Logistic Regression","KNeighbors Classifier", 'RandomForest']
        option = st.selectbox('Choix du modèle', choix)
        if option == "Logistic Regression" : 
           st.dataframe(joblib.load("confusion_matrix_reg.sav"))
        elif option == "Arbre de décision" :
           st.dataframe(joblib.load("confusion_matrix_dt.sav"))
        elif option == "Gradient Boosting Classifier":
           st.dataframe(joblib.load("confusion_matrix_gb.sav"))
        elif option == "KNeighbors Classifier":
           st.dataframe(joblib.load("confusion_matrix_knn.sav"))      
        elif option == "RandomForest":
           st.dataframe(joblib.load("confusion_matrix_rf.sav"))        
           


if page == pages[6] :
    st.title("Démo")
    #Classification à 2 classes 
    from nltk.corpus import stopwords

    # Initialiser la variable des mots vides
    stop_words = set(stopwords.words('french'))
    stop_words.update(["commande","livraison", "article", "service client","a"," service client","service","client"])
    stop_words.update(["produit"])
    st.write(stop_words)

    # 29
    #suppression des variables manquantes 
    df = df.dropna(axis = 0 , how="all", subset = ["Commentaire"])
    # Dénombrement des doublons
    doublons = df.duplicated().sum()

    #suppression de doublons 

    df.drop_duplicates(keep= "first")

    df["sentiment"]= df['star'].replace( ((1,2,3), (4,5)), (0,1) )


    # Séparer la variable explicative de la variable à prédire
    X, y = df.Commentaire, df.sentiment

    from sklearn.model_selection import train_test_split
    # Séparer le jeu de données en données d'entraînement et données test 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    import joblib 
    vectorizer = joblib.load("vectorizer_test.sav") 
    model = joblib.load("rf.sav")
    
    # Séparer la variable explicative de la variable à prédire
    X, y = df.Commentaire, df.sentiment

    from sklearn.model_selection import train_test_split
    # Séparer le jeu de données en données d'entraînement et données test 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    import joblib 
    vectorizer = joblib.load("vectorizer_test_new.sav") 
    model = joblib.load("rf_new.sav")
    
    X_train = vectorizer.fit_transform(X_train).todense()
    X_test = vectorizer.transform(X_test).todense()
    


    texte =list(df["Commentaire"])

    radio_choice = st.radio("Faites votre choix :", ("Insérer votre texte" , "Exemples"))
    if radio_choice =="Insérer votre texte" :
        text = st.text_input("Insérer votre texte", placeholder = "Texte")
        if text :
            st.text(type(text))
            vect_text = vectorizer.transform([text])
            st.text(model.predict(vect_text))
            st.text(len(text))
    elif radio_choice == "Exemples" :
        #choix =["Veepee a livré ma commande à une mauvaise adresse sans que je leur ai demandé. Suite à ma réclamation, le service client (long à réagir) ne veut rien entendre. Ni bon de réexpédition, ni geste commercial. A moi de me débrouiller pour récupérer le colis à 700km de chez moi  Je suis pourtant cliente depuis 2004!J'ai procédé à la suppression de mon compte, très en colère du mépris de ce site pour ses clients.", "Tres bon site . Produits pas chers , livraison rapide .","je suis très déçu de showroomprive c est la dernière fois que je commande le mobilier chez eux j ai acheter des chaises 1qui était cassé j ai reçu l étiquette du retour , pour les 2 autres ils m ont demandé de me débrouiller , alors que la 1 chaise elle traîne toujours chez le commerçant relais du coin parceque DPD ils déposent mais ils reprennent pas.je suis très très déçu je crois qu il est temps de me désabonner trop c est trop aucun geste commercial c est toujours nous qui acceptons tout de leurs part.En plus moi j ai acheté à showroomprive pas chez le commerçant ( Made sens ) c est pas normal c est a shroomprive de se débrouiller pour le retour comme ils ont ils ont fait avec la vente ."]
        choix = texte
        option = st.selectbox('Choix du modèle', choix)
        if choix : 
            vect_doc = vectorizer.transform([option])
            st.write("l'exemple choisi : ")
            st.write(option)

            st.write("Prédiction :")
            st.text(model.predict(vect_doc))  


            st.write("Le nombre de mots :")
            st.text(len(option))
            

    


#            vect_doc = vectorizer.transform([option])
#            st.text(model.predict(vect_doc))
#            st.text(len(option))
           
#    if "pas" in stop_words :
#        st.text("True")




if page == pages[5] : 
  st.write("### Interprétabilité SHAP")
  #from streamlit_shap import st_shap
  
  import joblib
  vectorizer=joblib.load("SP_vectorizer.sav") 
  X_train = vectorizer.fit_transform(X_train.apply(lambda x: np.str_(x)))
  X_test = vectorizer.transform(X_test.apply(lambda x: np.str_(x)))

  import joblib
  GB=joblib.load("SP_GB.sav") 
  RF=joblib.load("SP_RF.sav") 
  TD=joblib.load("SP_TD.sav") 
  SHAP_TD=joblib.load ('SP_SHAPTD.sav')
  import shap
  display = st.radio('Quel modèle souhaitez-vous montrer en SHAP ?', ("Arbre de décision", "GradientBoostingClassifier","Logistic Regression","KNeighborsClassifier"))
  if display ==  'Arbre de décision':
    #explainer_TD = shap.TreeExplainer(TD, data=X_test.toarray()) 
    shap_values_decision_tree = SHAP_TD
    fig =(plt.figure(figsize=(15,15)))
    shap.summary_plot(shap_values_decision_tree, X_test.toarray(), feature_names=vectorizer.get_feature_names_out())
    plt.title("Importance des variables en fonction de leurs valeurs Shap")
    st.pyplot(fig,display)
  elif display == 'GradientBoostingClassifier':
    explainer_GB = shap.TreeExplainer(GB, data=X_test.toarray())
    shap_values_GB = explainer_GB.shap_values(X_test.todense(), approximate=True)
    fig =(plt.figure(figsize=(15,15)))
    shap.summary_plot(shap_values_GB, X_test.toarray(), feature_names=vectorizer.get_feature_names_out())
    plt.title("Importance des variables en fonction de leurs valeurs Shap")
    st.pyplot(fig,display)
  elif display == "Logistic Regression" :
    explainer_LogisticRegression = shap.TreeExplainer(LogisticRegression, data=X_test.toarray())
    shap_values_LogisticRegression= explainer_LogisticRegression.shap_values(X_test.todense(), approximate=True)
    fig =(plt.figure(figsize=(15,15)))
    shap.summary_plot(shap_values_LogisticRegression, X_test.toarray(), feature_names=vectorizer.get_feature_names_out())
    plt.title("Importance des variables en fonction de leurs valeurs Shap")
    st.pyplot(fig,display)
  elif display == "KNeighborsClassifier" :
     explainer_KNeighborsClassifier = shap.TreeExplainer(KNeighborsClassifier, data=X_test.toarray())
     shap_values_KNeighborsClassifier= explainer_KNeighborsClassifier.shap_values(X_test.todense(), approximate=True)  
     explainer_GB = shap.TreeExplainer(GB, data=X_test.toarray())
     shap.summary_plot(shap_values_KNeighborsClassifier, X_test.toarray(), feature_names=vectorizer.get_feature_names_out())
     fig =(plt.figure(figsize=(15,15)))
     shap.summary_plot(shap_values_GB, X_test.toarray(), feature_names=vectorizer.get_feature_names_out())
     plt.title("Importance des variables en fonction de leurs valeurs Shap")
     st.pyplot(fig,display)
    

