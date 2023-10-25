import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#importer la base d'étude
df = pd.read_csv("reviews_trust (2).csv")



st.title("SUPPLY CHAIN")
st.sidebar.title("Sommaire")
pages=["Demo"]

page=st.sidebar.radio("Aller vers", pages)


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





