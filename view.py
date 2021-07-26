
#shell : streamlit run view.py 

import streamlit as st
from composants.lda_models import Lda_model

#set streamlit
st.set_page_config(layout="wide")


#set_title
st.title('Generateur des IA de propositions de films')

#create object
lda_model = Lda_model()
df = lda_model.df

#show df
st.markdown(
    """
    ---
    ### Voir le résultat du scrapping sur Les 1000 meilleurs films du site IMDB
    ---
    
    """
)
st.write(df)

#delete some words presents in each topics
lda_model.delete_words()

#show top words for each topics
st.markdown(
    """
    ---
    ### Voir les top_words pour chaque topics du classement via lda
    ---
    """)
#split dataset
train_shape, test_shape = lda_model.split()

#create model
train, topics_descriptions = lda_model.model_creation()

for topic_description in topics_descriptions:
    st.write(topic_description)
    st.write('')

#plot top words 1
st.markdown(
    """
    ---
    ### Visualisation de la pertinence des classements en topics pour chaque film
    ---
    """)
st.pyplot(lda_model.the_plot_proba(train))

#plot top words 2
st.markdown(
    """
    ---
    ### Visualisation de l'importances des tops words pour chaque topic
    ---
    """)
st.pyplot(lda_model.the_plot_main_word(train))

#plot top words 3
st.markdown(
    """
    ---
    ### Visualisation des nuages de tops words pour chaque topic
    ---
    """)
st.pyplot(lda_model.the_plot_wordcloud(train))

#interract for classement
st.markdown(
    """
    ---
    ### Classer les topics selon vos préférences
    ---
    """)
viz_topics = lda_model.viztopics(train)

cols = st.beta_columns(3)
vars = [0,0,0]

cols[0].dataframe(viz_topics[['topic1']], 700, 500)
choix_0 = cols[0].selectbox(f'topics {1}',[1,2,3], index = 0)
if choix_0:
    vars[0] = choix_0

cols[1].dataframe(viz_topics[['topic2']], 700, 500)
choix_1 = cols[1].selectbox(f'topics {2}',[1,2,3], index = 1)
if choix_1:
    vars[1] = choix_1

cols[2].dataframe(viz_topics[['topic3']], 700, 500)
choix_2 = cols[2].selectbox(f'topics {3}',[1,2,3], index = 2)
if choix_2:
    vars[2] = choix_2

validation = st.button('validation')

if validation:
    if len(set(vars)) != 3:
        st.write('Mettez un choix unique')
    else :

        dico = {
            'topic1':f"User Preference ({vars[0]})",
            'topic2':f"User Preference ({vars[1]})",
            'topic3':f"User Preference ({vars[2]})"
            }
        lda_model.topic_mapping(dico)

        #Show training samples with user choice topic name
        st.markdown(
            """
            ---
            ### Visualise les données d'entrainement selon les préférences de l'utilisateurs
            ---
            """)
        st.write(lda_model.viz_topics_user_choice())

        #Predict X_test with user choice topic name
        st.markdown(
            """
            ---
            ### Prédiction sur les données tests selon les préférences de l'utilisateurs
            ---
            """)
        st.write(lda_model.predict_topics_user_choice())


        





        
    
