#Import streamlit
import streamlit as st

#Import the packages used
import os
import pickle
import numpy as np
import pandas as pd

#Import the modules for plotting
import plotly.express as px
import plotly.graph_objs as go
import matplotlib.pyplot as plt


#Import the NLP packages
import nltk
import string
from nltk.util import ngrams
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

#Import the ML Packages
from sklearn.mixture import GaussianMixture
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples

#Import the sentence transformers
#from transformers import pipeline
#from sentence_transformers import SentenceTransformer


    


#Import the data
uploaded_file = st.file_uploader("Upload the Aha data file")

if uploaded_file is not None:
    # Unpack the data
    with uploaded_file as file:
        Save_list = pickle.load(file)
    df,Embeddings_dic,Column_list = Save_list
    
    
    #Create the sidebar
    with st.sidebar:
        #scrapped_website = st.selectbox('Which data would you like to use?',('Reddit',))
        Which_comments = st.selectbox("Select which column you'd like to use",tuple(Column_list))

    if Which_comments in ['NFR','Other']:
        st.write('There are not enough entries, printing all entries')
        for statement in df[df[Which_comments].isna()==False][Which_comments].tolist():
            #if '<a rpl=' not in str(statement):
            st.write(statement)
            st.write('--------------------------------------------------------------')
                #st.write(statement)
                #st.write('####')
    else:
        #Extract the comments from the dataframe
        All_comments = df[df[Which_comments].isna()==False][Which_comments].tolist()
        
        #Take the embeddings from the data file
        All_embeddings = Embeddings_dic[Which_comments+'_Semantic_embeddings']
        
        #Create the model
        num_clusters = 6
        gm = GaussianMixture(n_components=num_clusters, random_state=42)
        All_clust_num = gm.fit_predict(All_embeddings)
        clf = LDA(n_components=2)
        
        #Reduce the dimensionality of the embeddings
        All_lda = clf.fit_transform(All_embeddings, All_clust_num)
        
        #Relabel the clusters
        Embeddings = All_embeddings
        labels = All_clust_num
        
        #Calculate the metric on the clustering using silhouette scores
        silhouette_avg = silhouette_score(Embeddings, labels)
        silhouette_labels = silhouette_samples(Embeddings, labels)

        #Reduce the dimension
        clf = LDA(n_components=2)
        ans_lda = clf.fit_transform(Embeddings, labels)
        
        #Plot the figure of the clusters represented in 2D space
        fig = go.Figure()

        fig.add_trace(
                go.Scatter(
                    x=ans_lda[:,0],
                    y=ans_lda[:,1],
                    mode='markers',
                    marker=dict(color=np.array(px.colors.sequential.Plasma)[labels%10],size=10)
                    #marker=dict(color=labels,size=10)
                )
            )

        fig.update_layout(
            autosize=False,
            width=700,
            height=500,
            margin=dict(
                l=50,
                r=50,
                b=100,
                t=100,
                pad=4
            ),
        )

        #fig.show()
        st.plotly_chart(fig)
        
        
        
        
        #Plot the figure of the cluster metrics
        fig = go.Figure()
        named_colorscales = px.colors.named_colorscales()

        # The silhouette coefficient can range from -1, 1 but in this example all
        # lie within [-0.1, 1]
        x_min = -0.1
        x_max = 1
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        y_min = 0
        
        #if Which_comments == 'All':
        y_max = len(All_comments) + (num_clusters + 1) * 10
        #if Which_comments == 'Positive':
        #    y_max = len(Pos_comments) + (num_clusters + 1) * 10
        #if Which_comments == 'Negative':
        #    y_max = len(Neg_comments) + (num_clusters + 1) * 10

        y_lower = 10
        for i in range(num_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = silhouette_labels[labels == i]
            ith_cluster_silhouette_values.sort()
            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i
            #color = f'rgb{cm.nipy_spectral(float(i) / num_clusters)[:3]}'
            #color = named_colorscales[0]

            fig.add_trace(go.Scatter(
                x=ith_cluster_silhouette_values,
                y=np.arange(y_lower, y_upper),
                mode='lines',
                fill='tozerox',
                fillcolor=px.colors.sequential.Plasma[i%10],
                line=dict(color=px.colors.sequential.Plasma[i%10],width=0.5),
                showlegend=False
            ))

            # Label the silhouette plots with their cluster numbers at the middle
            fig.add_trace(go.Scatter(
                x=[-0.05],
                y=[y_lower + 0.5 * size_cluster_i],
                text=[str(i+1)],
                mode='text',
                showlegend=False
            ))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        # The vertical line for average silhouette score of all the values
        fig.add_shape(
            type="line",
            x0=silhouette_avg, y0=y_min, x1=silhouette_avg, y1=y_max,
            line=dict(color="Red", dash="dash")
        )

        fig.update_layout(
            #title="The silhouette plot for the various clusters.",
            xaxis=dict(
                title="The silhouette coefficient values",
                range=[x_min, x_max],
                tickvals=[-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1]
            ),
            yaxis=dict(
                title="Cluster label",
                range=[y_min, y_max],
                showticklabels=False
            ),
            width=700,
            height=500
        )

        #fig.show()
        st.plotly_chart(fig)
        
        
        
        #Redo the metric with the word counts and number of elements in the cluster
        ans_list_eng = All_comments
        
        
        #Do the text analysis
        ans_clustered = [[] for i in list(set(labels))]
        ans_clustered_eng = [[] for i in list(set(labels))]
        cluster_metric = [[] for i in list(set(labels))]

        for i in list(set(labels)):
            for j in range(len(ans_list_eng)):
                if labels[j] == i:
                    #ans_clustered[i].append(ans_list[j])
                    ans_clustered_eng[i].append(ans_list_eng[j])
                    cluster_metric[i].append(silhouette_labels[j])



        n_gram = 2
        nltk.download('stopwords')
        nltk.download('punkt')

        #if Which_comments == 'All':
        stop_words = set(stopwords.words('english'))#|{'ahahaahah','regeneron','ahahahahah','classrelative','pointereventsauto','nofollow','ugc','relnoopener','rpl','hi'}
        #if Which_comments == 'positive':
        #    stop_words = 

        exclude = set(string.punctuation)|{'・','→'}

        #ans_clustered_bog = [[] for i in list(set(labels))]
        ans_clustered_eng_bog = [[] for i in list(set(labels))]

        for i in range(len(ans_clustered_eng)):
            for j,statement in enumerate(ans_clustered_eng[i]):
                print(str(i+1)+'/'+str(len(ans_clustered_eng))+'-'+str(j+1)+'/'+str(len(ans_clustered_eng[i]))+'          ',end='\r')
                s = ''.join(ch for ch in str(statement) if ch not in exclude)
                word_tokens = word_tokenize(s)
                word_tokens_stopwords = [w.lower() for w in word_tokens if not w.lower() in stop_words]
                filtered_sentence = ngrams(word_tokens_stopwords,n_gram)
                ans_clustered_eng_bog[i] += filtered_sentence

        word_counts = []
        for i in range(len(ans_clustered_eng)):
            df_tmp = pd.DataFrame(ans_clustered_eng_bog[i])
            df_tmp['words'] = df_tmp[df_tmp.columns].agg(' '.join, axis=1)
            df_tmp = df_tmp[['words']]
            word_counts.append([df_tmp['words'].value_counts()[:10]])


        #Give a representation of how accurate each cluster is
        #st.subheader('Further analysis of the groups')
        #st.write('The graph below rates the groups on 3 factors, the size of each group, how closely aligned each froup is semantically and the number of commonly occuring words.')
        
        #Produce the new plot showing the cluster rankings
        fig = go.Figure()

        fig = px.bar(
                        x=np.arange(num_clusters)+1,
                        #y=[np.mean(i)**(1/3) for i in cluster_metric],
                        y=[np.mean(cluster_metric[i])**(1/3)*len(ans_clustered_eng[i])*sum(word_counts[i][0].values)**(1/3) for i in range(len(cluster_metric))],
                        #y=[np.mean(cluster_metric[i])*len(ans_clustered_eng[i]) for i in range(len(cluster_metric))],
                        color=[np.array(px.colors.sequential.Plasma)[i%10] for i in range(num_clusters)],
                        color_discrete_map="identity"
                    )

        fig.update_layout(
            autosize=False,
            width=700,
            height=500,
            margin=dict(
                l=50,
                r=50,
                b=100,
                t=100,
                pad=4
            ),
        )

        fig.update_layout(
            #title="The silhouette plot for the various clusters.",
            xaxis=dict(
                title="Cluster label"
            ),
            yaxis=dict(
                title="Cluster rating",
                showticklabels=False
            )
        )

        #fig.show()
        st.plotly_chart(fig)
        
        
        #Add something to investigate the individual clusters
        cluster_to_inspect = st.selectbox('Which cluster would you like to inspect',(i+1 for i in range(num_clusters)))
        
        
        comments_of_interest = []
        for i in range(len(All_comments)):
            if All_clust_num[i] == cluster_to_inspect-1:
                comments_of_interest.append(str(All_comments[i]))
                #print(Pos_comments[i])

        n_gram = 2
        nltk.download('stopwords')
        nltk.download('punkt')
        #stop_words = set(stopwords.words('english'))#|{'would','like'}
        stopwords = nltk.corpus.stopwords.words('english')
        #stopwords.append('’')
        exclude = set(string.punctuation)#|{'・','→'}

        #ans_clustered_bog = [[] for i in list(set(labels))]
        ans_clustered_eng_bog = []

        #for i in range(len(ans_clustered_eng)):
        for j,statement in enumerate(comments_of_interest):
            #print(str(i+1)+'/'+str(len(ans_clustered_eng))+'-'+str(j+1)+'/'+str(len(ans_clustered_eng[i]))+'          ',end='\r')
            s = ''.join(ch for ch in statement if ch not in exclude)
            word_tokens = word_tokenize(s)
            word_tokens_stopwords = [w.lower() for w in word_tokens if not w.lower() in stopwords]
            filtered_sentence = ngrams(word_tokens_stopwords,n_gram)
            ans_clustered_eng_bog += filtered_sentence

        word_counts = []
        #for i in range(len(ans_clustered_eng)):
        df_tmp = pd.DataFrame(ans_clustered_eng_bog)
        df_tmp['words'] = df_tmp[df_tmp.columns].agg(' '.join, axis=1)
        df_tmp = df_tmp[['words']]
        word_counts.append([df_tmp['words'].value_counts()[:10]])
        st.write(word_counts)

        #word_counts[cluster_to_inspect-1][0]
        #word_counts

        for statement in ans_clustered_eng[cluster_to_inspect-1]:
            #if '<a rpl=' not in str(statement):
            st.write(statement)
            st.write('--------------------------------------------------------------')
                #st.write(statement)
                #st.write('####')


























