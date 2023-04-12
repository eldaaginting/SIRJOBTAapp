from os import path, listdir
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import pdfplumber
import streamlit as st
import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords 
from string import punctuation
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from nltk.probability import FreqDist
from nltk.collocations import *
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from streamlit_option_menu import option_menu
from os import path, listdir
import glob
from collections import Counter
from sklearn.metrics.pairwise import linear_kernel
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Ignore warning
st.set_option('deprecation.showPyplotGlobalUse', False)
# Set wide layout
st.set_page_config(
    page_title = 'SIRJOBTA (Sistem Rekomendasi Job&Talent)',
    page_icon = '✅',
    layout = 'wide'
)


# Search Recommendation Job&Talent  
def main():
    #harozontal menu
    selected = option_menu(
        menu_title="SIRJOBTA",
        options=["Home",  "Exploration Job & Talent", "Recommendation Job & Talent", "Search jobs", "Developer"],
        icons=["house",  "bar-chart-line", "hand-thumbs-up", "search", "people-fill"],
        menu_icon="globe",
        default_index=0,
        orientation="horizontal",
        styles={
                    "container": {"padding": "0!important", "background-color": "#fafafa"},
                    "icon": {"color": "orange", "font-size": "25px"},
                    "nav-link": {
                        "font-size": "13px",
                        "text-align": "center",
                        "margin": "0px",
                        "--hover-color": "#eee",
                    },
                    "nav-link-selected": {"background-color": "green"},
                },
        )
        
    #Home
    if selected == "Home":
        # import Image 
        from PIL import Image
        img = Image.open("data/m1.png")
        st.date_input("")
        st.title('Welcome to SIRJOBTA')
        st.markdown("""------------------------------------------------------------------------------------------------""")
        
        st.image(img, width=1000)
        
        st.markdown("""------------------------------------------------------------------------------------------------""")
       
        imgm2 = Image.open("data/m2.png")
        st.image(imgm2, width=1000)
        st.markdown("""------------------------------------------------------------------------------------------------""")
        
        imgm3 = Image.open("data/m3.png")
        st.image(imgm3, width=1000)
        st.markdown("""------------------------------------------------------------------------------------------------""")
        
        # Sreaping
        imgm4 = Image.open("data/m4.png")
        st.image(imgm4, width=1000)
        
        st.markdown("""------------------------------------------------------------------------------------------------""")
        #proses
       
        img_convert = Image.open("data/m5.png")
        st.image(img_convert, width=1000)
        
        st.write("**Untuk mencari rekomendasi kerja Berdasarkan CV:**") 
        st.markdown("""
        * Pertama, pilih menu recommendation job&talent 
        * Masukkan CV dengan format pdf, kemudian akan diproses oleh sistem. 
        * Setelah itu, CV dan lowongan pekerjaan akan menjalani pemrosesan. 
        * Kemudian CV dan lowongan kerja akan dibandingkan dengan berbagai metode untuk menemukan kesamaan. 
        * Terakhir, sistem akan mencantumkan rekomendasi pekerjaan.
        

        **Untuk mencari lowongan kerja berdasarkan Search Jobs :** 
        * Pertama, Pilih menu search jobs 
        * Masukkan Job Tittle yang mau dicari
        * Masukkan jumlah rekomendasi pencarian yang diinginkan 
        * Kemudian akan diproses oleh sistem  
        * Setelah itu, input pencarian dan lowongan pekerjaan akan menjalani pemrosesan. 
        * Input pencarian dan lowongan kerja akan dibandingkan dengan berbagai metode untuk menemukan kesamaan. 
        * Terakhir, sistem akan mencantumkan hasil rekomendasi pekerjaan.
       
        ------------------------------------------------------------------------------------------------
        """)
        #DATASET
        st.write("**Dataset SIRJOBTA**")  # add a title
        st.write("Dataset yang kami gunakan diambil dari situs www.techinasia.com, kami mengambil data keseluruhan job yang di posting pada tahun 2022 dengan spesifikasi Years of Experience terdiri dari: Less than 1 years, 1 to 4 years, 4 to 7 years, 7 to 10 years, dan More than 10 years. Kami memiliki 16 kolom pada dataset yang telah kami scraping.")
        df = pd.read_csv("data/Dataset Techinasia Preprocessing.csv")  # read a CSV file inside the 'data"
        
        st.write(df)  # visualize my dataframe in the Streamlit app
        
        st.markdown("""

        **Kolom berikut menjelaskan data sebagai berikut.**

        *   Index: Indeks setiap baris (self-ecplanatory) mulai dari 0

        *   Job Type: Tipe dari pekerjaan yang dibutuhkan (Fulltime, Contract, Internship, dll)

        *   Job Experience: Total pengalaman yang dibutuhkan
        
        *   Position: Posisi dari pekerjaan yang dibutuhkan

        *   Vacancy Count: Jumlah lowongan yang dibutuhkan

        *   Company: Nama perusahaan yang membuka lowongan

        *   Location: Lokasi perusahaan yang membuka lowongan

        *   Industries: Industri dari perusahaan yang membuka lowongan

        *   Job Title: Judul pekerjaan yang sedang dibuka lowongan

        *   Job Requirement: Persyaratan yang dibutuhkan dari masing-masing pekerjaan

        *   Job Salary: Rentang salary dari masing-masing pekerjaan

        *   Skills: Keterampilan yang dibutuhkan dari masing-masing pekerjaan
        *   Career Level: Career level dari pekerjaan yang dibutuhkan
        *   Salary Min: Minimum salary
        *   Salary Max: Maximum Salary
        *   Date Created: Tanggal pekerjaan diiklankan
        *   Job Posting Link: Url setiap postingan pekerjaan di situs techinasia.com
        ------------------------------------------------------------------------------------------------
        """)
        st.write("**Thanks To:**")
        col1,col2,col3, col4, col5=st.columns(5)
        from PIL import Image
        img_km  = Image.open("data/km.png")
        img_orbit = Image.open("data/orbit.png")
        img_uwhs = Image.open("data/uwhs.jpg")
        img_um = Image.open("data/um.jpg")
        img_tec = Image.open("data/tec.jpg")
        col1.image(img_km,width=110)
        col2.image(img_orbit,width=180)
        col3.image(img_uwhs,width=100)
        col4.image(img_um,width=110)
        col5.image(img_tec,width=110)
        st.snow()
    
    # Exploration Job&Talent
    elif selected == "Exploration Job&Talent": 
        st.date_input("")
        # dashboard title
        st.title("Exploration Data Science Job & Talent")
        st.markdown("""------------------------------------------------------------------------------------------------""")
        # read csv 
        df = pd.read_csv("data/Dataset Techinasia Preprocessing.csv")
        df1 = df.copy()
        
        #Trends For All Indonesia
        st.header('Trends For All Indonesia: www.techinasia.com:')
        st.markdown("""
        """)

        companies = df['Company'].value_counts()
        companies = dict(companies)
        list1 = companies.keys()
        list2 = companies.values()
        companies_df = pd.DataFrame(list(zip(list1,list2)), columns=['Company','count'])

        fig = px.bar(companies_df[0:20],y='Company', x='count', text='count',orientation='h',
                    labels={'count':'Count'}, color='count', color_continuous_scale = 'Viridis') 

        fig.update_traces(textposition='outside')
        fig.update_layout(title_text="<b>Top Companies </b>",
                         title_font_size=25,
                         title_font_color='green',
                         title_font_family='Titillium Web',
                         title_x=0.6,
                         title_y=0.95,
                         title_xanchor='center',
                         title_yanchor='top',
                         yaxis={'categoryorder':'total ascending'}
                         )

        fig.update_xaxes(
                color='teal',
                title_text='Jobs',
                title_font_family='Open Sans',
                title_font_size=20,
                title_font_color='maroon',
                title_standoff = 15,
                showgrid=False,
                linecolor='red',
                linewidth=3,
                mirror=True)

        fig.update_yaxes(
                color='Teal',
                title_text='Company',
                title_font_family='Droid Sans',
                title_font_size=20,
                title_font_color='maroon',
                title_standoff = 15,
                tickfont_family='Arial',
                nticks = 20,
                showgrid=False,
                linecolor='red',
                linewidth=3,
                mirror = True)
        st.plotly_chart(fig)
        
        
        #Top Job Titles
        titles = df['Job Title'].value_counts()
        titles = dict(titles)
        list1 = titles.keys()
        list2 = titles.values()
        titles_df = pd.DataFrame(list(zip(list1,list2)), columns=['Job Title','count'])
        fig = px.bar(titles_df[0:20],y='Job Title', x='count', text='count',orientation='h',
                    labels={'count':'Count'}, color='count', color_continuous_scale = 'Viridis') 
        fig.update_traces(textposition='outside')
        fig.update_layout(title_text="<b>Top Job Titles </b>",
                         title_font_size=25,
                         title_font_color='green',
                         title_font_family='Titillium Web',
                         title_x=0.6,
                         title_y=0.95,
                         title_xanchor='center',
                         title_yanchor='top',
                         yaxis={'categoryorder':'total ascending'}
                         )
        fig.update_xaxes(
                color='teal',
                title_text='Jobs',
                title_font_family='Open Sans',
                title_font_size=20,
                title_font_color='maroon',
                title_standoff = 15,
                showgrid=False,
                tickmode='auto',
                linecolor='red',
                linewidth=3,
                mirror=True)
        fig.update_yaxes(
                color='Teal',
                title_text='Title',
                title_font_family='Droid Sans',
                title_font_size=20,
                title_font_color='maroon',
                title_standoff = 15,
                tickfont_family='Arial',
                nticks = 20,
                showgrid=False,
                linecolor='red',
                linewidth=3,
                mirror = True)
        st.plotly_chart(fig)
        
        #Top Locations 
        locations = df['Location'].value_counts()
        locations = dict(locations)
        list1 = locations.keys()
        list2 = locations.values()
        locations_df = pd.DataFrame(list(zip(list1,list2)), columns=['Location','count'])
        fig = px.bar(locations_df[0:20],y='Location', x='count', text='count',orientation='h',
                    labels={'count':'Count'}, color='count', color_continuous_scale = 'Viridis') 
        fig.update_traces(textposition='outside')
        fig.update_layout(title_text="<b>Top Locations </b>",
                         title_font_size=25,
                         title_font_color='green',
                         title_font_family='Titillium Web',
                         title_x=0.6,
                         title_y=0.95,
                         title_xanchor='center',
                         title_yanchor='top',
                         yaxis={'categoryorder':'total ascending'}
                         )
        fig.update_xaxes(
                color='teal',
                title_text='Jobs',
                title_font_family='Open Sans',
                title_font_size=20,
                title_font_color='maroon',
                title_standoff = 15,
                showgrid=False,
                tickmode='auto',
                linecolor='red',
                linewidth=3,
                mirror=True)
        fig.update_yaxes(
                color='Teal',
                title_text='Location',
                title_font_family='Droid Sans',
                title_font_size=20,
                title_font_color='maroon',
                title_standoff = 15,
                tickfont_family='Arial',
                nticks = 20,
                showgrid=False,
                linecolor='red',
                linewidth=3,
                mirror = True)
        st.plotly_chart(fig)
        
        #Job Titles For All Data Science
        st.subheader('Job Titles For All Data Science Related Roles')
        title_list = df['Job Title'].values.tolist()
        count = Counter(title_list)
        wordcloud = WordCloud(width = 1600, height = 800, background_color='white')\
        .generate_from_frequencies(count)
        plt.figure(figsize=(40,30))
        plt.imshow(wordcloud)
        plt.axis("off")
        plt.show() 
        st.pyplot()


       # Company Hiring For All Data Science 
        st.subheader('Company Hiring For All Data Science Related Roles')
        company_list = df['Company'].values.tolist()
        count = Counter(company_list)
        wordcloud = WordCloud(width = 1000, height = 500, background_color='lightblue')\
        .generate_from_frequencies(count)
        plt.figure(figsize=(40,30))
        plt.imshow(wordcloud)
        plt.axis("off")
        plt.show() 
        st.pyplot()
        st.snow()
        st.markdown("""------------------------------------------------------------------------------------------------""")
        st.title("Search For Job&Talent Exploration data by Location and Position")
        st.markdown("""------------------------------------------------------------------------------------------------""")
        # top-level filters 
        loc = dict(df1['Location'].value_counts()[0:100])
        loc_list = list(loc.keys())
        location = st.selectbox('Location',loc_list)
        df1 = df1.loc[df1['Location'] == location]

        tittle = dict(df1['Job Title'].value_counts()[0:100])
        tittle_list = list(tittle.keys())
        title = st.selectbox('Position',tittle_list)
        df1 = df1.loc[df1['Job Title'] == title]
        st.markdown("""------------------------------------------------------------------------------------------------""")
        st.markdown("""


        """)
        st.header('Hiring Trends For ' + title + ' role in ' + location + ':')
        

    
        #Top Company Hiring
        companies = df1['Company'].value_counts()
        companies = dict(companies)
        list1 = companies.keys()
        list2 = companies.values()
        companies_df = pd.DataFrame(list(zip(list1,list2)), columns=['Company','count'])
        fig = px.bar(companies_df[0:10],y='Company', x='count', text='count',orientation='h',
                    labels={'count':'Count'}, color='count', color_continuous_scale = 'Viridis') 
        fig.update_traces(textposition='outside')
        fig.update_layout(title_text="<b>Top Company Hiring</b>",
                         title_font_size=30,
                         title_font_color='green',
                         title_font_family='Titillium Web',
                         title_x=0.57,
                         title_y=0.95,
                         title_xanchor='center',
                         title_yanchor='top',
                         yaxis={'categoryorder':'total ascending'}
                         )
        fig.update_xaxes(
                color='teal',
                title_text='Jobs',
                title_font_family='Open Sans',
                title_font_size=20,
                title_font_color='maroon',
                title_standoff = 15,
                tickmode='auto',
                showgrid=False,
                linecolor='red',
                linewidth=3,
                mirror=True)
        fig.update_yaxes(
                color='Teal',
                title_text='Company',
                title_font_family='Droid Sans',
                title_font_size=20,
                title_font_color='maroon',
                title_standoff = 15,
                showgrid=False,
                tickfont_family='Arial',
                linecolor='red',
                linewidth=3,
                mirror = True)
        st.plotly_chart(fig)
        
        
        #Company worcloud
        company_list = df1['Company'].values.tolist()
        count = Counter(company_list)
        wordcloud = WordCloud(width = 1000, height = 500, background_color='lightblue')\
        .generate_from_frequencies(count)
        plt.figure(figsize=(40,30))
        plt.imshow(wordcloud)
        plt.axis("off")
        plt.show() 
        st.subheader('Company Hiring For ' + title + ' role in ' + location)
        st.pyplot()
        
        st.markdown("""------------------------------------------------------------------------------------------------""")

        
        # Recommendation Job&Talent
    elif selected == "Recommendation Job&Talent": 
        st.date_input("")
        # (OCR function)
        def extract_data(feed):
            text=''
            with pdfplumber.open(feed) as pdf:
                pages = pdf.pages
                for page in pages:
                    text+=page.extract_text(x_tolerance=2)
            return text

        # Import whole nlp csv
        @st.cache(allow_output_mutation=True) #method to get data once and store in cache.
        def get_jobcsv():
            url = 'data/Dataset Techinasia Preprocessing.csv'
            return pd.read_csv(url)

        starlight_job_df= get_jobcsv()
        
        # Title & select boxes--------------------------display##
        st.title('Job Recommendation')
        st.markdown("""------------------------------------------------------------------------------------------------""")
        c1, c2, c3 = st.columns((3,3,2))
        cv = c1.file_uploader('Upload your CV', type='pdf')

        # Career level-----------------------------------display##
        levels = starlight_job_df['Career Level'].unique().tolist()
        CL = c2.multiselect('Career level', levels, levels)

        # Number of job recommend slider------------------display##
        no_of_jobs = c3.slider('Number of Job Recommendations:', min_value=10, max_value=50, step=5)

        if cv is not None:
            cv_text = extract_data(cv)

            #(NLP keywords function)
            @st.cache
            def nlp(x):
                word_sent = word_tokenize(x.lower().replace("\n",""))
                _stopwords = set(stopwords.words('english') + list(punctuation))
                word_sent=[word for word in word_sent if word not in _stopwords]
                processed = [' '.join(word for word in sentence.split(' ') if len(word) > 1) for sentence in word_sent]
                lemmatizer = WordNetLemmatizer()
                NLP_Processed_CV = [lemmatizer.lemmatize(word) for word in word_tokenize(" ".join(processed))]
                return NLP_Processed_CV
            

            # (NLP keywords for CV workings)
            try:
                NLP_Processed_CV = nlp(cv_text)
            except NameError:
                st.error('Please enter a valid input')

            
            starlight_resume_df = pd.DataFrame()
            starlight_resume_df['Job Title'] = ['I']
            starlight_resume_df['Job Requirement'] = ['I']
            starlight_resume_df['Skills'] = ['I']

            starlight_resume_df['All'] = " ".join(NLP_Processed_CV)


            # Combine column Position, Job Title, Job Requirement, Skills in one column namely 'All'
            starlight_job_df['All'] = starlight_job_df['Job Type'] + ' ' + starlight_job_df['Job Experience'] + ' ' + starlight_job_df['Career Level'] + ' ' + \
                                    starlight_job_df['Job Title'] + ' ' + starlight_job_df['Job Requirement'] + ' ' + starlight_job_df['Skills']

            # Preprocessing function for result combine columns
            starlight_job_df['All'] = starlight_job_df['All'].str.replace('[^\w\s]','')
            starlight_job_df['All'] = starlight_job_df['All'].str.replace('\n','')

            _stopwords = set(stopwords.words('english') + list(punctuation))
            starlight_job_df['All'] = starlight_job_df['All'].apply(lambda x: ' '.join([word.lower() for word in x.split() if word not in (_stopwords)]))
            starlight_job_df['All'] = starlight_job_df['All'].apply(word_tokenize)

            def lemmatize_text(text):
                lemmatizer = WordNetLemmatizer()
                return [lemmatizer.lemmatize(w) for w in text]
            starlight_job_df['All'] = starlight_job_df['All'].apply(lemmatize_text)

            starlight_job_df['All'] = starlight_job_df['All'].apply(lambda x: ' '.join(x))

            @st.cache
            # Recommendation function
            def get_recommendation(top, df, scores):
                recommendation = pd.DataFrame(columns = ['Job Id', 'Job Type', 'Job Experience', 'Career Level', 'Position',
                                                        'Industries', 'Vacancy Count', 'Job Title', 'Job Requirement', 'Skills',
                                                        'Job Salary', 'Salary Min', 'Salary Max', 'Company', 'Location','Date Created', 
                                                        'Job Posting Link','Score'])
                count = 0
                for i in top:
                    recommendation.at[count, 'Job Id'] = df.index[i]
                    recommendation.at[count, 'Job Type'] = df['Job Type'][i]
                    recommendation.at[count, 'Job Experience'] = df['Job Experience'][i]
                    recommendation.at[count, 'Career Level'] = df['Career Level'][i]
                    recommendation.at[count, 'Position'] = df['Position'][i]
                    recommendation.at[count, 'Industries'] = df['Industries'][i]
                    recommendation.at[count, 'Vacancy Count'] = df['Vacancy Count'][i]
                    recommendation.at[count, 'Job Title'] = df['Job Title'][i]
                    recommendation.at[count, 'Job Requirement'] = df['Job Requirement'][i]
                    recommendation.at[count, 'Skills'] = df['Skills'][i]
                    recommendation.at[count, 'Job Salary'] = df['Job Salary'][i]
                    recommendation.at[count, 'Salary Min'] = df['Salary Min'][i]
                    recommendation.at[count, 'Salary Max'] = df['Salary Max'][i]
                    recommendation.at[count, 'Company'] = df['Company'][i]
                    recommendation.at[count, 'Location'] = df['Location'][i]
                    recommendation.at[count, 'Date Created'] = df['Date Created'][i]
                    recommendation.at[count, 'Job Posting Link'] = df['Job Posting Link'][i]
                    recommendation.at[count, 'score'] =  scores[count]
                    count += 1
                return recommendation
            
            @st.cache
            # Create a function for a content based filtering recommendation system using TFIDF Vectorizer + Cosine Similarity
            def TFIDF(job_dataset, resume_data):
                tfidf_vectorizer = TfidfVectorizer(stop_words='english')

                # TF-IDF calculation For column All in starlight_job_df
                tfidf_jobid = tfidf_vectorizer.fit_transform(job_dataset)

                # TF-IDF Calculation For Column All in starlight_resume_df
                user_tfidf = tfidf_vectorizer.transform(resume_data)

                # Calculate the data similarity between the results of tfidf_jobid and user_tfidf
                cos_similarity_tfidf = map(lambda x: cosine_similarity(user_tfidf,x),tfidf_jobid)

                # The output will be displayed in the form of a list
                output_tfidf_cosine = list(cos_similarity_tfidf)
                return output_tfidf_cosine      
            output_tfidf_cosine = TFIDF(starlight_job_df['All'], starlight_resume_df['All'])

            # Show top job recommendations using TF-IDF
            top = sorted(range(len(output_tfidf_cosine)), key=lambda i: output_tfidf_cosine[i], reverse=True)[:50]
            list_scores = [output_tfidf_cosine[i][0][0] for i in top]
            result_job_recommendation_tfidf = get_recommendation(top, starlight_job_df, list_scores)

            @st.cache
            # Create a function for a content based filtering recommendation system using Count Vectorizer + Cosine Similarity
            def CV(job_dataset, resume_data):
                count_vectorizer = CountVectorizer(stop_words='english')

                # Count Vectorizer Calculation For column All in starlight_job_df
                cv_jobid = count_vectorizer.fit_transform(job_dataset)

                # Count Vectorizer Calculation for Column All in starlight_resume_df
                user_cv = count_vectorizer.transform(resume_data)

                # Calculate the data similarity between the results of cv_jobid and user_cv
                cos_similarity_cv = map(lambda x: cosine_similarity(user_cv,x),cv_jobid)

                # The output will be displayed in the form of a list
                output_cv_cosine = list(cos_similarity_cv)
                return output_cv_cosine   
            output_cv_cosine = TFIDF(starlight_job_df['All'], starlight_resume_df['All'])

            # Show top job recommendations using Count Vectorizer
            top = sorted(range(len(output_cv_cosine)), key=lambda i: output_cv_cosine[i], reverse=True)[:50]
            list_scores = [output_cv_cosine[i][0][0] for i in top]
            result_job_recommendation_cv = get_recommendation(top, starlight_job_df, list_scores)

            @st.cache
            def KNN(job_dataset, resume_data):
                tfidf_vectorizer = TfidfVectorizer(stop_words='english')
                n_neighbors = 1000
                KNN = NearestNeighbors(n_neighbors=9)
                KNN.fit(tfidf_vectorizer.fit_transform(job_dataset))
                NNs = KNN.kneighbors(tfidf_vectorizer.transform(resume_data))
                top = NNs[1][0][1:]
                index_score = NNs[0][0][1:]
                output_knn = get_recommendation(top, starlight_job_df, index_score)
                return output_knn
            result_job_recommendation_knn = KNN(starlight_job_df['All'], starlight_resume_df['All'])

            # Combine 3 methods into a dataframe
            merge1 = result_job_recommendation_knn[['Job Id', 'Job Type', 'Job Experience', 'Career Level', 'Position',
                                                    'Industries', 'Vacancy Count', 'Job Title', 'Job Requirement', 'Skills',
                                                    'Job Salary', 'Salary Min', 'Salary Max', 'Company', 'Location','Date Created', 
                                                    'Job Posting Link','score']].merge(result_job_recommendation_tfidf[['Job Id','score']], on= "Job Id")
            final = merge1.merge(result_job_recommendation_cv[['Job Id','score']], on = "Job Id")
            final = final.rename(columns={"score_x": "KNN", "score_y": "TF-IDF","score": "CV"})

            slr = MinMaxScaler()
            final[["KNN", "TF-IDF", 'CV']] = slr.fit_transform(final[["KNN", "TF-IDF", 'CV']])

            # Multiply by weights
            final['KNN'] = (1-final['KNN'])/3
            final['TF-IDF'] = final['TF-IDF']/3
            final['CV'] = final['CV']/3
            final['Final'] = final['KNN']+final['TF-IDF']+final['CV']
            final.sort_values(by="Final", ascending=False, inplace=True) #make silde bar to change top N recommendations

            def Job_recomm(x):
                final_ = final[['Job Id', 'Job Type', 'Job Experience', 'Career Level', 'Position',
                                'Industries', 'Vacancy Count', 'Job Title', 'Job Requirement', 'Skills',
                                'Job Salary', 'Salary Min', 'Salary Max', 'Company', 'Location','Date Created', 
                                'Job Posting Link']]
                selected_levels = final_['Career Level'].isin(CL)
                cl_select = final_[selected_levels]
                return cl_select
            
            result_jd = Job_recomm(CL)
            final_jobrecomm = result_jd.head(no_of_jobs)

            # Qualification bar chart
            db_expander = st.expander(label='CV dashboard:')
            with db_expander:

                chart1, chart2 = st.columns(2)

                with chart1:
                    industry_count = final_jobrecomm.Industries.count()
                    count_with_null = final_jobrecomm.Industries.count() + final_jobrecomm.Industries.isnull().sum()
                    st.write("**INDUSTRIES PROVIDED FROM**", industry_count, "**OF**", count_with_null, "**JOBS**")
                    industry_count = final_jobrecomm.Industries.value_counts()
                    industry = pd.DataFrame(industry_count)
                    industry.reset_index(inplace=True)
                    industry.rename({'index': 'Industries', 'Industries': 'Count'}, axis=1, inplace=True)
                    fig = px.pie(industry, values = "Count", names = "Industries", width=600)
                    fig.update_layout(showlegend=True)
                    st.plotly_chart(fig, use_container_width=True)

                with chart2:
                    job_type_count = final_jobrecomm['Job Type'].count()
                    count_with_null = final_jobrecomm['Job Type'].count() + final_jobrecomm['Job Type'].isnull().sum()
                    st.write("**JOB TYPE PROVIDED FROM**", job_type_count, "**OF**", count_with_null, "**JOBS**")
                    job_type_count = final_jobrecomm['Job Type'].value_counts()
                    job_type = pd.DataFrame(job_type_count)
                    job_type.reset_index(inplace=True)
                    job_type.rename({'index': 'Job Type', 'Job Type': 'Count'}, axis=1, inplace=True)
                    fig = px.pie(job_type, values = "Count", names = "Job Type", width=600)
                    fig.update_layout(showlegend=True)
                    st.plotly_chart(fig, use_container_width=True)
                

                # Gabungkan kolom Skills dengan kolom Position
                data = final_jobrecomm.groupby('Position').agg(lambda col: ' '.join(col))
                data = data[['Skills']]

                # Buat fungsi untuk membersihkan data teks
                def clean_text(text):
                    text = re.sub('[%s]' % re.escape(string.punctuation), '', text).lower() #remove punctutations
                    text = re.sub('\w*\d\w*', '', text)
                    text = re.sub('[‘’“”…]', '', text)
                    text = re.sub('\n',' ',text)
                    return text

                # Bersihkan data teks
                clean = lambda x :clean_text(x)
                df_skills = pd.DataFrame(data['Skills'].apply(clean))
                
                # Lemmentasikan data teks untuk meningkatkan analisis
                lemmer = WordNetLemmatizer()
                df_skills['Skills'] = df_skills['Skills'].apply(lambda x: word_tokenize(x))
                df_skills['Skills'] = df_skills['Skills'].apply(lambda x : [lemmer.lemmatize(y) for y in x])
                df_skills['Skills'] = df_skills['Skills'].apply(lambda x: ' '.join(x))

                # Tambahkan kata-kata yang sering muncul dalam deskripsi tetapi tidak membawa nilai ke daftar kata stopwords
                extra_stopword = ['data','experience','work','team','will','skill','year','skills']
                stop_words = text.ENGLISH_STOP_WORDS.union(extra_stopword)
                
                st.write("**SKILLS OF THE RECOMMENDED JOBS**")

                wc = WordCloud(stopwords=stop_words, background_color="white", colormap="Dark2",
                            random_state=42, collocations = False)
                i = 0
                fig = plt.figure(figsize=(30,15))
                for x in df_skills['Skills'].index:
                    wc.generate(df_skills['Skills'][str(x)])  
                    i += 1
                    fig.add_subplot(4, 1, i)
                    plt.imshow(wc, interpolation="bilinear")
                    plt.axis("off")
                    plt.title(str(x), size = 12)
                st.pyplot(fig)



            # Expander for jobs df ---------------------------display#
            db_expander = st.expander(label='Job Recommendation:')

            def make_clickable(link):
                # Target _blank to open new window
                # Extract clickable text to display for your link
                text = 'Click here'
                return f'<a target="_blank" href="{link}">{text}</a>'

            with db_expander:
                final_jobrecomm['Job Posting Link'] = final_jobrecomm['Job Posting Link'].apply(make_clickable)
                final_jobrecomm['Job Salary'] = final_jobrecomm['Job Salary'].str.replace('0 – 0', 'Not Available')
                final_df = final_jobrecomm[['Job Title', 'Job Type', 'Career Level', 'Position', 'Job Salary', 'Company', 
                                            'Location','Date Created', 'Job Posting Link']]
                
                show_df = final_df.to_html(escape=False)
                st.write(show_df, unsafe_allow_html=True)
                
                
                
            st.snow()
        
    #Search jobs    
    elif selected == "Search jobs":
        st.date_input("")
        st.title(f"Search Jobs")
        data = pd.read_csv("data/Dataset Techinasia Preprocessing.csv", usecols=lambda c: not c.startswith('Unnamed:'))

        tfdif = TfidfVectorizer(stop_words='english')
        def search_jobs(search: str, item_count: int = 10) -> pd.DataFrame:
            jobs_list = data[data['Job Title'].str.contains(search)]

            return jobs_list.iloc[1:item_count + 1]
            
            
        with st.container():
            col1, col2, col3 = st.columns((2, 0.5, 2))

            with col1:
                search_input = st.text_input('Search jobs', '')
                st.write(f'Search results for: {search_input}')

            with col3:
                result_count = st.number_input('Results count', 1, 10, 10)
                st.write('')
            
    
        if search_input != '':
            results = search_jobs(search_input, result_count)
            st.snow()

            with st.container():
                for index, result in results.iterrows():
                    with st.expander(result['Job Title']):
                        st.write('**Location:** ' + result['Location'])
                        st.write('**Job Type:** ' + result['Job Type'])
                        st.write('**Company:** ' + result['Company'])
                        st.write('**Industries:** ' + result['Industries'])
                        st.write('**Skills:** ' + result['Skills'])
                        st.write('**Career Level:** ' + result['Career Level'])


                        st.write('**Salary**')
                        st.write(result['Job Salary'])

                        st.markdown('**Description**')
                        st.write(result['Job Requirement'])

                        st.write(f'**Link:** [{result["Job Posting Link"]}]({result["Job Posting Link"]})')
                       
                       

        
    #Developer
    elif selected == "Developer":
        st.title(f"TIM STARLIGHT")
        st.write("**CLASS GALAXY  || COACH ANNISA RIZKI LILIANDARI || MSIB2 ORBIT FUTURE ACADEMY**")

        st.markdown("""------------------------------------------------------------------------------------------------""")
        
        col1,col2,col3,col4,col5=st.columns(5)
        from PIL import Image
        img_likun  = Image.open("data/likun.jpg")
        img_Justin = Image.open("data/Justine.jpeg")
        img_anjana = Image.open("data/anjana.png")
        img_silvi  = Image.open("data/silvi.png")
        img_elda  = Image.open("data/elda.png")
        
        col1.image(img_likun,width=200)
        col1.write("**Sholikun**")
        col1.write("**Universitas Widya Husada**")
        
        col2.image(img_Justin, width=194)
        col2.write("**Justine**")
        col2.write("**Universitas Mikroskil**")
        
        col3.image(img_anjana, width=216)
        col3.write("**Anjanah Diah Andriani**")
        col3.write("**Universitas Widya Husada**")
        
        col4.image(img_silvi, width=240)
        col4.write("**Silvia Maysiska Saragih**")
        col4.write("**Universitas Widya Husada**")
        
        col5.image(img_elda, width=216)
        col5.write("**Elda Florenti Ginting**")
        col5.write("**Universitas Widya Husada**")
        
        st.snow()
        
        
    st.markdown("""------------------------------------------------------------------------------------------------""")
        
if __name__ == '__main__':
    main()

 
                
                

