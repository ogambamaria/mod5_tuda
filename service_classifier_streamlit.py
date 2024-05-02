import streamlit as st
import pandas as pd
import plotly.express as px
from joblib import load
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Define the feedback types and phrases likely to be used for flagging
feedback_types = {
    'positive_feedback': {
        'Friendly Staff': 'friendly',
        'Good Treatment': 'treatment',
        'Confidentiality Upholded': 'confidentiality',
    },
    'negative_feedback': {
        'Harassment from Staff': 'harassment',
        'Delays in Service': 'delay',
        'Lack of Nutritional Support': 'lack nutritional',
        'Septrin Out of Stock': 'septrin'
    }
}

# Load the trained model and vectorizer
model = load('naive_bayes_model.joblib')
vectorizer = load('tfidf_vectorizer.joblib')

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words and token.isalpha()]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(tokens)

def predict_input(user_input):
    cleaned_input = preprocess_text(user_input)
    transformed_input = vectorizer.transform([cleaned_input])
    prediction = model.predict(transformed_input)
    
    features_counts = {}
    for category in feedback_types.values():
        for feature, phrase in category.items():
            features_counts[feature] = cleaned_input.lower().count(phrase)
    
    return prediction, features_counts

def add_emoji_to_df(features_df):
    # Map updated features to emojis
    emoji_map = {
        'Friendly Staff': '😊',
        'Good Treatment': '😊',
        'Confidentiality Upholded': '😊',
        'Harassment from Staff': '😢',
        'Delays in Service': '😢',
        'Lack of Nutritional Support': '😢',
        'Septrin Out of Stock': '😢'
    }
    
    # Apply the emoji map to the DataFrame
    features_df['Type'] = features_df['Feature'].map(emoji_map)
    return features_df


# Set the configuration for the page
st.set_page_config(page_title='CLM Text Analytics App', layout='wide')

# Sidebar description and links
st.sidebar.header("About the App")
st.sidebar.markdown("This application is designed to flag certain features in the feedback test. The expected input is feedback from patients in HIV clinics who were interviewed about their experiences through the CLM (Community-led Monitoring) framework.")
st.sidebar.markdown("""
See the data preparation and model development steps: [GitHub Repository](https://github.com/ogambamaria/mod5_poa)
""")

col1, col2, col3 = st.columns([1, 2, 1])
with col1:
    st.write("")
with col2:
    st.image('images/page_icon.png', width=300)
with col3:
    st.write("")

# Center-align the title
st.markdown("<h2 style='text-align: center;'>CLM Text Analytics App</h2>", unsafe_allow_html=True)

col4, col5 = st.columns([1, 1])  # Giving equal width to both columns
with col4:
    user_input = st.text_area("Enter your feedback:", "")
with col5:
    uploaded_file = st.file_uploader("Or upload a CSV file with feedback:", type='csv')
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("Uploaded Data:")
        st.write(data.head())

if st.button('Submit Feedback'):
    prediction, features_counts = predict_input(user_input)
    
    features_df = pd.DataFrame(list(features_counts.items()), columns=['Feature', 'Count'])
    features_df = features_df.sort_values(by='Count', ascending=False)
    
    features_df = add_emoji_to_df(features_df)

    col6, col7 = st.columns([1, 1])  # Giving equal width to both columns
    # Use columns to display the dataframe and the plot side by side
    with col6:
        st.markdown("**Feature Counts**")
        st.write("Positive: 😊")
        st.write("Negative: 😢")
        st.dataframe(features_df)
    with col7:
        st.write("**Top 3 Flagged Features**")
        top_features = features_df.head(3)
        fig = px.bar(top_features, y='Feature', x='Count', orientation='h')
        st.plotly_chart(fig, use_container_width=True)