import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import nltk

# Download NLTK data
nltk.download('punkt')

# Check for missing libraries
try:
    from textblob import TextBlob
    from wordcloud import WordCloud
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sumy.parsers.plaintext import PlaintextParser
    from sumy.nlp.tokenizers import Tokenizer
    from sumy.summarizers.lsa import LsaSummarizer
except ModuleNotFoundError as e:
    st.error(f"Error: {e}. Please install the required libraries using 'pip install streamlit pandas numpy nltk textblob wordcloud scikit-learn sumy matplotlib'.")
    st.stop()

# Set page title and icon
st.set_page_config(page_title="Text Analyzer", page_icon="📊", layout="wide")

# Custom CSS for styling
st.markdown("""
<style>
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    .stTextArea>textarea {
        font-size: 16px;
    }
    .stMarkdown {
        font-size: 18px;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.title("📝 Text Analyzer")
st.markdown("Analyze your text with advanced features like sentiment analysis, keyword extraction, and more!")

# Input text area
text = st.text_area("Enter your text here:", height=200)

# Sidebar for feature selection
st.sidebar.header("Settings")
features = st.sidebar.multiselect(
    "Choose features to analyze:",
    ["Word Frequency", "Sentiment Analysis", "Keyword Extraction", "Text Summarization", "Readability Score", "Word Cloud"]
)

# Analyze button
if st.button("Analyze Text"):
    if text.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        st.success("Analysis complete! Here are the results:")

        # Word Frequency Analysis
        if "Word Frequency" in features:
            st.subheader("Word Frequency")
            words = text.split()
            word_freq = pd.Series(words).value_counts().reset_index()
            word_freq.columns = ["Word", "Frequency"]
            st.write(word_freq)

        # Sentiment Analysis
        if "Sentiment Analysis" in features:
            st.subheader("Sentiment Analysis")
            blob = TextBlob(text)
            sentiment = blob.sentiment
            st.write(f"Polarity: {sentiment.polarity} (Range: -1 to 1)")
            st.write(f"Subjectivity: {sentiment.subjectivity} (Range: 0 to 1)")
            if sentiment.polarity > 0:
                st.write("Sentiment: Positive 😊")
            elif sentiment.polarity < 0:
                st.write("Sentiment: Negative 😠")
            else:
                st.write("Sentiment: Neutral 😐")

        # Keyword Extraction
        if "Keyword Extraction" in features:
            st.subheader("Keyword Extraction")
            try:
                vectorizer = TfidfVectorizer(stop_words='english')
                tfidf_matrix = vectorizer.fit_transform([text])
                feature_names = vectorizer.get_feature_names_out()
                tfidf_scores = tfidf_matrix.toarray()[0]
                keywords = pd.DataFrame({"Keyword": feature_names, "TF-IDF Score": tfidf_scores})
                keywords = keywords.sort_values(by="TF-IDF Score", ascending=False).head(10)
                st.write(keywords)
            except Exception as e:
                st.error(f"Error in keyword extraction: {e}")

        # Text Summarization
        if "Text Summarization" in features:
            st.subheader("Text Summarization")
            try:
                parser = PlaintextParser.from_string(text, Tokenizer("english"))
                summarizer = LsaSummarizer()
                summary = summarizer(parser.document, 3)  # Summarize to 3 sentences
                st.write("Summary:")
                for sentence in summary:
                    st.write(f"- {str(sentence)}")
            except Exception as e:
                st.error(f"Error in text summarization: {e}")

        # Readability Score
        if "Readability Score" in features:
            st.subheader("Readability Score")
            try:
                sentences = text.count('.') + text.count('!') + text.count('?')
                words = len(text.split())
                syllables = sum([len(word) for word in text.split()])
                readability = 206.835 - (1.015 * (words / sentences)) - (84.6 * (syllables / words))
                st.write(f"Flesch-Kincaid Readability Score: {readability:.2f}")
            except Exception as e:
                st.error(f"Error in calculating readability score: {e}")

        # Word Cloud
        if "Word Cloud" in features:
            st.subheader("Word Cloud")
            try:
                wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
                plt.figure(figsize=(10, 5))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis("off")
                st.pyplot(plt)
            except Exception as e:
                st.error(f"Error in generating word cloud: {e}")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ by Nisa Iqbal")