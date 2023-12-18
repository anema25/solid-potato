import streamlit as st
from joblib import load
import requests
from bs4 import BeautifulSoup

# Load your models
presence_classifier = load('api/presence_classifier.joblib')
presence_vect = load('api/presence_vectorizer.joblib')
category_classifier = load('api/category_classifier.joblib')
category_vect = load('api/category_vectorizer.joblib')

# Streamlit app
def main():
    st.title("Dark Pattern Detection")

    # Get URL input from user
    url = st.text_input("Enter website URL:", "")

    if st.button("Fetch and Detect Dark Patterns"):
        text_from_website = fetch_text_from_website(url)
        tokens = text_from_website.split('\n')  # Assuming you want to split text into tokens

        output = detect_dark_patterns(tokens)
        st.json({"result": output})

def fetch_text_from_website(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        text = ' '.join([p.get_text() for p in soup.find_all('p')])  # Extract text from paragraphs
        return text
    except Exception as e:
        st.error(f"Error fetching text from the website: {e}")
        return ""

def detect_dark_patterns(tokens):
    # The rest of your detection code remains unchanged
    output = []

    for token in tokens:
        result = presence_classifier.predict(presence_vect.transform([token]))
        if result == 'Dark':
            cat = category_classifier.predict(category_vect.transform([token]))
            output.append(cat[0])
        else:
            output.append(result[0])

    dark_patterns = [tokens[i] for i in range(len(output)) if output[i] == 'Dark']
    st.text("Detected Dark Patterns:")
    for d in dark_patterns:
        st.text(d)
    
    st.text(f"Number of Detected Dark Patterns: {len(dark_patterns)}")

    st.text("Debugging:")
    st.text(f"Output of presence classifier: {output}")
    
    return output

if __name__ == '__main__':
    main()
