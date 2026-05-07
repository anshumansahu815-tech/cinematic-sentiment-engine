import streamlit as st
from transformers import pipeline

# 1. Page Configuration
st.set_page_config(page_title="Cinematic Sentiment Engine", page_icon="🎬", layout="centered")

# 2. App Header
st.title("🎬 Cinematic Sentiment Engine")
st.write("Drop a movie review below, and our NLP model will instantly analyze the sentiment and classify the audience reaction.")

# 3. Load the NLP Model (Cached so it only loads once)
@st.cache_resource
def load_model():
    # Using a fast, pre-trained model for the prototype
    return pipeline("sentiment-analysis")

analyzer = load_model()

# 4. User Input Area
user_review = st.text_area("Enter a Movie Review:", placeholder="e.g., The visual effects were absolutely stunning, but the plot felt a bit rushed...")

# 5. Analysis Logic
if st.button("Analyze Sentiment"):
    if user_review:
        with st.spinner("Analyzing neural pathways..."):
            result = analyzer(user_review)[0]
            label = result['label']
            score = result['score']
            
            # 6. Display Results
            st.markdown("### 📊 Analysis Results")
            if label == "POSITIVE":
                st.success(f"**Sentiment:** {label} (Confidence: {score:.2%})")
            else:
                st.error(f"**Sentiment:** {label} (Confidence: {score:.2%})")
    else:
        st.warning("Please enter a review first!")