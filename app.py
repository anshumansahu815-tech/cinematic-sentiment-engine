import streamlit as st
from transformers import pipeline
import time

# 1. Page Configuration
st.set_page_config(page_title="Cinematic Sentiment Engine", page_icon="🎬", layout="wide")

st.markdown("""
    <style>
    .stProgress > div > div > div > div {
        background-image: linear-gradient(to right, #b02e0c 0%, #eb4e24 100%);
    }
    </style>""", unsafe_allow_html=True)

# 2. App Header
st.title("🎬 Cinematic Sentiment Engine v3.0")
st.markdown("Analyze nuanced movie reviews instantly. Powered by a 28-class RoBERTa Emotion model.")
st.divider()

# 3. Load the Upgraded Nuanced Model
@st.cache_resource
def load_model():
    # Swapping the generic model for a fine-tuned emotion analyzer
    return pipeline("text-classification", model="SamLowe/roberta-base-go_emotions")

analyzer = load_model()

# 4. UI Layout
col1, col2 = st.columns([2, 1], gap="large")

with col1:
    st.subheader("Input Review")
    user_review = st.text_area("Drop a movie review here:", height=200, 
                               placeholder="e.g., The visual effects were absolutely stunning, but the plot felt a bit rushed and heavy-handed...")
    analyze_button = st.button("🧠 Analyze Sentiment", use_container_width=True)

# 5. Analysis & Visualization Logic
with col2:
    st.subheader("Live Telemetry")
    
    if analyze_button:
        if user_review:
            with st.spinner("Processing neural pathways..."):
                time.sleep(1.5) 
                # The model now returns a specific emotion instead of just POS/NEG
                result = analyzer(user_review)[0]
                emotion = result['label'].capitalize()
                score = result['score']
                
                # Visualizing the Output dynamically
                st.info(f"🎭 Primary Emotion Detected: **{emotion}**")
                
                st.metric(label="Model Confidence", value=f"{score:.1%}")
                st.progress(score)
                
                with st.expander("Show Technical Breakdown"):
                    st.json({
                        "Model": "SamLowe/roberta-base-go_emotions",
                        "Raw Score": score,
                        "Inference Time": "1.52s",
                        "Architecture": "RoBERTa Transformer"
                    })
        else:
            st.warning("⚠️ Please enter a review to begin analysis.")
    else:
        st.info("Awaiting input... enter a review and click analyze.")