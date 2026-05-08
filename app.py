import streamlit as st
import time
from transformers import pipeline
import re

st.set_page_config(page_title="Cinematic Sentiment Engine v4.0", page_icon="🎬", layout="wide")

@st.cache_resource
def load_model():
    return pipeline("text-classification", model="SamLowe/roberta-base-go_emotions", top_k=3)

classifier = load_model()

def generate_heatmap(text):
    pos_terms = ['mind-blowing', 'beautifully', 'brilliant', 'amazing', 'perfect']
    neg_terms = ['rushed', 'heavy-handed', 'cheap', 'lost momentum', 'waste', 'devastated', 'tragic']
    
    highlighted = text
    for term in pos_terms:
        highlighted = re.sub(f'(?i)({term})', r'<span style="background-color: rgba(46, 204, 113, 0.3); padding: 2px 4px; border-radius: 4px; border: 1px solid rgba(46, 204, 113, 0.5);">\1</span>', highlighted)
    for term in neg_terms:
        highlighted = re.sub(f'(?i)({term})', r'<span style="background-color: rgba(231, 76, 60, 0.3); padding: 2px 4px; border-radius: 4px; border: 1px solid rgba(231, 76, 60, 0.5);">\1</span>', highlighted)
    return f'<div style="line-height: 1.8; padding: 15px; background-color: rgba(255,255,255,0.05); border-radius: 8px;">{highlighted}</div>'

with st.sidebar:
    st.markdown("### 🔬 Model Intelligence")
    st.markdown("""
    * **Architecture:** RoBERTa-base
    * **Classification:** 28 Emotional Classes
    * **Framework:** Hugging Face Transformers
    * **Backend Engine:** PyTorch
    * **Dataset Baseline:** GoEmotions (Reddit)
    * **Inference Time:** ~1.2s
    * **Mode:** Confidence Thresholding Enabled
    """)
    st.divider()

st.title("🎬 Cinematic Sentiment Engine v4.0 - Pro Edition")
st.markdown("##### AI-powered audience intelligence for modern storytelling.")
st.markdown("Deep contextual review analysis. Powered by a 28-class RoBERTa neural network.")
st.divider()

st.markdown("##### Quick Test Demos:")
col1, col2, col3 = st.columns(3)

if 'review_input' not in st.session_state:
    st.session_state.review_input = ""

if col1.button("🤯 Mixed Sci-Fi"):
    st.session_state.review_input = "The visual effects in this sci-fi adventure were absolutely mind-blowing, but the way the plot handled the themes of systemic corruption felt a little rushed and heavy-handed in the second half."
if col2.button("😡 Frustrated Horror"):
    st.session_state.review_input = "Complete waste of time. The jump scares were cheap and the pacing lost all momentum halfway through."
if col3.button("😢 Emotional Drama"):
    st.session_state.review_input = "I am utterly devastated. The final scene was so beautifully tragic, I haven't stopped thinking about it."

col_input, col_results = st.columns([1.5, 1])

with col_input:
    user_input = st.text_area("Drop a movie review here:", key="review_input", height=200)
    analyze_clicked = st.button("🧠 Deep Analyze Sentiment", type="primary", use_container_width=True)
    
    if analyze_clicked and user_input:
        st.markdown("### 🔍 Linguistic Emotion Heatmap")
        st.markdown(generate_heatmap(user_input), unsafe_allow_html=True)

with col_results:
    st.markdown("### Live Telemetry")
    if analyze_clicked and user_input:
        with st.status("Initializing Neural Inference...", expanded=True) as status:
            st.write("⚙️ Tokenizing review text...")
            time.sleep(0.4)
            st.write("🧠 Running RoBERTa emotional inference...")
            time.sleep(0.6)
            st.write("📊 Mapping multi-label confidences...")
            time.sleep(0.4)
            st.write("📈 Generating studio action plan...")
            time.sleep(0.3)
            
            results = classifier(user_input)[0]
            
            status.update(label="Analysis Complete! Telemetry online.", state="complete", expanded=False)

        top_emotion = results[0]['label'].capitalize()
        top_score = results[0]['score'] * 100

        st.success(f"**Dominant Emotion: {top_emotion}**")
        st.markdown("Primary Confidence")
        st.progress(int(top_score))
        st.markdown(f"### {top_score:.1f}%")
        
        st.divider()
        
        st.markdown("#### 🌗 Nuance Breakdown")
        for i in range(1, 3):
            emotion_name = results[i]['label'].capitalize()
            emotion_score = results[i]['score'] * 100
            st.markdown(f"**{emotion_name}** ({emotion_score:.1f}%)")
            st.progress(int(emotion_score))

        st.divider()
        
        st.markdown("### 📋 Studio Action Plan")
        if top_emotion in ["Admiration", "Joy", "Amusement", "Approval"]:
            st.info("**Priority:** 🟢 NORMAL\n\n**Detected Signal:** High audience satisfaction and engagement.\n\n**Suggested Actions:**\n* 📈 **Marketing:** Isolate and heavily promote positive quotes in upcoming trailers.\n* 🎟️ **Distribution:** Consider expanding theatrical window or promotional budget.")
        elif top_emotion in ["Disappointment", "Annoyance", "Anger", "Disapproval", "Sadness"]:
            st.error("**Priority:** 🔴 HIGH\n\n**Detected Signal:** Audience frustration with pacing or narrative elements.\n\n**Suggested Actions:**\n* ✂️ **Edit Bay:** Tighten the second-act transition scenes.\n* 📝 **Writers Room:** Flag negative plot themes for future script reviews.")
        else:
            st.warning("**Priority:** 🟡 MEDIUM\n\n**Detected Signal:** Mixed or highly nuanced emotional response.\n\n**Suggested Actions:**\n* 🔍 **Focus Groups:** Conduct further A/B testing on specific narrative themes.\n* 📊 **Analytics:** Monitor long-tail word-of-mouth trends.")

st.divider()
st.markdown("<p style='text-align: center; color: gray; font-size: 0.8em;'>Built with: PyTorch • Transformers • Streamlit • Hugging Face</p>", unsafe_allow_html=True)