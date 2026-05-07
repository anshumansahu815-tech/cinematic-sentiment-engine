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
st.title("🎬 Cinematic Sentiment Engine v4.0 - Pro Edition")
st.markdown("Deep contextual review analysis. Powered by a 28-class RoBERTa neural network.")
st.divider()

# 3. Load the Upgraded Nuanced Model (Now pulling top 3 emotions)
@st.cache_resource
def load_model():
    # Adding top_k=3 forces the model to return the top 3 conflicting emotions!
    return pipeline("text-classification", model="SamLowe/roberta-base-go_emotions", top_k=3)

analyzer = load_model()

# 4. UI Layout
col1, col2 = st.columns([2, 1], gap="large")

with col1:
    st.subheader("Input Review")
    user_review = st.text_area("Drop a movie review here:", height=200, 
                               placeholder="e.g., The visual effects were absolutely stunning, but the plot felt a bit rushed and heavy-handed...")
    analyze_button = st.button("🧠 Deep Analyze Sentiment", use_container_width=True)

# 5. Analysis & Visualization Logic
with col2:
    st.subheader("Live Telemetry")
    
    if analyze_button:
        if user_review:
            with st.spinner("Processing neural pathways..."):
                time.sleep(1.5) 
                
                # Result is now a list of the top 3 dictionaries
                results = analyzer(user_review)[0]
                
                # --- PRIMARY EMOTION ---
                primary = results[0]
                primary_label = primary['label'].capitalize()
                
                st.success(f"🎭 Dominant Emotion: **{primary_label}**")
                st.metric(label="Primary Confidence", value=f"{primary['score']:.1%}")
                st.progress(primary['score'])
                
                st.divider()
                
                # --- SECONDARY EMOTIONS (The Machine Learning Flex) ---
                st.markdown("#### 🔍 Nuance Breakdown")
                st.caption("Secondary emotions detected in the subtext:")
                
                for res in results[1:]:
                    label = res['label'].capitalize()
                    score = res['score']
                    st.write(f"**{label}** ({score:.1%})")
                    st.progress(score)
                
                st.divider()

                # --- REAL-WORLD BUSINESS INSIGHT ---
                st.markdown("#### 🏢 Studio Recommendation")
                if primary_label in ["Admiration", "Joy", "Approval", "Excitement", "Amusement"]:
                    st.info("🟢 **Action:** Highlight quote in marketing materials. High audience retention expected.")
                elif primary_label in ["Disappointment", "Annoyance", "Disapproval", "Sadness"]:
                    st.error("🔴 **Action:** Flag for the writers' room. Post-production pacing edits recommended.")
                elif primary_label in ["Confusion", "Curiosity", "Surprise"]:
                    st.warning("🟡 **Action:** Mixed reaction. Consider revising the trailer to set better audience expectations.")
                else:
                    st.write("⚪ **Action:** Neutral response. Standard release strategy.")
                
                # --- TECHNICAL LOGS ---
                with st.expander("Show Technical Breakdown"):
                    st.json({
                        "Model": "SamLowe/roberta-base-go_emotions",
                        "Inference_Type": "Multi-Label Top 3",
                        "Raw_Outputs": results
                    })
        else:
            st.warning("⚠️ Please enter a review to begin analysis.")
    else:
        st.info("Awaiting input... enter a review and click analyze.")