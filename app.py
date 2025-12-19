import streamlit as st
from transformers import pipeline

# Page configuration
st.set_page_config(page_title="Sentiment Analyzer", page_icon="🎭")

# Load the model with caching to prevent reloading on every click
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis")

analyzer = load_model()

# --- UI Layout ---
st.title("🎭 AI Sentiment Analyzer")
st.markdown("""
    Welcome! This app uses a **Hugging Face** transformer model to detect if your text is 
    **Positive** or **Negative**. 
""")

st.divider()

# Input section
text_input = st.text_area("✍️ Enter your text below:", placeholder="e.g., I absolutely love this product!")

if st.button("Analyze Sentiment"):
    if text_input.strip() != "":
        with st.spinner("Analyzing... 💫"):
            # Model Inference
            result = analyzer(text_input)[0]
            label = result['label']
            score = result['score']

            # Display Results
            st.subheader("Results:")
            
            # Formatting based on sentiment
            if label == "POSITIVE":
                st.success(f"**Sentiment:** {label} 😊")
            else:
                st.error(f"**Sentiment:** {label} ☹️")
            
            st.info(f"**Confidence Score:** {score:.2%}")
            
            # Progress bar for visual appeal
            st.progress(score)
    else:
        st.warning("Please enter some text first! ⚠️")

# Sidebar info
st.sidebar.title("About")
st.sidebar.info("Built with Streamlit and Hugging Face Transformers. 🚀")
