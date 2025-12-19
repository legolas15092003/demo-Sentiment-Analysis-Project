import streamlit as st
from transformers import pipeline

# Page configuration
st.set_page_config(
    page_title="Sentiment Analysis App",
    page_icon="😊",
    layout="centered"
)

# Title and description
st.markdown("<h1 style='text-align: center;'>🧠 Sentiment Analysis App</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Analyze emotions in text using 🤗 Transformers</p>", unsafe_allow_html=True)

st.divider()

# Load model (cached for performance)
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis")

model = load_model()

# Text input
text = st.text_area(
    "✍️ Enter your text below:",
    placeholder="Example: I don't like this product 😞",
    height=120
)

# Button
if st.button("🔍 Analyze Sentiment"):
    if text.strip() == "":
        st.warning("⚠️ Please enter some text first!")
    else:
        with st.spinner("Analyzing sentiment... 🤔"):
            result = model(text)[0]

        label = result["label"]
        score = result["score"]

        st.success("✅ Analysis Complete!")

        # Display result nicely
        if label == "POSITIVE":
            st.markdown(f"### 😄 Sentiment: **{label}**")
        else:
            st.markdown(f"### 😡 Sentiment: **{label}**")

        st.markdown(f"**Confidence Score:** `{score:.4f}`")

        st.divider()
        st.markdown("📌 *This model uses a pre-trained Hugging Face Transformer*")

# Footer
st.markdown(
    "<hr><p style='text-align:center;'>Made with ❤️ using Streamlit & Transformers</p>",
    unsafe_allow_html=True
)
