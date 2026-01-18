import streamlit as st
import time
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download
import re

if "load_model" not in st.session_state or "TFIDF" not in st.session_state:
    model_path = hf_hub_download(repo_id="AbhijeetBeedikar/Fake-News-Detector-RFC",
                                 filename="Fake_News_detector_model.pkl")
    st.session_state.load_model = joblib.load(model_path)
    st.session_state.TFIDF = joblib.load('fake_news_detector_tfidf_vectorizer_model_3.joblib')

# Your actual model prediction function
def clean(listt):
  #works for removing links from texts, additional spaces & new lines, special unknown characters, and makes sure "a.b" turns to "a. b"
  empty = []
  for b in listt:
    if '(Reuters) -' in b:
      b = b[b.index('-')+1:]
    text = re.sub(r' #39;',"'",b)
    text = re.sub(r'https?://\S+|www\.\S+', '.', text)
    text = re.sub(r'\n','',text)
    text = re.sub(r'\b[^\s]*\.com[^\s]*\b','',text)
    text = re.sub(r'\b[^\s]*\.net[^\s]*\b','',text)
    text = re.sub(r'\b[^\s]*\.org[^\s]*\b','',text)
    text = re.sub(r'\b[^\s]*\.gov[^\s]*\b','',text)


    text= re.sub(r'\.','. ',text)
    text = re.sub("[^A-Za-z0-9$,%!\)\(—.;:'\"\&/ =\+-]","",text)
    text = re.sub(r' +',' ', text)
    text = text.strip()
    text = text.lower()
    text = text.strip()
    empty.append(text)
  return empty
def reality_check(text):
    """
    Returns True for Real news, False for Fake news.
    """
    news = clean([str(text)])
    print(news)
    if len(news[0]) <= 550:
        print('Please enter a longer text.')
        raise KeyboardInterrupt
        # return 'Please enter a longer text.'
    dict = {'name': news}
    df = pd.DataFrame(dict)
    vectorized_news = st.session_state.TFIDF.transform(df['name'])
    return bool(st.session_state.load_model.predict(vectorized_news)[0])


# Page config
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
<style>
    h1 {
        color: black !important;
    }
    /* Main background with gradient */
    .stApp {
        background: linear-gradient(135deg, #EFF6FF 0%, #FAF5FF 50%, #FDF2F8 100%);
    }

    /* Hide default streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Title styling */
    .main-title {
        text-align: center;
        font-size: 3.5rem;
        font-weight: bold;
        color: #1F2937;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }

    /* Subtitle styling */
    .subtitle {
        text-align: center;
        font-size: 1.2rem;
        color: #4B5563;
        margin-bottom: 0.5rem;
    }

    .warning-text {
        text-align: center;
        font-size: 1rem;
        color: #DC2626;
        font-weight: bold;
        margin-bottom: 2rem;
    }

    /* Decorative elements */
    .decorative-bg {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        pointer-events: none;
        z-index: 0;
        overflow: hidden;
    }

    .bubble {
        position: absolute;
        border-radius: 50%;
        opacity: 0.2;
        animation: pulse 3s ease-in-out infinite;
    }

    .bubble1 {
        top: 10%;
        left: 10%;
        width: 120px;
        height: 120px;
        background: #FCD34D;
        animation-delay: 0s;
    }

    .bubble2 {
        top: 30%;
        right: 15%;
        width: 90px;
        height: 90px;
        background: #93C5FD;
        animation-delay: 1s;
    }

    .bubble3 {
        bottom: 20%;
        left: 25%;
        width: 150px;
        height: 150px;
        background: #C4B5FD;
        animation-delay: 2s;
    }

    @keyframes pulse {
        0%, 100% { transform: scale(1); opacity: 0.2; }
        50% { transform: scale(1.1); opacity: 0.3; }
    }

    /* Meter container */
    .meter-container {
        text-align: center;
        padding: 2rem;
        margin: 2rem auto;
        max-width: 600px;
    }

    /* Result box */
    .result-box {
        text-align: center;
        padding: 1.5rem;
        border-radius: 50px;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 2rem auto;
        max-width: 600px;
        animation: fadeIn 0.5s ease-out;
    }

    .result-real {
        background-color: #D1FAE5;
        color: #065F46;
        border: 3px solid #6EE7B7;
    }

    .result-fake {
        background-color: #FEE2E2;
        color: #991B1B;
        border: 3px solid #FCA5A5;
    }

    @keyframes fadeIn {
        from {
            opacity: 0;
            transform: translateY(10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    /* Character counter */
    .char-counter {
        text-align: right;
        font-size: 0.9rem;
        margin-top: 0.5rem;
    }

    .char-valid {
        color: #059669;
        font-weight: bold;
    }

    .char-invalid {
        color: #6B7280;
    }

    /* Footer */
    .footer-text {
        text-align: center;
        color: #6B7280;
        font-size: 0.9rem;
        margin-top: 3rem;
        padding-bottom: 2rem;
    }
</style>

<div class="decorative-bg">
    <div class="bubble bubble1"></div>
    <div class="bubble bubble2"></div>
    <div class="bubble bubble3"></div>
</div>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<h1 class="main-title">📰 Fake News Detector</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Type the news that you have into the text box below and click enter to see if the news is fake or not.</p>',
    unsafe_allow_html=True)
st.markdown('<p class="warning-text">NOTE: Input must be over 600 characters.</p>', unsafe_allow_html=True)

# Initialize session state
if 'result' not in st.session_state:
    st.session_state.result = None
if 'analyzing' not in st.session_state:
    st.session_state.analyzing = False

# Create main container
col1, col2, col3 = st.columns([1, 3, 1])

with col2:
    # Text input
    news_text = st.text_area(
        "News Article",
        height=250,
        placeholder="Paste your news article here (minimum 600 characters)...",
        label_visibility="collapsed"
    )

    # Character counter
    char_count = len(news_text)
    if char_count >= 600:
        st.markdown(f'<p class="char-counter char-valid">Character count: {char_count} / 600 ✓</p>',
                    unsafe_allow_html=True)
    else:
        st.markdown(f'<p class="char-counter char-invalid">Character count: {char_count} / 600</p>',
                    unsafe_allow_html=True)

    # Analyze button
    if st.button("🔍 Analyze News", use_container_width=True, type="primary"):
        if char_count < 600:
            st.error("⚠️ Input must be over 600 characters. Please provide more text.")
        else:
            st.session_state.analyzing = True
            with st.spinner("Analyzing your news article..."):
                time.sleep(1.5)  # Simulate processing
                st.session_state.result = reality_check(news_text)
                st.session_state.analyzing = False
            st.rerun()

    # Display meter and results
    if st.session_state.result is not None:
        # SVG Meter
        needle_rotation = 90 if st.session_state.result else -90
        needle_color = '#10B981' if st.session_state.result else '#EF4444'

        meter_svg = f'''
        <div class="meter-container">
            <svg viewBox="0 0 200 110" style="width: 100%; max-width: 500px;">
                <path
                    d="M 10 100 A 90 90 0 0 1 100 10"
                    fill="none"
                    stroke="#EF4444"
                    stroke-width="20"
                    stroke-linecap="round"
                />
                <path
                    d="M 100 10 A 90 90 0 0 1 190 100"
                    fill="none"
                    stroke="#10B981"
                    stroke-width="20"
                    stroke-linecap="round"
                />
                <circle cx="100" cy="100" r="8" fill="#374151" />
                <line
                    x1="100"
                    y1="100"
                    x2="100"
                    y2="30"
                    stroke="#374151"
                    stroke-width="3"
                    stroke-linecap="round"
                    transform="rotate({needle_rotation} 100 100)"
                    style="transition: transform 1s cubic-bezier(0.34, 1.56, 0.64, 1);"
                />
                <circle
                    cx="100"
                    cy="30"
                    r="5"
                    fill="{needle_color}"
                    transform="rotate({needle_rotation} 100 100)"
                    style="transition: transform 1s cubic-bezier(0.34, 1.56, 0.64, 1);"
                />
                <text x="20" y="105" font-size="18" font-weight="bold" fill="#EF4444">FAKE</text>
                <text x="145" y="105" font-size="18" font-weight="bold" fill="#10B981">REAL</text>
            </svg>
        </div>
        '''

        st.markdown(meter_svg, unsafe_allow_html=True)

        # Result text
        if st.session_state.result:
            st.markdown(
                '<div class="result-box result-real">✅ Model\'s Opinion: The news is REAL</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div class="result-box result-fake">❌ Model\'s Opinion: The news is FAKE</div>',
                unsafe_allow_html=True
            )

        # Reset button
        if st.button("🔄 Check Another Article", use_container_width=True):
            st.session_state.result = None
            st.rerun()

# Footer
st.markdown("""
<div class="footer-text">
    <p>This tool uses machine learning to detect potentially fake news.</p>
    <p>Always verify information from multiple reliable sources.</p>
</div>
""", unsafe_allow_html=True)