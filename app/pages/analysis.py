import streamlit as st
import time
import streamlit.components.v1 as components


st.set_page_config(
    page_title="Analysis",
    layout="wide"
)
st.markdown("""
<style>

/* Main Background */
.stApp {
    background-color: #FFFFFF;   /* White */
    color: #353535;              /* Graphite text */
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #D9D9D9;   /* Alabaster Grey */
}

/* Buttons */
.stButton > button {
    background-color: #3C6E71;   /* Stormy Teal */
    color: white;
    font-weight: bold;
    border-radius: 10px;
    border: none;
}

.stButton > button:hover {
    background-color: #284B63;   /* Yale Blue */
    color: white;
}

/* Title */
.title {
    font-size: 56px;
    font-weight: 700;
    color: #284B63;
    margin-top: 0px; 
    padding-top: 0px;  
    text-align: center;    /* Yale Blue */
}

/* Text Inputs */
textarea, input {
    background-color: #F5F5F5 !important;
    color: #353535 !important;
    border-radius: 6px !important;
}

/* Cards (optional) */
.card {
    background-color: #D9D9D9;
    padding: 15px;
    border-radius: 10px;
    border: 1px solid #CCCCCC;
}

</style>
""", unsafe_allow_html=True)
st.markdown(
    """
    <style>
        /* Remove padding from the top of the main container */
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 0rem;
            padding-left: 5rem;
            padding-right: 5rem;
        }
        
        /* Optional: Remove the header bar's height/padding if needed */
        header {
            visibility: transparent;
            height: 0px;
        }
    </style>
    """,
    unsafe_allow_html=True
)
def get_model_output():
    for s in OUTPUTS:
        yield s
        time.sleep(0.0005)

if "outputs" in st.session_state:
    OUTPUTS = st.session_state["outputs"]
else:
    st.warning("No outputs found. Please run analysis first.")

if "graph" in st.session_state:
    GRAPH = st.session_state["graph"]
else:
    st.warning("No outputs found. Please run analysis first.")

st.markdown('<div class="title">InteractomeRx 🧬</div>', unsafe_allow_html=True)
st.divider()

col1, col2 = st.columns(spec=2)
with col1:
    st.header("Model Output", anchor=False)
    st.write_stream(get_model_output)
    st.session_state["outputs"] = ""
with col2:
    st.header("Interactome Visualised", anchor=False)
    components.html(GRAPH, height=780, scrolling=True)

if st.button("Try with another protein Set!"):
    st.switch_page("app.py")
   
dcol1, dcol2, dcol3 = st.columns(3)
with dcol2:
    if st.button("Analyse the drug Effects on this interactome"):
        st.switch_page("pages/drug_effect.py")