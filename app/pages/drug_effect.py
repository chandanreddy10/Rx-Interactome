import streamlit as st
import time
import streamlit.components.v1 as components
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import requests
import torch
import torch

st.set_page_config(
    page_title="Drug perturbation",
    layout="wide"
)
st.markdown("""
<style>
.stApp {
    background-color: #FFFFFF;
    color: #353535;
}

section[data-testid="stSidebar"] {
    background-color: #D9D9D9;
}

.stButton > button {
    background-color: #3C6E71;
    color: white;
    font-weight: bold;
    border-radius: 10px;
    border: none;
}

.stButton > button:hover {
    background-color: #284B63;
    color: white;
}

.title {
    font-size: 56px;
    font-weight: 700;
    color: #284B63;
    text-align: center;
}

textarea, input {
    background-color: #F5F5F5 !important;
    color: #353535 !important;
    border-radius: 6px !important;
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 0rem;
    padding-left: 5rem;
    padding-right: 5rem;
}
</style>
""", unsafe_allow_html=True)
@st.cache_resource
def load_model():
    model_id = "google/medgemma-4b-it"

    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_id)

    return model, processor

def generate_response(prompt, log_box):
    with log_box:
        with st.spinner("Loading the Model !", show_time=False):
            model, processor = load_model()
    with log_box:
        with st.spinner("Preparing the Model !", show_time=False):
            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You are a biomedical systems pharmacology expert."}]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"{prompt}"},
                    ]
                }
            ]

            inputs = processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt"
            ).to(model.device, dtype=torch.bfloat16)

            input_len = inputs["input_ids"].shape[-1]
    with log_box:
        with st.spinner("Running Medgemma!", show_time=False):
            with torch.inference_mode():
                generation = model.generate(**inputs, max_new_tokens=2000, do_sample=False)
                generation = generation[0][input_len:]

            decoded = processor.decode(generation, skip_special_tokens=True)
            return decoded 

if "graph" in st.session_state:
    GRAPH = st.session_state["graph"]
else:
    st.warning("No outputs found. Please run analysis first.")

if "interactome" in st.session_state:
    INTERACTOME = st.session_state["interactome"]
else:
    st.warning("Interactome failed to record.")

def stream_outputs():
    for output in st.session_state.effect_output:
        yield output
        time.sleep(0.005)
st.markdown('<div class="title">InteractomeRx 🧬</div>', unsafe_allow_html=True)
st.divider()
col1, col2, = st.columns(2)
with col1:
    st.header("Interactome", anchor=False)
    components.html(GRAPH, height=780, scrolling=True)

if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False

with col2:
    st.header("Test Effect", anchor=False)

    if not st.session_state.analysis_done:

        effect_prompt = st.text_area(
            "Enter the effects here",
            height=500,
            placeholder="add your prompt here..."
        )
        effect_prompt = f"{INTERACTOME} {effect_prompt} Note: Only output the Drug effects on the protein sets ! Do not explain the protein sets."
        button_col, log_col = st.columns([0.4, 0.7])

        with button_col:
            run_clicked = st.button("Test Drug effect")

        with log_col:
            log_box = st.empty()

        if run_clicked:
            outputs= generate_response(effect_prompt, log_box)
            st.session_state.effect_output = outputs
            st.session_state.analysis_done = True
            st.rerun()

    else:
        st.subheader("Results")
        st.write_stream(stream_outputs)
        st.session_state["effect_output"] = ""
        bcol1, bcol2 = st.columns(2)
        with bcol1:
            if st.button("Try New effect!", width="stretch"):
                st.session_state.analysis_done = False
                st.rerun()
        with bcol2:
            if st.button("Try New protein set!", width="stretch"):
                st.session_state.analysis_done = False
                st.session_state["outputs"] = ""
                st.switch_page("app.py")