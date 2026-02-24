import streamlit as st
import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import json
import re
from pyvis.network import Network

st.set_page_config(
    page_title="Rx-Interactome",
    page_icon="🧬",
    layout="wide"
)

if "running" not in st.session_state:
    st.session_state.running = False

#Loading the model.
@st.cache_resource
def load_model():
    base_model_id = "google/medgemma-4b-it"
    adapter_path = "../medgemma-4b-it-sft-lora-interactome/checkpoint-678"

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=quant_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()

    return model, tokenizer


def generate_response(prompt: str, log_box):
    with log_box:
        with st.spinner("Loading the Model !", show_time=False):
            model, tokenizer = load_model()
    with log_box:
        with st.spinner("Preparing the Model !", show_time=False):
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with log_box:
        with st.spinner("Running Fine-tuned Medgemma!", show_time=False):
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=2000,
                    do_sample=False,
                    repetition_penalty=1.1,
                )
            generated_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
            return tokenizer.decode(generated_tokens, skip_special_tokens=True)

def extract_json(response: str):
    match = re.search(r"```json(.*?)```", response, re.DOTALL)
    if match:
        return json.loads(match.group(1).strip())
    return None

def generate_graph(response):

    data = extract_json(response)
    if not data:
        return None

    interactome = data["interactome"]

    net = Network(
        height="750px",
        width="100%",
        bgcolor="#FFFFFF",
        font_color="#353535",
        directed=True
    )

    net.barnes_hut()

    # Modern styling
    net.set_options("""
    var options = {
      "nodes": {
        "shape": "dot",
        "size": 22,
        "font": {
          "size": 16,
          "color": "#353535",
          "face": "Arial"
        },
        "borderWidth": 2
      },
      "edges": {
        "arrows": {
          "to": { "enabled": true, "scaleFactor": 0.8 }
        },
        "smooth": {
          "type": "dynamic"
        },
        "font": {
          "size": 14,
          "align": "middle"
        }
      },
      "physics": {
        "barnesHut": {
          "gravitationalConstant": -5000,
          "centralGravity": 0.3,
          "springLength": 160
        }
      }
    }
    """)

    # Node color palette (modern)
    node_colors = [
        "#284B63",  # Yale Blue
        "#3C6E71",  # Stormy Teal
        "#2A9D8F",  # Accent Teal
        "#E9C46A",  # Soft Gold
        "#E76F51"   # Coral
    ]

    # Add nodes with different colors
    for i, node in enumerate(interactome["nodes"]):
        net.add_node(
            node,
            label=node,
            color=node_colors[i % len(node_colors)]
        )

    # Edge styling based on strength
    for edge in interactome["edges"]:

        if edge["strength"] == "strong":
            color =  "#E76F51"
            width = 4
        elif edge["strength"] == "medium":
            color = "#E9C46A"
            width = 3
        else:
            color = "#2A9D8F"
            width = 2

        net.add_edge(
            edge["source"],
            edge["target"],
            title=edge.get("justification", ""),  # interaction point on hover
            color=color,
            width=width
        )

    return interactome, net.generate_html()

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
}

.title {
    font-size: 56px;
    font-weight: 700;
    color: #284B63;
    text-align: center;
}

.block-container {
    padding-top: 1.5rem;
    padding-left: 5rem;
    padding-right: 5rem;
}
</style>
""", unsafe_allow_html=True)


st.markdown('<div class="title">Rx-Interactome 🧬</div>', unsafe_allow_html=True)
st.divider()

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div style="text-align: left; margin-top: 200px;">
        <h2 style="font-size: 38px; font-weight: 600;">
            Predicting Protein Interactomes and Drug Effect on the Interactomes.
        </h2>
        <h3 style="font-size: 40px; font-weight: 400; color: #284B63;">
            Powered by Med<span style="color:navy; font-weight:600;">Gemma</span>
            &
            Tx<span style="color:navy; font-weight:600;">Gemma</span>
        </h3>
    </div>
    """, unsafe_allow_html=True)


with col2:
    st.header("Input")

    prompt = st.text_area(
        "Enter Protein Sequences",
        height=400,
        placeholder="Add protein sequences here..."
    )
    prompt = f"""{prompt}"""
    button_col, log_col = st.columns([0.3, 0.7])

    with button_col:
        run_clicked = st.button("Run Analysis")

    with log_col:
        log_box = st.empty()

    if run_clicked:
        st.session_state.running = True

    if st.session_state.running:
        with log_box:
            with st.spinner("Starting Analysis !", show_time=True):
                time.sleep(0.5)

        response = generate_response(prompt, log_box)
        print(response)
        interactome, graph = generate_graph(response)
        st.session_state["outputs"] = response
        st.session_state["graph"] = graph
        st.session_state["interactome"] = interactome
        st.session_state.running = False

        st.switch_page("pages/analysis.py")