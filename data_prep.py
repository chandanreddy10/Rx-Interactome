import pandas as pd
import numpy as np
import os
import logging
import random
from collections import defaultdict
from ollama import chat
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    filename="data_prep.log",
)
os.makedirs("sft_input_size_3", exist_ok=True)
os.makedirs("sft_output_size_3", exist_ok=True)

DG_PROMPT_FILE_1="data_generation_prompt.txt"
SUMMARIZE_PROMPT="data_generation_prompt_2.txt"
with open(DG_PROMPT_FILE_1,"r") as file:
    PROMPT = file.read()

with open(SUMMARIZE_PROMPT,"r") as file:
    SUM_PROMPT=file.read()

def call_gemma_for_sample(sample_data):
    return gemma_call(sample_data, PROMPT)

def gemma_call(data :dict, prompt: str) -> str:
    """Call Gemma3 model to generate a response for SFT."""
    message = f"{prompt} Data:{data}"
    response = chat(
        model='gemma3:12b',
        messages=[{'role': 'user', 'content': message}],
    )
    return response.message.content

def build_sample(user_prompt: str, assistant_response: str) -> dict:
    """Constructs SFT data sample"""
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": user_prompt}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": assistant_response}
                ]
            }
        ]
    }

def load_file(file, type="parquet"):
    if type == "csv":
        return pd.read_csv(file)
    elif type == "parquet":
        return pd.read_parquet(file)
    else:
        raise "File type not supported"


# Save Reactome Filtered Data
def save_reactome_for_proteins(
    reactome_file,
    protein_links_file,
    cluster_file,
    output_folder="Processed",
):

    os.makedirs(output_folder, exist_ok=True)

    reactome_df = pd.read_parquet(reactome_file)
    protein_links_df = pd.read_parquet(protein_links_file)
    protein_cluster_df = pd.read_parquet(cluster_file)

    # Keep only ENSP proteins
    new_reactome = reactome_df[
        reactome_df["Worker_ID"].str.startswith("ENSP", na=False)
    ]

    reactome_ids = set(new_reactome["Worker_ID"])

    # Clean STRING prefixes
    protein_links_df["protein1"] = protein_links_df["protein1"].str.replace(
        "9606.", "", regex=False
    )
    protein_links_df["protein2"] = protein_links_df["protein2"].str.replace(
        "9606.", "", regex=False
    )

    # Filter links
    new_pl = protein_links_df[
        protein_links_df["protein1"].isin(reactome_ids)
        & protein_links_df["protein2"].isin(reactome_ids)
    ].reset_index(drop=True)

    # Clean clusters
    protein_cluster_df["protein_id"] = protein_cluster_df["protein_id"].str.replace(
        "9606.", "", regex=False
    )

    valid_proteins = set(new_pl["protein1"]).union(new_pl["protein2"])

    protein_cluster_df = protein_cluster_df[
        protein_cluster_df["protein_id"].isin(valid_proteins)
    ].reset_index(drop=True)

    # Save
    new_pl.to_parquet(os.path.join(output_folder, "protein_links.parquet"))
    protein_cluster_df.to_parquet(
        os.path.join(output_folder, "protein_clusters.parquet")
    )

    logging.info("Saved processed files to %s", output_folder)


# Build Pathway Dictionary
def build_pathway_dict(reactome_df):

    return reactome_df.groupby("Worker_ID")["Pathway_Name"].agg(set).to_dict()


# Build Location Dictionary
def build_location_dict(loc_df):

    loc_df = loc_df[loc_df["5"] < 5]
    loc_df = loc_df[~loc_df["4"].astype(str).str.startswith("GO", na=False)]

    return loc_df.groupby("1")["4"].agg(set).to_dict()


# Build Cluster Dictionary
def build_cluster_dict(cluster_df):

    return cluster_df.groupby("cluster_id")["protein_id"].agg(set).to_dict()


# Subgraph Per Cluster
def return_list_of_proteins_fast(
    cluster_id,
    cluster_to_proteins,
    links_df,
    num_proteins=3,
):

    cluster_proteins = cluster_to_proteins.get(cluster_id)

    if not cluster_proteins or len(cluster_proteins) >= 15:
        return None

    # Keep only edges fully inside cluster
    subgraph = links_df[
        links_df["protein1"].isin(cluster_proteins)
        & links_df["protein2"].isin(cluster_proteins)
    ]

    if len(subgraph) <= 5:
        return None

    # Proteins participating in internal edges
    proteins_in_graph = set(subgraph["protein1"]).union(subgraph["protein2"])

    if len(proteins_in_graph) < num_proteins:
        return None

    sampled = set(random.sample(list(proteins_in_graph), num_proteins))
    final_subgraph = subgraph[
        subgraph["protein1"].isin(sampled) & subgraph["protein2"].isin(sampled)
    ].copy()

    return final_subgraph


# Add Pathways (FAST)
def add_pathways(sample, reactome_dict):

    sample["string_pathway"] = [
        ",".join(reactome_dict.get(p1, set()).union(reactome_dict.get(p2, set())))
        for p1, p2 in zip(sample["protein1"], sample["protein2"])
    ]

    return sample


# Add Locations (FAST)
def add_locations(sample, location_dict):

    sample["location"] = [
        ",".join(
            list(
                location_dict.get(p1, set()).intersection(location_dict.get(p2, set()))
            )[:5]
        )
        for p1, p2 in zip(sample["protein1"], sample["protein2"])
    ]

    return sample


def build_sequence_dict(sequence_df):
    sequence_df["ID"] = sequence_df["ID"].str.replace("9606.", "")
    sequence_dict = sequence_df.groupby("ID")["Sequence"].agg("first").to_dict()
    return sequence_dict


def add_sequence(sample_cluster, sequence_dict):

    sample_cluster["protein1_sequence"] = sample_cluster["protein1"].apply(
        lambda x: sequence_dict[x]
    )
    sample_cluster["protein2_sequence"] = sample_cluster["protein2"].apply(
        lambda x: sequence_dict[x]
    )
    return sample_cluster


def build_info_dict(info_df):
    info_df["#string_protein_id"] = info_df["#string_protein_id"].str.replace(
        "9606.", ""
    )
    info_df.rename(columns={"#string_protein_id": "protein_id"}, inplace=True)
    info_dict = (
        info_df.set_index("protein_id")[["protein_size", "annotation"]]
        .agg(list, axis=1)
        .to_dict()
    )
    return info_dict


def add_info(sample_cluster, info_dict):

    sample_cluster["protein1_size"] = sample_cluster["protein1"].apply(
        lambda x: info_dict[x][0]
    )
    sample_cluster["protein1_description"] = sample_cluster["protein1"].apply(
        lambda x: info_dict[x][1]
    )

    sample_cluster["protein2_size"] = sample_cluster["protein2"].apply(
        lambda x: info_dict[x][0]
    )
    sample_cluster["protein2_description"] = sample_cluster["protein2"].apply(
        lambda x: info_dict[x][1]
    )

    return sample_cluster


# MAIN EXECUTION
# protein_cluster_file = "files/protein_clusters.parquet"
# protein_links_file = "files/protein_links_full.parquet"
# save_reactome_for_proteins(reactome_file, protein_links_file, protein_cluster_file)

reactome_file = "files/reactome_pathway.parquet"
cluster_file = "Processed/protein_clusters.parquet"
links_file = "Processed/protein_links.parquet"
compartment_loc_file = "files/human_compartment_loc.parquet"
protein_info_file = "files/protein_info.parquet"
sequence_file = "files/output.csv"

cluster_df = load_file(cluster_file)
links_df = load_file(links_file)
reactome_df = load_file(reactome_file)
loc_df = load_file(compartment_loc_file)
info_df = load_file(protein_info_file)
sequence_df = load_file(sequence_file, "csv")

# Clean location column names if needed
loc_df = loc_df.rename(
    columns={
        "18S_rRNA": "1",
        "18S_rRNA.1": "2",
        "GO:0005840": "3",
        "Ribosome": "4",
        "3.633": "5",
    }
)

# Build lookup structures ONCE
reactome_dict = build_pathway_dict(reactome_df)
location_dict = build_location_dict(loc_df)
cluster_to_proteins = build_cluster_dict(cluster_df)
sequence_dict = build_sequence_dict(sequence_df)
info_dict = build_info_dict(info_df)

print("All structures built successfully.")

# Run Per Cluster
total_clusters = 0

for index, cluster_id in enumerate(cluster_to_proteins):
    if total_clusters <= 250:
        subgraph = return_list_of_proteins_fast(
            cluster_id, cluster_to_proteins, links_df, num_proteins=3
        )
        if subgraph is None:
            continue
        logging.info("Working on Cluster : {}".format(total_clusters+1))
        logging.info(f"""Cluster Brief Info.
                    Number of Edges : {subgraph.shape[0]}
                    Number of Protein1 : {len(subgraph.protein1.unique())}
                    """)
        subgraph = add_pathways(subgraph, reactome_dict)
        subgraph = add_locations(subgraph, location_dict)

        subgraph = add_sequence(subgraph, sequence_dict)
        subgraph = add_info(subgraph, info_dict)

        logging.info("Added Pathways, Locations, Sequence and Protein Information.")
        total_clusters += 1
        subgraph = subgraph.drop(columns=['coexpression_transferred', 'experiments', 'experiments_transferred',
        'database', 'database_transferred', 'textmining',
        'textmining_transferred',"combined_score"])
        # all_data = list of protein subgraph dicts
        all_outputs = []
        with ThreadPoolExecutor(max_workers=16) as executor:  # adjust number of threads
            futures = [executor.submit(call_gemma_for_sample, sample) for sample in subgraph.iterrows()]
            for future in as_completed(futures):
                all_outputs.append(future.result())
        logging.info("All the Subgraph outputs are collected for the cluster.")
        
        sft_output = gemma_call(all_outputs, SUM_PROMPT)
        logging.info("SFT output Generated.")
        logging.info(sft_output[random.choice([10,25,50]):random.choice([70,100])])

        protein1 = subgraph["protein1"].unique().tolist()
        protein2 = subgraph["protein2"].unique().tolist()

        protein1.extend(protein2)
        sft_input = ""
        for protein in protein1:
            sft_input = sft_input + f"{protein}:{sequence_dict[protein]}"

        with open(f"sft_input_size_3/sample{index}.txt", "w") as file:
            file.write(sft_input)
        
        with open(f"sft_output_size_3/sample{index}.txt", "w") as file:
            file.write(sft_output)
        logging.info("Saved Input and output for the cluster!!!.:)")
        logging.info(f"Clusters count : {total_clusters}")
    else:
        break
logging.info(f"Total Clusters Processed: {total_clusters}")
