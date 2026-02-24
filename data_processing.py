import pandas as pd
import numpy as np
import os 
from Bio import SeqIO
import csv

def write_sequence_to_csv(output_folder_name):
    os.makedirs(output_folder_name, exist_ok=True)
    sequence_file = "Downloads/9606.protein.sequences.v12.0.fa"
    with open(f"{output_folder_name}/output.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["ID", "Description", "Sequence"])
        
        for record in SeqIO.parse(sequence_file, "fasta"):
            writer.writerow([record.id, record.description, str(record.seq)])
    print("Saved to CSV file.")

def process_string_db_data(output_folder_name):
    os.makedirs(output_folder_name, exist_ok=True)
    protein_info = "Downloads/9606.protein.info.v12.0.txt"
    protein_links = "Downloads/9606.protein.links.full.v12.0.txt"
    protein_clusters_info = "Downloads/9606.clusters.info.v12.0.txt"
    protein_clusters = "Downloads/9606.clusters.proteins.v12.0.txt"

    df_pi = pd.read_csv(protein_info, sep="\t")
    df_pl_load = pd.read_csv(protein_links, sep="\t")
    df_pc_info = pd.read_csv(protein_clusters_info, sep="\t")
    df_pc = pd.read_csv(protein_clusters, sep="\t")

    print("Protein Info file has shape : {}".format(df_pi.shape))
    print("Protein Links file has shape : {}".format(df_pl_load.shape))
    print("Protein Clusters file has shape : {}".format(df_pc_info.shape))
    print("Protein Clusters : {}".format(df_pc.shape))

    columns = df_pl_load.columns[0].split(" ")
    chunks = np.arange(0, df_pl_load.shape[0], df_pl_load.shape[0] / 75)
    chunks = [int(chunk) for chunk in chunks]
    data = []
    prev_chunk_start_index = chunks[0]
    for chunk in chunks[1:]:
        print("On Row : {}".format(chunk))
        temp = df_pl_load.iloc[prev_chunk_start_index: chunk, 0].str.split(" ")[:]
        data.extend(temp)
        prev_chunk_start_index = chunk

    df_protein_links = pd.DataFrame(data=data, columns=columns)
    df_pi.to_parquet(f"{output_folder_name}/protein_info.parquet")
    df_pc_info.to_parquet(f"{output_folder_name}/protein_clusters_info.parquet")
    df_pc.to_parquet(f"{output_folder_name}/protein_clusters.parquet")
    df_protein_links.to_parquet(f"{output_folder_name}/protein_links_full.parquet")

    print("All files saved to Parquet successfully.")
    print(columns)

def process_did_data(output_folder_name):
    os.makedirs(output_folder_name, exist_ok=True)

    did_flat = "Downloads/3did_flat.tsv"
    df = pd.read_csv(did_flat, sep="\t")
    df.to_parquet(f"{output_folder_name}/did.parquet")
    print("Saved Files to Parquet.")

def process_compartments_data(output_folder_name):
    os.makedirs(output_folder_name, exist_ok=True)

    did_flat = "Downloads/human_compartment_integrated_full.tsv"
    df = pd.read_csv(did_flat, sep="\t")
    df.to_parquet(f"{output_folder_name}/human_compartment_loc.parquet")
    print("Saved Files to Parquet.")

def process_reactome_data(output_folder_name):
    column_names = [
    'Worker_ID', 
    'Pathway_ID', 
    'URL', 
    'Pathway_Name', 
    'Evidence_Code', 
    'Species'
    ]
    column_names_pathway = ["Pathway_ID",
                            "Function",
                            "Species"]
    os.makedirs(output_folder_name, exist_ok=True)
    reactome_file = "Downloads/Ensembl2Reactome_All_Levels.txt"
    pathways_file = "Downloads/ReactomePathways.txt"
    df = pd.read_csv(reactome_file, sep="\t", header=None, names=column_names)
    df = df[df["Species"]=="Homo sapiens"]
    df.reset_index(drop=True, inplace=True)
    
    df2 = pd.read_csv(pathways_file, sep="\t", header=None, names=column_names_pathway)
    df2 = df2[df2["Species"]=="Homo sapiens"]
    df2.reset_index(drop=True, inplace=True)
    
    df.to_parquet(f"{output_folder_name}/reactome_pathway.parquet")
    df2.to_parquet(f"{output_folder_name}/pathways.parquet")
    print("Saved Files to Parquet.")

def process_biogrid_data(output_folder_name):
    os.makedirs(output_folder_name, exist_ok=True)
    biogrid_file = "Downloads/BIOGRID-ALL-5.0.254.tab3.txt"

    df = pd.read_csv(biogrid_file, header=0, sep="\t",dtype={
        'Entrez Gene Interactor A': str,
        'Entrez Gene Interactor B': str,
        'Score': str,
        'Throughput': str
    })
    df = df.loc[(df['Organism Name Interactor A']=="Homo sapiens") & (df['Organism Name Interactor B']== "Homo sapiens")]
    df.reset_index(drop=True, inplace=True)

    df.to_parquet(f"{output_folder_name}/human_interaction_biogrid.parquet")
    print("Saved Files to Parquet.")
# process_string_db_data("files")
# process_did_data("files")
# process_compartments_data("files")
# process_reactome_data("files")
# process_biogrid_data("files")
write_sequence_to_csv("files")