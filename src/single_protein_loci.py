def compute_esm2_embeddings_loci_per_protein(general_path, data_suffix='', add=False):
    """
    This function computes ESM-2 embeddings for each individual protein within loci, from the Locibase.json file.

    INPUTS:
    - general path to the project data folder
    - data suffix to optionally add to the saved file name (default='')
    OUTPUT: esm2_embeddings_loci_per_protein.csv (with one embedding per protein)
    """
    import json
    import pandas as pd
    import numpy as np
    import torch
    import esm
    from tqdm import tqdm

    # Load ESM-2 model
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model.eval()  # disables dropout for deterministic results

    # Load json file
    with open(general_path + '/Locibase' + data_suffix + '.json') as dict_file:
        loci_dict = json.load(dict_file)

    if add:
        old_embeddings_df = pd.read_csv(general_path + '/esm2_embeddings_loci_per_protein' + data_suffix + '.csv')
        processed_accession_proteins = set(zip(old_embeddings_df['accession'], old_embeddings_df['protein_index']))
        for key in list(loci_dict.keys()):
            loci_dict[key] = [seq for i, seq in enumerate(loci_dict[key]) if (key, i) not in processed_accession_proteins]
        print('Processing', sum(len(v) for v in loci_dict.values()), 'more protein sequences (add=True)')

    # Compute embeddings per protein
    protein_representations = []
    accessions = []
    protein_indices = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    for key in tqdm(loci_dict.keys(), desc="Embedding loci proteins"):
        for idx, sequence in enumerate(loci_dict[key]):
            data = [(f"{key}_prot_{idx}", sequence)]
            _, _, batch_tokens = batch_converter(data)
            batch_tokens = batch_tokens.to(device)
            try:
                with torch.no_grad():
                    results = model(batch_tokens, repr_layers=[33],
                                    return_contacts=False)   # <-- the fix
                rep = results["representations"][33]
                emb = rep[0, 1:len(sequence) + 1].mean(0).cpu().numpy()
            except torch.cuda.OutOfMemoryError:
                # rare very long locus protein: fall back to CPU for this one
                torch.cuda.empty_cache()
                with torch.no_grad():
                    results = model.cpu()(batch_tokens.cpu(), repr_layers=[33],
                                          return_contacts=False)
                rep = results["representations"][33]
                emb = rep[0, 1:len(sequence) + 1].mean(0).numpy()
                model = model.to(device)

            accessions.append(key)
            protein_indices.append(idx)
            protein_representations.append(emb)

    # Save results
    embeddings_df = pd.concat([
        pd.DataFrame({'accession': accessions, 'protein_index': protein_indices}),
        pd.DataFrame(protein_representations)
    ], axis=1)

    if add:
        embeddings_df = pd.concat([old_embeddings_df, embeddings_df], axis=0, ignore_index=True)

    embeddings_df.to_csv('./Data/esm2_embeddings_loci_per_protein' + data_suffix + '.csv', index=False)
    print("Saved embeddings to: ", './Data/esm2_embeddings_loci_per_protein' + data_suffix + '.csv')

    return embeddings_df


loci_path = "Data/" # add here your path to Locibase.json, downloaded from the Zenodo repository

compute_esm2_embeddings_loci_per_protein(loci_path)