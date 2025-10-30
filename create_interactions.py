import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski
import requests
import time

# ==================== METHOD 1: From DrugBank CSV ====================
def create_from_drugbank(drugbank_csv_path, your_drugs_csv, your_diseases_csv):
    """
    Create interactions from DrugBank download
    
    DrugBank CSV should have columns:
    - DrugBank ID, Name, Indication, SMILES
    """
    print("Loading DrugBank data...")
    drugbank_df = pd.read_csv(drugbank_csv_path)
    
    print("Loading your data...")
    drugs_df = pd.read_csv(your_drugs_csv, header=None, names=['smiles'])
    drugs_df['drug_name'] = [f"Drug_{i+1}" for i in range(len(drugs_df))]
    
    diseases_df = pd.read_csv(your_diseases_csv)
    diseases_df.columns = diseases_df.columns.str.lower().str.replace('_', '')
    diseases_df.rename(columns={'diseaseid': 'disease_id', 'diseasename': 'disease_name'}, inplace=True)
    
    interactions = []
    
    # Map DrugBank indications to your diseases
    for idx, drug in drugs_df.iterrows():
        # Find similar drug in DrugBank by SMILES
        for _, db_drug in drugbank_df.iterrows():
            if pd.notna(db_drug.get('SMILES')) and pd.notna(drug['smiles']):
                if db_drug['SMILES'].strip() == drug['smiles'].strip():
                    # Found matching drug
                    indication = str(db_drug.get('Indication', '')).lower()
                    
                    # Match indication to your diseases
                    for _, disease in diseases_df.iterrows():
                        disease_name_lower = disease['disease_name'].lower()
                        
                        # Positive label if indication mentions disease
                        if any(word in indication for word in disease_name_lower.split()):
                            interactions.append({
                                'drug_name': drug['drug_name'],
                                'disease_id': disease['disease_id'],
                                'label': 1
                            })
                        else:
                            # Add some negative samples
                            if np.random.random() < 0.1:  # 10% negative samples
                                interactions.append({
                                    'drug_name': drug['drug_name'],
                                    'disease_id': disease['disease_id'],
                                    'label': 0
                                })
    
    interactions_df = pd.DataFrame(interactions)
    interactions_df.to_csv('interactions.csv', index=False)
    print(f"\n✅ Created interactions.csv with {len(interactions_df)} interactions")
    print(f"   Positive samples: {sum(interactions_df['label']==1)}")
    print(f"   Negative samples: {sum(interactions_df['label']==0)}")
    
    return interactions_df


# ==================== METHOD 2: From KEGG API ====================
def create_from_kegg_api(your_drugs_csv, your_diseases_csv, max_drugs=100):
    """
    Create interactions using KEGG REST API
    Warning: Slow (API rate limits), but free!
    """
    print("Loading your data...")
    drugs_df = pd.read_csv(your_drugs_csv, header=None, names=['smiles'])
    drugs_df['drug_name'] = [f"Drug_{i+1}" for i in range(len(drugs_df))]
    
    diseases_df = pd.read_csv(your_diseases_csv)
    diseases_df.columns = diseases_df.columns.str.lower().str.replace('_', '')
    diseases_df.rename(columns={'diseaseid': 'disease_id', 'diseasename': 'disease_name'}, inplace=True)
    
    interactions = []
    
    print(f"Querying KEGG API (this will take ~{max_drugs} minutes)...")
    
    for idx, drug in drugs_df.head(max_drugs).iterrows():
        try:
            # Search KEGG for this compound
            smiles = drug['smiles']
            
            # This is a placeholder - KEGG API doesn't directly support SMILES search
            # You'd need to manually map your SMILES to KEGG drug IDs
            
            print(f"  Processing drug {idx+1}/{max_drugs}...")
            time.sleep(1)  # Rate limiting
            
        except Exception as e:
            print(f"  Error for drug {idx}: {e}")
            continue
    
    # For now, return empty
    print("\n⚠️  KEGG API method requires manual drug ID mapping")
    print("    Consider using DrugBank or ChEMBL instead")
    
    return pd.DataFrame()


# ==================== METHOD 3: Rule-Based Synthetic ====================
def create_rule_based_synthetic(your_drugs_csv, your_diseases_csv):
    """
    Create better synthetic interactions using molecular properties
    and disease name matching
    """
    print("Loading your data...")
    drugs_df = pd.read_csv(your_drugs_csv, header=None, names=['smiles'])
    drugs_df['drug_name'] = [f"Drug_{i+1}" for i in range(len(drugs_df))]
    
    diseases_df = pd.read_csv(your_diseases_csv)
    diseases_df.columns = diseases_df.columns.str.lower().str.replace('_', '')
    diseases_df.rename(columns={'diseaseid': 'disease_id', 'diseasename': 'disease_name'}, inplace=True)
    
    print("Analyzing molecular structures...")
    
    # Categorize diseases
    cancer_keywords = ['cancer', 'carcinoma', 'leukemia', 'lymphoma', 'sarcoma', 'tumor', 'malignant']
    infection_keywords = ['infection', 'bacterial', 'viral', 'fungal', 'sepsis']
    cardiac_keywords = ['heart', 'cardiac', 'cardiovascular', 'hypertension']
    metabolic_keywords = ['diabetes', 'metabolic', 'obesity', 'hyperlipidemia']
    inflammatory_keywords = ['arthritis', 'inflammation', 'autoimmune', 'lupus']
    neurological_keywords = ['alzheimer', 'parkinson', 'epilepsy', 'seizure', 'depression']
    
    def categorize_disease(disease_name):
        disease_lower = disease_name.lower()
        categories = []
        if any(kw in disease_lower for kw in cancer_keywords):
            categories.append('cancer')
        if any(kw in disease_lower for kw in infection_keywords):
            categories.append('infection')
        if any(kw in disease_lower for kw in cardiac_keywords):
            categories.append('cardiac')
        if any(kw in disease_lower for kw in metabolic_keywords):
            categories.append('metabolic')
        if any(kw in disease_lower for kw in inflammatory_keywords):
            categories.append('inflammatory')
        if any(kw in disease_lower for kw in neurological_keywords):
            categories.append('neurological')
        return categories if categories else ['other']
    
    # Add disease categories
    diseases_df['categories'] = diseases_df['disease_name'].apply(categorize_disease)
    
    interactions = []
    valid_drugs = 0
    
    for idx, drug in drugs_df.iterrows():
        mol = Chem.MolFromSmiles(drug['smiles'])
        if mol is None:
            continue
        
        valid_drugs += 1
        
        # Calculate molecular properties
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        num_aromatic_rings = Lipinski.NumAromaticRings(mol)
        num_h_acceptors = Lipinski.NumHAcceptors(mol)
        num_h_donors = Lipinski.NumHDonors(mol)
        
        # Rule-based assignment
        likely_categories = []
        
        # Large molecules with many aromatic rings → cancer drugs
        if num_aromatic_rings >= 3 and mw > 400:
            likely_categories.append('cancer')
        
        # Small molecules, high LogP → cardiac/metabolic
        if mw < 400 and logp > 2:
            likely_categories.extend(['cardiac', 'metabolic'])
        
        # Many H-bond donors/acceptors → inflammatory
        if num_h_donors + num_h_acceptors > 8:
            likely_categories.append('inflammatory')
        
        # Medium size, moderate LogP → neurological
        if 300 < mw < 500 and 1 < logp < 4:
            likely_categories.append('neurological')
        
        # If no specific category, assign to general
        if not likely_categories:
            likely_categories = ['other']
        
        # Create interactions based on matching categories
        for _, disease in diseases_df.iterrows():
            disease_cats = disease['categories']
            
            # Positive label if categories match
            if any(cat in disease_cats for cat in likely_categories):
                # High confidence match
                interactions.append({
                    'drug_name': drug['drug_name'],
                    'disease_id': disease['disease_id'],
                    'label': 1
                })
            else:
                # Add some negative samples (20% chance)
                if np.random.random() < 0.2:
                    interactions.append({
                        'drug_name': drug['drug_name'],
                        'disease_id': disease['disease_id'],
                        'label': 0
                    })
        
        if (idx + 1) % 500 == 0:
            print(f"  Processed {idx+1}/{len(drugs_df)} drugs...")
    
    interactions_df = pd.DataFrame(interactions)
    interactions_df = interactions_df.sample(frac=1).reset_index(drop=True)  # Shuffle
    interactions_df.to_csv('interactions.csv', index=False)
    
    print(f"\n✅ Created interactions.csv with {len(interactions_df)} interactions")
    print(f"   Valid drugs processed: {valid_drugs}")
    print(f"   Positive samples: {sum(interactions_df['label']==1)}")
    print(f"   Negative samples: {sum(interactions_df['label']==0)}")
    print(f"   Ratio: {sum(interactions_df['label']==1)/len(interactions_df)*100:.1f}% positive")
    
    return interactions_df


# ==================== METHOD 4: From ChEMBL Database ====================
def create_from_chembl(your_drugs_csv, your_diseases_csv):
    """
    Query ChEMBL database for drug-disease associations
    Requires: pip install chembl_webresource_client
    """
    try:
        from chembl_webresource_client.new_client import new_client
    except ImportError:
        print("❌ Please install: pip install chembl_webresource_client")
        return pd.DataFrame()
    
    print("Loading your data...")
    drugs_df = pd.read_csv(your_drugs_csv, header=None, names=['smiles'])
    drugs_df['drug_name'] = [f"Drug_{i+1}" for i in range(len(drugs_df))]
    
    diseases_df = pd.read_csv(your_diseases_csv)
    diseases_df.columns = diseases_df.columns.str.lower().str.replace('_', '')
    diseases_df.rename(columns={'diseaseid': 'disease_id', 'diseasename': 'disease_name'}, inplace=True)
    
    print("Querying ChEMBL database...")
    
    molecule = new_client.molecule
    interactions = []
    
    for idx, drug in drugs_df.head(100).iterrows():  # Limit to 100 for testing
        try:
            # Search ChEMBL by SMILES
            results = molecule.filter(molecule_structures__canonical_smiles__exact=drug['smiles'])
            
            if results:
                chembl_id = results[0]['molecule_chembl_id']
                
                # Get drug indications
                drug_indication = new_client.drug_indication
                indications = drug_indication.filter(molecule_chembl_id=chembl_id)
                
                for indication in indications:
                    indication_text = indication.get('mesh_heading', '').lower()
                    
                    # Match to your diseases
                    for _, disease in diseases_df.iterrows():
                        if any(word in indication_text for word in disease['disease_name'].lower().split()):
                            interactions.append({
                                'drug_name': drug['drug_name'],
                                'disease_id': disease['disease_id'],
                                'label': 1
                            })
            
            if (idx + 1) % 10 == 0:
                print(f"  Processed {idx+1} drugs...")
            
        except Exception as e:
            print(f"  Error for drug {idx}: {e}")
            continue
    
    if interactions:
        interactions_df = pd.DataFrame(interactions)
        interactions_df.to_csv('interactions.csv', index=False)
        print(f"\n✅ Created interactions.csv with {len(interactions_df)} interactions")
        return interactions_df
    else:
        print("\n⚠️  No interactions found in ChEMBL")
        return pd.DataFrame()


# ==================== MAIN ====================
if __name__ == "__main__":
    print("="*60)
    print("Drug-Disease Interactions Creator")
    print("="*60 + "\n")
    
    # Choose your method
    print("Available methods:")
    print("1. Rule-based synthetic (RECOMMENDED - works immediately)")
    print("2. From DrugBank CSV (requires DrugBank account)")
    print("3. From ChEMBL API (slow but accurate)")
    print("4. From KEGG API (requires manual mapping)")
    
    method = input("\nChoose method (1-4): ").strip()
    
    your_drugs_csv = 'drugs.csv'
    your_diseases_csv = 'diseases.csv'
    
    if method == '1':
        print("\n🔧 Using rule-based synthetic method...")
        interactions_df = create_rule_based_synthetic(your_drugs_csv, your_diseases_csv)
    
    elif method == '2':
        drugbank_csv = input("Path to DrugBank CSV: ").strip()
        interactions_df = create_from_drugbank(drugbank_csv, your_drugs_csv, your_diseases_csv)
    
    elif method == '3':
        interactions_df = create_from_chembl(your_drugs_csv, your_diseases_csv)
    
    elif method == '4':
        interactions_df = create_from_kegg_api(your_drugs_csv, your_diseases_csv)
    
    else:
        print("Invalid choice!")
        exit()
    
    if not interactions_df.empty:
        print("\n" + "="*60)
        print("✅ Success! You can now train with:")
        print("   python train_gnn.py")
        print("="*60)
        
        # Show sample
        print("\nSample interactions:")
        print(interactions_df.head(10))
    else:
        print("\n❌ Failed to create interactions.csv")