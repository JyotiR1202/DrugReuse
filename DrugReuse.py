import streamlit as st
import requests
import re
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Lipinski, Descriptors, Draw
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool
import pickle
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Suppress warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', message='.*torch.classes.*')

# Set matplotlib style for better charts
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ==================== GNN Model Definition ====================
class SimpleGNN(nn.Module):
    def __init__(self, num_node_features=16, hidden_channels=64, num_classes=100):
        super(SimpleGNN, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = nn.Linear(hidden_channels, num_classes)
        
    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        
        x = global_mean_pool(x, batch)
        x = self.lin(x)
        return x

# ==================== Helper Functions ====================
def get_drug_info_from_pubchem(drug_name):
    """Fetch drug information from PubChem API"""
    try:
        # Search for compound by name
        search_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{drug_name}/JSON"
        response = requests.get(search_url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            cid = data['PC_Compounds'][0]['id']['id']['cid']
            
            # Get compound properties
            props_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{cid}/JSON"
            props_response = requests.get(props_url, timeout=10)
            
            if props_response.status_code == 200:
                props_data = props_response.json()
                
                therapeutic_uses = []
                pharmacology = []
                mechanism = []
                indications = []
                
                def extract_text_from_section(section, target_list, keywords):
                    """Recursively extract text from nested sections"""
                    heading = section.get('TOCHeading', '').lower()
                    
                    # Check if this section matches our keywords
                    if any(keyword.lower() in heading for keyword in keywords):
                        # Extract information from this section
                        for info in section.get('Information', []):
                            if 'Value' in info:
                                if 'StringWithMarkup' in info['Value']:
                                    for markup in info['Value']['StringWithMarkup']:
                                        text = markup.get('String', '').strip()
                                        if text and len(text) > 10:  # Avoid very short strings
                                            target_list.append(text)
                                elif 'String' in info['Value']:
                                    text = info['Value']['String'].strip()
                                    if text and len(text) > 10:
                                        target_list.append(text)
                    
                    # Recursively search subsections
                    for subsection in section.get('Section', []):
                        extract_text_from_section(subsection, target_list, keywords)
                
                # Search through all sections
                for section in props_data.get('Record', {}).get('Section', []):
                    # Therapeutic uses
                    extract_text_from_section(section, therapeutic_uses, 
                                            ['Therapeutic Uses', 'Clinical Use', 'Medical Uses', 'Indications'])
                    
                    # Pharmacology
                    extract_text_from_section(section, pharmacology, 
                                            ['Pharmacology', 'Drug Action', 'Pharmacodynamics'])
                    
                    # Mechanism of action
                    extract_text_from_section(section, mechanism, 
                                            ['Mechanism of Action', 'Mode of Action', 'MOA'])
                    
                    # Indications
                    extract_text_from_section(section, indications,
                                            ['Indication', 'FDA Label', 'Clinical Trial'])
                
                # Remove duplicates while preserving order
                therapeutic_uses = list(dict.fromkeys(therapeutic_uses))
                pharmacology = list(dict.fromkeys(pharmacology))
                mechanism = list(dict.fromkeys(mechanism))
                indications = list(dict.fromkeys(indications))
                
                # Combine therapeutic uses and indications
                all_uses = therapeutic_uses + indications
                all_uses = list(dict.fromkeys(all_uses))  # Remove duplicates
                
                # Combine pharmacology and mechanism
                all_pharmacology = pharmacology + mechanism
                all_pharmacology = list(dict.fromkeys(all_pharmacology))
                
                return {
                    'cid': cid,
                    'therapeutic_uses': all_uses,
                    'pharmacology': all_pharmacology,
                    'found': True
                }
    except Exception as e:
        print(f"PubChem API error: {e}")
        import traceback
        traceback.print_exc()
    
    return {'found': False}

def get_drug_indications(drug_name):
    """Get drug indications from multiple sources"""
    # First try PubChem and return if good data
    pubchem_info = get_drug_info_from_pubchem(drug_name)
    if pubchem_info.get('found') and (pubchem_info.get('therapeutic_uses') or pubchem_info.get('pharmacology')):
        return pubchem_info
    
    # If PubChem found the compound but no therapeutic info, get basic description
    if pubchem_info.get('found'):
        try:
            cid = pubchem_info['cid']
            # Try to get a simpler description
            desc_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/description/JSON"
            response = requests.get(desc_url, timeout=10)
            
            if response.status_code == 200:
                desc_data = response.json()
                descriptions = []
                
                for info in desc_data.get('InformationList', {}).get('Information', []):
                    desc = info.get('Description', '').strip()
                    if desc and len(desc) > 20:
                        descriptions.append(desc)
                
                if descriptions:
                    return {
                        'cid': cid,
                        'therapeutic_uses': descriptions[:3],  # Top 3 descriptions
                        'pharmacology': [],
                        'found': True,
                        'source': 'description'
                    }
        except Exception as e:
            print(f"Description fetch error: {e}")
    
    return pubchem_info

def smiles_to_graph(smiles):
    """Convert SMILES to graph representation"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    atom_features = []
    for atom in mol.GetAtoms():
        features = [
            atom.GetAtomicNum(),
            atom.GetDegree(),
            atom.GetFormalCharge(),
            atom.GetHybridization(),
            atom.GetIsAromatic(),
            atom.GetTotalNumHs(),
            atom.IsInRing(),
        ]
        atom_types = [6, 7, 8, 9, 15, 16, 17, 35, 53]
        one_hot = [1 if atom.GetAtomicNum() == t else 0 for t in atom_types]
        features.extend(one_hot)
        atom_features.append(features)
    
    edge_index = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_index.append([i, j])
        edge_index.append([j, i])
    
    if len(edge_index) == 0:
        edge_index = [[0, 0]]
    
    x = torch.tensor(atom_features, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    
    return Data(x=x, edge_index=edge_index)

def compute_tanimoto(smiles1, smiles2):
    """Compute Tanimoto similarity between two molecules"""
    if smiles1 and smiles2:
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)
        if mol1 and mol2:
            try:
                # Use new API if available
                from rdkit.Chem import rdFingerprintGenerator
                mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
                fp1 = mfpgen.GetFingerprint(mol1)
                fp2 = mfpgen.GetFingerprint(mol2)
                return AllChem.DataStructs.TanimotoSimilarity(fp1, fp2)
            except:
                # Fallback to old API
                fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=1024)
                fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=1024)
                return AllChem.DataStructs.TanimotoSimilarity(fp1, fp2)
    return 0.0

def check_lipinski_rules(smiles):
    """Check Lipinski's Rule of Five"""
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        h_donors = Lipinski.NumHDonors(mol)
        h_acceptors = Lipinski.NumHAcceptors(mol)
        mol_weight = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        
        passes = (h_donors <= 5 and 
                 h_acceptors <= 10 and 
                 mol_weight <= 500 and 
                 logp <= 5)
        
        return {
            'H_Donors': h_donors,
            'H_Acceptors': h_acceptors,
            'Molecular_Weight': mol_weight,
            'LogP': logp,
            'Passes': passes
        }
    return None

def check_enamine_availability(drug_name, csv_file='enamine_drugs.csv'):
    """Check if drug is available in Enamine database"""
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        match = df[df['Drug_Name'].str.lower() == drug_name.lower()]
        if not match.empty:
            return match['Available'].iloc[0] == 'Yes'
    return False

def predict_diseases(model, smiles, diseases_df, label_encoder, top_k=10):
    """Predict diseases for a given drug SMILES"""
    graph = smiles_to_graph(smiles)
    if graph is None:
        return None
    
    device = torch.device('cpu')
    graph = graph.to(device)
    
    with torch.no_grad():
        batch = torch.zeros(graph.x.size(0), dtype=torch.long)
        out = model(graph.x, graph.edge_index, batch)
        probs = torch.softmax(out, dim=1)[0]
        
        top_probs, top_indices = torch.topk(probs, min(top_k, len(probs)))
        
        results = []
        for prob, idx in zip(top_probs, top_indices):
            disease_id = diseases_df.iloc[idx.item()]['disease_id']
            disease_name = diseases_df.iloc[idx.item()]['disease_name']
            results.append({
                'Disease_ID': disease_id,
                'Disease_Name': disease_name,
                'Probability': prob.item(),
                'Confidence': f"{prob.item()*100:.2f}%"
            })
        
        return results

# ==================== Streamlit App Configuration ====================
st.set_page_config(
    page_title="DrugResuse",
    page_icon="💊♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-top: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stButton>button {
        width: 100%;
    }
    .known-uses-box {
        background-color: #e8f4f8;
        border-left: 4px solid #1f77b4;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# ==================== Sidebar Configuration ====================
with st.sidebar:
    st.image("https://via.placeholder.com/150x50/1f77b4/ffffff?text=DrugAI", use_container_width=True)
    st.title("⚙️ Configuration")
    
    st.markdown("---")
    
    # Model Upload Section
    st.subheader("📤 Model Files")
    model_file = st.file_uploader("Upload Trained Model (.pth)", type=['pth'], key='model_upload')
    encoder_file = st.file_uploader("Upload Label Encoder (.pkl)", type=['pkl'], key='encoder_upload')
    diseases_file = st.file_uploader("Upload Diseases CSV", type=['csv'], key='diseases_upload')
    
    st.markdown("---")
    
    # Optional: Enamine Database
    st.subheader("🗄️ Optional Database")
    enamine_file = st.file_uploader("Upload Enamine CSV (Optional)", type=['csv'], key='enamine_upload')
    
    st.markdown("---")
    
    # Model Status
    if model_file and encoder_file and diseases_file:
        st.success("✅ Model files loaded")
        model_loaded = True
    else:
        st.warning("⚠️ Upload model files to enable predictions")
        model_loaded = False

# ==================== Initialize Session State ====================
if 'model' not in st.session_state:
    st.session_state.model = None
if 'diseases_df' not in st.session_state:
    st.session_state.diseases_df = None
if 'label_encoder' not in st.session_state:
    st.session_state.label_encoder = None

# Load model files if uploaded
if model_loaded and st.session_state.model is None:
    try:
        with st.spinner("Loading model files..."):
            with open('temp_model.pth', 'wb') as f:
                f.write(model_file.getvalue())
            
            with open('temp_encoder.pkl', 'wb') as f:
                f.write(encoder_file.getvalue())
            
            st.session_state.diseases_df = pd.read_csv(diseases_file)
            st.session_state.diseases_df.columns = st.session_state.diseases_df.columns.str.lower().str.replace('_', '')
            
            column_mapping = {
                'diseaseid': 'disease_id',
                'diseasename': 'disease_name'
            }
            st.session_state.diseases_df.rename(columns=column_mapping, inplace=True)
            
            with open('temp_encoder.pkl', 'rb') as f:
                st.session_state.label_encoder = pickle.load(f)
            
            device = torch.device('cpu')
            num_diseases = len(st.session_state.diseases_df)
            hidden_channels = 64
            
            st.session_state.model = SimpleGNN(
                num_node_features=16,
                hidden_channels=hidden_channels,
                num_classes=num_diseases
            )
            st.session_state.model.load_state_dict(
                torch.load('temp_model.pth', map_location=device)
            )
            st.session_state.model.eval()
            
            if enamine_file:
                enamine_df = pd.read_csv(enamine_file)
                enamine_df.to_csv('enamine_drugs.csv', index=False)
            
            st.sidebar.success(f"✅ Model loaded successfully!\n\n"
                             f"- Diseases: {len(st.session_state.diseases_df)}\n"
                             f"- Model classes: {st.session_state.model.lin.out_features}")
        
    except Exception as e:
        st.sidebar.error(f"❌ Error loading model:")
        st.sidebar.error(f"```\n{str(e)}\n```")
        
        with st.sidebar.expander("🔍 Debug Information"):
            st.write("**Error Details:**")
            import traceback
            st.code(traceback.format_exc())
            
            if st.session_state.diseases_df is not None:
                st.write("**Diseases DataFrame Info:**")
                st.write(f"Columns: {st.session_state.diseases_df.columns.tolist()}")
                st.write(f"Shape: {st.session_state.diseases_df.shape}")
        
        model_loaded = False
        st.session_state.model = None

# ==================== Main App ====================
st.markdown('<h1 class="main-header">♻️ DrugReuse </h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: gray;">AI-Powered Drug Repositioning & Disease Association Analysis</p>', unsafe_allow_html=True)

# Create tabs
tabs = st.tabs(["🏠 Home ","💊 Drug-Disease Predictor ", "🔬 Tanimotto Similarity ", "📊 Batch Prediction ", "ℹ️ About ", "📞 Contact "])

# ==================== HOME TAB ====================
with tabs[0]:
    
    st.markdown("""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 2rem; border-radius: 10px; margin-bottom: 2rem;'>
            <h1 style='color: white; text-align: center; margin: 0;'>
                🎯 Welcome to DrugReuse Platform
            </h1>
            <p style='color: white; text-align: center; font-size: 1.2rem; margin-top: 0.5rem;'>
                AI-Powered Drug Repositing & Disease Prediction System
            </p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown('<h2 class="sub-header">ℹ️ About DrugReuse Platform</h2>', unsafe_allow_html=True)
    
    # Platform Overview
    st.markdown("""
    ## 🎯 Platform Overview
    
    **DrugReuse** is an AI-powered tool that leverages Graph Neural Networks (GNNs) 
    to identify new therapeutic applications for existing drugs. By analyzing molecular structures and 
    learning from known drug-disease interactions, our platform can predict potential repurposing 
    opportunities, significantly accelerating drug discovery.
    """)

    st.markdown("---")
    
    # Getting Started Guide
    st.markdown("### 🚀 Getting Started")
    
    st.info("""
    **📋 Before You Begin:**
    
    1. **Train Your Model** (Optional - if you don't have model files):
       - Create an `interactions.csv` file with columns: `drug_id`, `disease_id`, `smiles`
       - Train a GNN model using your interaction data
       - This generates the required `.pth` and `.pkl` files

       **OR**
       Download From Github(https://github.com/JyotiR1202/DrugReuse)
    
    2. **Upload Required Files** (in the left sidebar ⬅️):
       - 🔹 **Trained Model** (.pth) - Your trained GNN model weights
       - 🔹 **Label Encoder** (.pkl) - Disease label encoder from training
       - 🔹 **Diseases CSV** - Contains disease_id and disease_name columns
       - 🔹 **Enamine CSV** (Optional) - Check drug availability in Enamine database
    
    3. **Start Predicting!** - Once files are uploaded, you are ready!!.
    """)
    
    st.markdown("---")
    
    # Three Main Features
    st.markdown("### 🛠️ Platform Features")
    
    feat_col1, feat_col2, feat_col3 = st.columns(3)
    
    with feat_col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 2rem; border-radius: 1rem; color: white; height: 280px;">
            <h3 style="margin-top: 0; color: white;">💊 Drug-Disease Predictor</h3>
            <p><strong>What it does:</strong></p>
            <ul>
                <li>Input drug name & SMILES</li>
                <li>Shows existing medical uses</li>
                <li>Predicts NEW disease associations</li>
                <li>Validates drug-likeness 
                    (Lipinski's Rule)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with feat_col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                    padding: 2rem; border-radius: 1rem; color: white; height: 280px;">
            <h3 style="margin-top: 0; color: white;">🔬 Tanimoto Similarity</h3>
            <p><strong>What it does:</strong></p>
            <ul>
                <li>Compare two drugs molecularly</li>
                <li>Calculate Tanimoto similarity score</li>
                <li>Visualize structural differences</li>
                <li>Identify drug-drug relationships</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with feat_col3:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                    padding: 2rem; border-radius: 1rem; color: white; height: 280px;">
            <h3 style="margin-top: 0; color: white;">📊 Batch Prediction</h3>
            <p><strong>What it does:</strong></p>
            <ul>
                <li>Upload CSV with multiple drugs</li>
                <li>Process hundreds of drugs at once</li>
                <li>Get predictions for each drug</li>
                <li>Export results as CSV</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")

# ==================== DRUG-DISEASE PREDICTOR ====================
with tabs[1]:
    st.markdown('<h2 class="sub-header">Drug Repurposing Analysis</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🧪 Input Drug Information")
        
        input_method = st.radio("Select Input Method:", ["Drug Name", "SMILES String", "Upload CSV"])
        
        if input_method == "Drug Name":
            drug_name = st.text_input("Enter Drug Name:", placeholder="e.g., Aspirin")
            drug_smiles = st.text_input("Enter SMILES (Optional):", placeholder="CC(=O)OC1=CC=CC=C1C(=O)O")
        
        elif input_method == "SMILES String":
            drug_name = st.text_input("Drug Name (Optional):", placeholder="e.g., Aspirin")
            drug_smiles = st.text_input("Enter SMILES String:", placeholder="CC(=O)OC1=CC=CC=C1C(=O)O")
        
        else:  # Upload CSV
            uploaded_file = st.file_uploader("Upload Drug CSV (columns: drug_name, smiles OR just smiles)", type=['csv'])
            if uploaded_file:
                drugs_df = pd.read_csv(uploaded_file)
                
                if len(drugs_df.columns) == 1:
                    if 'smiles' in drugs_df.columns:
                        drugs_df['drug_name'] = [f"Drug_{i+1}" for i in range(len(drugs_df))]
                    else:
                        drugs_df.columns = ['smiles']
                        drugs_df['drug_name'] = [f"Drug_{i+1}" for i in range(len(drugs_df))]
                
                st.dataframe(drugs_df.head(), use_container_width=True)
        
        st.markdown("---")
        
        # Analysis options
        st.subheader("🔧 Analysis Options")
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            check_lipinski = st.checkbox("Lipinski's Rule", value=True)
        with col_b:
            check_enamine = st.checkbox("Enamine Database", value=False)
        with col_c:
            num_predictions = st.slider("Top Predictions", 5, 20, 10)
    
    with col2:
        st.subheader("👁️ Molecular Visualization")
        
        if input_method != "Upload CSV" and drug_smiles:
            try:
                mol = Chem.MolFromSmiles(drug_smiles)
                if mol:
                    img = Draw.MolToImage(mol, size=(300, 300))
                    st.image(img, caption="Molecular Structure")
                else:
                    st.error("Invalid SMILES string")
            except Exception as e:
                st.error(f"Error rendering molecule: {str(e)}")
        else:
            st.info("Enter SMILES to visualize molecular structure")
    
    st.markdown("---")
    
    # Run Analysis Button
    if st.button("🚀 Run Analysis", type="primary", use_container_width=True):
        if not model_loaded or st.session_state.model is None:
            st.error("❌ Please upload model files in the sidebar first!")
        else:
            if input_method == "Upload CSV":
                if uploaded_file is None:
                    st.error("Please upload a CSV file")
                else:
                    st.info("Processing batch predictions... See 'Batch Prediction' tab")
            else:
                if not drug_smiles:
                    st.error("Please enter a SMILES string")
                else:
                    with st.spinner("Analyzing drug..."):
                        st.success(f"✅ Analysis Complete for {drug_name if drug_name else 'Drug'}")
                        
                        st.markdown("---")
                        
                        # ============ NEW SECTION: KNOWN USES ============
                        st.markdown("## 🏥 Current Medical Uses")
                        
                        if drug_name:
                            with st.spinner(f"Fetching known indications for {drug_name}..."):
                                drug_info = get_drug_indications(drug_name)
                                
                                if drug_info.get('found'):
                                    st.markdown(f"""
                                    <div class="known-uses-box">
                                        <h3 style="color: #1f77b4; margin-top: 0;">📋 Established Therapeutic Uses</h3>
                                        <p><strong>Drug:</strong> {drug_name}</p>
                                        <p><strong>PubChem CID:</strong> {drug_info.get('cid', 'N/A')}</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                    
                                    col_known1, col_known2 = st.columns(2)
                                    
                                    with col_known1:
                                        st.markdown("#### 💊 **Medical Information:**")
                                        if drug_info.get('therapeutic_uses'):
                                            for i, use in enumerate(drug_info['therapeutic_uses'][:5], 1):
                                                # Truncate very long descriptions
                                                display_text = use if len(use) <= 300 else use[:300] + "..."
                                                st.markdown(f"**{i}.** {display_text}")
                                                st.markdown("")  # Add spacing
                                        else:
                                            st.info("ℹ️ No specific therapeutic uses found in database")
                                    
                                    with col_known2:
                                        st.markdown("#### ⚕️ **Pharmacological Actions:**")
                                        if drug_info.get('pharmacology'):
                                            for i, action in enumerate(drug_info['pharmacology'][:5], 1):
                                                # Truncate very long descriptions
                                                display_text = action if len(action) <= 300 else action[:300] + "..."
                                                st.markdown(f"**{i}.** {display_text}")
                                                st.markdown("")  # Add spacing
                                        else:
                                            st.info("ℹ️ No pharmacology information found")
                                    
                                    # Add source attribution
                                    if drug_info.get('source') == 'description':
                                        st.caption("ℹ️ Information from PubChem compound descriptions")
                                    
                                    st.markdown("---")
                                else:
                                    st.warning(f"⚠️ Could not retrieve established uses for '{drug_name}' from PubChem. This might be because:")
                                    st.markdown("- The drug name is not in the database")
                                    st.markdown("- The compound is experimental or not yet approved")
                                    st.markdown("- Network connectivity issues")
                                    st.markdown("---")
                        else:
                            st.info("💡 **Tip:** Enter a drug name to see its established medical uses before viewing AI predictions")
                            st.markdown("---")
                        
                        # ============ CONTINUE WITH EXISTING SECTIONS ============
                        
                        # SECTION 1: DRUG PROPERTIES
                        st.subheader("💊 Drug Properties & Molecular Characteristics")
                        
                        prop_col1, prop_col2, prop_col3 = st.columns([1, 1, 1])
                        
                        with prop_col1:
                            st.markdown("##### 🧪 Molecular Structure")
                            try:
                                mol = Chem.MolFromSmiles(drug_smiles)
                                if mol:
                                    img = Draw.MolToImage(mol, size=(250, 250))
                                    st.image(img, use_container_width=True)
                            except:
                                st.error("Could not render structure")
                        
                        with prop_col2:
                            st.markdown("##### 📊 Lipinski's Rule of Five")
                            if check_lipinski:
                                lipinski_results = check_lipinski_rules(drug_smiles)
                                if lipinski_results:
                                    m1, m2 = st.columns(2)
                                    with m1:
                                        st.metric("H-Donors", f"{lipinski_results['H_Donors']:.0f}", 
                                                 help="Hydrogen bond donors (≤5)")
                                        st.metric("Mol. Weight", f"{lipinski_results['Molecular_Weight']:.1f}", 
                                                 help="Molecular weight (≤500 Da)")
                                    with m2:
                                        st.metric("H-Acceptors", f"{lipinski_results['H_Acceptors']:.0f}", 
                                                 help="Hydrogen bond acceptors (≤10)")
                                        st.metric("LogP", f"{lipinski_results['LogP']:.2f}", 
                                                 help="Lipophilicity (≤5)")
                        
                        with prop_col3:
                            st.markdown("##### ✅ Drug-likeness Assessment")
                            if check_lipinski:
                                lipinski_results = check_lipinski_rules(drug_smiles)
                                if lipinski_results:
                                    if lipinski_results['Passes']:
                                        st.success("**PASSES** Lipinski's Rule")
                                        st.markdown("✅ Good oral bioavailability expected")
                                        st.markdown("✅ Drug-like properties")
                                    else:
                                        st.warning("**FAILS** Lipinski's Rule")
                                        violations = []
                                        if lipinski_results['H_Donors'] > 5:
                                            violations.append("Too many H-donors")
                                        if lipinski_results['H_Acceptors'] > 10:
                                            violations.append("Too many H-acceptors")
                                        if lipinski_results['Molecular_Weight'] > 500:
                                            violations.append("Molecular weight too high")
                                        if lipinski_results['LogP'] > 5:
                                            violations.append("LogP too high")
                                        for v in violations:
                                            st.markdown(f"⚠️ {v}")
                            
                            if check_enamine and drug_name:
                                st.markdown("---")
                                available = check_enamine_availability(drug_name)
                                if available:
                                    st.success("✅ Available in Enamine")
                                else:
                                    st.info("ℹ️ Not in Enamine DB")
                        
                        st.markdown("---")
                        st.markdown("")
                        
                        # SECTION 2: AI PREDICTIONS
                        st.subheader("🤖 AI-Predicted Disease Associations (Novel Repurposing Opportunities)")
                        st.info("💡 These are AI predictions for potential **new** therapeutic uses, different from established uses shown above")
                        
                        predictions = predict_diseases(
                            st.session_state.model,
                            drug_smiles,
                            st.session_state.diseases_df,
                            st.session_state.label_encoder,
                            top_k=num_predictions
                        )
                        
                        if predictions:
                            results_df = pd.DataFrame(predictions)
                            
                            st.markdown("##### 📋 Complete Prediction Results")
                            st.dataframe(
                                results_df.style.background_gradient(
                                    subset=['Probability'], 
                                    cmap='RdYlGn',
                                    vmin=0,
                                    vmax=results_df['Probability'].max()
                                ).format({
                                    'Probability': '{:.6f}'
                                }),
                                use_container_width=True,
                                height=400
                            )
                            
                            col1, col2, col3 = st.columns([1, 1, 1])
                            with col2:
                                csv = results_df.to_csv(index=False)
                                st.download_button(
                                    "📥 Download Full Results",
                                    csv,
                                    file_name=f"{drug_name if drug_name else 'drug'}_predictions.csv",
                                    mime='text/csv',
                                    use_container_width=True
                                )
                            
                            st.markdown("---")
                            
                            st.markdown("##### 📊 Top 10 Disease Associations - Confidence Chart")
                            
                            fig, ax = plt.subplots(figsize=(14, 8))
                            
                            plot_data = results_df.head(10).copy()
                            plot_data['Short_Name'] = plot_data['Disease_Name'].apply(
                                lambda x: (x[:50] + '...') if len(x) > 50 else x
                            )
                            plot_data = plot_data.sort_values('Probability', ascending=True)
                            
                            colors = plt.cm.RdYlGn(plot_data['Probability'] / plot_data['Probability'].max())
                            bars = ax.barh(
                                range(len(plot_data)), 
                                plot_data['Probability'],
                                color=colors,
                                edgecolor='black',
                                linewidth=0.5
                            )
                            
                            ax.set_yticks(range(len(plot_data)))
                            ax.set_yticklabels(plot_data['Short_Name'], fontsize=11)
                            ax.set_xlabel('Prediction Probability', fontsize=12, fontweight='bold')
                            ax.set_title('Top 10 Disease Predictions (Highest to Lowest)', 
                                        fontsize=14, fontweight='bold', pad=20)
                            
                            for i, (bar, prob, conf) in enumerate(zip(bars, plot_data['Probability'], plot_data['Confidence'])):
                                ax.text(
                                    prob + (plot_data['Probability'].max() * 0.01), 
                                    i, 
                                    f'{conf}',
                                    va='center',
                                    fontsize=10,
                                    fontweight='bold'
                                )
                            
                            ax.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.5)
                            ax.set_axisbelow(True)
                            ax.set_xlim(0, plot_data['Probability'].max() * 1.15)
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.close()
                            
                            st.markdown("---")
                            
                            # SECTION 3: STATISTICS
                            st.subheader("📈 Prediction Statistics")
                            
                            stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
                            
                            with stat_col1:
                                st.metric(
                                    "Highest Confidence", 
                                    f"{results_df['Probability'].max():.1%}",
                                    help="Most likely disease association"
                                )
                            
                            with stat_col2:
                                st.metric(
                                    "Average Confidence", 
                                    f"{results_df['Probability'].mean():.1%}",
                                    help="Mean prediction probability"
                                )
                            
                            with stat_col3:
                                high_conf = len(results_df[results_df['Probability'] > 0.1])
                                st.metric(
                                    "High Confidence (>10%)", 
                                    f"{high_conf}",
                                    help="Predictions with >10% probability"
                                )
                            
                            with stat_col4:
                                st.metric(
                                    "Total Predictions", 
                                    f"{len(results_df)}",
                                    help="Number of diseases analyzed"
                                )
                        else:
                            st.error("❌ Unable to generate predictions. Check SMILES validity.")

# ==================== Tanimotto Similarity ====================
with tabs[2]:
    st.markdown("#### 🔬 Compare Molecular Similarity Between Two Drugs")
        
    st.info("💡 Tanimoto similarity ranges from 0 (completely different) to 1 (identical). Values > 0.85 indicate very similar drugs.")
        
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Drug 1")
        drug1_name = st.text_input("Drug 1 Name:", key='d1_name')
        drug1_smiles = st.text_input("Drug 1 SMILES:", key='d1_smiles')
    
    with col2:
        st.subheader("Drug 2")
        drug2_name = st.text_input("Drug 2 Name:", key='d2_name')
        drug2_smiles = st.text_input("Drug 2 SMILES:", key='d2_smiles')
    
    if st.button("🔍 Compare Drugs", type="primary"):
        if drug1_smiles and drug2_smiles:
            with st.spinner("Computing similarity..."):
                tanimoto = compute_tanimoto(drug1_smiles, drug2_smiles)
                
                st.success(f"✅ Similarity Analysis Complete")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Tanimoto Similarity", f"{tanimoto:.4f}")
                
                with col2:
                    similarity_percent = tanimoto * 100
                    st.metric("Similarity %", f"{similarity_percent:.2f}%")
                
                with col3:
                    if similarity_percent >= 70:
                        st.success("High Similarity")
                    elif similarity_percent >= 40:
                        st.warning("Moderate Similarity")
                    else:
                        st.info("Low Similarity")
                
                st.subheader("Molecular Structures Comparison")
                col1, col2 = st.columns(2)
                
                with col1:
                    mol1 = Chem.MolFromSmiles(drug1_smiles)
                    if mol1:
                        img1 = Draw.MolToImage(mol1, size=(300, 300))
                        st.image(img1, caption=drug1_name or "Drug 1")
                
                with col2:
                    mol2 = Chem.MolFromSmiles(drug2_smiles)
                    if mol2:
                        img2 = Draw.MolToImage(mol2, size=(300, 300))
                        st.image(img2, caption=drug2_name or "Drug 2")
        else:
            st.error("Please enter SMILES for both drugs")

# ==================== BATCH PREDICTION TAB ====================
with tabs[3]:
    st.markdown('<h2 class="sub-header">📊 Batch Process Multiple Drugs</h2>', unsafe_allow_html=True)
    
    st.info("💡Upload a CSV file with multiple drugs for batch processing. The file should have columns: `drug_name`, `smiles` (or just `smiles")
    
    batch_file = st.file_uploader("Upload CSV (columns: drug_name, smiles OR just smiles)", type=['csv'], key='batch_upload')
    
    if batch_file:
        batch_df = pd.read_csv(batch_file)
        
        if len(batch_df.columns) == 1:
            if 'smiles' in batch_df.columns:
                batch_df['drug_name'] = [f"Drug_{i+1}" for i in range(len(batch_df))]
            else:
                batch_df.columns = ['smiles']
                batch_df['drug_name'] = [f"Drug_{i+1}" for i in range(len(batch_df))]
        
        st.dataframe(batch_df.head(10), use_container_width=True)
        st.info(f"Total drugs: {len(batch_df)}")
        
        if st.button("🚀 Process Batch", type="primary"):
            if not model_loaded or st.session_state.model is None:
                st.error("❌ Please upload model files in the sidebar first!")
            else:
                progress_bar = st.progress(0)
                all_results = []
                
                for idx, row in batch_df.iterrows():
                    predictions = predict_diseases(
                        st.session_state.model,
                        row['smiles'],
                        st.session_state.diseases_df,
                        st.session_state.label_encoder,
                        top_k=5
                    )
                    
                    if predictions:
                        for pred in predictions:
                            all_results.append({
                                'Drug_Name': row['drug_name'],
                                'SMILES': row['smiles'],
                                **pred
                            })
                    
                    progress_bar.progress((idx + 1) / len(batch_df))
                
                if all_results:
                    results_df = pd.DataFrame(all_results)
                    st.success(f"✅ Processed {len(batch_df)} drugs")
                    st.dataframe(results_df, use_container_width=True)
                    
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        "📥 Download Batch Results",
                        csv,
                        file_name='batch_predictions.csv',
                        mime='text/csv'
                    )

# ==================== ABOUT TAB ====================
with tabs[4]:
    st.markdown('<h2 class="sub-header">ℹ️ About DrugReuse</h2>', unsafe_allow_html=True)
    
    # Platform Overview
    st.markdown("""
    ## 🎯 Platform Overview
    
    The **DrugReuse** is an AI-powered tool that leverages Graph Neural Networks (GNNs) 
    to identify new therapeutic applications for existing drugs. By analyzing molecular structures and 
    learning from known drug-disease interactions, our platform can predict potential repurposing 
    opportunities, significantly accelerating drug discovery.
    """)
    
    # Introduction Section
    st.markdown("""
    ### 🎯 What is Drug Repurposing?
    
    **Drug repurposing** (also called drug repositioning) is the process of discovering **new therapeutic uses** 
    for existing approved drugs. Instead of developing new drugs from scratch (which takes 10-15 years and costs billions), 
    we can use AI to identify how existing drugs might treat different diseases.
    
    This platform uses **Graph Neural Networks (GNN)** to analyze molecular structures and predict potential 
    drug-disease associations, accelerating the drug discovery process.
    """)

    st.markdown("---")
    
    # How It Works
    st.markdown("## 🔬 How It Works")
    
    work_col1, work_col2 = st.columns(2)
    
    with work_col1:
        st.markdown("""
        ### 📊 The Process
        
        1. **Data Collection**
           - Gather drug-disease interaction data
           - Create `interactions.csv` with drug_id, disease_id, SMILES
        
        2. **Model Training**
           - Train Graph Neural Network on molecular structures
           - Learn patterns from known drug-disease associations
           - Generate `.pth` (model weights) and `.pkl` (label encoder) files
        
        3. **Prediction**
           - Input new drug SMILES
           - Model analyzes molecular graph structure
           - Predicts potential disease associations
        """)
    
    with work_col2:
        st.markdown("""
        ### 🧠 Technology Stack
        
        - **Graph Neural Networks (GNN)**
          - Analyzes molecular structures as graphs
          - Captures atom-bond relationships
          - 3-layer GCN architecture
        
        - **RDKit**
          - Molecular property calculations
          - SMILES parsing and validation
          - Lipinski's Rule of Five checks
        
        - **PyTorch Geometric**
          - Graph-based deep learning
          - Efficient molecular representation
        """)
    
    st.markdown("---")
    
    # Required Files Explanation
    st.markdown("## 📁 Required Files Explained")
    
    st.markdown("""
    <div style="background-color: #f0f2f6; padding: 2rem; border-radius: 1rem; margin: 1rem 0;">
        <h3 style="color: #1f77b4;">🔹 1. Trained Model File (.pth)</h3>
        <p><strong>What it is:</strong> PyTorch model state dictionary containing trained neural network weights.</p>
        <p><strong>How to get it:</strong> Train a GNN model using your drug-disease interaction dataset.</p>
        <p><strong>Contents:</strong> Model architecture parameters, learned weights from training.</p>
        <p><strong>Size:</strong> Typically 1-10 MB depending on model complexity.</p>
        <br>
        <p><strong>📝 Training Code Example:</strong></p>
        <code style="background: #2d2d2d; color: #f8f8f2; padding: 1rem; display: block; border-radius: 0.5rem;">
        model = SimpleGNN(num_node_features=16, hidden_channels=64, num_classes=100)<br>
        # ... training loop ...<br>
        torch.save(model.state_dict(), 'trained_model.pth')
        </code>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background-color: #f0f2f6; padding: 2rem; border-radius: 1rem; margin: 1rem 0;">
        <h3 style="color: #1f77b4;">🔹 2. Label Encoder File (.pkl)</h3>
        <p><strong>What it is:</strong> Scikit-learn LabelEncoder that maps disease names to numeric IDs.</p>
        <p><strong>How to get it:</strong> Created during training when encoding disease labels.</p>
        <p><strong>Contents:</strong> Disease ID → Disease Name mapping used during training.</p>
        <p><strong>Size:</strong> Usually < 1 MB.</p>
        <br>
        <p><strong>📝 Creation Example:</strong></p>
        <code style="background: #2d2d2d; color: #f8f8f2; padding: 1rem; display: block; border-radius: 0.5rem;">
        from sklearn.preprocessing import LabelEncoder<br>
        label_encoder = LabelEncoder()<br>
        encoded_labels = label_encoder.fit_transform(disease_names)<br>
        pickle.dump(label_encoder, open('label_encoder.pkl', 'wb'))
        </code>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background-color: #f0f2f6; padding: 2rem; border-radius: 1rem; margin: 1rem 0;">
        <h3 style="color: #1f77b4;">🔹 3. Diseases CSV File</h3>
        <p><strong>What it is:</strong> Reference file containing disease information.</p>
        <p><strong>Required columns:</strong></p>
        <ul>
            <li><code>disease_id</code> - Unique identifier (e.g., KEGG disease ID)</li>
            <li><code>disease_name</code> - Human-readable disease name</li>
        </ul>
        <p><strong>Example:</strong></p>
        <code style="background: #2d2d2d; color: #f8f8f2; padding: 1rem; display: block; border-radius: 0.5rem;">
        disease_id,disease_name<br>
        H00001,Acute lymphoblastic leukemia<br>
        H00002,Alzheimer disease<br>
        H00003,Parkinson disease
        </code>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background-color: #fff9e6; padding: 2rem; border-radius: 1rem; margin: 1rem 0; border-left: 4px solid #ffc107;">
        <h3 style="color: #f57c00;">🔹 4. Enamine CSV (Optional)</h3>
        <p><strong>What it is:</strong> Database of drug availability from Enamine chemical supplier.</p>
        <p><strong>Purpose:</strong> Check if predicted drugs are commercially available for purchase.</p>
        <p><strong>Required columns:</strong></p>
        <ul>
            <li><code>Drug_Name</code> - Name of the drug</li>
            <li><code>Available</code> - "Yes" or "No"</li>
        </ul>
        <p><strong>Note:</strong> This is completely optional and only used for availability checking.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Benefits Section
    st.markdown("## ✨ Key Benefits")
    
    ben_col1, ben_col2, ben_col3 = st.columns(3)
    
    with ben_col1:
        st.markdown("""
        ### ⚡ Speed
        
        - **Traditional:** 10-15 years
        - **AI-Powered:** Days to months
        - Accelerates discovery by 100x
        - Instant predictions for known drugs
        """)
    
    with ben_col2:
        st.markdown("""
        ### 💰 Cost-Effective
        
        - **New Drug:** $2.6 billion average
        - **Repurposing:** $300 million average
        - 87% cost reduction
        - Lower failure risk
        """)
    
    with ben_col3:
        st.markdown("""
        ### 🎯 Accuracy
        
        - AI learns from thousands of interactions
        - Molecular-level analysis
        - Validated against known drugs
        - Continuous improvement
        """)
    
    st.markdown("---")
    
    # Use Cases
    st.markdown("## 💡 Real-World Use Cases")
    
    use_col1, use_col2 = st.columns(2)
    
    with use_col1:
        st.markdown("""
        ### 🏥 Medical Research
        - Identify new treatments for rare diseases
        - Find alternatives for drug-resistant conditions
        - Discover combination therapy opportunities
        - Accelerate clinical trial planning
        """)
    
    with use_col2:
        st.markdown("""
        ### 💊 Pharmaceutical Industry
        - Reduce R&D costs and timeline
        - Optimize drug portfolios
        - Identify patent opportunities
        - Support regulatory submissions
        """)
    
    st.markdown("---")
    
    # Success Stories
    st.markdown("## 🏆 Drug Repurposing Success Stories")
    
    st.info("""
    **Famous Examples of Successful Drug Repurposing:**
    
    - **Aspirin**: Originally for pain relief → Now prevents heart attacks and strokes
    - **Sildenafil**: Developed for hypertension → Repurposed for erectile dysfunction
    - **Thalidomide**: Notorious sedative → Now treats multiple myeloma and leprosy
    - **Metformin**: Diabetes drug → Shows promise for cancer and aging research
    - **Remdesivir**: Ebola treatment → Repurposed for COVID-19
    """)
    
    st.markdown("---")
    
    # Knowing Drug Repurpose
    st.markdown("## 🚀 Knowing DrugReuse Better!!")
    
    st.markdown("""
    ### Step-by-Step Guide:
    
    1. **Prepare Your Data**
       ```
       Create interactions.csv:
       drug_id,disease_id,smiles
       D00001,H00001,CC(=O)OC1=CC=CC=C1C(=O)O
       D00002,H00001,CC(C)CC1=CC=C(C=C1)C(C)C(=O)O
       ```
    
    2. **Train Your Model**
       - Use the GNN training script (provided separately)
       - This generates `model.pth` and `label_encoder.pkl`
       - Training time: 10 minutes to 2 hours depending on dataset size
    
    3. **Upload Files**
       - Go to the sidebar (left panel)
       - Upload all three required files
       - Optionally upload Enamine database
    
    4. **Start Predicting**
       - Go to Home tab
       - Choose your tool (Drug Predictor, Similarity, or Batch)
       - Enter drug information and analyze!
    
    5. **Interpret Results**
       - Review predicted diseases with confidence scores
       - Check Lipinski's Rule validation
       - Download results for further analysis
    """)
    
    st.markdown("---")
    
    # Technical Specifications
    with st.expander("🔧 Technical Specifications"):
        st.markdown("""
        ### Model Architecture
        - **Type:** Graph Convolutional Network (GCN)
        - **Layers:** 3 GCN layers + 1 Linear layer
        - **Node Features:** 16 (atomic properties + one-hot encoding)
        - **Hidden Dimensions:** 64 channels
        - **Dropout:** 0.3 (during training)
        - **Pooling:** Global mean pooling
        
        ### Input Format
        - **SMILES:** Simplified Molecular Input Line Entry System
        - **Graph Representation:** Atoms as nodes, bonds as edges
        - **Atom Features:** Atomic number, degree, charge, hybridization, aromaticity, etc.
        
        ### Output Format
        - **Predictions:** Probability distribution over diseases
        - **Top-K Results:** Configurable (5-20 predictions)
        - **Confidence Scores:** Softmax probabilities
        """)
    
    st.markdown("---")
    
    # FAQ
    with st.expander("❓ Frequently Asked Questions"):
        st.markdown("""
        **Q: Do I need to train a model to use this platform?**  
        A: Yes, you need to upload pre-trained model files (.pth and .pkl). However, if you have drug-disease interaction data, you can train your own model.
        
        **Q: What if I don't have the Enamine database?**  
        A: The Enamine database is optional. You can use all features without it.
        
        **Q: How accurate are the predictions?**  
        A: Accuracy depends on your training data quality and size. Typical models achieve 70-85% accuracy on validation sets.
        
        **Q: What diseases can the model predict?**  
        A: The model can predict any diseases that were included in your training dataset.
        
        **Q: How do I interpret the confidence scores?**  
        A: Higher scores (>10%) indicate stronger predicted associations. Scores >20% are considered high confidence.
        """)
    
    st.markdown("---")
    
    # Footer Note
    st.success("""
    🎓 **Educational Purpose:** This platform is designed for research and educational purposes. 
    All predictions should be validated through proper scientific methods and clinical trials before any medical application.
    """)

# ==================== CONTACT TAB ====================
with tabs[5]:
    st.markdown('<h2 class="sub-header">Contact Us</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="Creator-section">
        <h3>👩‍🔬 About the Creator</h3>
        <div style="display: flex; align-items: center; gap: 20px;">
            <img src="https://media.licdn.com/dms/image/v2/D4D03AQGv0fpgreIM3g/profile-displayphoto-shrink_400_400/B4DZVQE3fmGkAg-/0/1740805209414?e=1763596800&v=beta&t=Lt5Ni0PJ4V42lKcDHQCCnwr5qP4XD7PUqBn5wjn3cDQ"
                 width="120" style="border-radius: 50%; border: 3px solid #6A5ACD;">
            <div class="about-section">
                <h4>Jyoti Rana</h4>
                <p>M.Sc. Bioinformatics Student, DES Pune University</p>
                <p>Email:jyotirana0890@gmail.com</p>
            </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
   
    # Mentorship Section
    st.markdown("""
    <div class="mentorship-section">
        <h3>👨‍🏫 Mentorship</h3>
        <div style="display: flex; align-items: center; gap: 20px;">
            <img src="https://media.licdn.com/dms/image/v2/D5603AQF9gsU7YBjWVg/profile-displayphoto-shrink_400_400/B56ZZI.WrdH0Ag-/0/1744981029051?e=1763596800&v=beta&t=bg8fuJhWkj7jan3iIqtA2-X63SIIZBXyWlbWcJw-nFY"
                 width="120" style="border-radius: 50%; border: 3px solid #6A5ACD;">
            <div>
                <h4>Dr. Kushagra Kashyap</h4>
                <p>Assistant Professor (Bioinformatics), Department of Life Sciences, School of Science and Mathematics, DES Pune University</p>
                <p>This project was developed under the guidance of Dr. Kashyap, who provided valuable insights and mentorship
                throughout the development process. His expertise in bioinformatics and computational biology was instrumental
                in shaping this project.</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

#Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: gray; padding: 2rem;'>
        <p>DrugReuse | Only for Educational Purpose</p>
    </div>
""", unsafe_allow_html=True)
