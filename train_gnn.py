import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
from tqdm import tqdm
import time

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

# ==================== Data Processing Functions ====================
def smiles_to_graph(smiles):
    """Convert SMILES string to graph representation"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # Get atom features
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
        
        # Get edges
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
    except Exception as e:
        return None

def load_and_standardize_data(drugs_csv, diseases_csv):
    """Load and standardize CSV files"""
    diseases_df = pd.read_csv(diseases_csv)
    diseases_df.columns = diseases_df.columns.str.lower().str.replace('_', '')
    column_mapping = {'diseaseid': 'disease_id', 'diseasename': 'disease_name'}
    diseases_df.rename(columns=column_mapping, inplace=True)
    
    print(f"✓ Loaded {len(diseases_df)} diseases")
    
    drugs_df = pd.read_csv(drugs_csv)
    if drugs_df.columns[0].startswith('C'):
        drugs_df = pd.read_csv(drugs_csv, header=None)
        drugs_df.columns = ['smiles']
    
    drugs_df.columns = drugs_df.columns.str.lower().str.replace('_', '')
    
    if len(drugs_df.columns) == 1:
        if 'smiles' not in drugs_df.columns:
            drugs_df.columns = ['smiles']
        drugs_df['drugname'] = [f"Drug_{i+1}" for i in range(len(drugs_df))]
    
    column_mapping = {'drugname': 'drug_name', 'smiles': 'smiles'}
    drugs_df.rename(columns=column_mapping, inplace=True)
    
    if 'drug_name' not in drugs_df.columns:
        drugs_df['drug_name'] = [f"Drug_{i+1}" for i in range(len(drugs_df))]
    
    print(f"✓ Loaded {len(drugs_df)} drugs")
    
    return drugs_df, diseases_df

def prepare_training_data(drugs_csv, diseases_csv, interactions_csv=None, max_samples_per_drug=5):
    """Prepare training data with progress tracking and sampling"""
    
    drugs_df, diseases_df = load_and_standardize_data(drugs_csv, diseases_csv)
    
    # Encode diseases
    le = LabelEncoder()
    diseases_df['disease_encoded'] = le.fit_transform(diseases_df['disease_id'])
    
    with open('disease_label_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)
    
    diseases_df[['disease_id', 'disease_name']].to_csv('diseases_standardized.csv', index=False)
    print("✓ Saved standardized files\n")
    
    graph_data = []
    labels = []
    drug_names = []
    
    # Pre-compute drug graphs with progress bar
    print("📊 Converting SMILES to molecular graphs...")
    drug_graphs = {}
    invalid_count = 0
    
    for idx, row in tqdm(drugs_df.iterrows(), total=len(drugs_df), desc="Processing drugs"):
        graph = smiles_to_graph(row['smiles'])
        if graph is not None:
            drug_graphs[row['drug_name']] = graph
        else:
            invalid_count += 1
    
    print(f"✓ Created {len(drug_graphs)} valid graphs ({invalid_count} invalid)")
    
    if interactions_csv and pd.io.common.file_exists(interactions_csv):
        print(f"\n📋 Loading interactions from: {interactions_csv}")
        interactions_df = pd.read_csv(interactions_csv)
        interactions_df.columns = interactions_df.columns.str.lower().str.replace('_', '')
        
        column_mapping = {
            'drugname': 'drug_name',
            'diseaseid': 'disease_id',
            'label': 'label'
        }
        interactions_df.rename(columns=column_mapping, inplace=True)
        
        print(f"Processing {len(interactions_df)} interactions...")
        
        valid_interactions = 0
        for idx, row in tqdm(interactions_df.iterrows(), total=len(interactions_df), desc="Building dataset"):
            if row['drug_name'] not in drug_graphs:
                continue
            
            disease_row = diseases_df[diseases_df['disease_id'] == row['disease_id']]
            if disease_row.empty:
                continue
            
            graph = drug_graphs[row['drug_name']]
            disease_label = disease_row.iloc[0]['disease_encoded']
            
            graph_data.append(graph)
            labels.append(disease_label)
            drug_names.append(row['drug_name'])
            valid_interactions += 1
        
        print(f"✓ Created {valid_interactions} training samples from interactions")
    
    else:
        print("\n⚠️  No interactions file - creating synthetic data...")
        print(f"Creating {max_samples_per_drug} samples per drug...")
        
        for drug_name, graph in tqdm(drug_graphs.items(), desc="Creating samples"):
            # Create multiple samples per drug
            for _ in range(max_samples_per_drug):
                disease_label = np.random.randint(0, len(diseases_df))
                graph_data.append(graph)
                labels.append(disease_label)
                drug_names.append(drug_name)
        
        print(f"✓ Created {len(graph_data)} synthetic training samples")
    
    return graph_data, labels, drug_names, len(diseases_df)

# ==================== Training Function ====================
def train_model(drugs_csv, diseases_csv, interactions_csv=None, 
                epochs=50, batch_size=64, learning_rate=0.001, max_samples_per_drug=5):
    """Train the GNN model with progress tracking"""
    
    print("\n" + "="*70)
    print("🧬 DRUG REPURPOSING GNN TRAINING")
    print("="*70)
    
    start_time = time.time()
    
    print("\n📥 LOADING DATA...")
    graph_data, labels, drug_names, num_diseases = prepare_training_data(
        drugs_csv, diseases_csv, interactions_csv, max_samples_per_drug
    )
    
    load_time = time.time() - start_time
    print(f"\n✓ Data loaded in {load_time/60:.1f} minutes")
    
    print(f"\n📊 DATASET SUMMARY:")
    print(f"   Total samples: {len(graph_data)}")
    print(f"   Number of diseases: {num_diseases}")
    print(f"   Unique drugs: {len(set(drug_names))}")
    
    # Add labels to graphs
    for graph, label in zip(graph_data, labels):
        graph.y = torch.tensor([label], dtype=torch.long)
    
    # Split data
    train_data, val_data = train_test_split(graph_data, test_size=0.2, random_state=42)
    print(f"\n📊 DATA SPLIT:")
    print(f"   Training: {len(train_data)} samples")
    print(f"   Validation: {len(val_data)} samples")
    
    # Create loaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  Device: {device}")
    
    model = SimpleGNN(num_node_features=16, hidden_channels=64, num_classes=num_diseases)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
    
    print(f"\n⚙️  TRAINING CONFIG:")
    print(f"   Epochs: {epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Learning rate: {learning_rate}")
    
    print("\n" + "="*70)
    print("🚀 STARTING TRAINING...")
    print("="*70 + "\n")
    
    best_val_acc = 0.0
    train_start = time.time()
    
    for epoch in range(epochs):
        # Training
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(out, batch.y)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = out.argmax(dim=1)
            correct += (pred == batch.y).sum().item()
            total += batch.y.size(0)
        
        train_acc = correct / total
        avg_loss = total_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index, batch.batch)
                pred = out.argmax(dim=1)
                val_correct += (pred == batch.y).sum().item()
                val_total += batch.y.size(0)
        
        val_acc = val_correct / val_total
        
        # Progress update every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == 0:
            elapsed = time.time() - train_start
            eta = elapsed / (epoch + 1) * (epochs - epoch - 1)
            print(f'Epoch {epoch+1:3d}/{epochs} | Loss: {avg_loss:.4f} | '
                  f'Train: {train_acc:.4f} | Val: {val_acc:.4f} | '
                  f'ETA: {eta/60:.1f}m')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_gnn_model.pth')
            if (epoch + 1) % 5 == 0:
                print(f'   ✓ New best model saved!')
    
    total_time = time.time() - start_time
    
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE!")
    print("="*70)
    print(f"   Best validation accuracy: {best_val_acc:.4f}")
    print(f"   Total time: {total_time/60:.1f} minutes")
    print(f"\n📁 FILES CREATED:")
    print(f"   ✓ best_gnn_model.pth")
    print(f"   ✓ disease_label_encoder.pkl")
    print(f"   ✓ diseases_standardized.csv")
    print("="*70)
    
    return model

# ==================== Main ====================
if __name__ == "__main__":
    # For large datasets, reduce samples per drug if no interactions file
    model = train_model(
        drugs_csv='drugs.csv',
        diseases_csv='diseases.csv',
        interactions_csv='interactions.csv',  # Set to None if you don't have this
        epochs=50,
        batch_size=64,
        learning_rate=0.001
        #max_samples_per_drug=3  # Only used if no interactions file
    )