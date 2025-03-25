import streamlit as st
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, rdDepictor, DataStructs, rdmolops
from rdkit.Chem.Draw import rdMolDraw2D
from PIL import Image
import io
import base64

st.set_page_config(page_title="Recherche Analogues", layout="wide")

# Configuration pour maintenir l'état entre les actions
if 'initialized' not in st.session_state:
    st.session_state.initialized = True
    st.session_state.all_results = pd.DataFrame()
    st.session_state.selected_fragment_names = []

def sanitize_mol(mol):
    """Effectue une sanitization des molécules et initialise les infos de ring"""
    if mol is None:
        return None
    
    try:
        mol.UpdatePropertyCache(strict=False)
        Chem.GetSSSR(mol)
        return mol
    except:
        # Si la sanitization échoue, essayer de recréer la molécule
        try:
            smiles = Chem.MolToSmiles(mol)
            new_mol = Chem.MolFromSmiles(smiles)
            if new_mol is not None:
                new_mol.UpdatePropertyCache(strict=False)
                Chem.GetSSSR(new_mol)
                return new_mol
        except:
            pass
    return mol  # Retourner la molécule originale si tout échoue

def get_download_link(selected_results, filename, text):
    """
    Crée un lien de téléchargement pour les molécules sélectionnées
    en utilisant les données originales du CSV d'entrée
    """
    if selected_results is None or selected_results.empty:
        return "Aucune molécule sélectionnée pour le téléchargement"
    
    # Extraire les données originales des lignes sélectionnées
    rows_to_download = []
    for _, row in selected_results.iterrows():
        if 'Original_Row' in row:
            rows_to_download.append(row['Original_Row'])
    
    # Si aucune donnée originale n'a été trouvée, retourner un message d'erreur
    if not rows_to_download:
        return "Aucune donnée originale trouvée pour les molécules sélectionnées"
    
    # Créer un DataFrame à partir des données originales
    original_df = pd.DataFrame(rows_to_download)
    
    # Ajouter une colonne de similarité pour information
    if 'Similarite' in selected_results.columns:
        for i, (_, row) in enumerate(selected_results.iterrows()):
            if i < len(original_df):
                original_df.loc[i, 'Similarite'] = f"{row['Similarite']*100:.1f}%"
    
    # Convertir en CSV
    csv = original_df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href

def mol_from_smiles(smiles):
    """Cree un objet RDKit mol a partir d'un SMILES"""
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        # Sanitize et initialiser la molécule
        mol = sanitize_mol(mol)
        # Calculer les coordonnees 2D si necessaire
        if not mol.GetNumConformers():
            rdDepictor.Compute2DCoords(mol)
    return mol

def generate_molecule_image(mol, size=(400, 300), highlightAtoms=None, molName=""):
    """Genere une image d'une molecule, avec surlignage optionnel"""
    if mol is None:
        return None
    
    try:
        drawer = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
        
        # Definir des options d'affichage
        opts = drawer.drawOptions()
        if highlightAtoms and len(highlightAtoms) > 0:
            # Definir une couleur pour le surlignage
            opts.highlightColour = (0.8, 0.2, 0.2, 1.0)  # Rouge vif
            opts.highlightRadius = 0.5
        
        # Dessiner la molecule
        drawer.DrawMolecule(mol, legend=molName, highlightAtoms=highlightAtoms)
        drawer.FinishDrawing()
        
        png_data = drawer.GetDrawingText()
        return Image.open(io.BytesIO(png_data))
    except Exception as e:
        st.error(f"Erreur lors de la generation de l'image: {str(e)}")
        return None

def get_canonical_atom_description(mol, atom_indices):
    """Génère une description canonique des atomes pour éviter les doublons"""
    # Créer une liste de tuples (symbole, indice)
    atom_tuples = []
    for idx in atom_indices:
        atom = mol.GetAtomWithIdx(idx)
        atom_tuples.append((atom.GetSymbol(), idx+1))
    
    # Trier d'abord par symbole atomique, puis par indice
    atom_tuples.sort()
    
    # Convertir en description textuelle
    atom_symbols = [f"{symbol}{idx}" for symbol, idx in atom_tuples]
    return ", ".join(atom_symbols)

def identify_cycles(mol):
    """Identifie les cycles dans une molecule et retourne un dictionnaire des atomes par cycle"""
    if mol is None:
        return {}
    
    # S'assurer que les informations de cycle sont calculées
    mol = sanitize_mol(mol)
    
    cycle_atoms = {}
    
    try:
        # Utiliser RingInfo pour obtenir les atomes des cycles
        ring_info = mol.GetRingInfo()
        ring_atoms = ring_info.AtomRings()
        
        for i, ring in enumerate(ring_atoms):
            cycle_atoms[f"Cycle {i+1}"] = list(ring)
    except Exception as e:
        st.warning(f"Attention lors de l'identification des cycles: {str(e)}")
    
    return cycle_atoms

def identify_functional_groups(mol):
    """Identifie les groupes fonctionnels communs dans une molecule sans inclure les atomes des cycles"""
    if mol is None:
        return {}
    
    # S'assurer que la molécule est sanitizée
    mol = sanitize_mol(mol)
    
    # Identifier d'abord les atomes des cycles pour pouvoir les exclure
    ring_info = mol.GetRingInfo()
    ring_atoms = set()
    for ring in ring_info.AtomRings():
        ring_atoms.update(ring)
    
    # Definir des SMARTS pour des groupes fonctionnels communs
    functional_groups = {
        "Alcool": "[OX2H]",
        "Acide carboxylique": "[CX3](=O)[OX2H1]",
        "Amine": "[NX3;H2,H1,H0;!$(NC=O)]",
        "Amide": "[NX3][CX3](=[OX1])",
        "Ester": "[#6][CX3](=O)[OX2][#6]",
        "Ether": "[OD2]([#6])[#6]",
        "Aldehyde": "[CX3H1](=O)[#6]",
        "Ketone": "[#6][CX3](=O)[#6]",
        "Nitrile": "[NX1]#[CX2]",
        "Halogene": "[F,Cl,Br,I]"
    }
    
    # Trouver les occurrences de chaque groupe fonctionnel
    group_atoms = {}
    
    for name, smarts in functional_groups.items():
        try:
            pattern = Chem.MolFromSmarts(smarts)
            if pattern:
                matches = mol.GetSubstructMatches(pattern)
                if matches:
                    # Pour chaque occurrence, créer une entrée séparée
                    for i, match in enumerate(matches):
                        # Filtrer pour exclure les atomes qui font partie d'un cycle
                        non_ring_atoms = [idx for idx in match if idx not in ring_atoms]
                        
                        # Ne conserver le groupe que s'il contient des atomes hors cycle
                        if non_ring_atoms:
                            group_atoms[f"{name} {i+1}"] = non_ring_atoms
        except Exception as e:
            continue  # Ignorer les erreurs spécifiques à certains motifs
    
    return group_atoms

def identify_atom_specific_fragments(mol):
    """Identifie des fragments dans la molecule basés sur des caractéristiques structurelles"""
    if mol is None:
        return {}
    
    # S'assurer que la molécule est sanitizée
    mol = sanitize_mol(mol)
    
    # Utiliser un dictionnaire pour les fragments canoniques
    # La clé sera un tuple trié des atomes pour garantir l'unicité
    canonical_fragments = {}
    fragments = {}
    
    # 1. Identifier les cycles
    cycles = identify_cycles(mol)
    for name, atoms in cycles.items():
        # Créer une clé canonique (tuple trié des atomes)
        canonical_key = tuple(sorted(atoms))
        canonical_fragments[canonical_key] = (name, atoms)
    
    # Créer un ensemble de tous les atomes de cycle pour référence rapide
    cycle_atoms_flat = []
    for atoms in cycles.values():
        cycle_atoms_flat.extend(atoms)
    cycle_atoms_set = set(cycle_atoms_flat)
    
    # Créer une carte des cycles pour chaque atome
    atom_to_cycles = {}
    for cycle_name, atoms in cycles.items():
        for atom_idx in atoms:
            if atom_idx in atom_to_cycles:
                atom_to_cycles[atom_idx].append(cycle_name)
            else:
                atom_to_cycles[atom_idx] = [cycle_name]
    
    # 2. Identifier les groupes fonctionnels de manière spécifique
    func_groups = identify_functional_groups(mol)
    
    # Pour chaque groupe fonctionnel, vérifier sa position par rapport aux cycles
    for group_name, atoms in func_groups.items():
        # Trouver les points de connexion aux cycles
        connection_to_cycle = False
        cycle_connection_info = ""
        
        for atom_idx in atoms:
            atom = mol.GetAtomWithIdx(atom_idx)
            for neighbor in atom.GetNeighbors():
                neighbor_idx = neighbor.GetIdx()
                if neighbor_idx in cycle_atoms_set and neighbor_idx not in atoms:
                    connection_to_cycle = True
                    if neighbor_idx in atom_to_cycles:
                        cycle_names = atom_to_cycles[neighbor_idx]
                        if len(cycle_names) == 1:
                            cycle_connection_info = f" sur {cycle_names[0]}"
                        else:
                            cycle_connection_info = f" a l'intersection {'+'.join(cycle_names)}"
                    break
            if connection_to_cycle:
                break
        
        # Ajouter l'information de position au nom du groupe
        new_group_name = f"{group_name}{cycle_connection_info}"
        
        # Créer une clé canonique pour ce groupe
        canonical_key = tuple(sorted(atoms))
        
        # Vérifier si nous avons déjà un fragment avec ces mêmes atomes
        if canonical_key not in canonical_fragments:
            canonical_fragments[canonical_key] = (new_group_name, atoms)
    
    # 3. Identifier les substituants (chaînes latérales)
    # Trouver les atomes de connexion (atomes cycliques avec voisins non cycliques)
    connection_points = {}
    for atom_idx in cycle_atoms_set:
        atom = mol.GetAtomWithIdx(atom_idx)
        for neighbor in atom.GetNeighbors():
            n_idx = neighbor.GetIdx()
            if n_idx not in cycle_atoms_set:
                # C'est un point de connexion
                if atom_idx in connection_points:
                    connection_points[atom_idx].append(n_idx)
                else:
                    connection_points[atom_idx] = [n_idx]
    
    # Pour chaque point de connexion, explorer la chaîne latérale
    substituent_idx = 1
    processed = set()
    
    # Ensemble pour détecter les doublons de liaisons
    unique_connections = set()
    
    for atom_idx, neighbors in connection_points.items():
        position_info = ""
        if atom_idx in atom_to_cycles:
            cycle_names = atom_to_cycles[atom_idx]
            if len(cycle_names) == 1:
                position_info = f" sur {cycle_names[0]}"
            else:
                position_info = f" a l'intersection {'+'.join(cycle_names)}"
        
        for start_idx in neighbors:
            if start_idx in processed:
                continue
            
            # Créer une clé canonique pour cette connexion
            connection_key = tuple(sorted([atom_idx, start_idx]))
            
            # Vérifier si cette connexion a déjà été traitée
            if connection_key in unique_connections:
                continue
            
            # Marquer cette connexion comme traitée
            unique_connections.add(connection_key)
            
            # Explorer la chaîne à partir de ce point de départ
            chain_atoms = [start_idx]  # Ne pas inclure l'atome du cycle
            to_process = [start_idx]
            processed.add(start_idx)
            
            while to_process:
                current = to_process.pop(0)
                atom = mol.GetAtomWithIdx(current)
                for neighbor in atom.GetNeighbors():
                    n_idx = neighbor.GetIdx()
                    if n_idx not in cycle_atoms_set and n_idx not in processed and n_idx not in to_process:
                        chain_atoms.append(n_idx)
                        processed.add(n_idx)
                        to_process.append(n_idx)
            
            # Créer une clé canonique pour cette chaîne
            canonical_key = tuple(sorted(chain_atoms))
            
            # Vérifier si nous avons déjà un fragment avec ces mêmes atomes
            if canonical_key not in canonical_fragments:
                chain_name = f"Chaine {substituent_idx}{position_info}"
                canonical_fragments[canonical_key] = (chain_name, chain_atoms)
                substituent_idx += 1
    
    # Construire le dictionnaire final des fragments à partir des fragments canoniques
    for i, (_, (name, atoms)) in enumerate(canonical_fragments.items()):
        fragments[name] = atoms
    
    return fragments

def extract_scaffold(mol, modification_atoms):
    """
    Extrait le scaffold (partie invariante) de la molecule
    en excluant les atomes qui peuvent etre modifies
    
    Retourne:
    - scaffold_mol: la molecule scaffold
    - modification_exit_points: indices des atomes dans le scaffold 
                               connectes aux atomes de modification
    """
    if mol is None or not isinstance(modification_atoms, list):
        return None, []
    
    # S'assurer que la molécule est sanitizée
    mol = sanitize_mol(mol)
    
    # Créer une copie de travail de la molécule
    mol_copy = Chem.RWMol(mol)
    
    # Identifier les atomes à conserver (tous sauf ceux à modifier)
    num_atoms = mol.GetNumAtoms()
    atoms_to_keep = list(set(range(num_atoms)) - set(modification_atoms))
    
    # Identifier les points de connexion (atomes conservés connectés à des atomes à supprimer)
    exit_points = []
    
    for idx in atoms_to_keep:
        atom = mol.GetAtomWithIdx(idx)
        for bond in atom.GetBonds():
            other_idx = bond.GetOtherAtomIdx(idx)
            if other_idx in modification_atoms:
                exit_points.append(idx)
                break
    
    # Convertir les indices pour le nouveau scaffold
    original_to_scaffold = {}
    scaffold_atoms = []
    
    for i, idx in enumerate(atoms_to_keep):
        original_to_scaffold[idx] = i
        scaffold_atoms.append(mol.GetAtomWithIdx(idx))
    
    # Créer la nouvelle molécule scaffold
    scaffold = Chem.RWMol()
    
    # Ajouter les atomes
    for atom in scaffold_atoms:
        new_atom = Chem.Atom(atom.GetAtomicNum())
        new_atom.SetFormalCharge(atom.GetFormalCharge())
        new_atom.SetNumExplicitHs(atom.GetNumExplicitHs())
        new_atom.SetIsAromatic(atom.GetIsAromatic())
        new_atom.SetHybridization(atom.GetHybridization())
        new_atom.SetNoImplicit(atom.GetNoImplicit())
        new_atom.SetChiralTag(atom.GetChiralTag())  # Préserver la chiralité
        scaffold.AddAtom(new_atom)
    
    # Ajouter les liaisons
    for i, orig_idx in enumerate(atoms_to_keep):
        atom = mol.GetAtomWithIdx(orig_idx)
        for bond in atom.GetBonds():
            other_idx = bond.GetOtherAtomIdx(orig_idx)
            # Ne considérer que les liaisons entre atomes conservés
            if other_idx in atoms_to_keep and orig_idx < other_idx:
                j = original_to_scaffold[other_idx]
                scaffold.AddBond(i, j, bond.GetBondType())
    
    # Convertir les points de sortie vers les indices du scaffold
    scaffold_exit_points = [original_to_scaffold[idx] for idx in exit_points]
    
    # Finaliser et retourner le scaffold
    scaffold_mol = scaffold.GetMol()
    
    # Sanitize et calculer les infos de cycle
    try:
        Chem.SanitizeMol(scaffold_mol)
        scaffold_mol.UpdatePropertyCache(strict=False)
        Chem.GetSSSR(scaffold_mol)
        # Essayer de calculer des coordonnées 2D
        rdDepictor.Compute2DCoords(scaffold_mol)
    except Exception as e:
        st.warning(f"Attention lors de la finalisation du scaffold: {str(e)}")
        # Essayer une approche plus sûre
        try:
            smiles = Chem.MolToSmiles(scaffold_mol)
            scaffold_mol = Chem.MolFromSmiles(smiles)
            if scaffold_mol is not None:
                scaffold_mol.UpdatePropertyCache(strict=False)
                Chem.GetSSSR(scaffold_mol)
                rdDepictor.Compute2DCoords(scaffold_mol)
        except:
            pass
    
    return scaffold_mol, scaffold_exit_points

def compare_scaffolds(ref_scaffold, test_mol):
    """
    Compare chimiquement si le scaffold de référence est présent dans la molécule test.
    
    Retourne:
    - is_match: True si le scaffold est chimiquement identique, False sinon
    - match_atoms: Liste des indices des atomes dans test_mol qui correspondent au scaffold
    - differences: Description des différences si non identique
    """
    # S'assurer que les infos de cycle sont calculées pour les deux molécules
    try:
        ref_scaffold.UpdatePropertyCache(strict=False)
        Chem.GetSSSR(ref_scaffold)
        test_mol.UpdatePropertyCache(strict=False)
        Chem.GetSSSR(test_mol)
    except Exception as e:
        # Si l'initialisation échoue, essayer de recréer les molécules
        try:
            # Recréer le scaffold
            ref_smiles = Chem.MolToSmiles(ref_scaffold)
            ref_scaffold = Chem.MolFromSmiles(ref_smiles)
            ref_scaffold.UpdatePropertyCache(strict=False)
            Chem.GetSSSR(ref_scaffold)
            
            # Recréer la molécule test
            test_smiles = Chem.MolToSmiles(test_mol)
            test_mol = Chem.MolFromSmiles(test_smiles)
            test_mol.UpdatePropertyCache(strict=False)
            Chem.GetSSSR(test_mol)
        except:
            return False, [], f"Erreur lors de l'initialisation des molécules: {str(e)}"
    
    # Rechercher les correspondances de sous-structure
    try:
        # Utiliser isomericSmiles=True pour conserver les informations stéréochimiques
        scaffold_smiles = Chem.MolToSmiles(ref_scaffold, isomericSmiles=True)
        scaffold_mol_clean = Chem.MolFromSmiles(scaffold_smiles)
        
        if scaffold_mol_clean is None:
            return False, [], "Impossible de générer une molécule propre à partir du scaffold"
        
        scaffold_smarts = Chem.MolToSmarts(scaffold_mol_clean)
        scaffold_query = Chem.MolFromSmarts(scaffold_smarts)
        
        if scaffold_query is None:
            return False, [], "Erreur lors de la création de la requête SMARTS"
    except Exception as e:
        return False, [], f"Erreur lors de la préparation de la requête: {str(e)}"
    
    # Rechercher dans la molécule test
    try:
        matches = test_mol.GetSubstructMatches(scaffold_query)
    except Exception as e:
        return False, [], f"Erreur lors de la recherche de sous-structure: {str(e)}"
    
    if not matches:
        return False, [], "Aucune correspondance de scaffold trouvée"
    
    # Vérifier chaque correspondance pour une identité chimique exacte
    scaffold_smiles_canonical = Chem.MolToSmiles(ref_scaffold, isomericSmiles=True)
    
    for match in matches:
        try:
            # Créer un fragment basé sur les atomes correspondants
            match_indices = list(match)
            rwmol = Chem.RWMol()
            
            # Cartographie des indices du test_mol vers le nouveau mol
            atom_map = {}
            
            # Ajouter les atomes
            for i, idx in enumerate(match_indices):
                atom = test_mol.GetAtomWithIdx(idx)
                new_idx = rwmol.AddAtom(Chem.Atom(atom.GetAtomicNum()))
                atom_map[idx] = new_idx
                # Copier les propriétés importantes
                rwmol.GetAtomWithIdx(new_idx).SetFormalCharge(atom.GetFormalCharge())
                rwmol.GetAtomWithIdx(new_idx).SetNumExplicitHs(atom.GetNumExplicitHs())
                rwmol.GetAtomWithIdx(new_idx).SetIsAromatic(atom.GetIsAromatic())
                rwmol.GetAtomWithIdx(new_idx).SetChiralTag(atom.GetChiralTag())
            
            # Ajouter les liaisons
            for i, orig_idx in enumerate(match_indices):
                for j in range(i+1, len(match_indices)):
                    other_idx = match_indices[j]
                    bond = test_mol.GetBondBetweenAtoms(orig_idx, other_idx)
                    if bond:
                        rwmol.AddBond(atom_map[orig_idx], atom_map[other_idx], bond.GetBondType())
            
            # Convertir en molécule finale
            match_mol = rwmol.GetMol()
            
            # Sanitize la molécule
            try:
                Chem.SanitizeMol(match_mol)
                match_mol.UpdatePropertyCache(strict=False)
                Chem.GetSSSR(match_mol)
            except:
                continue  # Si la sanitization échoue, passer à la prochaine correspondance
            
            # Comparer les SMILES canoniques
            match_smiles = Chem.MolToSmiles(match_mol, isomericSmiles=True)
            
            if match_smiles == scaffold_smiles_canonical:
                return True, match_indices, ""
            
            # Si ce n'est pas une correspondance exacte, calculer les différences
            differences = []
            
            # Différence dans les types d'atomes
            match_atoms = [match_mol.GetAtomWithIdx(i).GetSymbol() for i in range(match_mol.GetNumAtoms())]
            ref_atoms = [ref_scaffold.GetAtomWithIdx(i).GetSymbol() for i in range(ref_scaffold.GetNumAtoms())]
            
            if sorted(match_atoms) != sorted(ref_atoms):
                atom_counts_match = {}
                atom_counts_ref = {}
                
                for atom in match_atoms:
                    atom_counts_match[atom] = atom_counts_match.get(atom, 0) + 1
                
                for atom in ref_atoms:
                    atom_counts_ref[atom] = atom_counts_ref.get(atom, 0) + 1
                
                diff_elements = []
                for atom, count in atom_counts_ref.items():
                    if atom not in atom_counts_match or atom_counts_match[atom] != count:
                        diff_elements.append(f"{atom}: {count} vs {atom_counts_match.get(atom, 0)}")
                
                differences.append(f"Différence d'éléments: {', '.join(diff_elements)}")
            
            # Si nous trouvons des différences, stocker pour le rapport final
            if differences:
                return False, match_indices, "\n".join(differences)
            
        except Exception as e:
            continue  # En cas d'erreur, passer à la correspondance suivante
    
    # Si nous arrivons ici, c'est qu'aucune correspondance exacte n'a été trouvée
    if matches:
        # Retourner la première correspondance avec un message générique
        return False, list(matches[0]), "Structure chimique différente dans le scaffold"
    
    return False, [], "Aucune correspondance chimique valide trouvée"

def find_analogs_by_fixed_scaffold(df, ref_mol, modification_atoms, ref_id):
    """
    Trouve les analogues en s'assurant que la partie non sélectionnée
    est chimiquement identique (scaffold invariant) et ne contient pas
    d'éléments supplémentaires non présents dans le scaffold de référence.
    
    Cette version ne retourne que les analogues exacts sans gestion séparée des déviants.
    """
    if ref_mol is None or not modification_atoms:
        return pd.DataFrame()
    
    # Extraire le scaffold de référence (partie qui doit rester identique)
    scaffold_mol, exit_points = extract_scaffold(ref_mol, modification_atoms)
    
    if scaffold_mol is None:
        st.error("Erreur lors de l'extraction du scaffold.")
        return pd.DataFrame()
    
    # S'assurer que le scaffold est correctement initialisé
    scaffold_mol = sanitize_mol(scaffold_mol)
    
    # Nombre d'atomes dans le scaffold de référence
    scaffold_atom_count = scaffold_mol.GetNumAtoms()
    scaffold_smiles = Chem.MolToSmiles(scaffold_mol, isomericSmiles=True)
    
    # Stocker le scaffold pour utilisation ultérieure
    st.session_state.ref_scaffold_mol = scaffold_mol
    st.session_state.ref_scaffold_smiles = scaffold_smiles
    
    # Afficher le scaffold pour le débogage
    scaffold_img = generate_molecule_image(scaffold_mol, (400, 300), highlightAtoms=exit_points, 
                                         molName="Scaffold (partie invariante)")
    st.image(scaffold_img, caption=f"Partie qui doit rester identique dans les analogues ({scaffold_atom_count} atomes)")
    st.text(f"SMILES du scaffold: {scaffold_smiles}")
    
    # Cartographier les points de sortie attendus (où les modifications sont permises)
    expected_exit_points = set(exit_points)
    
    results = []
    progress_bar = st.progress(0)
    total = len(df)
    
    for i, row in enumerate(df.itertuples()):
        # Mettre à jour la barre de progression
        progress_bar.progress(min(1.0, i / total))
        
        mol_id = getattr(row, 'ID')
        smiles = getattr(row, 'SMILES')
        
        # Récupérer la ligne complète du CSV d'entrée pour une utilisation ultérieure
        original_row = df.iloc[i].to_dict()
        
        # Ignorer la molécule de référence
        if mol_id == ref_id:
            continue
        
        # Créer la molécule test
        test_mol = mol_from_smiles(smiles)
        if test_mol is None:
            continue
        
        # S'assurer que la molécule test est correctement initialisée
        test_mol = sanitize_mol(test_mol)
        
        # Calculer la similarité globale
        try:
            fp_ref = AllChem.GetMorganFingerprintAsBitVect(ref_mol, 3)
            fp_test = AllChem.GetMorganFingerprintAsBitVect(test_mol, 3)
            similarity = DataStructs.TanimotoSimilarity(fp_ref, fp_test)
        except:
            # En cas d'erreur, utiliser une similarité par défaut
            similarity = 0.5
        
        # Vérifier si le scaffold est présent dans cette molécule
        is_match, match_atoms, diff_message = compare_scaffolds(scaffold_mol, test_mol)
        
        # Si on a un match, procéder aux vérifications supplémentaires
        if is_match and match_atoms:
            # 1. Trouver tous les atomes modifiés (ceux qui ne font pas partie du scaffold)
            match_atoms_set = set(match_atoms)
            all_atoms_test = set(range(test_mol.GetNumAtoms()))
            modified_atoms_test = list(all_atoms_test - match_atoms_set)
            
            # 2. Cartographier les atomes du scaffold dans la molécule analogue vers
            # les atomes correspondants dans le scaffold de référence
            match_to_scaffold = {}
            if len(match_atoms) == scaffold_atom_count:
                # Si le nombre d'atomes correspond, supposons une correspondance ordonnée
                for i, atom_idx in enumerate(match_atoms):
                    match_to_scaffold[atom_idx] = i
            
            # 3. Vérifier si des substituants inattendus sont attachés au scaffold
            has_extra_substituents = False
            
            # Vérifier chaque atome du scaffold dans l'analogue
            for atom_idx in match_atoms:
                atom = test_mol.GetAtomWithIdx(atom_idx)
                
                # Vérifier si cet atome est un point de sortie attendu
                is_expected_exit = False
                if atom_idx in match_to_scaffold:
                    scaffold_idx = match_to_scaffold[atom_idx]
                    is_expected_exit = scaffold_idx in expected_exit_points
                
                # Parcourir tous les voisins de cet atome
                for neighbor in atom.GetNeighbors():
                    neighbor_idx = neighbor.GetIdx()
                    
                    # Si le voisin n'est pas dans le scaffold
                    if neighbor_idx not in match_atoms_set:
                        # S'il s'agit d'un point de sortie attendu, c'est normal
                        if is_expected_exit:
                            continue
                        
                        # Sinon, c'est un substituant inattendu
                        has_extra_substituents = True
                        break
                
                if has_extra_substituents:
                    break
            
            # 4. Vérifier si le scaffold est chimiquement identique
            scaffold_match_exact = False
            
            if not has_extra_substituents and len(match_atoms) == scaffold_atom_count:
                # Extraire le scaffold trouvé pour une comparaison SMILES
                match_mol = Chem.RWMol()
                atom_map = {}
                
                # Ajouter les atomes du match au nouveau mol
                for i, atom_idx in enumerate(match_atoms):
                    atom = test_mol.GetAtomWithIdx(atom_idx)
                    new_idx = match_mol.AddAtom(Chem.Atom(atom.GetAtomicNum()))
                    atom_map[atom_idx] = new_idx
                    # Copier les propriétés importantes
                    match_mol.GetAtomWithIdx(new_idx).SetFormalCharge(atom.GetFormalCharge())
                    match_mol.GetAtomWithIdx(new_idx).SetNumExplicitHs(atom.GetNumExplicitHs())
                    match_mol.GetAtomWithIdx(new_idx).SetIsAromatic(atom.GetIsAromatic())
                    match_mol.GetAtomWithIdx(new_idx).SetChiralTag(atom.GetChiralTag())
                
                # Ajouter les liaisons
                for i, atom_idx in enumerate(match_atoms):
                    for j in range(i+1, len(match_atoms)):
                        other_idx = match_atoms[j]
                        bond = test_mol.GetBondBetweenAtoms(atom_idx, other_idx)
                        if bond:
                            match_mol.AddBond(atom_map[atom_idx], atom_map[other_idx], bond.GetBondType())
                
                # Convertir en molécule finale et sanitizer
                try:
                    extracted_scaffold = match_mol.GetMol()
                    Chem.SanitizeMol(extracted_scaffold)
                    
                    # Comparer les SMILES
                    extracted_smiles = Chem.MolToSmiles(extracted_scaffold, isomericSmiles=True)
                    
                    if extracted_smiles == scaffold_smiles:
                        scaffold_match_exact = True
                except Exception as e:
                    scaffold_match_exact = False
            
            # 5. Si le scaffold est exactement identique, ajouter la molécule aux résultats
            if scaffold_match_exact:
                # Créer les données de résultat
                result_data = {
                    'ID': mol_id,
                    'SMILES': smiles,
                    'Similarite': similarity,
                    'Atomes_Modifies_Test': modified_atoms_test,
                    'Match_Atoms': match_atoms,
                    'Mol_Obj': test_mol,
                    'Original_Row': original_row  # Conserver les données originales
                }
                
                results.append(result_data)
    
    # Terminer la barre de progression
    progress_bar.progress(1.0)
    
    # Trier par similarité décroissante
    results_df = pd.DataFrame(results) if results else pd.DataFrame()
    
    if not results_df.empty:
        results_df = results_df.sort_values(by='Similarite', ascending=False)
    
    # Stocker dans session_state pour utilisation future
    st.session_state.all_results = results_df
    
    return results_df

# Interface utilisateur
st.title("Recherche Analogues")

# Zone pour sélectionner la molécule de référence
st.header("Sélection de la molécule de référence")

# Option par défaut pour I0
default_ref_id = "Molport-001-492-296"
use_default = st.checkbox("Utiliser I0 comme molécule de référence", value=True)

ref_id = default_ref_id
ref_smiles = ""

if not use_default:
    # Permettre à l'utilisateur de spécifier une référence personnalisée
    ref_input_type = st.radio(
        "Spécifier la molécule de référence par:",
        ["ID", "SMILES"], horizontal=True
    )
    
    if ref_input_type == "ID":
        ref_id = st.text_input("ID de la molécule de référence:", key="custom_ref_id")
    else:
        ref_smiles = st.text_input("SMILES de la molécule de référence:", key="custom_ref_smiles")

# Upload du fichier
uploaded_file = st.file_uploader("Charger le fichier CSV/TSV de molecules", type=["csv", "tsv"])

if uploaded_file is not None:
    try:
        # Detecter le delimiteur
        file_ext = uploaded_file.name.split('.')[-1].lower()
        delimiter = '\t' if file_ext == 'tsv' else ','
        
        # Charger le fichier
        df = pd.read_csv(uploaded_file, delimiter=delimiter)
        
        # Verifier les colonnes requises
        if 'SMILES' not in df.columns or 'ID' not in df.columns:
            st.error("Le fichier doit contenir les colonnes 'SMILES' et 'ID'")
        else:
            st.success(f"Fichier charge avec succes! {len(df)} molecules trouvees.")
            
            # Trouver la molecule de référence
            if use_default:
                # Chercher I0 dans le fichier
                ref_data = df[df['ID'] == ref_id]
                if len(ref_data) == 0:
                    st.warning(f"La molécule I0 ({ref_id}) n'a pas été trouvée dans le fichier.")
                    use_custom_smiles = st.checkbox("Entrer manuellement le SMILES de I0")
                    if use_custom_smiles:
                        ref_smiles = st.text_input("SMILES de I0:", key="i0_smiles_input")
                else:
                    st.success(f"Molécule de référence I0 trouvée: {ref_id}")
                    ref_smiles = ref_data['SMILES'].values[0]
            else:
                # Utiliser la référence personnalisée
                if ref_input_type == "ID":
                    if not ref_id:
                        st.error("Veuillez entrer un ID de molécule de référence valide.")
                    else:
                        ref_data = df[df['ID'] == ref_id]
                        if len(ref_data) == 0:
                            st.error(f"La molécule avec l'ID '{ref_id}' n'a pas été trouvée dans le fichier.")
                        else:
                            st.success(f"Molécule de référence trouvée: {ref_id}")
                            ref_smiles = ref_data['SMILES'].values[0]
                else:  # SMILES
                    if not ref_smiles:
                        st.error("Veuillez entrer un SMILES de molécule de référence valide.")
            
            # Créer la molécule de référence
            if ref_smiles:
                ref_mol = mol_from_smiles(ref_smiles)
                if ref_mol:
                    # Afficher l'ID ou SMILES de la molécule de référence
                    if use_default:
                        display_name = f"{ref_id} (I0)"
                    else:
                        display_name = ref_id if ref_input_type == "ID" else "Molécule personnalisée"
                    
                    st.subheader(f"Molécule de référence: {display_name}")
                    ref_img = generate_molecule_image(ref_mol, (600, 400), molName=display_name)
                    if ref_img:
                        st.image(ref_img, caption=f"SMILES: {ref_smiles}")
                    
                    # Extraire les fragments avec identification précise
                    st.subheader("Fragments de la molécule de référence")
                    fragments = identify_atom_specific_fragments(ref_mol)
                    
                    # Stocker les fragments dans la session state
                    st.session_state.fragments = fragments
                    
                    # Visualisation des fragments disponibles
                    st.write("### Selectionnez les fragments a MODIFIER")
                    st.warning("Attention: Sélectionnez uniquement les fragments que vous souhaitez voir varier. Les parties non sélectionnées resteront identiques.")
                    
                    # Mise en page en grille
                    col_count = min(3, len(fragments))
                    
                    # Pour chaque fragment, créer une rangée avec l'image et la case à cocher
                    fragment_rows = []
                    for i in range(0, len(fragments), col_count):
                        fragment_rows.append(list(fragments.keys())[i:i+col_count])
                    
                    for row in fragment_rows:
                        cols = st.columns(col_count)
                        for i, fragment_name in enumerate(row):
                            if i < len(row):  # Vérifier que l'index est valide
                                atom_indices = fragments[fragment_name]
                                with cols[i]:
                                    # Générer l'image du fragment surligné
                                    fragment_img = generate_molecule_image(
                                        ref_mol, 
                                        (250, 200), 
                                        highlightAtoms=atom_indices, 
                                        molName=fragment_name
                                    )
                                    
                                    if fragment_img:
                                        st.image(fragment_img)
                                    
                                    # Ajouter des informations sur les atomes impliqués (de façon canonique)
                                    atom_info = get_canonical_atom_description(ref_mol, atom_indices)
                                    st.write(f"Atomes: {atom_info}")
                                    
                                    # Case à cocher pour sélectionner ce fragment
                                    fragment_selected = st.checkbox(
                                        fragment_name, 
                                        value=fragment_name in st.session_state.selected_fragment_names,
                                        key=f"checkbox_{fragment_name}"
                                    )
                                    
                                    # Mettre à jour la liste des fragments sélectionnés
                                    if fragment_selected and fragment_name not in st.session_state.selected_fragment_names:
                                        st.session_state.selected_fragment_names.append(fragment_name)
                                    elif not fragment_selected and fragment_name in st.session_state.selected_fragment_names:
                                        st.session_state.selected_fragment_names.remove(fragment_name)
                    
                    # Afficher les fragments sélectionnés combinés
                    if st.session_state.selected_fragment_names:
                        st.subheader("Zones de modification selectionnees")
                        
                        # Récupérer les indices des atomes pour tous les fragments sélectionnés
                        modification_atoms = []
                        
                        for frag_name in st.session_state.selected_fragment_names:
                            if frag_name in fragments:
                                frag_atoms = fragments[frag_name]
                                modification_atoms.extend(frag_atoms)
                        
                        # Éliminer les doublons
                        modification_atoms = list(set(modification_atoms))
                        
                        # Stocker les atomes de modification dans la session state
                        st.session_state.modification_atoms = modification_atoms
                        
                        # Créer une visualisation des fragments sélectionnés combinés
                        combined_img = generate_molecule_image(
                            ref_mol, 
                            (600, 400), 
                            highlightAtoms=modification_atoms, 
                            molName="Zones de modifications autorisees"
                        )
                        
                        if combined_img:
                            st.image(combined_img)
                            
                        st.write(f"Fragments selectionnes: {', '.join(st.session_state.selected_fragment_names)}")
                        
                        # Créer et afficher le scaffold (ce qui ne changera pas)
                        scaffold_mol, exit_points = extract_scaffold(ref_mol, modification_atoms)
                        if scaffold_mol:
                            # Stocker le scaffold dans la session state
                            st.session_state.scaffold_mol = scaffold_mol
                            st.session_state.scaffold_exit_points = exit_points
                            
                            st.subheader("Partie invariante (scaffold)")
                            scaffold_img = generate_molecule_image(
                                scaffold_mol, 
                                (600, 400), 
                                highlightAtoms=exit_points,
                                molName="Scaffold (partie qui restera identique)"
                            )
                            if scaffold_img:
                                st.image(scaffold_img)
                                scaffold_atom_count = scaffold_mol.GetNumAtoms()
                                scaffold_smiles = Chem.MolToSmiles(scaffold_mol)
                                st.info(f"Les points de connexion aux zones de modification sont surlignés en rouge. Le scaffold contient {scaffold_atom_count} atomes.")
                                st.text(f"SMILES du scaffold: {scaffold_smiles}")
                    else:
                        st.info("Selectionnez au moins un fragment a modifier pour commencer l'analyse")
                    
                    # Bouton pour rechercher les analogues avec modifications spécifiques
                    if st.session_state.selected_fragment_names:
                        search_button = st.button(
                            "Rechercher les analogues", 
                            key="search_analogs_button"
                        )
                        
                        if search_button:
                            # Récupérer les atomes à modifier
                            modification_atoms = st.session_state.modification_atoms
                            
                            st.info(f"Recherche d'analogues où la partie invariante ({ref_mol.GetNumAtoms() - len(modification_atoms)} atomes) reste chimiquement identique...")
                            
                            results = find_analogs_by_fixed_scaffold(df, ref_mol, modification_atoms, ref_id)
                            
                            # Stocker les résultats dans la session state
                            st.session_state.results = results
                    
                    # Si des résultats existent dans la session, les afficher
                    if 'results' in st.session_state and not st.session_state.results.empty:
                        results = st.session_state.results
                        st.success(f"Trouvé {len(results)} analogues avec des modifications uniquement dans les zones sélectionnées")
                        
                        # Créer un DataFrame pour l'affichage avec des cases à cocher
                        display_df = results[['ID', 'SMILES', 'Similarite']].copy()
                        # Ajouter une colonne de sélection
                        display_df['Selectionner'] = True
                        # Convertir la similarité en pourcentage pour une meilleure lisibilité
                        display_df['Similarite'] = display_df['Similarite'].apply(lambda x: f"{x*100:.1f}%")
                        
                        # Afficher le dataframe avec la possibilité de sélectionner des lignes
                        st.subheader("Analogues identifiés")
                        st.write("Cochez les molécules que vous souhaitez inclure dans le CSV final")
                        
                        # Utiliser st.data_editor pour créer un dataframe modifiable avec cases à cocher
                        edited_df = st.data_editor(
                            display_df,
                            column_config={
                                "Selectionner": st.column_config.CheckboxColumn(
                                    "Inclure dans CSV",
                                    help="Cochez pour inclure cette molécule dans le CSV final",
                                    default=True,
                                )
                            },
                            hide_index=True,
                            key="results_editor"
                        )
                        
                        # Stocker le DataFrame édité dans la session state
                        st.session_state.edited_df = edited_df
                        
                        # Bouton pour télécharger les molécules sélectionnées
                        if st.button("Télécharger les molécules sélectionnées", key="download_selected"):
                            # Récupérer les IDs des molécules sélectionnées
                            selected_ids = edited_df[edited_df['Selectionner']]['ID'].tolist()
                            
                            if selected_ids:
                                # Filtrer les résultats originaux
                                download_df = results[results['ID'].isin(selected_ids)]
                                
                                # Générer le lien de téléchargement
                                st.markdown(get_download_link(download_df, "molecules_selectionnees.csv", 
                                                            "Télécharger les molécules sélectionnées en CSV"), unsafe_allow_html=True)
                            else:
                                st.error("Aucune molécule sélectionnée pour le téléchargement")
                        
                        # Afficher TOUTES les molécules trouvées
                        st.markdown("---")
                        st.subheader("Visualisation de tous les analogues")
                        
                        # Calculer le nombre de colonnes en fonction du nombre de molécules
                        num_cols = min(3, len(results))  # Maximum 3 colonnes
                        
                        # Grouper les molécules par lignes
                        molecule_rows = []
                        for i in range(0, len(results), num_cols):
                            molecule_rows.append(results.iloc[i:i+num_cols])
                        
                        # Afficher chaque groupe de molécules
                        for idx, row_group in enumerate(molecule_rows):
                            cols = st.columns(num_cols)
                            
                            for i, (_, row) in enumerate(row_group.iterrows()):
                                if i < len(cols):  # Vérifier que l'index est valide
                                    with cols[i]:
                                        mol_idx = idx * num_cols + i + 1
                                        st.write(f"**{mol_idx}. {row['ID']}**")
                                        st.write(f"Similarité de Tanimoto (morgan R=3): {row['Similarite']*100:.1f}%")
                                        
                                        # Obtenir l'objet molécule
                                        analog_mol = row['Mol_Obj']
                                        
                                        # Générer l'image avec les différences surlignées
                                        analog_img = generate_molecule_image(
                                            analog_mol, 
                                            (300, 250), 
                                            highlightAtoms=row['Atomes_Modifies_Test'], 
                                            molName=row['ID']
                                        )
                                        
                                        if analog_img:
                                            st.image(analog_img)
                                        
                                        # Afficher les détails des atomes modifiés
                                        with st.expander("Voir détails"):
                                            st.write(f"**Nombre d'atomes modifiés:** {len(row['Atomes_Modifies_Test'])}")
                                            
                                            # Générer une version moins encombrante du SMILES
                                            canonical_smiles = Chem.MolToSmiles(analog_mol, isomericSmiles=True)
                                            st.code(canonical_smiles, language="python")
                                            
                        # Ajouter une légende pour les visualisations
                        st.info("""
                        **Légende**: 
                        - Les parties surlignées en rouge représentent les modifications par rapport à la molécule de référence
                        """)
                else:
                    st.error("SMILES invalide pour la molécule de référence.")
            else:
                st.error("Impossible de trouver ou de définir une molécule de référence valide.")

    except Exception as e:
        st.error(f"Une erreur s'est produite lors du traitement du fichier: {str(e)}")
        st.error(f"Détails: {type(e).__name__}")
        import traceback
        st.code(traceback.format_exc())
else:
    st.info("""
    ## Comment utiliser cet outil
    
    1. **Choisissez votre molécule de référence**:
       - Utilisez I0 (par défaut)
       - Ou spécifiez une autre molécule par ID ou SMILES
    
    2. **Chargez votre fichier CSV/TSV** contenant les molécules à analyser (doit contenir les colonnes 'SMILES' et 'ID')
    
    3. **Sélectionnez les fragments à MODIFIER** - toutes les parties non sélectionnées resteront identiques
    
    4. **Examinez le scaffold** qui montre la partie qui restera identique dans tous les analogues
    
    5. **Recherchez les analogues** qui ont exactement le même scaffold avec des modifications uniquement aux endroits sélectionnés
    
    6. **Téléchargez les résultats** au format CSV
    
    ### Principe de fonctionnement
    
    Cette approche garantit que les modifications se produisent uniquement aux endroits spécifiés :
    - L'outil extrait le "scaffold" (partie non sélectionnée)
    - Il recherche des molécules contenant exactement ce scaffold
    - Le fichier CSV exporté contient toutes les colonnes du fichier d'origine plus une information de similarité (Tanimoto - Morgan R 3)
    """)
