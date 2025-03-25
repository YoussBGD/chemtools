import streamlit as st
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, rdFMCS, rdDepictor, DataStructs, Descriptors, rdMolAlign
from rdkit.Chem.Draw import rdMolDraw2D
from PIL import Image
import io
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import random

st.set_page_config(page_title="Comparaison avec I0", layout="wide")

@st.cache_data
def load_data(file):
    """Charge et met en cache les données du fichier CSV"""
    return pd.read_csv(file)

def mol_from_smiles(smiles):
    """Crée un objet RDKit mol à partir d'un SMILES"""
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        # Calculer les coordonnées 2D si nécessaire
        if not mol.GetNumConformers():
            rdDepictor.Compute2DCoords(mol)
    return mol

def generate_molecule_image(mol, size=(400, 300), legend=""):
    """Génère une image d'une seule molécule"""
    if mol is None:
        return None
    
    try:
        # Utiliser la méthode MolToImage plus simple et robuste
        img = Draw.MolToImage(mol, size=size, legend=legend)
        return img
    except Exception as e:
        st.error(f"Erreur lors de la génération de l'image: {str(e)}")
        return None

def find_rotatable_bonds(mol):
    """Identifie les liaisons rotatives dans une molécule"""
    if mol is None:
        return []
    
    rot_bonds = []
    for bond in mol.GetBonds():
        if bond.GetBondType() == Chem.BondType.SINGLE and \
           not bond.IsInRing() and \
           bond.GetBeginAtom().GetAtomicNum() > 1 and \
           bond.GetEndAtom().GetAtomicNum() > 1:
            rot_bonds.append(bond.GetIdx())
    
    return rot_bonds

def optimize_rotatable_bonds(ref_mol, mol):
    """Optimise les angles des liaisons rotatives pour une meilleure superposition avec la référence"""
    if ref_mol is None or mol is None:
        return mol
    
    # Créer une copie de la molécule pour ne pas modifier l'original
    optimized_mol = Chem.Mol(mol)
    
    try:
        # Trouver les liaisons rotatives
        rot_bonds = find_rotatable_bonds(optimized_mol)
        
        if not rot_bonds:
            # Pas de liaisons rotatives, retourner la molécule telle quelle
            return optimized_mol
        
        # Pour simplifier, on essaie quelques angles pour chaque liaison rotative
        best_similarity = 0
        best_conformer = optimized_mol
        
        # Générer plusieurs conformations en faisant varier les angles des liaisons rotatives
        confs = []
        for _ in range(10):  # Essayer 10 conformations différentes
            mol_copy = Chem.Mol(optimized_mol)
            
            # Appliquer des rotations aléatoires aux liaisons rotatives
            for bond_idx in rot_bonds:
                # Angle aléatoire entre 0 et 360 degrés
                angle = random.uniform(0, 360)
                
                # Essayer de faire pivoter la liaison (peut échouer si la molécule n'a pas de conformère 3D)
                try:
                    AllChem.EmbedMolecule(mol_copy)
                    AllChem.SetDihedralDeg(mol_copy.GetConformer(), *list(mol_copy.GetBondWithIdx(bond_idx).GetBeginAtomIdx()), angle)
                except:
                    pass
            
            # Aplatir la molécule en 2D pour la visualisation
            AllChem.Compute2DCoords(mol_copy)
            
            confs.append(mol_copy)
        
        # Évaluer chaque conformation en termes de superposition avec la référence
        for conf in confs:
            # Calculer une mesure de similarité
            fp1 = AllChem.GetMorganFingerprintAsBitVect(ref_mol, 2)
            fp2 = AllChem.GetMorganFingerprintAsBitVect(conf, 2)
            similarity = DataStructs.TanimotoSimilarity(fp1, fp2)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_conformer = conf
        
        # Retourner la meilleure conformation
        return best_conformer
            
    except Exception as e:
        # st.error(f"Erreur lors de l'optimisation des liaisons rotatives: {str(e)}")
        return optimized_mol  # En cas d'erreur, retourner la molécule non modifiée

def align_mol_to_ref(ref_mol, mol):
    """Aligne une molécule sur la référence en utilisant le MCS"""
    if ref_mol is None or mol is None:
        return mol  # Retourne la molécule non modifiée
    
    try:
        # Créer une copie pour ne pas modifier l'original
        aligned_mol = Chem.Mol(mol)
        
        # S'assurer que les deux molécules ont des coordonnées 2D
        if not ref_mol.GetNumConformers():
            rdDepictor.Compute2DCoords(ref_mol)
        if not aligned_mol.GetNumConformers():
            rdDepictor.Compute2DCoords(aligned_mol)
        
        # Trouver la sous-structure commune maximum (MCS)
        mcs = rdFMCS.FindMCS([ref_mol, aligned_mol], 
                           completeRingsOnly=True,
                           ringMatchesRingOnly=True,
                           timeout=1)
        
        if mcs.numAtoms < 3:  # Trop peu d'atomes communs
            return aligned_mol
            
        # Créer un objet molécule à partir du SMARTS du MCS
        mcs_mol = Chem.MolFromSmarts(mcs.smartsString)
        
        # Trouver les correspondances dans les deux molécules
        ref_match = ref_mol.GetSubstructMatch(mcs_mol)
        mol_match = aligned_mol.GetSubstructMatch(mcs_mol)
        
        if not ref_match or not mol_match:
            return aligned_mol
        
        # Utiliser AllChem.GenerateDepictionMatching2DStructure qui aligne TOUTE la molécule
        try:
            # Cette fonction déplace toute la molécule pour aligner les parties correspondantes
            rdDepictor.GenerateDepictionMatching2DStructure(aligned_mol, ref_mol, mol_match, ref_match)
        except:
            # En cas d'échec, conserver les coordonnées originales
            pass
            
        return aligned_mol
    except Exception as e:
        # st.error(f"Erreur lors de l'alignement: {str(e)}")
        return mol  # En cas d'erreur, retourner la molécule non modifiée

def rotate_molecule(mol, angle_degrees):
    """Fait pivoter une molécule 2D d'un angle donné"""
    if mol is None:
        return None
    
    try:
        # Créer une copie de la molécule
        rotated_mol = Chem.Mol(mol)
        
        # S'assurer que la molécule a des coordonnées 2D
        if not rotated_mol.GetNumConformers():
            rdDepictor.Compute2DCoords(rotated_mol)
        
        # Convertir l'angle en radians
        angle_radians = np.radians(angle_degrees)
        
        # Matrice de rotation 2D
        cos_theta = np.cos(angle_radians)
        sin_theta = np.sin(angle_radians)
        
        # Centre de la molécule
        conformer = rotated_mol.GetConformer()
        coords = [conformer.GetAtomPosition(i) for i in range(rotated_mol.GetNumAtoms())]
        center_x = np.mean([coord.x for coord in coords])
        center_y = np.mean([coord.y for coord in coords])
        
        # Appliquer la rotation à chaque atome
        for i in range(rotated_mol.GetNumAtoms()):
            pos = conformer.GetAtomPosition(i)
            
            # Translater au centre
            x = pos.x - center_x
            y = pos.y - center_y
            
            # Rotation
            x_new = x * cos_theta - y * sin_theta
            y_new = x * sin_theta + y * cos_theta
            
            # Translater en arrière
            x_new += center_x
            y_new += center_y
            
            # Définir la nouvelle position
            conformer.SetAtomPosition(i, Chem.rdGeometry.Point3D(x_new, y_new, 0))
        
        return rotated_mol
    except Exception as e:
        # st.error(f"Erreur lors de la rotation: {str(e)}")
        return mol  # En cas d'erreur, retourner la molécule originale

def flip_molecule_horizontal(mol):
    """Inverse horizontalement une molécule 2D"""
    if mol is None:
        return None
    
    try:
        # Créer une copie de la molécule
        flipped_mol = Chem.Mol(mol)
        
        # S'assurer que la molécule a des coordonnées 2D
        if not flipped_mol.GetNumConformers():
            rdDepictor.Compute2DCoords(flipped_mol)
        
        # Inverser les coordonnées x de chaque atome
        conformer = flipped_mol.GetConformer()
        
        # Calculer le centre pour faire le flip autour du centre
        coords = [conformer.GetAtomPosition(i) for i in range(flipped_mol.GetNumAtoms())]
        center_x = np.mean([coord.x for coord in coords])
        
        for i in range(flipped_mol.GetNumAtoms()):
            pos = conformer.GetAtomPosition(i)
            
            # Inverser la position par rapport au centre
            new_x = 2 * center_x - pos.x
            
            # Définir la nouvelle position
            conformer.SetAtomPosition(i, Chem.rdGeometry.Point3D(new_x, pos.y, pos.z))
        
        return flipped_mol
    except Exception as e:
        # st.error(f"Erreur lors de l'inversion horizontale: {str(e)}")
        return mol  # En cas d'erreur, retourner la molécule originale

def flip_molecule_vertical(mol):
    """Inverse verticalement une molécule 2D"""
    if mol is None:
        return None
    
    try:
        # Créer une copie de la molécule
        flipped_mol = Chem.Mol(mol)
        
        # S'assurer que la molécule a des coordonnées 2D
        if not flipped_mol.GetNumConformers():
            rdDepictor.Compute2DCoords(flipped_mol)
        
        # Inverser les coordonnées y de chaque atome
        conformer = flipped_mol.GetConformer()
        
        # Calculer le centre pour faire le flip autour du centre
        coords = [conformer.GetAtomPosition(i) for i in range(flipped_mol.GetNumAtoms())]
        center_y = np.mean([coord.y for coord in coords])
        
        for i in range(flipped_mol.GetNumAtoms()):
            pos = conformer.GetAtomPosition(i)
            
            # Inverser la position par rapport au centre
            new_y = 2 * center_y - pos.y
            
            # Définir la nouvelle position
            conformer.SetAtomPosition(i, Chem.rdGeometry.Point3D(pos.x, new_y, pos.z))
        
        return flipped_mol
    except Exception as e:
        # st.error(f"Erreur lors de l'inversion verticale: {str(e)}")
        return mol  # En cas d'erreur, retourner la molécule originale

def calculate_similarity(mol1, mol2, radius=2):
    """Calcule la similarité de Tanimoto entre deux molécules"""
    if mol1 is None or mol2 is None:
        return 0.0
    
    try:
        # Générer les empreintes Morgan
        fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, radius)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, radius)
        
        # Calculer la similarité de Tanimoto
        return DataStructs.TanimotoSimilarity(fp1, fp2)
    except:
        return 0.0

def find_similar_molecules(df, mol, selected_molecules, threshold=0.6, max_results=5):
    """Trouve les molécules similaires parmi les molécules sélectionnées"""
    if mol is None or not selected_molecules:
        return []
    
    similarities = []
    for mol_id in selected_molecules:
        mol_data = df[df['ID'] == mol_id]
        if not mol_data.empty:
            mol_smiles = mol_data['SMILES'].values[0]
            comparison_mol = mol_from_smiles(mol_smiles)
            if comparison_mol:
                similarity = calculate_similarity(mol, comparison_mol)
                if similarity >= threshold:
                    similarities.append((mol_id, similarity, comparison_mol))
    
    # Trier par similarité décroissante et limiter le nombre de résultats
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:max_results]

def generate_side_by_side(mol1, mol2, id1, id2, size=(800, 400)):
    """Génère une image côte à côte des deux molécules avec leurs IDs"""
    if mol1 is None or mol2 is None:
        return None
    
    try:
        # Générer les images avec les légendes incluant les IDs
        img1 = Draw.MolToImage(mol1, size=(size[0]//2, size[1]), legend=f"{id1}")
        img2 = Draw.MolToImage(mol2, size=(size[0]//2, size[1]), legend=f"{id2}")
        
        # Combiner les images côte à côte
        combined = Image.new('RGB', (size[0], size[1]), (255, 255, 255))
        combined.paste(img1, (0, 0))
        combined.paste(img2, (size[0]//2, 0))
        
        return combined
    except Exception as e:
        # st.error(f"Erreur lors de la génération de l'image côte à côte: {str(e)}")
        return None

def generate_difference_highlight_image(mol1, mol2, id1, id2, size=(800, 400)):
    """Génère deux images côte à côte avec les différences mises en évidence et IDs affichés"""
    if mol1 is None or mol2 is None:
        return None
    
    try:
        # Trouver la sous-structure commune maximum (MCS)
        mcs = rdFMCS.FindMCS([mol1, mol2], 
                           completeRingsOnly=True,
                           ringMatchesRingOnly=True,
                           timeout=1)
        
        if mcs.numAtoms > 0:
            # Créer un objet molécule à partir du SMARTS du MCS
            mcs_mol = Chem.MolFromSmarts(mcs.smartsString)
            
            # Trouver les sous-structures dans les deux molécules
            mol1_match = mol1.GetSubstructMatch(mcs_mol)
            mol2_match = mol2.GetSubstructMatch(mcs_mol)
            
            # Trouver les atomes qui diffèrent
            mol1_atoms = set(range(mol1.GetNumAtoms()))
            mol2_atoms = set(range(mol2.GetNumAtoms()))
            
            mol1_diff = list(mol1_atoms - set(mol1_match))
            mol2_diff = list(mol2_atoms - set(mol2_match))
            
            # Utiliser la méthode standard Draw.MolToImage avec highlight
            img1 = Draw.MolToImage(mol1, size=(size[0]//2, size[1]), 
                                  highlightAtoms=mol1_diff, 
                                  highlightColor=(1, 0, 0),  # Rouge
                                  legend=f"{id1}")
            
            img2 = Draw.MolToImage(mol2, size=(size[0]//2, size[1]), 
                                  highlightAtoms=mol2_diff, 
                                  highlightColor=(0, 0.7, 0),  # Vert
                                  legend=f"{id2}")
            
            # Combiner les images côte à côte
            combined = Image.new('RGB', (size[0], size[1]), (255, 255, 255))
            combined.paste(img1, (0, 0))
            combined.paste(img2, (size[0]//2, 0))
            
            return combined
            
        else:
            # Si pas de MCS, retourner les images sans surlignage
            return generate_side_by_side(mol1, mol2, id1, id2, size)
    except Exception as e:
        # st.error(f"Erreur lors de la génération des images avec surlignage: {str(e)}")
        return generate_side_by_side(mol1, mol2, id1, id2, size)  # Fallback à l'affichage standard

def create_download_link(df, filename):
    """Crée un lien de téléchargement pour un dataframe"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Télécharger {filename}</a>'
    return href

# Interface utilisateur
st.title("Comparaison avec I0")

st.sidebar.header("Options")
image_width = st.sidebar.slider("Largeur d'image", 600, 1200, 800, key="image_width_slider")
image_height = st.sidebar.slider("Hauteur d'image", 300, 600, 400, key="image_height_slider")

# Initialisation des variables de session
if 'current_index' not in st.session_state:
    st.session_state.current_index = 0
if 'rotation_angle' not in st.session_state:
    st.session_state.rotation_angle = 0
if 'flip_h' not in st.session_state:
    st.session_state.flip_h = False
if 'flip_v' not in st.session_state:
    st.session_state.flip_v = False
if 'auto_align' not in st.session_state:
    st.session_state.auto_align = True
if 'optimize_rotatable_bonds' not in st.session_state:
    st.session_state.optimize_rotatable_bonds = True
if 'selected_molecules' not in st.session_state:
    st.session_state.selected_molecules = set()
if 'similar_molecules_index' not in st.session_state:
    st.session_state.similar_molecules_index = 0

# Upload du fichier
uploaded_file = st.file_uploader("Charger le fichier CSV de molécules", type=["csv"])

if uploaded_file is not None:
    try:
        # Charger les données
        df = load_data(uploaded_file)
        
        # Vérifier que les colonnes nécessaires existent
        required_columns = ['SMILES', 'ID']
        if not all(col in df.columns for col in required_columns):
            missing = [col for col in required_columns if col not in df.columns]
            st.error(f"Le fichier manque des colonnes requises: {', '.join(missing)}")
        else:
            st.success(f"Fichier chargé avec succès! {len(df)} molécules trouvées.")
            
            # Rechercher I0 dans le fichier
            i0_id = "Molport-001-492-296"
            i0_data = df[df['ID'] == i0_id]
            
            if len(i0_data) == 0:
                st.error(f"La molécule de référence I0 (ID: {i0_id}) n'a pas été trouvée dans le fichier.")
                # Option pour entrer manuellement le SMILES de I0
                i0_smiles = st.text_input("Entrez manuellement le SMILES de la molécule I0:", key="i0_smiles_input")
                if i0_smiles:
                    i0_mol = mol_from_smiles(i0_smiles)
                    if i0_mol:
                        st.success("Molécule I0 définie manuellement.")
                    else:
                        st.error("SMILES invalide pour I0.")
                        i0_mol = None
                else:
                    i0_mol = None
            else:
                st.success(f"Molécule de référence I0 trouvée: {i0_id}")
                i0_smiles = i0_data['SMILES'].values[0]
                i0_mol = mol_from_smiles(i0_smiles)
                
                if i0_mol:
                    # Afficher I0 seule
                    st.subheader("Molécule de référence I0")
                    i0_img = generate_molecule_image(i0_mol, (image_width//2, image_height//2), f"I0: {i0_id}")
                    if i0_img:
                        st.image(i0_img, caption=f"I0: {i0_smiles}")
                    
                    # Préparer la navigation
                    df_without_i0 = df[df['ID'] != i0_id].reset_index(drop=True)
                    total_mols = len(df_without_i0)
                    
                    # Fonctions pour les boutons de navigation
                    def next_molecule():
                        st.session_state.current_index = (st.session_state.current_index + 1) % total_mols
                        # Réinitialiser les transformations
                        st.session_state.rotation_angle = 0
                        st.session_state.flip_h = False
                        st.session_state.flip_v = False
                        # Réinitialiser l'index des molécules similaires
                        st.session_state.similar_molecules_index = 0
                    
                    def prev_molecule():
                        st.session_state.current_index = (st.session_state.current_index - 1) % total_mols
                        # Réinitialiser les transformations
                        st.session_state.rotation_angle = 0
                        st.session_state.flip_h = False
                        st.session_state.flip_v = False
                        # Réinitialiser l'index des molécules similaires
                        st.session_state.similar_molecules_index = 0
                    
                    def jump_to_molecule(index):
                        st.session_state.current_index = index
                        # Réinitialiser les transformations
                        st.session_state.rotation_angle = 0
                        st.session_state.flip_h = False
                        st.session_state.flip_v = False
                        # Réinitialiser l'index des molécules similaires
                        st.session_state.similar_molecules_index = 0
                        
                    def jump_to_first():
                        jump_to_molecule(0)
                        
                    def jump_to_last():
                        jump_to_molecule(total_mols - 1)
                    
                    # Assurer que l'index actuel est valide
                    st.session_state.current_index = max(0, min(st.session_state.current_index, total_mols - 1))
                    
                    # Section pour les molécules sélectionnées
                    st.sidebar.subheader("Molécules sélectionnées")
                    st.sidebar.write(f"Nombre de molécules sélectionnées: {len(st.session_state.selected_molecules)}")
                    
                    if len(st.session_state.selected_molecules) > 0:
                        # Créer un DataFrame avec les molécules sélectionnées
                        selected_df = df[df['ID'].isin(st.session_state.selected_molecules)]
                        
                        # Créer un lien de téléchargement
                        download_link = create_download_link(selected_df, "molecules_selectionnees.csv")
                        st.sidebar.markdown(download_link, unsafe_allow_html=True)
                        
                        # Option pour effacer la sélection
                        if st.sidebar.button("Effacer la sélection", key="clear_selection_button"):
                            st.session_state.selected_molecules = set()
                            st.session_state.similar_molecules_index = 0
                            st.rerun()
                    
                    # Obtenir la molécule actuelle
                    current_mol = df_without_i0.iloc[st.session_state.current_index]
                    current_id = current_mol['ID']
                    
                    # Créer l'objet mol pour la molécule sélectionnée
                    original_mol = mol_from_smiles(current_mol['SMILES'])
                    
                    if original_mol:
                        # En-tête pour la molécule actuelle
                        status = "✓ Sélectionnée" if current_id in st.session_state.selected_molecules else ""
                        st.subheader(f"Molécule actuelle: {current_id} {status}")
                        
                        # Afficher les informations sur la navigation avec bouton de sélection plus visible
                        st.info(f"Molécule {st.session_state.current_index + 1} sur {total_mols}")
                        
                        # Nouvelle section pour la navigation directe
                        st.subheader("Navigation directe")
                        
                        # Layout pour les options de navigation avancée
                        nav_cols = st.columns([1, 1, 1])
                        
                        with nav_cols[0]:
                            # Aller au début
                            if st.button("⏮️ Première", key="first_mol_button", use_container_width=True):
                                jump_to_first()
                                st.rerun()
                        
                        with nav_cols[1]:
                            # Navigation par index
                            target_index = st.number_input(
                                "Aller à l'index:", 
                                min_value=1, 
                                max_value=total_mols, 
                                value=st.session_state.current_index + 1,
                                key="target_index_input"
                            )
                            
                            if st.button("🔍 Aller", key="go_to_index_button", use_container_width=True):
                                # Convertir l'index affiché (1-based) en index interne (0-based)
                                jump_to_molecule(target_index - 1)
                                st.rerun()
                                
                        with nav_cols[2]:
                            # Aller à la fin
                            if st.button("⏭️ Dernière", key="last_mol_button", use_container_width=True):
                                jump_to_last()
                                st.rerun()
                        
                        # Recherche par ID
                        st.subheader("Recherche par ID")
                        search_cols = st.columns([3, 1])
                        
                        with search_cols[0]:
                            # Liste déroulante avec recherche pour les IDs
                            all_ids = df_without_i0['ID'].tolist()
                            selected_id = st.selectbox(
                                "Rechercher un ID:",
                                options=all_ids,
                                index=st.session_state.current_index,
                                key="molecule_id_search"
                            )
                        
                        with search_cols[1]:
                            if st.button("🔎 Rechercher", key="search_id_button", use_container_width=True):
                                # Trouver l'index de la molécule avec cet ID
                                try:
                                    target_idx = df_without_i0[df_without_i0['ID'] == selected_id].index[0]
                                    jump_to_molecule(target_idx)
                                    st.rerun()
                                except IndexError:
                                    st.error(f"ID non trouvé: {selected_id}")
                        
                        # Interface de navigation standard
                        st.subheader("Navigation séquentielle")
                        col1, col2, col3 = st.columns([1, 1, 1])
                        with col1:
                            st.button("❮ Précédente", on_click=prev_molecule, key="prev_button", use_container_width=True)
                        
                        with col2:
                            # Bouton de sélection plus visible et explicite
                            if current_id in st.session_state.selected_molecules:
                                if st.button("❌ Désélectionner cette molécule", 
                                             key="deselect_button",
                                             use_container_width=True):
                                    st.session_state.selected_molecules.remove(current_id)
                                    st.rerun()
                            else:
                                if st.button("✅ Sélectionner cette molécule", 
                                             key="select_button",
                                             use_container_width=True):
                                    st.session_state.selected_molecules.add(current_id)
                                    st.rerun()
                        
                        with col3:
                            st.button("Suivante ❯", on_click=next_molecule, key="next_button", use_container_width=True)
                        
                        # Options de transformation de la molécule
                        st.sidebar.subheader("Options d'alignement")
                        st.session_state.auto_align = st.sidebar.checkbox("Alignement automatique", 
                                                                        value=st.session_state.auto_align, 
                                                                        key="auto_align_checkbox")
                        st.session_state.optimize_rotatable_bonds = st.sidebar.checkbox("Optimiser les liaisons rotatives", 
                                                                                     value=st.session_state.optimize_rotatable_bonds, 
                                                                                     key="optimize_bonds_checkbox")
                        
                        # Options de rotation manuelle
                        st.sidebar.subheader("Rotation manuelle")
                        col1, col2 = st.sidebar.columns(2)
                        with col1:
                            if st.button("⟲ Rotation -90°", key="rotate_left_button"):
                                st.session_state.rotation_angle = (st.session_state.rotation_angle - 90) % 360
                        with col2:
                            if st.button("⟳ Rotation +90°", key="rotate_right_button"):
                                st.session_state.rotation_angle = (st.session_state.rotation_angle + 90) % 360
                        
                        # Slider pour la rotation fine
                        rotation_angle = st.sidebar.slider("Rotation", -180, 180, st.session_state.rotation_angle, 10, 
                                                        key="rotation_slider")
                        if rotation_angle != st.session_state.rotation_angle:
                            st.session_state.rotation_angle = rotation_angle
                        
                        # Boutons pour inverser
                        col1, col2 = st.sidebar.columns(2)
                        with col1:
                            if st.button("↔️ Inverser H", key="flip_h_button"):
                                st.session_state.flip_h = not st.session_state.flip_h
                        with col2:
                            if st.button("↕️ Inverser V", key="flip_v_button"):
                                st.session_state.flip_v = not st.session_state.flip_v
                                
                        # Bouton pour réinitialiser les transformations
                        if st.sidebar.button("Réinitialiser", key="reset_transforms_button"):
                            st.session_state.rotation_angle = 0
                            st.session_state.flip_h = False
                            st.session_state.flip_v = False
                            st.session_state.auto_align = True
                            st.session_state.optimize_rotatable_bonds = True
                        
                        # Appliquer les transformations à la molécule
                        mol = Chem.Mol(original_mol)
                        
                        # D'abord l'alignement automatique si activé
                        if st.session_state.auto_align:
                            mol = align_mol_to_ref(i0_mol, mol)
                            
                        # Ensuite optimiser les liaisons rotatives si activé
                        if st.session_state.optimize_rotatable_bonds:
                            mol = optimize_rotatable_bonds(i0_mol, mol)
                        
                        # Puis les transformations manuelles
                        if st.session_state.rotation_angle != 0:
                            mol = rotate_molecule(mol, st.session_state.rotation_angle)
                        
                        if st.session_state.flip_h:
                            mol = flip_molecule_horizontal(mol)
                        
                        if st.session_state.flip_v:
                            mol = flip_molecule_vertical(mol)
                        
                        # Afficher d'abord la comparaison avec I0
                        st.subheader("Comparaison avec I0")
                        st.markdown("""
                        Visualisation côte à côte avec les différences mises en évidence:
                        - **Rouge**: Parties présentes uniquement dans I0
                        - **Vert**: Parties présentes uniquement dans la molécule comparée
                        - **Noir**: Structure commune aux deux molécules
                        """)
                        
                        highlight_img = generate_difference_highlight_image(
                            i0_mol, 
                            mol,
                            i0_id,
                            current_id,
                            (image_width, image_height)
                        )
                        
                        if highlight_img:
                            st.image(highlight_img)
                        else:
                            st.warning("Impossible de générer l'image avec surlignage.")
                            
                        # Informations sur les liaisons rotatives
                        rot_bonds = find_rotatable_bonds(mol)
                        if rot_bonds and st.session_state.optimize_rotatable_bonds:
                            st.info(f"Cette molécule possède {len(rot_bonds)} liaisons rotatives qui ont été optimisées automatiquement pour une meilleure superposition avec I0.")
                        
                        # Trouver des molécules similaires parmi celles déjà sélectionnées
                        similar_molecules = []
                        if len(st.session_state.selected_molecules) > 0 and current_id not in st.session_state.selected_molecules:
                            similar_molecules = find_similar_molecules(
                                df,
                                original_mol, 
                                st.session_state.selected_molecules, 
                                threshold=0.5, 
                                max_results=10
                            )
                        
                        # Afficher les molécules similaires si elles existent
                        if similar_molecules:
                            st.subheader("Molécules similaires (déjà sélectionnées)")
                            st.markdown("Cette molécule ressemble aux molécules suivantes que vous avez déjà sélectionnées:")
                            
                            # Assurer que l'index est dans les limites
                            if st.session_state.similar_molecules_index >= len(similar_molecules):
                                st.session_state.similar_molecules_index = 0
                            
                            # Obtenir la molécule similaire actuelle
                            similar_id, similarity, similar_mol = similar_molecules[st.session_state.similar_molecules_index]
                            
                            # Afficher l'information sur les molécules similaires
                            st.info(f"Affichage de la molécule similaire {st.session_state.similar_molecules_index + 1} sur {len(similar_molecules)}: {similar_id} (Similarité: {similarity:.4f})")
                            
                            # Navigation entre les molécules similaires
                            def next_similar():
                                st.session_state.similar_molecules_index = (st.session_state.similar_molecules_index + 1) % len(similar_molecules)
                            
                            def prev_similar():
                                st.session_state.similar_molecules_index = (st.session_state.similar_molecules_index - 1) % len(similar_molecules)
                            
                            # Boutons de navigation pour les molécules similaires
                            col1, col2 = st.columns(2)
                            with col1:
                                st.button("⬅️ Similaire précédente", on_click=prev_similar, key="prev_similar_button", use_container_width=True)
                            with col2:
                                st.button("Similaire suivante ➡️", on_click=next_similar, key="next_similar_button", use_container_width=True)
                            
                            # Aligner la molécule similaire par rapport à la molécule actuelle
                            aligned_similar_mol = align_mol_to_ref(original_mol, similar_mol)
                            
                            # Afficher la comparaison avec la molécule similaire
                            st.markdown("### Comparaison entre la molécule actuelle et la molécule similaire")
                            
                            comparison_img = generate_difference_highlight_image(
                                original_mol,
                                aligned_similar_mol,
                                current_id,
                                similar_id,
                                (image_width, image_height)
                            )
                            
                            if comparison_img:
                                st.image(comparison_img)
                                st.markdown(f"""
                                **Interprétation:**
                                - **Molécule actuelle** ({current_id}) à gauche - parties uniquement présentes sont en rouge
                                - **Molécule similaire** ({similar_id}) à droite - parties uniquement présentes sont en vert
                                - **Structure commune** en noir
                                """)
                            else:
                                st.warning("Impossible de générer l'image de comparaison avec la molécule similaire.")
                            
                            # Option pour voir toutes les molécules similaires
                            with st.expander("Voir toutes les molécules similaires"):
                                st.write("Liste complète des molécules similaires déjà sélectionnées:")
                                for i, (sim_id, sim_score, _) in enumerate(similar_molecules):
                                    st.write(f"{i+1}. **{sim_id}** - Similarité: {sim_score:.4f}")
                                
                                # Option pour choisir une molécule spécifique dans la liste
                                selected_option = st.selectbox(
                                    "Voir une molécule spécifique:",
                                    [f"{i+1}. {sim_id} (Similarité: {sim:.4f})" for i, (sim_id, sim, _) in enumerate(similar_molecules)],
                                    index=st.session_state.similar_molecules_index,
                                    key="similar_molecule_selector"
                                )
                                
                                # Mettre à jour l'index si une molécule est sélectionnée dans la liste déroulante
                                selected_index = int(selected_option.split(".")[0]) - 1
                                if selected_index != st.session_state.similar_molecules_index:
                                    st.session_state.similar_molecules_index = selected_index
                                    st.rerun()
                        
                        # Si aucune molécule similaire n'est trouvée
                        elif len(st.session_state.selected_molecules) > 0 and current_id not in st.session_state.selected_molecules:
                            st.info("Aucune molécule similaire n'a été trouvée parmi les molécules déjà sélectionnées (seuil de similarité: 0.5).")
                    else:
                        st.error(f"Impossible de traiter la molécule: {current_id}")
                else:
                    st.error("Impossible de traiter la molécule I0.")
    except Exception as e:
        st.error(f"Une erreur s'est produite lors du traitement du fichier: {str(e)}")
else:
    st.info("Veuillez charger un fichier CSV contenant vos molécules.")
    st.markdown("""
    ### Format attendu du fichier
    
    Le fichier CSV doit contenir au minimum les colonnes suivantes:
    - `SMILES`: Notation SMILES des molécules
    - `ID`: Identifiant unique de chaque molécule
    
    La molécule de référence I0 doit avoir l'ID `Molport-001-492-296`.
    """)
    
    # Afficher un exemple d'utilisation
    st.markdown("""
    ### Comment utiliser cet outil
    
    1. **Chargez votre fichier CSV** contenant les molécules à comparer
    2. **Naviguez** entre les molécules avec les boutons de navigation
    3. **Sélectionnez** les molécules intéressantes
    
    #### Nouvelles fonctionnalités de navigation
    
    Vous pouvez maintenant accéder directement à n'importe quelle molécule:
    
    1. **Navigation directe**: Utilisez les boutons "Première" et "Dernière" pour aller au début ou à la fin
    2. **Aller à l'index**: Spécifiez un numéro d'index et cliquez sur "Aller"
    3. **Recherche par ID**: Recherchez une molécule par son identifiant
    
    #### Fonctionnalité de comparaison avec molécules similaires
    
    Pour chaque nouvelle molécule, l'outil affiche automatiquement:
    1. Une comparaison avec I0 (en haut)
    2. Une comparaison avec les molécules similaires déjà sélectionnées (en bas)
    
    Vous pouvez naviguer entre les molécules similaires déjà sélectionnées grâce aux boutons "Similaire précédente" et "Similaire suivante".
    
    Cette fonctionnalité vous permet d'identifier rapidement si la molécule actuelle ressemble à des molécules que vous avez déjà sélectionnées.
    """)
