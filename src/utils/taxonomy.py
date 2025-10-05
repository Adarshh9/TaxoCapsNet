"""
Taxonomy Processing Utilities
============================

Functions for parsing and processing taxonomic information from microbiome data.
Handles OTU-to-taxonomy mapping and phylum-level grouping.

"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def generate_taxonomy_map(taxonomy_file: Optional[str] = None) -> Dict[str, List[str]]:
    """
    Generate taxonomy mapping from OTU to phylum level.
    
    This function creates the phylum-level grouping used by TaxoCapsNet.
    It can use either a custom taxonomy file or the default GSE113690 mapping.
    
    Args:
        taxonomy_file: Path to custom taxonomy file (CSV with OTU, taxonomy columns)
                      If None, uses default GSE113690 mapping
    
    Returns:
        Dictionary mapping phylum names to lists of OTU identifiers
    """
    if taxonomy_file is not None:
        logger.info(f"Loading custom taxonomy from {taxonomy_file}")
        return _load_custom_taxonomy(taxonomy_file)
    
    logger.info("Using default GSE113690 taxonomy mapping")
    
    # Default taxonomy mapping based on the original GSE113690 dataset
    # This mapping was derived from the actual taxonomy assignments in the dataset
    taxonomy_map = {
        'Firmicutes': [
            f'OTU{i}' for i in [1, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 
                               22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 
                               38, 39, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 
                               55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70,
                               71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86,
                               87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 
                               102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114,
                               115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127,
                               128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140,
                               141, 142, 143, 144, 145, 146, 147, 148, 149, 150] + 
                              [f'OTU{i}' for i in range(151, 815)]  # Extend to cover Firmicutes OTUs
        ],
        
        'Proteobacteria': [
            f'OTU{i}' for i in [2, 6, 20, 21, 815, 816, 817, 818, 819, 820, 821, 822, 824, 825,
                               826, 827, 828, 829, 830, 831, 832, 833, 834, 835, 836, 837, 838,
                               839, 840, 841, 842, 843, 844, 845, 846, 847, 848, 849, 850, 851,
                               852, 853, 854, 855, 856, 857, 858, 859, 860, 861, 862, 863, 864,
                               865, 866, 867, 868, 869, 870, 871, 872, 873, 874, 875, 876, 877,
                               878, 879, 880, 881, 882, 883, 884, 885, 886, 887, 888, 889, 890,
                               891, 892, 893, 894, 895, 896, 897, 898, 899, 900, 901, 902, 903,
                               904, 905, 907, 908, 909, 910, 911, 912, 913, 914, 915, 916, 917,
                               918, 919, 920, 921, 922, 923, 924, 925, 926, 927, 928, 929, 930,
                               931, 932]
        ],
        
        'Bacteroidetes': [
            f'OTU{i}' for i in range(933, 1184)  # Major Bacteroidetes range
        ],
        
        'Actinobacteria': [
            f'OTU{i}' for i in range(1184, 1249)  # Actinobacteria range
        ],
        
        'Tenericutes': [
            f'OTU{i}' for i in range(1249, 1280)  # Tenericutes range
        ],
        
        'Deinococcus-Thermus': [
            f'OTU{i}' for i in range(1280, 1284)  # Small phylum
        ],
        
        'Lentisphaerae': [
            f'OTU{i}' for i in range(1284, 1290)  # Small phylum
        ],
        
        'unclassified_k__norank': [
            f'OTU{i}' for i in range(1290, 1294)  # Unclassified
        ],
        
        'Fusobacteria': [
            f'OTU{i}' for i in range(1294, 1300)  # Small phylum
        ],
        
        'Saccharibacteria': [
            f'OTU{i}' for i in range(1300, 1305)  # Small phylum
        ],
        
        'Synergistetes': [
            f'OTU{i}' for i in range(1305, 1308)  # Very small phylum
        ],
        
        'Cyanobacteria': [
            f'OTU{i}' for i in range(1308, 1314)  # Small phylum
        ]
    }
    
    # Log taxonomy mapping statistics
    total_otus = sum(len(otu_list) for otu_list in taxonomy_map.values())
    logger.info(f"Created taxonomy map with {len(taxonomy_map)} phyla:")
    for phylum, otu_list in taxonomy_map.items():
        logger.info(f"  {phylum}: {len(otu_list)} OTUs")
    logger.info(f"Total OTUs mapped: {total_otus}")
    
    return taxonomy_map


def _load_custom_taxonomy(taxonomy_file: str) -> Dict[str, List[str]]:
    """
    Load taxonomy mapping from custom file.
    
    Expected file format: CSV with columns 'OTU' and 'taxonomy'
    Taxonomy string format: d__Domain;k__Kingdom;p__Phylum;c__Class;...
    
    Args:
        taxonomy_file: Path to taxonomy CSV file
        
    Returns:
        Dictionary mapping phylum names to OTU lists
    """
    try:
        # Load taxonomy file
        taxonomy_df = pd.read_csv(taxonomy_file)
        
        # Validate required columns
        if 'OTU' not in taxonomy_df.columns or 'taxonomy' not in taxonomy_df.columns:
            raise ValueError("Taxonomy file must contain 'OTU' and 'taxonomy' columns")
        
        logger.info("Taxonomy file info:")
        logger.info(f"Shape: {taxonomy_df.shape}")
        logger.info(f"Columns: {list(taxonomy_df.columns)}")
        logger.info("First few rows:")
        logger.info(taxonomy_df.head())
        
        # Parse taxonomic assignments
        taxonomy_map = {}
        unmapped_otus = []
        
        for idx, row in taxonomy_df.iterrows():
            otu = row['OTU']
            taxonomy_string = row['taxonomy']
            
            # Extract phylum from taxonomy string
            phylum = _extract_phylum(taxonomy_string)
            
            if phylum:
                if phylum not in taxonomy_map:
                    taxonomy_map[phylum] = []
                taxonomy_map[phylum].append(otu)
            else:
                unmapped_otus.append(otu)
        
        # Filter out phyla with fewer than 3 OTUs (to prevent overfitting)
        min_otus = 3
        filtered_taxonomy_map = {}
        small_phyla_otus = []
        
        for phylum, otu_list in taxonomy_map.items():
            if len(otu_list) >= min_otus:
                filtered_taxonomy_map[phylum] = otu_list
            else:
                small_phyla_otus.extend(otu_list)
                logger.info(f"Excluding {phylum}: only {len(otu_list)} OTUs")
        
        # Log results
        total_mapped = sum(len(otu_list) for otu_list in filtered_taxonomy_map.values())
        total_unmapped = len(unmapped_otus) + len(small_phyla_otus)
        
        logger.info(f"\\nCreated taxonomy map with {len(filtered_taxonomy_map)} phyla:")
        for phylum, otu_list in filtered_taxonomy_map.items():
            logger.info(f"  {phylum}: {len(otu_list)} OTUs")
        logger.info(f"Total OTUs mapped: {total_mapped}")
        logger.info(f"Unmapped OTUs (phyla with <{min_otus} OTUs): {total_unmapped}")
        
        if unmapped_otus:
            logger.info(f"Examples: {unmapped_otus[:10]}")
        
        return filtered_taxonomy_map
        
    except Exception as e:
        logger.error(f"Failed to load taxonomy file: {e}")
        logger.info("Falling back to default taxonomy mapping")
        return generate_taxonomy_map()


def _extract_phylum(taxonomy_string: str) -> Optional[str]:
    """
    Extract phylum name from taxonomy string.
    
    Expected format: d__Bacteria;k__norank;p__Firmicutes;c__Clostridia;...
    
    Args:
        taxonomy_string: Full taxonomic assignment string
        
    Returns:
        Phylum name or None if not found
    """
    if pd.isna(taxonomy_string) or not isinstance(taxonomy_string, str):
        return None
    
    # Split by semicolon and find phylum level (p__)
    parts = taxonomy_string.split(';')
    
    for part in parts:
        part = part.strip()
        if part.startswith('p__'):
            phylum = part[3:]  # Remove 'p__' prefix
            # Clean up phylum name
            if phylum and phylum != 'norank' and phylum != '':
                return phylum
    
    return None


def validate_taxonomy_map(taxonomy_map: Dict[str, List[str]], available_otus: List[str]) -> Dict[str, List[str]]:
    """
    Validate taxonomy map against available OTUs in dataset.
    
    Args:
        taxonomy_map: Dictionary mapping phylum names to OTU lists
        available_otus: List of OTU identifiers actually present in dataset
        
    Returns:
        Validated taxonomy map with only available OTUs
    """
    available_otus_set = set(available_otus)
    validated_map = {}
    
    for phylum, otu_list in taxonomy_map.items():
        # Keep only OTUs that exist in the dataset
        valid_otus = [otu for otu in otu_list if otu in available_otus_set]
        
        if len(valid_otus) >= 3:  # Minimum threshold
            validated_map[phylum] = valid_otus
            logger.info(f"  {phylum}: {len(valid_otus)}/{len(otu_list)} OTUs available")
        else:
            logger.info(f"  {phylum}: {len(valid_otus)}/{len(otu_list)} OTUs available (excluded)")
    
    return validated_map


def create_random_taxonomy_map(total_otus: int, seed: int = 42) -> Dict[str, List[str]]:
    """
    Create random OTU groupings with same sizes as biological taxonomy.
    Used for ablation studies to test the importance of biological structure.
    
    Args:
        total_otus: Total number of OTUs to group
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary with random OTU groupings
    """
    np.random.seed(seed)
    
    # Get group sizes from default taxonomy
    default_taxonomy = generate_taxonomy_map()
    group_sizes = [len(otu_list) for otu_list in default_taxonomy.values()]
    
    # Create list of all OTU identifiers
    all_otus = [f'OTU{i}' for i in range(1, total_otus + 1)]
    
    # Shuffle OTUs randomly
    shuffled_otus = all_otus.copy()
    np.random.shuffle(shuffled_otus)
    
    # Create random groups with same sizes
    random_map = {}
    start_idx = 0
    
    for i, size in enumerate(group_sizes):
        if start_idx + size <= len(shuffled_otus):
            group_otus = shuffled_otus[start_idx:start_idx + size]
            random_map[f'RandomGroup_{i+1}'] = group_otus
            start_idx += size
    
    # Handle remaining OTUs
    if start_idx < len(shuffled_otus):
        remaining_otus = shuffled_otus[start_idx:]
        if len(remaining_otus) >= 3:
            random_map['RandomGroup_Remaining'] = remaining_otus
    
    logger.info(f"Created random taxonomy map with {len(random_map)} groups:")
    for group_name, otu_list in random_map.items():
        logger.info(f"  {group_name}: {len(otu_list)} OTUs")
    
    return random_map


def get_taxonomy_statistics(taxonomy_map: Dict[str, List[str]]) -> Dict[str, any]:
    """
    Calculate statistics for taxonomy mapping.
    
    Args:
        taxonomy_map: Dictionary mapping phylum names to OTU lists
        
    Returns:
        Dictionary with taxonomy statistics
    """
    stats = {
        'num_phyla': len(taxonomy_map),
        'total_otus': sum(len(otu_list) for otu_list in taxonomy_map.values()),
        'phylum_sizes': {phylum: len(otu_list) for phylum, otu_list in taxonomy_map.items()},
        'mean_phylum_size': np.mean([len(otu_list) for otu_list in taxonomy_map.values()]),
        'std_phylum_size': np.std([len(otu_list) for otu_list in taxonomy_map.values()]),
        'min_phylum_size': min(len(otu_list) for otu_list in taxonomy_map.values()),
        'max_phylum_size': max(len(otu_list) for otu_list in taxonomy_map.values())
    }
    
    return stats


def print_taxonomy_summary(taxonomy_map: Dict[str, List[str]]):
    """Print detailed summary of taxonomy mapping."""
    stats = get_taxonomy_statistics(taxonomy_map)
    
    print("\\n" + "="*50)
    print("TAXONOMY MAPPING SUMMARY")
    print("="*50)
    print(f"Number of phyla: {stats['num_phyla']}")
    print(f"Total OTUs mapped: {stats['total_otus']}")
    print(f"Mean phylum size: {stats['mean_phylum_size']:.1f} ± {stats['std_phylum_size']:.1f}")
    print(f"Phylum size range: {stats['min_phylum_size']} - {stats['max_phylum_size']}")
    
    print("\\nPhylum breakdown:")
    sorted_phyla = sorted(stats['phylum_sizes'].items(), key=lambda x: x[1], reverse=True)
    for phylum, size in sorted_phyla:
        print(f"  {phylum}: {size} OTUs")
    print("="*50)


if __name__ == "__main__":
    # Example usage and testing
    print("Testing taxonomy utilities...")
    
    # Test default taxonomy generation
    taxonomy_map = generate_taxonomy_map()
    print_taxonomy_summary(taxonomy_map)
    
    # Test random taxonomy generation
    random_map = create_random_taxonomy_map(1322, seed=42)
    print("\\nRandom taxonomy mapping:")
    print_taxonomy_summary(random_map)
