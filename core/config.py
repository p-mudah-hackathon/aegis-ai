import os

# DIRECTORY PATHS 
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = PROJECT_ROOT
CSV_PATH = os.path.join(PROJECT_ROOT, 'aegis_qris_transactions.csv')
GRAPH_META_PATH = os.path.join(PROJECT_ROOT, 'aegis_graph_meta.json')
PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'processed')
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, 'checkpoints')
EXPORT_DIR = os.path.join(PROJECT_ROOT, 'exported')
EXPLANATIONS_DIR = os.path.join(PROJECT_ROOT, 'explanations')


# DEFAULT HYPERPARAMETERS
DEFAULTS = {
    'epochs': 200,
    'lr': 5e-4,
    'weight_decay': 5e-4,
    'd_model': 64,
    'n_heads': 4,
    'n_layers': 3,
    'dropout': 0.2,
    'patience': 40,
    'focal_alpha': 0.75,
    'focal_gamma': 2.0,
}


# TEMPORAL SPLIT DATES
TRAIN_END = '2026-02-10'
VAL_END = '2026-02-20'
# Test = everything after VAL_END
