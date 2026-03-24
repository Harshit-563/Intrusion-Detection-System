from pathlib import Path
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import TensorDataset

# ==============================
# Config
# ==============================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

TRAIN_FILE = DATA_DIR / "KDDTrain+.txt"
TEST_FILE = DATA_DIR / "KDDTest+.txt"

INPUT_DIM = 41
CLASS_NAMES = ["normal", "dos", "probe", "r2l", "u2r"]
NUM_CLASSES = len(CLASS_NAMES)
NUM_CLIENTS = 3

# ==============================
# Attack Mapping
# ==============================
ATTACK_GROUPS = {
    "normal": {"normal"},
    "dos": {
        "back", "land", "neptune", "pod", "smurf", "teardrop",
        "mailbomb", "apache2", "processtable", "udpstorm",
    },
    "probe": {"satan", "ipsweep", "nmap", "portsweep", "mscan", "saint"},
    "r2l": {
        "guess_passwd", "ftp_write", "imap", "phf", "multihop",
        "warezmaster", "warezclient", "spy", "xlock", "xsnoop",
        "snmpguess", "snmpgetattack", "httptunnel", "sendmail",
        "named", "worm",
    },
    "u2r": {
        "buffer_overflow", "loadmodule", "rootkit",
        "perl", "sqlattack", "xterm", "ps",
    },
}

ATTACK_TO_CATEGORY = {
    attack: category
    for category, attacks in ATTACK_GROUPS.items()
    for attack in attacks
}

# ==============================
# Model
# ==============================
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(INPUT_DIM, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, NUM_CLASSES),
        )

    def forward(self, x):
        return self.model(x)


# ==============================
# Data Loading
# ==============================
def load_dataframe(path: Path) -> pd.DataFrame:
    """Load NSL-KDD dataset and map attack labels."""
    columns = [f"f{i}" for i in range(INPUT_DIM)] + ["label", "difficulty"]
    df = pd.read_csv(path, names=columns)

    # Drop unused column
    df.drop(columns=["difficulty"], inplace=True)

    # Map attack types → categories
    original_labels = df["label"].copy()
    df["label"] = df["label"].map(ATTACK_TO_CATEGORY)

    # Check for unmapped labels
    if df["label"].isna().any():
        missing = sorted(original_labels[df["label"].isna()].unique())
        raise ValueError(f"Unmapped labels in {path.name}: {missing}")

    return df


# ==============================
# Preprocessing
# ==============================
def encode_features(train_df, test_df):
    """Encode categorical features."""
    categorical_cols = (
        train_df.drop(columns=["label"])
        .select_dtypes(include=["object", "string"])
        .columns
    )

    encoders = {}
    for col in categorical_cols:
        encoder = LabelEncoder()
        encoder.fit(pd.concat([train_df[col], test_df[col]]))

        train_df[col] = encoder.transform(train_df[col])
        test_df[col] = encoder.transform(test_df[col])
        encoders[col] = encoder

    return train_df, test_df, encoders


def encode_labels(train_df, test_df):
    """Encode target labels."""
    label_encoder = LabelEncoder()
    label_encoder.fit(CLASS_NAMES)

    train_df["label"] = label_encoder.transform(train_df["label"])
    test_df["label"] = label_encoder.transform(test_df["label"])

    return train_df, test_df, label_encoder


def scale_features(train_df, test_df):
    """Normalize feature values."""
    scaler = StandardScaler()

    X_train = scaler.fit_transform(train_df.drop(columns=["label"]).values)
    X_test = scaler.transform(test_df.drop(columns=["label"]).values)

    y_train = train_df["label"].values
    y_test = test_df["label"].values

    return X_train, X_test, y_train, y_test, scaler


# ==============================
# Dataset Preparation
# ==============================
def prepare_datasets():
    """Full pipeline: load → encode → scale → tensor."""
    train_df = load_dataframe(TRAIN_FILE)
    test_df = load_dataframe(TEST_FILE)

    train_df, test_df, feature_encoders = encode_features(train_df, test_df)
    train_df, test_df, label_encoder = encode_labels(train_df, test_df)

    X_train, X_test, y_train, y_test, scaler = scale_features(train_df, test_df)

    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )

    test_dataset = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long),
    )

    return train_dataset, test_dataset, feature_encoders, label_encoder, scaler


# ==============================
# Federated Split
# ==============================
def split_train_dataset(train_dataset, client_id, num_clients=NUM_CLIENTS):
    """Split dataset among clients (IID)."""
    if not (0 <= client_id < num_clients):
        raise ValueError(f"client_id must be between 0 and {num_clients - 1}")

    features, labels = train_dataset.tensors

    # Shuffle
    perm = torch.randperm(len(features))
    features, labels = features[perm], labels[perm]

    # Split
    total = len(features)
    start = client_id * total // num_clients
    end = (client_id + 1) * total // num_clients

    return TensorDataset(features[start:end], labels[start:end])