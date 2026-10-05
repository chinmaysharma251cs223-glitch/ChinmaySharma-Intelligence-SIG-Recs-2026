# ============================================================
# FINALE — MULTIMODAL SHOPEE PRODUCT MATCHING
# ============================================================
#
# Pipeline:
#   Product titles -> TF-IDF -> cosine similarity ----\
#                                                     \
#                                                      -> OOF reranker
#                                                     /       |
#   Product images -> ResNet50 -> 2048-D -> cosine --/        |
#                                                             v
#                                                   final score + threshold
#
# The script:
#   1. Loads and cleans train.csv
#   2. Creates a group-aware train/validation listing split
#   3. Creates one shared pair dataset
#   4. Builds the TF-IDF text branch
#   5. Builds/caches frozen ResNet50 image embeddings
#   6. Computes text/image pair scores
#   7. Runs 5-fold OOF scoring
#   8. Trains a logistic-regression multimodal reranker on OOF scores
#   9. Selects thresholds using OOF predictions
#  10. Evaluates text-only, image-only and multimodal systems
#  11. Produces ablation/results tables
#  12. Saves false positives / false negatives
#  13. Produces a reduced visualization set
#
# Expected input:
#   train.csv
#   train_images/
#
# Required columns:
#   posting_id, image, title, label_group
#
# ============================================================

import os
import re
import json
import random
import warnings
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
from PIL import Image, ImageFile
from tqdm.auto import tqdm

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet50, ResNet50_Weights


# ============================================================
# 1. CONFIGURATION
# ============================================================

DATA_PATH = "/kaggle/input/competitions/shopee-product-matching/train.csv"
IMAGE_DIR = "/kaggle/input/competitions/shopee-product-matching/train_images"

IMAGE_COLUMN = "image"
IMAGE_ID_COLUMN = "posting_id"
PRODUCT_ID_COLUMN = "label_group"
TEXT_COLUMN = "title"

RESULTS_DIR = Path("results")
CACHE_DIR = RESULTS_DIR / "cache"
METRICS_DIR = RESULTS_DIR / "metrics"
PLOTS_DIR = RESULTS_DIR / "plots"
ERROR_DIR = RESULTS_DIR / "errors"
RETRIEVAL_DIR = RESULTS_DIR / "retrieval"

for directory in [
    RESULTS_DIR,
    CACHE_DIR,
    METRICS_DIR,
    PLOTS_DIR,
    ERROR_DIR,
    RETRIEVAL_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

# Validation split
VALIDATION_FRACTION = 0.25

# Pair construction
MAX_POSITIVE_PAIRS_PER_GROUP = 10
N_NEGATIVE_PAIRS = 10000

# OOF
N_SPLITS = 5

# TF-IDF
TFIDF_NGRAM_RANGE = (1, 2)
TFIDF_MIN_DF = 2
TFIDF_SUBLINEAR_TF = True
TFIDF_MAX_FEATURES = None

# Image extraction
IMAGE_BATCH_SIZE = 32
NUM_WORKERS = 0
IMAGE_SIZE = 224

# Retrieval
TOP_K = 5
N_RETRIEVAL_EXAMPLES = 5

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", DEVICE)


# ============================================================
# 2. REPRODUCIBILITY
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Deterministic behavior where possible.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(RANDOM_STATE)


# ============================================================
# 3. DATA LOADING AND CLEANING
# ============================================================

def normalize_title(text):
    """
    Lightweight product-title normalization.

    We deliberately preserve product-specific tokens such as:
    model numbers, capacities, dimensions and variant identifiers.
    """
    if pd.isna(text):
        return ""

    text = str(text).lower()
    text = re.sub(r"\s+", " ", text)
    text = text.strip()

    return text


def load_and_clean_data():
    df = pd.read_csv(DATA_PATH)

    required_columns = [
        IMAGE_ID_COLUMN,
        IMAGE_COLUMN,
        TEXT_COLUMN,
        PRODUCT_ID_COLUMN,
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    print("Original dataframe shape:", df.shape)

    # Remove rows missing essential information.
    df = df.dropna(
        subset=[
            IMAGE_ID_COLUMN,
            IMAGE_COLUMN,
            TEXT_COLUMN,
            PRODUCT_ID_COLUMN,
        ]
    ).copy()

    # Remove duplicate listing IDs.
    df = df.drop_duplicates(
        subset=[IMAGE_ID_COLUMN]
    ).copy()

    df["clean_title"] = df[TEXT_COLUMN].apply(normalize_title)

    # Remove empty titles after normalization.
    df = df[df["clean_title"].str.len() > 0].copy()

    # Resolve image paths.
    df["image_path"] = df[IMAGE_COLUMN].apply(
        lambda x: str(Path(IMAGE_DIR) / str(x))
    )

    # Check image availability.
    exists_mask = df["image_path"].apply(os.path.exists)

    missing_images = int((~exists_mask).sum())

    if missing_images > 0:
        print(
            f"Warning: {missing_images} rows have missing image files "
            "and will be removed."
        )

    df = df[exists_mask].copy()

    df = df.reset_index(drop=True)

    print("Clean dataframe shape:", df.shape)

    return df


df = load_and_clean_data()


# ============================================================
# 4. GROUP-AWARE TRAIN / VALIDATION SPLIT
# ============================================================

def create_group_aware_split(
    df,
    validation_fraction=0.25,
    random_state=42,
):
    """
    Create train_df and validation_df while splitting inside
    label_group.

    Rules:
      - group size 1: remains in train
      - group size 2: 1 validation, 1 train
      - group size >= 3: reserve approximately validation_fraction,
        but try to keep at least 2 validation examples.
    """
    rng = np.random.default_rng(random_state)

    train_indices = []
    validation_indices = []

    for label, group in df.groupby(PRODUCT_ID_COLUMN):
        indices = group.index.to_numpy()
        rng.shuffle(indices)

        n = len(indices)

        if n == 1:
            n_val = 0

        elif n == 2:
            n_val = 1

        else:
            n_val = max(
                2,
                int(round(n * validation_fraction))
            )

            # Leave at least one listing in train.
            n_val = min(n_val, n - 1)

        validation_indices.extend(indices[:n_val])
        train_indices.extend(indices[n_val:])

    train_df = (
        df.loc[train_indices]
        .sample(frac=1, random_state=random_state)
        .reset_index(drop=True)
    )

    validation_df = (
        df.loc[validation_indices]
        .sample(frac=1, random_state=random_state)
        .reset_index(drop=True)
    )

    return train_df, validation_df


train_df, validation_df = create_group_aware_split(
    df,
    validation_fraction=VALIDATION_FRACTION,
    random_state=RANDOM_STATE,
)

print("\nTrain listings:", len(train_df))
print("Validation listings:", len(validation_df))

train_df.to_csv(
    RESULTS_DIR / "train_listings.csv",
    index=False
)

validation_df.to_csv(
    RESULTS_DIR / "validation_listings.csv",
    index=False
)


# ============================================================
# 5. SHARED PAIR CONSTRUCTION
# ============================================================

def construct_positive_pairs(
    listing_df,
    max_pairs_per_group=10,
    random_state=42,
):
    rng = np.random.default_rng(random_state)

    positive_pairs = []

    for label, group in listing_df.groupby(PRODUCT_ID_COLUMN):
        rows = group[
            [
                IMAGE_ID_COLUMN,
                PRODUCT_ID_COLUMN,
            ]
        ].to_dict("records")

        all_pairs = list(combinations(rows, 2))

        if len(all_pairs) > max_pairs_per_group:
            selected_indices = rng.choice(
                len(all_pairs),
                size=max_pairs_per_group,
                replace=False,
            )
            all_pairs = [
                all_pairs[i]
                for i in selected_indices
            ]

        for a, b in all_pairs:
            positive_pairs.append({
                "id_1": a[IMAGE_ID_COLUMN],
                "id_2": b[IMAGE_ID_COLUMN],
                "label": 1,
            })

    return positive_pairs


def construct_negative_pairs(
    listing_df,
    n_negative_pairs=10000,
    random_state=42,
):
    rng = np.random.default_rng(random_state)

    records = listing_df[
        [
            IMAGE_ID_COLUMN,
            PRODUCT_ID_COLUMN,
        ]
    ].to_dict("records")

    if len(records) < 2:
        raise ValueError("Not enough listings to create negative pairs.")

    negative_pairs = []
    seen = set()

    max_attempts = max(
        1000,
        n_negative_pairs * 20
    )

    attempts = 0

    while (
        len(negative_pairs) < n_negative_pairs
        and attempts < max_attempts
    ):
        attempts += 1

        i, j = rng.integers(
            0,
            len(records),
            size=2
        )

        if i == j:
            continue

        a = records[i]
        b = records[j]

        if a[PRODUCT_ID_COLUMN] == b[PRODUCT_ID_COLUMN]:
            continue

        id_pair = tuple(sorted([
            a[IMAGE_ID_COLUMN],
            b[IMAGE_ID_COLUMN],
        ]))

        if id_pair in seen:
            continue

        seen.add(id_pair)

        negative_pairs.append({
            "id_1": a[IMAGE_ID_COLUMN],
            "id_2": b[IMAGE_ID_COLUMN],
            "label": 0,
        })

    if len(negative_pairs) < n_negative_pairs:
        print(
            f"Warning: requested {n_negative_pairs} negatives but "
            f"generated {len(negative_pairs)}."
        )

    return negative_pairs


def build_pair_dataframe(
    listing_df,
    max_positive_pairs=10,
    n_negative_pairs=10000,
    random_state=42,
):
    positives = construct_positive_pairs(
        listing_df,
        max_pairs_per_group=max_positive_pairs,
        random_state=random_state,
    )

    negatives = construct_negative_pairs(
        listing_df,
        n_negative_pairs=n_negative_pairs,
        random_state=random_state + 1,
    )

    pairs = positives + negatives

    pairs_df = pd.DataFrame(pairs)

    pairs_df = pairs_df.sample(
        frac=1,
        random_state=random_state,
    ).reset_index(drop=True)

    return pairs_df


# IMPORTANT:
# The final multimodal experiment uses ONE shared pair list.
#
# We construct pairs from the validation listings because these
# are the listings on which final matching performance is evaluated.
#
# For OOF reranker training, we separately create training pairs.
train_pairs = build_pair_dataframe(
    train_df,
    max_positive_pairs=MAX_POSITIVE_PAIRS_PER_GROUP,
    n_negative_pairs=N_NEGATIVE_PAIRS,
    random_state=RANDOM_STATE,
)

validation_pairs = build_pair_dataframe(
    validation_df,
    max_positive_pairs=MAX_POSITIVE_PAIRS_PER_GROUP,
    n_negative_pairs=N_NEGATIVE_PAIRS,
    random_state=RANDOM_STATE + 10,
)

print("\nTraining pairs:")
print(train_pairs["label"].value_counts())

print("\nValidation pairs:")
print(validation_pairs["label"].value_counts())

train_pairs.to_csv(
    RESULTS_DIR / "train_pairs.csv",
    index=False
)

validation_pairs.to_csv(
    RESULTS_DIR / "validation_pairs.csv",
    index=False
)


# ============================================================
# 6. DATAFRAME LOOKUP
# ============================================================

all_df = pd.concat(
    [train_df, validation_df],
    ignore_index=True
)

all_df = all_df.drop_duplicates(
    subset=[IMAGE_ID_COLUMN]
)

id_to_row = all_df.set_index(
    IMAGE_ID_COLUMN
).to_dict("index")


def add_pair_metadata(pairs_df):
    pairs = pairs_df.copy()

    pairs["title_1"] = pairs["id_1"].map(
        lambda x: id_to_row[x][TEXT_COLUMN]
    )

    pairs["title_2"] = pairs["id_2"].map(
        lambda x: id_to_row[x][TEXT_COLUMN]
    )

    pairs["clean_title_1"] = pairs["id_1"].map(
        lambda x: id_to_row[x]["clean_title"]
    )

    pairs["clean_title_2"] = pairs["id_2"].map(
        lambda x: id_to_row[x]["clean_title"]
    )

    pairs["label_group_1"] = pairs["id_1"].map(
        lambda x: id_to_row[x][PRODUCT_ID_COLUMN]
    )

    pairs["label_group_2"] = pairs["id_2"].map(
        lambda x: id_to_row[x][PRODUCT_ID_COLUMN]
    )

    return pairs


train_pairs = add_pair_metadata(train_pairs)
validation_pairs = add_pair_metadata(validation_pairs)


# ============================================================
# 7. TEXT MODEL — TF-IDF
# ============================================================

def fit_tfidf_vectorizer(texts):
    vectorizer = TfidfVectorizer(
        ngram_range=TFIDF_NGRAM_RANGE,
        min_df=TFIDF_MIN_DF,
        sublinear_tf=TFIDF_SUBLINEAR_TF,
        max_features=TFIDF_MAX_FEATURES,
    )

    matrix = vectorizer.fit_transform(texts)

    return vectorizer, matrix


def transform_tfidf(vectorizer, texts):
    return vectorizer.transform(texts)


def compute_text_pair_scores(
    pairs_df,
    vectorizer,
    title_to_vector=None,
):
    """
    Efficiently computes TF-IDF cosine similarity for pairs.

    If title_to_vector is provided, it should map clean title -> row
    index in the fitted TF-IDF matrix.
    """
    if title_to_vector is None:
        raise ValueError("title_to_vector is required.")

    scores = []

    for _, row in pairs_df.iterrows():
        idx1 = title_to_vector[row["clean_title_1"]]
        idx2 = title_to_vector[row["clean_title_2"]]

        v1 = vectorizer_matrix[idx1]
        v2 = vectorizer_matrix[idx2]

        score = float(v1.multiply(v2).sum())

        scores.append(score)

    return np.asarray(scores, dtype=np.float32)


# ============================================================
# 8. TEXT EMBEDDING CACHE FOR ALL LISTINGS
# ============================================================

def fit_global_text_model(df):
    """
    This is used for baseline/validation scoring after the training
    pipeline has been defined.

    For strict OOF training of the reranker, fit_tfidf_vectorizer()
    is called independently inside each fold.
    """
    vectorizer = TfidfVectorizer(
        ngram_range=TFIDF_NGRAM_RANGE,
        min_df=TFIDF_MIN_DF,
        sublinear_tf=TFIDF_SUBLINEAR_TF,
        max_features=TFIDF_MAX_FEATURES,
    )

    matrix = vectorizer.fit_transform(
        df["clean_title"].tolist()
    )

    return vectorizer, matrix


global_text_vectorizer, global_text_matrix = fit_global_text_model(
    train_df
)

global_title_to_vector = {}

for idx, title in enumerate(train_df["clean_title"]):
    global_title_to_vector[title] = idx


def score_pairs_with_fitted_text_model(
    pairs_df,
    vectorizer,
    matrix,
    df_for_matrix,
):
    """
    Compute pairwise cosine similarity from an already-fitted TF-IDF
    matrix.

    The dataframe must contain all listing IDs used by pairs.
    """
    id_to_matrix_index = {
        listing_id: idx
        for idx, listing_id in enumerate(
            df_for_matrix[IMAGE_ID_COLUMN]
        )
    }

    scores = []

    for _, row in pairs_df.iterrows():
        id1 = row["id_1"]
        id2 = row["id_2"]

        idx1 = id_to_matrix_index[id1]
        idx2 = id_to_matrix_index[id2]

        v1 = matrix[idx1]
        v2 = matrix[idx2]

        score = float(v1.multiply(v2).sum())
        scores.append(score)

    return np.asarray(scores, dtype=np.float32)


# ============================================================
# 9. RESNET50 IMAGE MODEL
# ============================================================

print("\nLoading pretrained ResNet50...")

weights = ResNet50_Weights.DEFAULT

image_preprocess = weights.transforms()

image_model = resnet50(weights=weights)

# Remove classification head.
image_model.fc = nn.Identity()

image_model = image_model.to(DEVICE)
image_model.eval()

for parameter in image_model.parameters():
    parameter.requires_grad = False

EMBEDDING_DIM = 2048

print("Image embedding dimension:", EMBEDDING_DIM)


# ============================================================
# 10. IMAGE DATASET / DATALOADER
# ============================================================

class ProductImageDataset(Dataset):

    def __init__(
        self,
        dataframe,
        transform,
    ):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image_id = row[IMAGE_ID_COLUMN]
        image_path = row["image_path"]

        try:
            image = Image.open(
                image_path
            ).convert("RGB")

            image = self.transform(image)

        except Exception as exc:
            raise RuntimeError(
                f"Could not load image: {image_path}"
            ) from exc

        return image_id, image


def extract_image_embeddings(
    dataframe,
    model,
    transform,
    batch_size=32,
    num_workers=0,
):
    dataset = ProductImageDataset(
        dataframe,
        transform,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    embedding_ids = []
    embedding_chunks = []

    with torch.no_grad():

        for image_ids, images in tqdm(
            loader,
            desc="Extracting ResNet50 embeddings",
        ):
            images = images.to(DEVICE)

            embeddings = model(images)

            # L2 normalize embeddings.
            embeddings = torch.nn.functional.normalize(
                embeddings,
                p=2,
                dim=1,
            )

            embedding_chunks.append(
                embeddings.cpu().numpy()
            )

            embedding_ids.extend(image_ids)

    embeddings = np.concatenate(
        embedding_chunks,
        axis=0,
    )

    embedding_ids = np.asarray(
        embedding_ids
    )

    return embedding_ids, embeddings


# ============================================================
# 11. CACHE / LOAD IMAGE EMBEDDINGS
# ============================================================

EMBEDDING_FILE = CACHE_DIR / "image_embeddings.npy"
EMBEDDING_IDS_FILE = CACHE_DIR / "image_embedding_ids.csv"

if (
    EMBEDDING_FILE.exists()
    and EMBEDDING_IDS_FILE.exists()
):

    print("\nLoading cached image embeddings...")

    image_embeddings = np.load(
        EMBEDDING_FILE
    )

    embedding_ids = pd.read_csv(
        EMBEDDING_IDS_FILE
    )[IMAGE_ID_COLUMN].astype(str).to_numpy()

else:

    print("\nExtracting image embeddings...")

    # Keep identifiers consistently string-based.
    all_df[IMAGE_ID_COLUMN] = (
        all_df[IMAGE_ID_COLUMN].astype(str)
    )

    embedding_ids, image_embeddings = extract_image_embeddings(
        all_df,
        image_model,
        image_preprocess,
        batch_size=IMAGE_BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )

    np.save(
        EMBEDDING_FILE,
        image_embeddings,
    )

    pd.DataFrame({
        IMAGE_ID_COLUMN: embedding_ids
    }).to_csv(
        EMBEDDING_IDS_FILE,
        index=False
    )

print(
    "Cached image embedding matrix:",
    image_embeddings.shape
)


# Make sure pair IDs are strings.
train_pairs["id_1"] = train_pairs["id_1"].astype(str)
train_pairs["id_2"] = train_pairs["id_2"].astype(str)

validation_pairs["id_1"] = (
    validation_pairs["id_1"].astype(str)
)
validation_pairs["id_2"] = (
    validation_pairs["id_2"].astype(str)
)

embedding_index = {
    str(image_id): idx
    for idx, image_id in enumerate(embedding_ids)
}


def compute_image_pair_scores(
    pairs_df,
    embeddings,
    embedding_index,
):
    scores = []

    for _, row in pairs_df.iterrows():

        idx1 = embedding_index[str(row["id_1"])]
        idx2 = embedding_index[str(row["id_2"])]

        score = float(
            np.dot(
                embeddings[idx1],
                embeddings[idx2],
            )
        )

        scores.append(score)

    return np.asarray(
        scores,
        dtype=np.float32
    )


# ============================================================
# 12. GLOBAL BASELINE SCORES
# ============================================================

# For the baseline comparison, the text representation is fitted
# on train_df and applied to validation listings.

validation_ids = set(
    validation_df[IMAGE_ID_COLUMN].astype(str)
)

train_ids = set(
    train_df[IMAGE_ID_COLUMN].astype(str)
)

# Create a combined dataframe for validation scoring while fitting
# the TF-IDF vocabulary on training titles only.
train_titles = train_df["clean_title"].tolist()

baseline_text_vectorizer = TfidfVectorizer(
    ngram_range=TFIDF_NGRAM_RANGE,
    min_df=TFIDF_MIN_DF,
    sublinear_tf=TFIDF_SUBLINEAR_TF,
    max_features=TFIDF_MAX_FEATURES,
)

baseline_text_vectorizer.fit(train_titles)

# Transform all validation titles.
validation_text_matrix = (
    baseline_text_vectorizer.transform(
        validation_df["clean_title"].tolist()
    )
)

validation_id_to_text_index = {
    str(listing_id): idx
    for idx, listing_id in enumerate(
        validation_df[IMAGE_ID_COLUMN]
    )
}


def compute_text_scores_from_matrix(
    pairs_df,
    matrix,
    id_to_index,
):
    scores = []

    for _, row in pairs_df.iterrows():

        idx1 = id_to_index[str(row["id_1"])]
        idx2 = id_to_index[str(row["id_2"])]

        v1 = matrix[idx1]
        v2 = matrix[idx2]

        score = float(
            v1.multiply(v2).sum()
        )

        scores.append(score)

    return np.asarray(
        scores,
        dtype=np.float32
    )


validation_text_scores = compute_text_scores_from_matrix(
    validation_pairs,
    validation_text_matrix,
    validation_id_to_text_index,
)

validation_image_scores = compute_image_pair_scores(
    validation_pairs,
    image_embeddings,
    embedding_index,
)

validation_pairs["text_score"] = validation_text_scores
validation_pairs["image_score"] = validation_image_scores


# ============================================================
# 13. OOF BASE SCORE GENERATION
# ============================================================

def generate_oof_base_scores(
    train_df,
    train_pairs,
    n_splits=5,
    random_state=42,
):
    """
    Generate strict OOF text scores.

    ResNet50 is frozen, so image scores can be obtained directly
    from the cached embeddings.

    TF-IDF is fitted separately inside each fold.
    """
    pairs = train_pairs.reset_index(drop=True).copy()

    y = pairs["label"].astype(int).to_numpy()

    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    oof_text = np.zeros(
        len(pairs),
        dtype=np.float32
    )

    oof_image = compute_image_pair_scores(
        pairs,
        image_embeddings,
        embedding_index,
    )

    fold_records = []

    for fold, (fit_idx, holdout_idx) in enumerate(
        skf.split(
            pairs,
            y,
        ),
        start=1,
    ):

        fold_fit_pairs = pairs.iloc[
            fit_idx
        ]

        fold_holdout_pairs = pairs.iloc[
            holdout_idx
        ]

        # ----------------------------------------------------
        # Fit TF-IDF only on listings participating in the
        # fold's training pairs.
        # ----------------------------------------------------
        fold_train_ids = pd.unique(
            pd.concat([
                fold_fit_pairs["id_1"],
                fold_fit_pairs["id_2"],
            ])
        )

        fold_train_listing_df = train_df[
            train_df[IMAGE_ID_COLUMN].astype(str).isin(
                set(map(str, fold_train_ids))
            )
        ].copy()

        fold_vectorizer = TfidfVectorizer(
            ngram_range=TFIDF_NGRAM_RANGE,
            min_df=TFIDF_MIN_DF,
            sublinear_tf=TFIDF_SUBLINEAR_TF,
            max_features=TFIDF_MAX_FEATURES,
        )

        fold_matrix = fold_vectorizer.fit_transform(
            fold_train_listing_df["clean_title"]
        )

        fold_id_to_text_index = {
            str(listing_id): idx
            for idx, listing_id in enumerate(
                fold_train_listing_df[IMAGE_ID_COLUMN]
            )
        }

        # ----------------------------------------------------
        # Transform all pair listings needed by this fold.
        # ----------------------------------------------------
        required_ids = pd.unique(
            pd.concat([
                fold_fit_pairs["id_1"],
                fold_fit_pairs["id_2"],
                fold_holdout_pairs["id_1"],
                fold_holdout_pairs["id_2"],
            ])
        )

        required_listing_df = train_df[
            train_df[IMAGE_ID_COLUMN].astype(str).isin(
                set(map(str, required_ids))
            )
        ].copy()

        required_text_matrix = fold_vectorizer.transform(
            required_listing_df["clean_title"]
        )

        required_id_to_text_index = {
            str(listing_id): idx
            for idx, listing_id in enumerate(
                required_listing_df[IMAGE_ID_COLUMN]
            )
        }

        # ----------------------------------------------------
        # Training fold score generation
        # ----------------------------------------------------
        # These scores are not used as OOF predictions.
        # They are generated only so the fold's text model is
        # fully defined and can be inspected if required.
        #
        # ----------------------------------------------------
        # Held-out fold scores
        # ----------------------------------------------------
        holdout_scores = []

        for _, row in fold_holdout_pairs.iterrows():

            idx1 = required_id_to_text_index[
                str(row["id_1"])
            ]

            idx2 = required_id_to_text_index[
                str(row["id_2"])
            ]

            v1 = required_text_matrix[idx1]
            v2 = required_text_matrix[idx2]

            score = float(
                v1.multiply(v2).sum()
            )

            holdout_scores.append(score)

        oof_text[
            holdout_idx
        ] = np.asarray(
            holdout_scores,
            dtype=np.float32
        )

        fold_records.append({
            "fold": fold,
            "n_train_pairs": len(fit_idx),
            "n_holdout_pairs": len(holdout_idx),
        })

        print(
            f"OOF fold {fold}/{n_splits}: "
            f"train={len(fit_idx)}, "
            f"holdout={len(holdout_idx)}"
        )

    result = pairs.copy()

    result["fold"] = -1

    # Reconstruct fold assignment.
    for fold, (_, holdout_idx) in enumerate(
        skf.split(
            pairs,
            y,
        ),
        start=1,
    ):
        result.loc[
            holdout_idx,
            "fold"
        ] = fold

    result["text_score_oof"] = oof_text
    result["image_score_oof"] = oof_image

    fold_df = pd.DataFrame(fold_records)

    return result, fold_df


oof_pairs, cv_structure = generate_oof_base_scores(
    train_df=train_df,
    train_pairs=train_pairs,
    n_splits=N_SPLITS,
    random_state=RANDOM_STATE,
)

oof_pairs.to_csv(
    CACHE_DIR / "pair_scores_oof.csv",
    index=False
)

cv_structure.to_csv(
    METRICS_DIR / "cv_structure.csv",
    index=False
)


# ============================================================
# 14. THRESHOLD SEARCH
# ============================================================

def find_best_threshold(
    y_true,
    scores,
    metric="f1",
    n_thresholds=501,
):
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores)

    thresholds = np.linspace(
        scores.min(),
        scores.max(),
        n_thresholds,
    )

    best_threshold = None
    best_value = -np.inf

    threshold_records = []

    for threshold in thresholds:

        predictions = (
            scores >= threshold
        ).astype(int)

        precision = precision_score(
            y_true,
            predictions,
            zero_division=0,
        )

        recall = recall_score(
            y_true,
            predictions,
            zero_division=0,
        )

        f1 = f1_score(
            y_true,
            predictions,
            zero_division=0,
        )

        threshold_records.append({
            "threshold": threshold,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })

        value = f1 if metric == "f1" else f1

        if value > best_value:
            best_value = value
            best_threshold = threshold

    threshold_df = pd.DataFrame(
        threshold_records
    )

    return best_threshold, threshold_df


# ============================================================
# 15. OOF THRESHOLDS FOR TEXT AND IMAGE
# ============================================================

text_oof_threshold, text_threshold_df = (
    find_best_threshold(
        oof_pairs["label"].values,
        oof_pairs["text_score_oof"].values,
    )
)

image_oof_threshold, image_threshold_df = (
    find_best_threshold(
        oof_pairs["label"].values,
        oof_pairs["image_score_oof"].values,
    )
)

print("\nOOF text threshold:", text_oof_threshold)
print("OOF image threshold:", image_oof_threshold)


# ============================================================
# 16. MULTIMODAL RERANKER
# ============================================================

X_oof = oof_pairs[
    [
        "text_score_oof",
        "image_score_oof",
    ]
].to_numpy()

y_oof = oof_pairs["label"].astype(int).to_numpy()

reranker = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=RANDOM_STATE,
)

reranker.fit(
    X_oof,
    y_oof,
)

oof_pairs["final_score_oof"] = reranker.predict_proba(
    X_oof
)[:, 1]

print("\nReranker coefficients:")
print(
    "text coefficient =",
    reranker.coef_[0][0]
)
print(
    "image coefficient =",
    reranker.coef_[0][1]
)
print(
    "intercept =",
    reranker.intercept_[0]
)


# ============================================================
# 17. FINAL MULTIMODAL THRESHOLD
# ============================================================

final_oof_threshold, final_threshold_df = (
    find_best_threshold(
        y_oof,
        oof_pairs["final_score_oof"].values,
    )
)

print(
    "\nFinal multimodal OOF threshold:",
    final_oof_threshold
)

text_threshold_df.to_csv(
    METRICS_DIR / "text_threshold_curve.csv",
    index=False
)

image_threshold_df.to_csv(
    METRICS_DIR / "image_threshold_curve.csv",
    index=False
)

final_threshold_df.to_csv(
    METRICS_DIR / "final_threshold_curve.csv",
    index=False
)


# ============================================================
# 18. VALIDATION FINAL SCORES
# ============================================================

validation_pairs["final_score"] = reranker.predict_proba(
    validation_pairs[
        [
            "text_score",
            "image_score",
        ]
    ].to_numpy()
)[:, 1]

validation_pairs["text_prediction"] = (
    validation_pairs["text_score"]
    >= text_oof_threshold
).astype(int)

validation_pairs["image_prediction"] = (
    validation_pairs["image_score"]
    >= image_oof_threshold
).astype(int)

validation_pairs["prediction"] = (
    validation_pairs["final_score"]
    >= final_oof_threshold
).astype(int)


# ============================================================
# 19. METRIC FUNCTION
# ============================================================

def calculate_metrics(
    y_true,
    scores,
    threshold,
):
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores)

    predictions = (
        scores >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    accuracy = accuracy_score(
        y_true,
        predictions,
    )

    balanced_accuracy = balanced_accuracy_score(
        y_true,
        predictions,
    )

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0,
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    fpr = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else 0.0
    )

    fnr = (
        fn / (fn + tp)
        if (fn + tp) > 0
        else 0.0
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0,
    )

    mcc = matthews_corrcoef(
        y_true,
        predictions,
    )

    try:
        roc_auc = roc_auc_score(
            y_true,
            scores,
        )
    except ValueError:
        roc_auc = np.nan

    try:
        pr_auc = average_precision_score(
            y_true,
            scores,
        )
    except ValueError:
        pr_auc = np.nan

    return {
        "Accuracy": accuracy,
        "Balanced Accuracy": balanced_accuracy,
        "Precision": precision,
        "Recall": recall,
        "Specificity": specificity,
        "FPR": fpr,
        "FNR": fnr,
        "F1": f1,
        "MCC": mcc,
        "ROC-AUC": roc_auc,
        "PR-AUC": pr_auc,
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "Threshold": threshold,
        "N": len(y_true),
    }


# ============================================================
# 20. VALIDATION METRICS
# ============================================================

y_validation = validation_pairs[
    "label"
].astype(int).to_numpy()

text_metrics = calculate_metrics(
    y_validation,
    validation_pairs["text_score"].to_numpy(),
    text_oof_threshold,
)

image_metrics = calculate_metrics(
    y_validation,
    validation_pairs["image_score"].to_numpy(),
    image_oof_threshold,
)

multimodal_metrics = calculate_metrics(
    y_validation,
    validation_pairs["final_score"].to_numpy(),
    final_oof_threshold,
)

metrics_df = pd.DataFrame(
    [
        {
            "Model": "Text only",
            **text_metrics,
        },
        {
            "Model": "Image only",
            **image_metrics,
        },
        {
            "Model": "Text + Image Reranker",
            **multimodal_metrics,
        },
    ]
)

print("\n================ VALIDATION RESULTS ================")
print(
    metrics_df[
        [
            "Model",
            "Accuracy",
            "Balanced Accuracy",
            "Precision",
            "Recall",
            "Specificity",
            "F1",
            "MCC",
            "ROC-AUC",
            "PR-AUC",
        ]
    ].to_string(index=False)
)

metrics_df.to_csv(
    METRICS_DIR / "model_comparison.csv",
    index=False
)

validation_pairs.to_csv(
    METRICS_DIR / "validation_pair_scores.csv",
    index=False
)


# ============================================================
# 21. OOF METRICS
# ============================================================

oof_text_metrics = calculate_metrics(
    y_oof,
    oof_pairs["text_score_oof"].values,
    text_oof_threshold,
)

oof_image_metrics = calculate_metrics(
    y_oof,
    oof_pairs["image_score_oof"].values,
    image_oof_threshold,
)

oof_multimodal_metrics = calculate_metrics(
    y_oof,
    oof_pairs["final_score_oof"].values,
    final_oof_threshold,
)

oof_metrics_df = pd.DataFrame([
    {
        "Model": "Text only OOF",
        **oof_text_metrics,
    },
    {
        "Model": "Image only OOF",
        **oof_image_metrics,
    },
    {
        "Model": "Multimodal OOF",
        **oof_multimodal_metrics,
    },
])

oof_metrics_df.to_csv(
    METRICS_DIR / "oof_metrics.csv",
    index=False
)


# ============================================================
# 22. ABLATION TABLE
# ============================================================

ablation_df = pd.DataFrame([
    {
        "Configuration": "A",
        "Text": True,
        "Image": False,
        "Reranker": False,
        "F1": text_metrics["F1"],
        "MCC": text_metrics["MCC"],
        "ROC-AUC": text_metrics["ROC-AUC"],
        "PR-AUC": text_metrics["PR-AUC"],
    },
    {
        "Configuration": "B",
        "Text": False,
        "Image": True,
        "Reranker": False,
        "F1": image_metrics["F1"],
        "MCC": image_metrics["MCC"],
        "ROC-AUC": image_metrics["ROC-AUC"],
        "PR-AUC": image_metrics["PR-AUC"],
    },
    {
        "Configuration": "C",
        "Text": True,
        "Image": True,
        "Reranker": True,
        "F1": multimodal_metrics["F1"],
        "MCC": multimodal_metrics["MCC"],
        "ROC-AUC": multimodal_metrics["ROC-AUC"],
        "PR-AUC": multimodal_metrics["PR-AUC"],
    },
])

print("\n================ ABLATION ================")
print(ablation_df.to_string(index=False))

ablation_df.to_csv(
    METRICS_DIR / "ablation.csv",
    index=False
)


# ============================================================
# 23. RERANKER DETAILS
# ============================================================

reranker_details = {
    "model": "LogisticRegression",
    "class_weight": "balanced",
    "max_iter": 1000,
    "text_coefficient": float(
        reranker.coef_[0][0]
    ),
    "image_coefficient": float(
        reranker.coef_[0][1]
    ),
    "intercept": float(
        reranker.intercept_[0]
    ),
    "text_threshold": float(
        text_oof_threshold
    ),
    "image_threshold": float(
        image_oof_threshold
    ),
    "final_threshold": float(
        final_oof_threshold
    ),
}

with open(
    METRICS_DIR / "reranker_config.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        reranker_details,
        f,
        indent=4,
    )


# ============================================================
# 24. ERROR ANALYSIS
# ============================================================

def save_error_tables(
    pairs_df,
    error_dir,
):
    errors = pairs_df[
        pairs_df["prediction"]
        != pairs_df["label"]
    ].copy()

    false_positives = errors[
        (errors["prediction"] == 1)
        & (errors["label"] == 0)
    ].copy()

    false_negatives = errors[
        (errors["prediction"] == 0)
        & (errors["label"] == 1)
    ].copy()

    # Useful qualitative categories.
    def infer_error_type(row):

        text_high = row["text_score"] >= text_oof_threshold
        image_high = row["image_score"] >= image_oof_threshold

        if row["label"] == 0 and image_high and not text_high:
            return "visually similar"

        if row["label"] == 0 and text_high and not image_high:
            return "textually similar"

        if row["label"] == 0 and text_high and image_high:
            return "multimodal similarity / variant confusion"

        if row["label"] == 1 and not image_high and text_high:
            return "same product / different photography"

        if row["label"] == 1 and image_high and not text_high:
            return "same product / noisy title"

        if row["label"] == 1 and not text_high and not image_high:
            return "weak evidence / missing information"

        return "conflicting modalities"

    if len(false_positives) > 0:
        false_positives["error_type"] = (
            false_positives.apply(
                infer_error_type,
                axis=1,
            )
        )

    if len(false_negatives) > 0:
        false_negatives["error_type"] = (
            false_negatives.apply(
                infer_error_type,
                axis=1,
            )
        )

    false_positives.to_csv(
        error_dir / "false_positives.csv",
        index=False,
    )

    false_negatives.to_csv(
        error_dir / "false_negatives.csv",
        index=False,
    )

    return false_positives, false_negatives


false_positives, false_negatives = save_error_tables(
    validation_pairs,
    ERROR_DIR,
)

print(
    "\nFalse positives:",
    len(false_positives)
)

print(
    "False negatives:",
    len(false_negatives)
)


# ============================================================
# 25. REDUCED VISUALIZATION SET
# ============================================================

sns.set_theme(style="whitegrid")


# ------------------------------------------------------------
# Plot 1 — Score distributions
# ------------------------------------------------------------

fig, axes = plt.subplots(
    1,
    3,
    figsize=(18, 5),
)

score_specs = [
    ("text_score", "Text similarity"),
    ("image_score", "Image similarity"),
    ("final_score", "Multimodal final score"),
]

for ax, (column, title) in zip(
    axes,
    score_specs,
):
    sns.kdeplot(
        data=validation_pairs[
            validation_pairs["label"] == 0
        ],
        x=column,
        label="Non-match",
        fill=True,
        ax=ax,
    )

    sns.kdeplot(
        data=validation_pairs[
            validation_pairs["label"] == 1
        ],
        x=column,
        label="Match",
        fill=True,
        ax=ax,
    )

    ax.set_title(title)
    ax.legend()

plt.tight_layout()
plt.savefig(
    PLOTS_DIR / "score_distributions.png",
    dpi=200,
    bbox_inches="tight",
)
plt.close()


# ------------------------------------------------------------
# Plot 2 — Threshold metrics
# ------------------------------------------------------------

plt.figure(figsize=(9, 6))

plt.plot(
    final_threshold_df["threshold"],
    final_threshold_df["f1"],
    label="F1",
)

plt.plot(
    final_threshold_df["threshold"],
    final_threshold_df["precision"],
    label="Precision",
)

plt.plot(
    final_threshold_df["threshold"],
    final_threshold_df["recall"],
    label="Recall",
)

plt.axvline(
    final_oof_threshold,
    linestyle="--",
    label="Selected threshold",
)

plt.xlabel("Threshold")
plt.ylabel("Score")
plt.title("Multimodal Threshold Analysis")
plt.legend()

plt.tight_layout()

plt.savefig(
    PLOTS_DIR / "threshold_metrics.png",
    dpi=200,
    bbox_inches="tight",
)

plt.close()


# ------------------------------------------------------------
# Plot 3 — ROC curve
# ------------------------------------------------------------

plt.figure(figsize=(8, 7))

for name, scores in [
    ("Text", validation_pairs["text_score"].values),
    ("Image", validation_pairs["image_score"].values),
    ("Multimodal", validation_pairs["final_score"].values),
]:

    fpr, tpr, _ = roc_curve(
        y_validation,
        scores,
    )

    auc = roc_auc_score(
        y_validation,
        scores,
    )

    plt.plot(
        fpr,
        tpr,
        label=f"{name} (AUC={auc:.4f})",
    )

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()

plt.tight_layout()

plt.savefig(
    PLOTS_DIR / "roc_curve.png",
    dpi=200,
    bbox_inches="tight",
)

plt.close()


# ------------------------------------------------------------
# Plot 4 — Precision-Recall curve
# ------------------------------------------------------------

plt.figure(figsize=(8, 7))

for name, scores in [
    ("Text", validation_pairs["text_score"].values),
    ("Image", validation_pairs["image_score"].values),
    ("Multimodal", validation_pairs["final_score"].values),
]:

    precision, recall, _ = precision_recall_curve(
        y_validation,
        scores,
    )

    ap = average_precision_score(
        y_validation,
        scores,
    )

    plt.plot(
        recall,
        precision,
        label=f"{name} (AP={ap:.4f})",
    )

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curve")
plt.legend()

plt.tight_layout()

plt.savefig(
    PLOTS_DIR / "pr_curve.png",
    dpi=200,
    bbox_inches="tight",
)

plt.close()


# ------------------------------------------------------------
# Plot 5 — Confusion matrix
# ------------------------------------------------------------

final_predictions = (
    validation_pairs["prediction"]
    .astype(int)
    .values
)

cm = confusion_matrix(
    y_validation,
    final_predictions,
    labels=[0, 1],
)

plt.figure(figsize=(7, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    square=True,
    xticklabels=[
        "Non-match",
        "Match",
    ],
    yticklabels=[
        "Non-match",
        "Match",
]

)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Final Multimodal Confusion Matrix")

plt.tight_layout()

plt.savefig(
    PLOTS_DIR / "confusion_matrix.png",
    dpi=200,
    bbox_inches="tight",
)

plt.close()


# ------------------------------------------------------------
# Plot 6 — Modality disagreement
# ------------------------------------------------------------

plt.figure(figsize=(9, 7))

sns.scatterplot(
    data=validation_pairs,
    x="text_score",
    y="image_score",
    hue="label",
    alpha=0.65,
)

plt.axvline(
    text_oof_threshold,
    linestyle="--",
)

plt.axhline(
    image_oof_threshold,
    linestyle="--",
)

plt.xlabel("Text cosine similarity")
plt.ylabel("Image cosine similarity")
plt.title("Text–Image Modality Agreement / Disagreement")

plt.tight_layout()

plt.savefig(
    PLOTS_DIR / "modality_disagreement.png",
    dpi=200,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# 26. TOP-K IMAGE RETRIEVAL
# ============================================================

def retrieve_top_k(
    query_id,
    embedding_ids,
    embeddings,
    k=5,
):
    query_id = str(query_id)

    query_index = embedding_index[
        query_id
    ]

    query_embedding = embeddings[
        query_index
    ]

    similarities = (
        embeddings @ query_embedding
    )

    # Exclude the query itself.
    similarities[query_index] = -np.inf

    top_indices = np.argsort(
        similarities
    )[::-1][:k]

    return [
        {
            "image_id": str(
                embedding_ids[idx]
            ),
            "similarity": float(
                similarities[idx]
            ),
        }
        for idx in top_indices
    ]


retrieval_records = []

# Pick validation queries.
query_ids = validation_df[
    IMAGE_ID_COLUMN
].astype(str).tolist()

query_ids = query_ids[
    :min(
        N_RETRIEVAL_EXAMPLES,
        len(query_ids),
    )
]

for query_id in query_ids:

    results = retrieve_top_k(
        query_id,
        embedding_ids,
        image_embeddings,
        k=TOP_K,
    )

    for rank, result in enumerate(
        results,
        start=1,
    ):
        retrieval_records.append({
            "query_id": query_id,
            "rank": rank,
            "neighbor_id": result["image_id"],
            "similarity": result["similarity"],
        })

retrieval_df = pd.DataFrame(
    retrieval_records
)

retrieval_df.to_csv(
    RETRIEVAL_DIR / "topk_results.csv",
    index=False,
)


# ============================================================
# 27. TOP-K VISUALIZATION
# ============================================================

def show_retrieval_grid(
    query_id,
    retrieval_df,
    all_df,
    save_path,
):
    query_row = all_df[
        all_df[IMAGE_ID_COLUMN].astype(str)
        == str(query_id)
    ].iloc[0]

    neighbors = retrieval_df[
        retrieval_df["query_id"].astype(str)
        == str(query_id)
    ].sort_values("rank")

    n_images = len(neighbors) + 1

    fig, axes = plt.subplots(
        1,
        n_images,
        figsize=(4 * n_images, 4),
    )

    if n_images == 1:
        axes = [axes]

    # Query
    query_image = Image.open(
        query_row["image_path"]
    ).convert("RGB")

    axes[0].imshow(query_image)
    axes[0].set_title("Query")
    axes[0].axis("off")

    # Neighbors
    for ax, (_, neighbor) in zip(
        axes[1:],
        neighbors.iterrows(),
    ):

        neighbor_row = all_df[
            all_df[IMAGE_ID_COLUMN].astype(str)
            == str(neighbor["neighbor_id"])
        ].iloc[0]

        neighbor_image = Image.open(
            neighbor_row["image_path"]
        ).convert("RGB")

        ax.imshow(neighbor_image)

        ax.set_title(
            f"Top-{int(neighbor['rank'])}\n"
            f"sim={neighbor['similarity']:.3f}"
        )

        ax.axis("off")

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close()


for query_id in query_ids:
    safe_name = re.sub(
        r"[^a-zA-Z0-9_-]",
        "_",
        str(query_id),
    )

    show_retrieval_grid(
        query_id,
        retrieval_df,
        all_df,
        RETRIEVAL_DIR / f"{safe_name}_topk.png",
    )


# ============================================================
# 28. UNLABELED TEST-SET INFERENCE / LABEL-GROUP PREDICTION
# ============================================================
#
# Expected test.csv columns:
#   posting_id, image, title
#
# The official test set normally has no label_group column. Therefore
# we do NOT calculate supervised test metrics. Instead we:
#   1. Build TF-IDF representations using the training vocabulary.
#   2. Extract frozen ResNet50 embeddings for test images.
#   3. Retrieve top-K candidates independently with text and image.
#   4. Fuse the two similarities with the trained OOF reranker.
#   5. Keep candidate edges whose multimodal score exceeds the OOF
#      threshold.
#   6. Form connected components of matched listings.
#   7. Assign one predicted label_group to each component.
#
# This produces:
#   results/test_predictions.csv
#   results/test_submission.csv
#
# If test.csv happens to contain a label_group column, it is deliberately
# ignored during prediction so that the inference path remains blind to
# the ground-truth groups.

TEST_DATA_PATH = "/kaggle/input/competitions/shopee-product-matching/test.csv"
TEST_IMAGE_DIR = "/kaggle/input/competitions/shopee-product-matching/test_images"
TEST_TOP_K_TEXT = 20
TEST_TOP_K_IMAGE = 20
TEST_BATCH_SIZE = 256


def load_test_data():
    if not Path(TEST_DATA_PATH).exists():
        print(
            f"\nTest inference skipped: {TEST_DATA_PATH} was not found."
        )
        return None

    test_df = pd.read_csv(TEST_DATA_PATH)

    required_test_columns = [
        IMAGE_ID_COLUMN,
        IMAGE_COLUMN,
        TEXT_COLUMN,
    ]

    missing = [
        col for col in required_test_columns
        if col not in test_df.columns
    ]

    if missing:
        raise ValueError(
            f"test.csv is missing required columns: {missing}"
        )

    test_df = test_df.dropna(
        subset=required_test_columns
    ).copy()

    test_df[IMAGE_ID_COLUMN] = test_df[IMAGE_ID_COLUMN].astype(str)
    test_df["clean_title"] = test_df[TEXT_COLUMN].apply(normalize_title)
    test_df = test_df[
        test_df["clean_title"].str.len() > 0
    ].copy()

    test_df["image_path"] = test_df[IMAGE_COLUMN].apply(
        lambda x: str(Path(TEST_IMAGE_DIR) / str(x))
    )

    exists_mask = test_df["image_path"].apply(os.path.exists)
    missing_images = int((~exists_mask).sum())

    if missing_images:
        print(
            f"Warning: {missing_images} test rows have missing images "
            "and will be removed."
        )

    test_df = test_df[exists_mask].copy()
    test_df = test_df.drop_duplicates(
        subset=[IMAGE_ID_COLUMN]
    ).reset_index(drop=True)

    if len(test_df) == 0:
        raise ValueError("No usable test listings remain after cleaning.")

    print("\nTest listings:", len(test_df))
    return test_df


def load_or_extract_test_embeddings(test_df):
    test_embedding_file = CACHE_DIR / "test_image_embeddings.npy"
    test_embedding_ids_file = CACHE_DIR / "test_image_embedding_ids.csv"

    expected_ids = set(test_df[IMAGE_ID_COLUMN].astype(str))
    use_cache = (
        test_embedding_file.exists()
        and test_embedding_ids_file.exists()
    )

    if use_cache:
        cached_embeddings = np.load(test_embedding_file)
        cached_ids = pd.read_csv(
            test_embedding_ids_file
        )[IMAGE_ID_COLUMN].astype(str).to_numpy()

        cache_valid = (
            cached_embeddings.ndim == 2
            and cached_embeddings.shape[1] == EMBEDDING_DIM
            and len(cached_ids) == len(cached_embeddings)
            and set(cached_ids) == expected_ids
        )

        if cache_valid:
            print("Loading cached test image embeddings...")
            return cached_ids, cached_embeddings

        print("Cached test embeddings are stale/incompatible; recomputing.")

    print("Extracting ResNet50 embeddings for test set...")

    embedding_ids, embeddings = extract_image_embeddings(
        test_df,
        image_model,
        image_preprocess,
        batch_size=IMAGE_BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )

    embedding_ids = np.asarray(embedding_ids).astype(str)

    np.save(test_embedding_file, embeddings)
    pd.DataFrame({
        IMAGE_ID_COLUMN: embedding_ids
    }).to_csv(
        test_embedding_ids_file,
        index=False,
    )

    return embedding_ids, embeddings


def build_candidate_pairs(test_df, test_text_matrix, test_image_embeddings):
    """Generate a manageable candidate edge set from both modalities."""

    n = len(test_df)
    if n < 2:
        return pd.DataFrame(
            columns=["id_1", "id_2", "text_score", "image_score"]
        )

    k_text = min(TEST_TOP_K_TEXT + 1, n)
    k_image = min(TEST_TOP_K_IMAGE + 1, n)

    print(
        f"\nGenerating test candidates: text top-{k_text - 1}, "
        f"image top-{k_image - 1}..."
    )

    # Text candidates. TF-IDF vectors are L2-normalized, so cosine
    # similarity is directly equivalent to their dot product.
    text_nn = NearestNeighbors(
        n_neighbors=k_text,
        metric="cosine",
        algorithm="brute",
        n_jobs=-1,
    )
    text_nn.fit(test_text_matrix)
    text_distances, text_indices = text_nn.kneighbors(
        test_text_matrix,
        return_distance=True,
    )

    # Image candidates. ResNet embeddings were L2-normalized during
    # extraction, but cosine distance is used explicitly here.
    image_nn = NearestNeighbors(
        n_neighbors=k_image,
        metric="cosine",
        algorithm="brute",
        n_jobs=-1,
    )
    image_nn.fit(test_image_embeddings)
    image_distances, image_indices = image_nn.kneighbors(
        test_image_embeddings,
        return_distance=True,
    )

    id_values = test_df[IMAGE_ID_COLUMN].astype(str).to_numpy()

    # Union candidate edges from both modalities. A dictionary avoids
    # duplicate edges while retaining both modality scores.
    candidates = {}

    for row_idx in range(n):
        for rank in range(1, k_text):
            neighbor_idx = int(text_indices[row_idx, rank])
            a = str(id_values[row_idx])
            b = str(id_values[neighbor_idx])
            key = tuple(sorted((a, b)))
            candidates.setdefault(key, {})["text_score"] = float(
                1.0 - text_distances[row_idx, rank]
            )

        for rank in range(1, k_image):
            neighbor_idx = int(image_indices[row_idx, rank])
            a = str(id_values[row_idx])
            b = str(id_values[neighbor_idx])
            key = tuple(sorted((a, b)))
            candidates.setdefault(key, {})["image_score"] = float(
                1.0 - image_distances[row_idx, rank]
            )

    candidate_records = []
    for (id1, id2), values in candidates.items():
        candidate_records.append({
            "id_1": id1,
            "id_2": id2,
            # A missing modality means this pair was not in that
            # modality's top-K retrieval. Zero is a conservative score.
            "text_score": float(values.get("text_score", 0.0)),
            "image_score": float(values.get("image_score", 0.0)),
        })

    candidates_df = pd.DataFrame(candidate_records)

    if len(candidates_df) == 0:
        return pd.DataFrame(
            columns=["id_1", "id_2", "text_score", "image_score"]
        )

    return candidates_df


def union_find_label_groups(test_df, matched_edges):
    """Convert predicted matching edges into connected-component groups."""

    ids = test_df[IMAGE_ID_COLUMN].astype(str).tolist()
    parent = {item: item for item in ids}
    rank = {item: 0 for item in ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        root_a = find(a)
        root_b = find(b)
        if root_a == root_b:
            return

        if rank[root_a] < rank[root_b]:
            root_a, root_b = root_b, root_a

        parent[root_b] = root_a
        if rank[root_a] == rank[root_b]:
            rank[root_a] += 1

    for _, row in matched_edges.iterrows():
        union(str(row["id_1"]), str(row["id_2"]))

    components = {}
    for item in ids:
        root = find(item)
        components.setdefault(root, []).append(item)

    # Use deterministic integer group IDs. Singleton products are also
    # assigned their own group, which is required for a complete output.
    ordered_components = sorted(
        components.values(),
        key=lambda members: min(members),
    )

    group_id_by_listing = {}
    for group_number, members in enumerate(ordered_components):
        for item in members:
            group_id_by_listing[item] = group_number

    output = test_df[[IMAGE_ID_COLUMN]].copy()
    output["predicted_label_group"] = output[IMAGE_ID_COLUMN].astype(str).map(
        group_id_by_listing
    ).astype(int)

    return output, ordered_components


def run_test_inference():
    test_df = load_test_data()
    if test_df is None:
        return None

    # Fit TF-IDF using TRAIN titles only. This keeps the representation
    # consistent with the supervised validation pipeline and prevents
    # using test labels or target information.
    test_vectorizer = TfidfVectorizer(
        ngram_range=TFIDF_NGRAM_RANGE,
        min_df=TFIDF_MIN_DF,
        sublinear_tf=TFIDF_SUBLINEAR_TF,
        max_features=TFIDF_MAX_FEATURES,
    )
    test_vectorizer.fit(train_df["clean_title"].tolist())

    test_text_matrix = test_vectorizer.transform(
        test_df["clean_title"].tolist()
    )

    test_embedding_ids, test_image_embeddings = (
        load_or_extract_test_embeddings(test_df)
    )

    # Keep embedding rows aligned with test_df.
    test_embedding_index = {
        str(item): idx
        for idx, item in enumerate(test_embedding_ids)
    }

    missing_embedding_ids = [
        item for item in test_df[IMAGE_ID_COLUMN].astype(str)
        if item not in test_embedding_index
    ]
    if missing_embedding_ids:
        raise ValueError(
            "Test embedding cache does not contain all test listing IDs."
        )

    ordered_indices = [
        test_embedding_index[str(item)]
        for item in test_df[IMAGE_ID_COLUMN].astype(str)
    ]
    test_image_embeddings = test_image_embeddings[ordered_indices]

    candidates = build_candidate_pairs(
        test_df,
        test_text_matrix,
        test_image_embeddings,
    )

    if len(candidates) == 0:
        # Every listing becomes a singleton group if no candidates exist.
        prediction_df, components = union_find_label_groups(
            test_df,
            candidates,
        )
        print("No candidate pairs were generated; all test listings are singletons.")
    else:
        candidates["final_score"] = reranker.predict_proba(
            candidates[["text_score", "image_score"]].to_numpy()
        )[:, 1]

        candidates["prediction"] = (
            candidates["final_score"] >= final_oof_threshold
        ).astype(int)

        matched_edges = candidates[
            candidates["prediction"] == 1
        ].copy()

        prediction_df, components = union_find_label_groups(
            test_df,
            matched_edges,
        )

        candidates.to_csv(
            RESULTS_DIR / "test_candidate_scores.csv",
            index=False,
        )

    # Main prediction file: one predicted group per test posting.
    prediction_df.to_csv(
        RESULTS_DIR / "test_predictions.csv",
        index=False,
    )

    # Competition-style submission file.
    submission_df = prediction_df.rename(
        columns={"predicted_label_group": PRODUCT_ID_COLUMN}
    )
    submission_df.to_csv(
        RESULTS_DIR / "test_submission.csv",
        index=False,
    )

    group_sizes = (
        prediction_df.groupby("predicted_label_group")
        .size()
        .sort_values(ascending=False)
    )

    print("\n================ TEST PREDICTION RESULTS ================")
    print("Test listings predicted:", len(prediction_df))
    print("Predicted label groups:", prediction_df["predicted_label_group"].nunique())
    print("Predicted matched edges:", int(
        prediction_df["predicted_label_group"].value_counts().gt(1).sum()
    ), "multi-listing groups")
    print("Largest predicted group:", int(group_sizes.iloc[0]) if len(group_sizes) else 0)
    print("\nSaved:")
    print("  results/test_predictions.csv")
    print("  results/test_submission.csv")
    if len(candidates) > 0:
        print("  results/test_candidate_scores.csv")

    return prediction_df


test_prediction_df = run_test_inference()


# ============================================================
# 28. FINAL CONFIGURATION / SUMMARY
# ============================================================

summary = {
    "data_path": DATA_PATH,
    "image_dir": IMAGE_DIR,
    "random_state": RANDOM_STATE,
    "device": str(DEVICE),
    "validation_fraction": VALIDATION_FRACTION,
    "n_train_listings": int(len(train_df)),
    "n_validation_listings": int(len(validation_df)),
    "n_train_pairs": int(len(train_pairs)),
    "n_validation_pairs": int(len(validation_pairs)),
    "n_splits": N_SPLITS,
    "tfidf_ngram_range": list(TFIDF_NGRAM_RANGE),
    "tfidf_min_df": TFIDF_MIN_DF,
    "tfidf_sublinear_tf": TFIDF_SUBLINEAR_TF,
    "image_model": "ResNet50 ImageNet pretrained, frozen",
    "image_embedding_dimension": EMBEDDING_DIM,
    "reranker": "LogisticRegression",
    "text_threshold": float(text_oof_threshold),
    "image_threshold": float(image_oof_threshold),
    "final_threshold": float(final_oof_threshold),
    "reranker_text_coefficient": float(
        reranker.coef_[0][0]
    ),
    "reranker_image_coefficient": float(
        reranker.coef_[0][1]
    ),
}

with open(
    RESULTS_DIR / "run_summary.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        summary,
        f,
        indent=4,
    )


# ============================================================
# 29. FINAL CONSOLE SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINALE MULTIMODAL PRODUCT MATCHING — COMPLETE")
print("=" * 70)

print("\nValidation metrics:")
print(
    metrics_df[
        [
            "Model",
            "Accuracy",
            "Balanced Accuracy",
            "Precision",
            "Recall",
            "Specificity",
            "F1",
            "MCC",
            "ROC-AUC",
            "PR-AUC",
        ]
    ].to_string(index=False)
)

print("\nReranker:")
print(
    "Text coefficient:",
    reranker_details["text_coefficient"]
)
print(
    "Image coefficient:",
    reranker_details["image_coefficient"]
)
print(
    "Final threshold:",
    reranker_details["final_threshold"]
)

print("\nFiles saved under:")
print(RESULTS_DIR.resolve())

print("\nMain outputs:")
print("  results/metrics/model_comparison.csv")
print("  results/metrics/ablation.csv")
print("  results/cache/pair_scores_oof.csv")
print("  results/errors/false_positives.csv")
print("  results/errors/false_negatives.csv")
print("  results/plots/")
print("  results/retrieval/")

print("\nDone.")
