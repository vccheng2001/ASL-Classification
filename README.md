# American Sign Language Classification

1. [Background](#background)
2. [File Hierarchy](#file-hierarchy)
3. [Dataset](#dataset)
4. [Feature Extraction](#feature-extraction)
5. [Classification](#classification)

## Background


## File Hierarchy
 
```bash
# dataset
data/
```

## Dataset

We use the [Hand Sign Images](https://www.kaggle.com/datasets/ash2703/handsignimages) dataset from Kaggle. This dataset contains 27,455 gray-scale images of size 28*28 pixels whose value range between 0-255. Each case represents a label (0-25) as a one-to-one map for each alphabetic letter A-Z (and no cases for 9=J or 25=Z because of gesture motions).


## Feature Extraction
1. KMeans Clustering (kmeans.py)
2. t-SNE (tsne.py)
3. PCA (pca.py)
4. Standard CV features (preprocess.py)

## Classification

0. MLP (src/train_mlp.py)
1. CNN (src/train_cnn.py)
2. Random Forest (src/train_rf.py)
