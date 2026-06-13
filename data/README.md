# Data

This repository includes only lightweight dataset metadata.

Included:

```text
data/splits/rawch/
data/splits/rawstrict/
```

Not included:

```text
raw audio files
WavLM .npy feature files
old balanced datasets
large archives
```

## Expected Local Feature Folder

To train or run inference, place WavLM embeddings here:

```text
data/features/WavLM_embeddings_unified/
```

Example:

```text
data/features/WavLM_embeddings_unified/hi/fake/audio_1285.npy
```

The `.gitignore` file prevents these large feature files from being committed.

## About The CSVs

The split CSVs contain:

```text
feature_path,label,language,audio_path
```

For model training and inference, the important columns are:

```text
feature_path
label
language
```

The `audio_path` column is provenance metadata from the original machine and may not point to valid paths after cloning. Use it as reference only unless you rebuild the dataset locally.

