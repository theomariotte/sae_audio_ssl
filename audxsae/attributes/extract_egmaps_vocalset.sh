#!/bin/bash

SPLIT=$1 # split on which to extract features

path=/lium/home/tmario/opensmile-3.0.2-linux-x86_64
cmd=${path}/bin/SMILExtract
wavpath=${DATA_ROOT}/VocalSet/FULL/${SPLIT}
feature_path=${DATA_ROOT}/VocalSet/features/${SPLIT}

# Create features directory if it doesn't exist
mkdir -p ${feature_path}

# Find all .wav files in the dataset structure recursively
find ${wavpath} -name "*.wav" -type f | while read -r f; do
    # Get the basename without extension
    bname=$(basename "$f" .wav)
    
    # Get the relative path from wavpath and replace / with _ for feature filename
    # This creates unique filenames that preserve the directory structure
    rel_path=$(realpath --relative-to="$wavpath" "$f")
    feature_name=$(echo "$rel_path" | sed 's/\//_/g' | sed 's/\.wav$//')
    
    echo "Processing: $f -> ${feature_path}/${feature_name}.csv"
    
    # Extract features
    $cmd -C ${path}/config/egemaps/v02/eGeMAPSv02.conf -I "$f" -O "${feature_path}/${feature_name}.csv"
done
