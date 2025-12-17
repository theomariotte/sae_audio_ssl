#!/bin/bash

SPLIT=$1 # split on which to extract features
LIST=$2

path=/lium/home/tmario/opensmile-3.0.2-linux-x86_64
cmd=${path}/bin/SMILExtract
wavpath=${DATA_ROOT}/VocalSet/FULL/${SPLIT}
feature_path=${DATA_ROOT}/VocalSet/features/${SPLIT}

# Create features directory if it doesn't exist
mkdir -p ${feature_path}

# Read wav file list from test_list.txt
while read -r rel_path; do
    # Skip empty lines or comments
    [[ -z "$rel_path" || "$rel_path" =~ ^# ]] && continue

    f="${wavpath}/${rel_path}"

    if [[ ! -f "$f" ]]; then
        echo "Warning: File not found -> $f"
        continue
    fi

    # Generate unique feature filename
    feature_name=$(echo "$rel_path" | sed 's/\//_/g' | sed 's/\.wav$//')

    echo "Processing: $f -> ${feature_path}/${feature_name}.csv"

    # Extract features
    $cmd -C ${path}/config/egemaps/v02/eGeMAPSv02.conf -I "$f" -O "${feature_path}/${feature_name}.csv"

done < ${DATA_ROOT}/VocalSet/${LIST}


# #!/bin/bash

# SPLIT=$1 # split on which to extract features
# LIST=$2
# path=/lium/home/tmario/opensmile-3.0.2-linux-x86_64
# cmd=${path}/bin/SMILExtract
# feature_path=${DATA_ROOT}/VocalSet/features/v2/${SPLIT}

# # Create features directory if it doesn't exist
# mkdir -p ${feature_path}

# # Read wav file list from test_list.txt
# while read -r f; do
#     # Skip empty lines or comments
#     [[ -z "$f" || "$f" =~ ^# ]] && continue

#     if [[ ! -f "$f" ]]; then
#         echo "Warning: File not found -> $f"
#         continue
#     fi

#     # Generate unique feature filename from absolute path
#     feature_name=$(echo "$f" | sed 's/\//_/g' | sed 's/\.wav$//')

#     echo "Processing: $f -> ${feature_path}/${feature_name}.csv"

#     # Extract features
#     $cmd -C ${path}/config/egemaps/v02/eGeMAPSv02.conf -I "$f" -O "${feature_path}/${feature_name}.csv"

# done < ${DATA_ROOT}/VocalSet/${LIST}
