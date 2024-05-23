#!/bin/bash

# Function to change file extension and add directory number postfix
rename_files() {
    local dir=$1
    local postfix=$2

    for file in "$dir"/*; do
        if [ -f "$file" ]; then
            filename=$(basename -- "$file")
            extension="${filename##*.}"
            name="${filename%.*}"
            newname="${name}${postfix}.png"
            mv "$file" "${dir}/${newname}"
        fi
    done
}

# Process each directory
for dir in */; do
    # Remove trailing slash from directory name
    dir=${dir%/}

    # Call the rename function with the directory and its number
    rename_files "$dir" "$dir"
done
