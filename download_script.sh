#!/bin/bash

FILE="download_files.txt"
DEST="Downloads"
if [ ! -d "$DEST" ]; then
    echo "Creating directory $DEST..."
    mkdir -p "$DEST"
fi

while read -r line || [ -n "$line" ]; do
    echo "Downloading: $line"
    curl -L --output-dir "$DEST" -O "$line"
done < "$FILE"

echo "All downloads complete!"
