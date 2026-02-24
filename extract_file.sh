FOLDER="Downloads"
FILES=( "$FOLDER"/* )

for FILE in "${FILES[@]}"; do
    echo "Current File : $FILE"
    gunzip "$FILE"
    echo "unzipping Done : $FILE"
done

echo "$FILES"