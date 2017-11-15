#!/bin/bash
#run command "unzip.sh <file_name>" to decompress 7z x files. For now file needs to be in same directory as this script  

brew install p7zip

FILE="$1"

7z x "$FILE" 

echo "unzipped file!"
