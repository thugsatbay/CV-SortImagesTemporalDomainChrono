jupyter nbconvert --to script $1
file_name="$(cut -d'.' -f1 <<<$1)"
python $file_name".py"
