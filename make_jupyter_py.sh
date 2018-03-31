for i in *.ipynb;do
	jupyter nbconvert --to script $i
	file_name="$(cut -d'.' -f1 <<<$i)"
	echo $file_name
done
