for directory in ../src/*/; do
	if [ "$directory" != "../src/common_headers/" ]; then	
		cp makefile "$directory";
	fi	
done
