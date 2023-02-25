ulimit -m 1048576

correct_cnt=0

for i in {1..3}
do
	printf "\nworking on case ${i}:\n"
	test_data=sample${i}.in
	correct_file=sample${i}.out
	timeout 60s bash compile.sh
	time timeout 600s taskset -c 1-8 bash run.sh ${test_data} output_file
	diff -qEwB output_file ${correct_file} > /dev/null
	res=$?
	if [ "${res}" -eq "0" ]; then
		correct_cnt=$((correct_cnt+1))
	else
		echo "incorrect on case ${i}"
	fi
done
printf "correct: ${correct_cnt}\n"
