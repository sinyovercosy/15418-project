#!/bin/bash
EVENTS=task-clock:u,cache-references:u,cache-misses:u

while getopts "e:p:f:" opt; do
    case "$opt" in
        (e) e="$OPTARG" ;;
        (p) p="$OPTARG" ;;
        (f) f="$OPTARG" ;;
    esac
done

k=$(echo $e | cut -d'-' -f 2)
if [[ "$k" == "omp" ]]; then exe="$k/$e -p $p"; else exe="mpirun -np $p $k/$e"; fi
if (("$p" > 24)); then ppn=24; else ppn=$p; fi
file=${f%.*}
script="${e}-${file##*/}-p${p}.sh"
output="${e}-${file##*/}-p${p}.o"
errout="${e}-${file##*/}-p${p}.e"
echo "
#PBS -l walltime=0:30:00
#PBS -l nodes=1:ppn=24
#PBS -o $output
#PBS -e $errout
cd \$PBS_O_WORKDIR;
perf stat -e $EVENTS $exe $f
" > $script
qsub $script && rm $script
