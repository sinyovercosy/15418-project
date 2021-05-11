#!/bin/bash

while getopts "e:f:" opt; do
    case "$opt" in
        (e) e="$OPTARG" ;;
        (f) f="$OPTARG" ;;
    esac
done

file=${f%.*}
combined_out="${e}-${file##*/}.o"
combined_err="${e}-${file##*/}.e"
for p in 1 2 4 8 16 24 32; do
    output="${e}-${file##*/}-p${p}.o"
    errout="${e}-${file##*/}-p${p}.e"
    echo p=$p >> $combined_out
    cat $output >> $combined_out
    echo p=$p >> $combined_err
    cat $errout >> $combined_err
    rm $output
    rm $errout
done;
