#!/bin/bash

path=/home/lium/tmario/opensmile-3.0.2-linux-x86_64/opensmile-3.0.2-linux-x86_64
cmd=${path}/bin/SMILExtract
wavpath=audio
for f in ${wavpath}/*.wav;do
	bname=$(basename $f .wav) 
	echo $bname
	$cmd -C ${path}/config/egemaps/v02/eGeMAPSv02.conf -I $f -O features/${bname}.csv
done

