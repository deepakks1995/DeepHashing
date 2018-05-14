#!/bin/bash
if [ "$1" != "" ]; then
	echo "GPU entered is"
else
	echo "Please input gpu type"
	exit
fi


GPU=

if [ "$1" = "bio" ];
then
	GPU="biometric@10.8.10.142"
elif [ "$1" = "gpu" ]
then
	GPU="gpuuser@10.8.1.77"
elif [ "$1" = "ter" ] 
then
	GPU="terone@10.8.10.13"
else
	GPU="Nothing"
fi
echo $GPU
scp -r bin/ $GPU:deepak/DeepHashing/ 
echo "Files copied successfully"