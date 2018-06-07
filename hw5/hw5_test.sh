#!/bin/bash

if [ "$3" = "public" ]
then
	python3 hw5_test.py $1 $2 rp5_1_semi_best.h5
else
	python3 hw5_test.py $1 $2 rp5_1_semi_best.h5
fi