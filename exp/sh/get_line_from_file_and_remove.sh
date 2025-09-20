#!/usr/bin/env bash

line=$(head -n 1 $1)
if [ "$(uname)" == "Darwin" ]; then
	sed -i '' '1d' $1
else
	sed -i '1d' $1
fi
echo $line
