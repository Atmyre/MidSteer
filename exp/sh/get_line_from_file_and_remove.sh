#!/usr/bin/env bash

line=$(head -n 1 $1)
sed -i '' '1d' $1
echo $line
