#!/bin/bash

echo "Running test"
sh runFindOpt.sh > opt.dat
echo "Plotting graph"
gnuplot opt.p
echo "Oppening plot"
evince opt.eps
