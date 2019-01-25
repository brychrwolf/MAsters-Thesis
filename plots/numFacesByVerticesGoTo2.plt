reset
load "gnuplot-palettes-20190125/dark2.pal"
load "configs/grid.cfg"

set datafile separator ","
data = "../data/meshSizes.csv"
set terminal pngcairo size 1000, 680 #enhanced font "Times New Roman, 10"
set output "numFacesByVerticesGoTo2.png"

set title "Ratio of Faces to Vertices by Increasing Vertex Count" font ",24"

set logscale x
set xlabel "Vertices in Mesh - Log Scale"
set ylabel "Faces / Vertices in Mesh"
set yrange [0.8:2.2]

set key center right noenhanced

set label at 397210,1.98737695425594 point pointtype 7 pointsize 1.5
set label noenhanced "Unisiegel" at 397210+50000,1.98737695425594-.03

set label at 56215,1.98009428088588 point pointtype 7 pointsize 1.5
set label noenhanced "ILATO_1A" at 56215-20000,1.98009428088588-.04

plot 2 title "F/V=2" dt 2 lc rgb "black",\
	 for [i=2:12:3] data using i:i+2 with linespoints ls (i/3+3) title columnhead(i)

