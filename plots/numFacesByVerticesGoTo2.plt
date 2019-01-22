reset
set datafile separator ","
data = "../experiments/computeTimes.csv"
set terminal pngcairo size 640, 640 #enhanced font "Times New Roman, 10"
set output "numFacesByVerticesGoTo2.png"



set title "Ratio of Faces to Vertices by Increasing Vertex Count"
set logscale x
set xlabel "Vertices in Mesh\nLog Scale"
set ylabel "Faces / Vertices in Mesh"
set yrange [0.8:2.2]

data = "../data/meshSizes.csv"
set label at 397210,1.98737695425594 point pointtype 7 pointsize 1.5
set label noenhanced "Unisiegel" at 397210+50000,1.98737695425594-.03

set label at 56215,1.98009428088588 point pointtype 7 pointsize 1.5
set label noenhanced "ILATO_1A" at 56215-20000,1.98009428088588-.04

set key center right noenhanced

plot 2 title "F/V=2" dt 2 lc rgb "black",\
	 for [i=2:12:3] data using i:i+2 with linespoints title columnhead(i)

