reset
set datafile separator ","
data = "../experiments/computeTimes.csv"
set terminal pngcairo size 640, 640 #enhanced font "Times New Roman, 10"
set output "numFacesByVerticesGoTo2.png"



set title "Ratio of Faces to Vertices by Increasing Vertex Count"
set logscale x
set xlabel "Vertices in Mesh\nLog Scale"
set ylabel "Faces / Vertices in Mesh"

data = "../data/meshSizes.csv"
set label at 397210,1.98737695425594 point pointtype 7 pointsize 1
set label "Unisiegel" at 397210,1.98737695425594-.02

set label at 56215,1.98009428088588 point pointtype 7 pointsize 1
set label "ILATO_1A" at 56215,1.98009428088588-.04

set key center right

#plot data using 2:4 with linespoints title columnhead(2)
plot for [i=2:12:3] data using i:i+2 with linespoints title columnhead(i)


