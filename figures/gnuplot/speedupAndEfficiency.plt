reset
load "gnuplot-palettes-20190125/moreland.pal"
#load "gnuplot-palettes-20190125/ylorrd.pal"
#load "gnuplot-palettes-20190125/rdylbu.pal"
load "configs/grid.cfg"
round(x, n) = round(x*10**n)*10.0**(-n)

set datafile separator ","
data = "../../data/computeTimesLinespoints.csv"
set terminal pngcairo size 1000, 680 enhanced font "FreeSerif,14"
set output "../speedupAndEfficiency.png"

set key Left outside #enhanced title "Experiment Code †"

set logscale xy
set xlabel "Mesh Sizes (Point Count) - Log Scale"
set ylabel "Speedup - Log Scale"

plot for [i=3:12] data every ::33::40  using 2:i with linespoints title "M.".columnhead(i) ls ceil(9-(i-1)*8/10), \
	 for [i=3:12] data every ::41::48 using 2:i with linespoints title "T.".columnhead(i) ls ceil(9-(i-1)*8/10)

