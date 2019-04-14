reset
load "gnuplot-palettes-20190125/moreland.pal"
#load "gnuplot-palettes-20190125/ylorrd.pal"
#load "gnuplot-palettes-20190125/rdylbu.pal"
load "configs/grid.cfg"
round(x, n) = round(x*10**n)*10.0**(-n)

set datafile separator ","
data = "../../data/computeTimesLinespoints.csv"
set terminal pngcairo size 1000, 540 enhanced font "FreeSerif,14"
set output "../speedup.png"

set key Left outside enhanced title "Experiment Code †"

set logscale x
set xlabel "Fast One-Ring Smothing Filter Convolutions - Log Scale"
set ylabel "Speedup"
set yrange [0:210]

plot for [i=3:12] data every ::33::40  using 2:i with linespoints title "M.".columnhead(i) ls ceil(5-i*4/10) pt (i==8 ? 5 : (i==10 ? 7 : 1)), \
	 for [i=3:12] data every ::41::48 using 2:i with linespoints title "T.".columnhead(i) ls ceil(4+i*4/10) pt (i==8 ? 5 : (i==10 ? 7 : 1))

