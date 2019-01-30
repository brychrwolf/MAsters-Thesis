reset
load "gnuplot-palettes-20190125/moreland.pal"
#load "gnuplot-palettes-20190125/ylorrd.pal"
#load "gnuplot-palettes-20190125/rdylbu.pal"
load "configs/grid.cfg"
round(x, n) = round(x*10**n)*10.0**(-n)

set datafile separator ","
data = "../../data/computeTimesLinespoints.csv"
set terminal pngcairo size 1000, 680 #enhanced font "Times New Roman, 10"
set output "../computeTimesLinespoints.png"

set title font ",16" offset 8 \
"Compute Times of Applying the One-Ring Filter for Selected Numbers of Iterations\n \
onto Acquired and Synthetic 3D Meshes of Varying Sizes"

set key outside

set logscale xy
set ylabel "Compute Time (seconds) - Log Scale"
set xlabel "One-Ring Filter Iterations - Log Scale"

plot for [i=3:12] data every ::1::8  using 2:i with linespoints title "TG.".columnhead(i) ls ceil(9-(i-1)*8/10), \
	 for [i=3:12] data every ::9::16 using 2:i with linespoints title "MG.".columnhead(i) ls ceil(9-(i-1)*8/10)

