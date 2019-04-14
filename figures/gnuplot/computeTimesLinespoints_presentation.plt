reset
load "gnuplot-palettes-20190125/moreland.pal"
#load "gnuplot-palettes-20190125/ylorrd.pal"
#load "gnuplot-palettes-20190125/rdylbu.pal"
load "configs/grid.cfg"
round(x, n) = round(x*10**n)*10.0**(-n)

set datafile separator ","
data = "../../data/computeTimesLinespoints.csv"
set terminal pngcairo size 1200, 600 enhanced font "FreeSerif,16"
set output "../computeTimesLinespoints_presentation.png"

#set title font ",16" \ #offset 8 \
#"Compute Times of Applying the One-Ring Filter for Selected Numbers of Iterations\n \
#onto Acquired and Synthetic 3D Meshes of Varying Sizes"

set key Left outside spacing 1.25 samplen 1 font ",10" enhanced title "Experiment Code †"
set logscale xy
set xlabel "Compute Time (seconds) - Log Scale"
set ylabel "Iterations Count - Log Scale" 
#set ylabel "Fast One-Ring Smoothing Filter Iterations - Log Scale"
#set xrange [-100:100000]
set yrange [1:3000]

plot for [i=3:12] data every ::1::8  using i:2 with linespoints title "MS.".columnhead(i) ls ceil(9-(i-1)*8/10) pt (i==8 ? 5 : (i==10 ? 7 : 1)), \
	 for [i=3:12] data every ::9::16 using i:2 with linespoints title "TS.".columnhead(i) ls ceil(9-(i-1)*8/10) pt (i==8 ? 5 : (i==10 ? 7 : 1)), \
	 for [i=3:12] data every ::17::24 using i:2 with linespoints title "MP.".columnhead(i) ls ceil(9-(i-1)*8/10) pt (i==8 ? 5 : (i==10 ? 7 : 1)), \
	 for [i=3:12] data every ::25::32 using i:2 with linespoints title "TP.".columnhead(i) ls ceil(9-(i-1)*8/10) pt (i==8 ? 5 : (i==10 ? 7 : 1))

