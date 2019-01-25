reset
set datafile separator ","
data = "../data/computeTimesLinespoints.csv"
set terminal pngcairo size 900, 640 #enhanced font "Times New Roman, 10"
set output "computeTimesLinespoints.png"

set title "Compute Times of Applying the One-Ring Filter for Selected Numbers of Iterations\n\
onto Acquired and Synthetic 3D Meshes of Varying Sizes" font ",16" offset 8

set logscale xy
set key outside
set ylabel "Compute Time (seconds) - Log Scale"
set xlabel "One-Ring Filter Iterations - Log Scale"
plot for [i=3:12] data every ::0::7  using 2:i with linespoints title "TG.".columnhead(i), \
	 for [i=3:12] data every ::8::15 using 2:i with linespoints title "MG.".columnhead(i)

