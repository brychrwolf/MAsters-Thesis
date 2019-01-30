reset
#load "gnuplot-palettes-20190125/brbg.pal"
load "gnuplot-palettes-20190125/rdylbu.pal"
load "configs/grid.cfg"

set datafile separator ","
data = "../../data/computeTimesScatter.csv"
set terminal pngcairo size 1000, 680 #enhanced font "Times New Roman, 10"
set output "../computeTimesScatter.png"

set title font ",16" offset 0 \
"Compute Times for Different Hardware Configurations\n \
by increaseing Mesh Size and Filter Iterations"

set key outside box samplen 3 width 1.5 height 0.5
set key title "Machine /\nProcessor"

set logscale xy
set xrange [3:1e8]
set yrange [0.3:3e4]
set xlabel "Mesh Size (Vertex Count) - Log Scale"
set ylabel "One-Ring Filter Iterations - Log Scale"

plot for [device = 4:7] for [i=0:79:10] data every ::i::(i+9) \
	using "MeshSize":"Iters":(column(device)**0.30) with points \
	linestyle (device < 6 ? 12-device : 8-device) \
	pointtype 6 pointsize variable \
	title (i == 0 ? columnhead(device) : "")
	
	#NORMAL   linestyle (device-3)  \
	#REVERSE  linestyle (12-device) \
	#EXTREMES linestyle (device < 6 ? device-3 : device+1) \
	#EXTM REV linestyle (device < 6 ? 12-device : 8-device) \

