reset
#load "gnuplot-palettes-20190125/brbg.pal"
load "gnuplot-palettes-20190125/rdylbu.pal"
load "configs/grid.cfg"

set datafile separator ","
data = "../../data/computeTimesScatter.csv"
set terminal pngcairo size 1000, 680 enhanced font "FreeSerif, 16"
set output "../computeTimesScatter.png"

#set title font ",16" offset 0 \
#"Compute Times for Different Hardware Configurations\n \
#by increaseing Mesh Size and Filter Iterations"

set key outside box samplen 3 width 1.5 height 0.5
set key title "Machine /\nAlg.Variant"

set logscale xy
set xrange [0.3:1.3e4]
set yrange [3:1.75e8]
set xlabel "Fast One-Ring Smothing Filter Convolutions - Log Scale"
set ylabel "Mesh Size (Point Count) - Log Scale"

plot for [device = 4:7] for [i=0:79:10] data every ::i::(i+9) \
	using "Iters":"MeshSize":(column(device)**0.30) with points \
	linestyle (device < 6 ? 12-device : 8-device) \
	pointtype 6 pointsize variable \
	title (i == 0 ? columnhead(device) : "")

	#NORMAL   linestyle (device-3)  \
	#REVERSE  linestyle (12-device) \
	#EXTREMES linestyle (device < 6 ? device-3 : device+1) \
	#EXTM REV linestyle (device < 6 ? 12-device : 8-device) \

