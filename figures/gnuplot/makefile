PLOTS = ../computeTimesLinespoints.png \
		../computeTimesLinespoints_presentation.png \
		../computeTimesScatter.png \
		../computeTimesScatter_presentation.png \
		../speedup.png \
		../efficiency.png \
		../numFacesByVerticesGoTo2.png
		#../computeTimesLinespoints4x4.png \
		#../outputSerialVsGPGPU.png \



all: $(PLOTS)

../%.png: %.plt
	gnuplot $<



clean: cleanPlots

cleanPlots:
	rm -f $(PLOTS)
