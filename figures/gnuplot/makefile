PLOTS =  ../computeTimesLinespoints.png \
		../computeTimesScatter.png \
		../speedupAndEfficiency.png \
		../numFacesByVerticesGoTo2.png
		#../computeTimesLinespoints4x4.png \
		#../outputSerialVsGPGPU.png \



all: $(PLOTS)

../%.png: %.plt
	gnuplot $<



clean: cleanPlots

cleanPlots:
	rm -f $(PLOTS)
