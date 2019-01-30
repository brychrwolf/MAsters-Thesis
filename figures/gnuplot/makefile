PLOTS =  ../computeTimesLinespoints4x4.png \
		../computeTimesLinespoints.png \
		../computeTimesScatter.png \
		../numFacesByVerticesGoTo2.png \
		../outputSerialVsGPGPU.png



all: $(PLOTS)

../%.png: %.plt
	gnuplot $<



clean: cleanPlots

cleanPlots:
	rm -f $(PLOTS)
