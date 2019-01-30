SVGS_OUTPUTS =  ../geodesicDisc.png



all: $(SVGS_OUTPUTS)

../%.png: %.svg
	inkscape -f $< -e $@



clean: cleanSvgs

cleanSvgs:
	rm -f $(SVGS_OUTPUTS)
