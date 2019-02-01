SVGS_OUTPUTS =  ../geodesicDisc.png \
				../centerOfGravity.png \
				../functionValues.png \
				../interpolation.png



all: $(SVGS_OUTPUTS)

../%.png: %.svg
	inkscape -f $< -e $@



clean: cleanSvgs

cleanSvgs:
	rm -f $(SVGS_OUTPUTS)
