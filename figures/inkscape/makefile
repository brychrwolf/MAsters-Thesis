SVGS_OUTPUTS =  ../geodesicDisc.png \
				../anglesAndCenterOfGravity.png \
				../funcValVolumes.png \
				../interpolation.png \
				../circleSector.png \
				../neighborhoods.png \
				../triangularFaces.png \
				../interpolatedFunctionValues.png \
				../dataDependencyOfL2Norm.png



all: $(SVGS_OUTPUTS)

../%.png: %.svg
	inkscape -f $< -e $@



clean: cleanSvgs

cleanSvgs:
	rm -f $(SVGS_OUTPUTS)
