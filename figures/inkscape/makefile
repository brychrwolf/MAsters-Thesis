SVGS_OUTPUTS =  ../geodesicDisc.png \
				../anglesAndCenterOfGravity.png \
				../funcValVolumes.png \
				../interpolation.png \
				../circleSector.png \
				../neighborhoods.png \
				../triangularFaces.png \
				../interpolatedFunctionValues.png \
				../definePositionVector.png

SVGS_OUTPUTS_PRESENTATION = \
				../geodesicDisc_presentation.png \
				../anglesAndCenterOfGravity_presentation.png \
				../funcValVolumes_presentation.png \
				../neighborhoods_presentation.png \
				../neighborhoods_presentation_acq.png \
				../triangularFaces_presentation.png \
				../interpolatedFunctionValues_presentation1.png \
				../interpolatedFunctionValues_presentation2.png \
				../interpolatedFunctionValues_presentation3.png \
				../interpolatedFunctionValues_presentation4.png \
				../amdahlsLaw.png


all: $(SVGS_OUTPUTS) ${SVGS_OUTPUTS_PRESENTATION}

presentation: ${SVGS_OUTPUTS_PRESENTATION}

../%.png: %.svg
	inkscape -f $< -e $@



clean: cleanSvgs cleanSvgsPresentation

cleanSvgs:
	rm -f $(SVGS_OUTPUTS)

cleanPresentation:
	rm -f $(SVGS_OUTPUTS_PRESENTATION)
