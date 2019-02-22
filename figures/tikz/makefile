OUTPUTS = ../unionsOfSimpleBuildNeighborhoods.pdf



all: $(OUTPUTS)

../%.pdf: %.tex
	pdflatex -f $< -e $@



clean: cleanTIKZs

cleanTIKZs:
	rm -f $(OUTPUTS)
