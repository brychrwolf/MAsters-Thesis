OUTPUTS = unionsOfSimpleBuildNeighborhoods.pdf \
		  sabnDataDependencies.pdf \
		  sacelDataDependencies.pdf \
		  sacfDataDependencies.pdf



all: $(OUTPUTS)

%.pdf: %.tex
	pdflatex $<

#convert -density 300 unionsOfSimpleBuildNeighborhoods.pdf -quality 90 unionsOfSimpleBuildNeighborhoods.png



clean: cleanTIKZs

cleanTIKZs:
	find . -type f -not -name '*tex' -not -name 'makefile' -delete
