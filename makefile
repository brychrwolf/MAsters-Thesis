all: svgs plots thesis

thesis: plots
	pdflatex --no-pdf thesis
	makeindex thesis
	makeindex thesis.nlo -s nomencl.ist -o thesis.nls
	bibtex thesis
	pdflatex --no-pdf thesis
	pdflatex thesis

once:
	pdflatex thesis

cleanThesis:
	rm -f thesis.aux thesis.bbl thesis.blg thesis.dvi thesis.idx thesis.ilg thesis.ind thesis.lof thesis.log thesis.lot thesis.out thesis.pdf thesis.toc thesis.loa thesis.nlo thesis.nls thesis.tdo thesis.xdv

cleanChapters:
	rm -f chapters/*.aux

cleanNom:
	rm -f thesis.nl*



.PHONY: svgs
svgs:
	cd figures/inkscape && $(MAKE)

cleanSvgs:
	cd figures/inkscape && make clean



.PHONY: tikzs
tikzs:
	cd figures/tikz && $(MAKE)

cleanTIKZs:
	cd figures/tikz && make clean



.PHONY: plots
plots:
	cd figures/gnuplot && $(MAKE)

cleanPlots:
	cd figures/gnuplot && make clean



clean: cleanThesis

cleanAll: cleanThesis cleanChapters cleanPlots cleanSvgs

