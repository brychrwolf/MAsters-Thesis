all: svgs plots thesis

thesis: plots
	xelatex --no-pdf thesis
	makeindex thesis
	makeindex thesis.nlo -s nomencl.ist -o thesis.nls
	bibtex thesis
	xelatex --no-pdf thesis
	xelatex thesis

once:
	xelatex thesis

cleanThesis:
	rm -f thesis.aux thesis.bbl thesis.blg thesis.dvi thesis.idx thesis.ilg thesis.ind thesis.lof thesis.log thesis.lot thesis.out thesis.pdf thesis.toc thesis.loa thesis.nlo thesis.nls thesis.tdo thesis.xdv chapters/*.aux



.PHONY: svgs
svgs:
	cd figures/inkscape && $(MAKE)

cleanSvgs:
	cd figures/inkscape && make clean



.PHONY: plots
plots:
	cd figures/gnuplot && $(MAKE)

cleanPlots:
	cd figures/gnuplot && make clean



clean: cleanThesis

cleanAll: cleanThesis cleanPlots cleanSvgs

