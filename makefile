all: plots thesis

thesis: plots
	xelatex --no-pdf thesis
	makeindex thesis
	bibtex thesis
	xelatex --no-pdf thesis
	xelatex thesis

once:
	xelatex thesis

cleanThesis:
	rm -f thesis.aux thesis.bbl thesis.blg thesis.dvi thesis.idx thesis.ilg thesis.ind thesis.lof thesis.log thesis.lot thesis.out thesis.pdf thesis.toc chapters/*.aux



.PHONY: plots
plots:
	cd plots && $(MAKE)

cleanPlots:
	cd plots && make clean



clean: cleanThesis

cleanAll: cleanThesis cleanPlots

