STIME = date '+%s' > $@_time
ETIME = read st < $@_time ; echo make recipe completed in $$((`date '+%s'`-$$st)) seconds ; rm -f $@_time

all: svgs tikzs plots thesis presentation

thesis: plots
	$(STIME)
	pdflatex -draftmode thesis
	makeindex thesis
	makeindex thesis.nlo -s nomencl.ist -o thesis.nls
	makeglossaries thesis
	bibtex thesis
	pdflatex -draftmode thesis
	pdflatex thesis
	$(ETIME)

once:
	pdflatex thesis

cleanThesis:
	rm -f thesis.aux thesis.bbl thesis.blg thesis.dvi thesis.idx thesis.ilg thesis.ind thesis.lof thesis.log thesis.lot thesis.out thesis.pdf thesis.toc thesis.loa thesis.nlo thesis.nls thesis.tdo thesis.xdv thesis.acn thesis.acr thesis.alg thesis.glg thesis.glo thesis.gls thesis.glsdefs thesis.ist

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

cleanTikzs:
	cd figures/tikz && make clean



.PHONY: plots
plots:
	cd figures/gnuplot && $(MAKE)

cleanPlots:
	cd figures/gnuplot && make clean




presentation: svgs tikzs plots
	$(STIME)
	pdflatex -draftmode presentation
	makeglossaries presentation
	pdflatex -draftmode presentation
	pdflatex presentation
	$(ETIME)

presentationOnce:
	$(STIME)
	pdflatex presentation
	$(ETIME)

cleanPresentation:
	cd figures/inkscape && $(MAKE) cleanPresentation
	rm -f presentation.aux presentation.log presentation.nav presentation.out presentation.pdf presentation.snm presentation.toc presentation.acn presentation.acr presentation.alg presentation.glg presentation.glo presentation.gls presentation.glsdefs presentation.ist



clean: cleanThesis

cleanAll: cleanThesis cleanChapters cleanPlots cleanSvgs cleanPresentation

