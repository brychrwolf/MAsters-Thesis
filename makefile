thesis:
	xelatex thesis
	makeindex thesis
	bibtex thesis
	xelatex thesis
	xelatex thesis
	
clean:
	rm -f thesis.aux thesis.bbl thesis.blg thesis.dvi thesis.idx thesis.ilg thesis.ind thesis.lof thesis.log thesis.lot thesis.out thesis.pdf thesis.toc
