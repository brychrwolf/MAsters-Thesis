STIME = date '+%s' > $@_time
ETIME = read st < $@_time ; echo make recipe completed in $$((`date '+%s'`-$$st)) seconds ; rm -f $@_time

presentation:
	$(STIME)
	pdflatex -draftmode presentation
	pdflatex -draftmode presentation
	pdflatex presentation
	$(ETIME)

once:
	pdflatex presentation

clean:
	rm -f presentation.aux presentation.log presentation.nav presentation.out presentation.pdf presentation.snm presentation.toc

