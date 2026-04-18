MAIN = main
LATEX = pdflatex
BIBTEX = bibtex

.PHONY: all clean wordcount

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex abstract.tex introduction.tex method.tex experiments.tex conclusion.tex appendix.tex references.bib
	$(LATEX) $(MAIN)
	$(BIBTEX) $(MAIN)
	$(LATEX) $(MAIN)
	$(LATEX) $(MAIN)

clean:
	rm -f $(MAIN).{aux,bbl,blg,log,out,pdf,synctex.gz,fls,fdb_latexmk}
	rm -f *.aux

wordcount:
	@texcount -inc -total $(MAIN).tex 2>/dev/null || echo "texcount not installed"
