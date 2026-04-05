MAIN = main
LATEX = pdflatex
BIBTEX = bibtex

.PHONY: all clean check wordcount view

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex *.tex references.bib
	$(LATEX) $(MAIN)
	-$(BIBTEX) $(MAIN)
	$(LATEX) $(MAIN)
	$(LATEX) $(MAIN)

quick:
	$(LATEX) $(MAIN)

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.toc *.fls *.fdb_latexmk *.synctex.gz *.nav *.snm $(MAIN).pdf

check:
	chktex -q $(MAIN).tex 2>&1 || true

wordcount:
	@texcount -inc -total $(MAIN).tex 2>/dev/null | grep "Words in text" || echo "texcount not installed"

view: $(MAIN).pdf
	xdg-open $(MAIN).pdf 2>/dev/null || open $(MAIN).pdf 2>/dev/null || echo "Cannot open PDF"
