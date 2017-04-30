
NAME = main

all: paper

paper:
	pdflatex $(NAME).tex
	bibtex $(NAME)
	pdflatex $(NAME).tex
	pdflatex $(NAME).tex

clean:
	rm -f $(NAME).bbl
	rm -f $(NAME).aux
	rm -f $(NAME).blg
	rm -f $(NAME).out
	rm -f $(NAME).log
