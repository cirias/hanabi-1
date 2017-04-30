
NAME=main
#LATEXCMDS="--output-dir=temp"

all: paper

paper:
	pdflatex $(NAME).tex
	bibtex $(NAME)
	pdflatex $(NAME).tex
	pdflatex $(NAME).tex

clean:
	rm -f *.bbl
	rm -f *.aux
	rm -f *.blg
	rm -f *.out
	rm -f *.log
