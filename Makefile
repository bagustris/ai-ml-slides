PANDOC ?= pandoc
PANDOC_FLAGS ?= -t beamer

SLIDES := $(shell find lessons -type f -name 'slides.md')
PDFS := $(SLIDES:.md=.pdf)

.PHONY: all pdf clean list

all: pdf

pdf: $(PDFS)

%.pdf: %.md
	cd "$(dir $<)" && $(PANDOC) "$(notdir $<)" $(PANDOC_FLAGS) -o "$(notdir $@)"

clean:
	rm -f $(PDFS)

list:
	@printf '%s\n' $(PDFS)
