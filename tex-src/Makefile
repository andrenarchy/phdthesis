# ./Makefile

# ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###

LATEX:=lualatex
LATEX_OPTIONS:=-shell-escape
BIBTEX:=biber
FIGURES_TEX:=$(wildcard figures/*.tex)
FIGURES_CACHEDIR:=cache/
AUTOGEN:=autogen_commands

# ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###

TARGET:=diss

# ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
# 

default: diss
pdf: diss

diss: revision.tex $(FIGURES_CACHEDIR)/stamp $(AUTOGEN).tex
	@$(LATEX) $(LATEX_OPTIONS) $(TARGET)
	@$(BIBTEX) $(TARGET)
	#@make -f main.makefile -j 4
	@$(LATEX) $(LATEX_OPTIONS) $(TARGET)
	@$(LATEX) $(LATEX_OPTIONS) $(TARGET)

$(FIGURES_CACHEDIR)/stamp: $(FIGURES_TEX)
	@rm -f \
		$(FIGURES_CACHEDIR)/*.log \
		$(FIGURES_CACHEDIR)/*.pdf \
		$(FIGURES_CACHEDIR)/*.dpth
	@touch $(FIGURES_CACHEDIR)/stamp

revision.tex: revision.tex.in ../.git/index
	@GITID=$$(git log -1 | grep -e '^commit' -e '^Date:' | sed  -e 's/^[^ ]* *//g' | tr '\n' ' '); \
	echo $$GITID; \
	sed -e "s/GITID/$$GITID/g" $< >| $@

$(AUTOGEN).tex: $(AUTOGEN).py
	@python $< > $@

.PHONY: clean experiments

clean:
	@rm -f $(TARGET)-blx.bib \
	       $(TARGET).aux \
	       $(TARGET).auxlock \
	       $(TARGET).bcf \
	       $(TARGET).bbl \
	       $(TARGET).blg \
	       $(TARGET).log \
	       $(TARGET).pdf \
	       $(TARGET)-figure*.pdf \
	       $(TARGET)-figure*.log \
	       $(TARGET)-figure*.dpth \
	       $(TARGET).snm \
	       $(TARGET).nav \
		   $(TARGET).out \
		   $(TARGET).toc \
		   $(TARGET)-blx.bib \
		   $(TARGET).run.xml
	@rm -f \
		$(FIGURES_CACHEDIR)/*.log \
		$(FIGURES_CACHEDIR)/*.pdf \
		$(FIGURES_CACHEDIR)/*.dpth \
		$(FIGURES_CACHEDIR)/stamp
	@rm -f missfont.log \
	       x.log
	@rm -f revision.tex
	@rm -f *~

experiments:
	python experiments/run_experiments.py
