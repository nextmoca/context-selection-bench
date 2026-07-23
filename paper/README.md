# Paper source

LaTeX source for *A Matched-Protocol Benchmark Program for Context Selection in Agentic LLM
Systems*.

## Build

```bash
cd paper
tectonic main.tex
```

Produces `main.pdf`. A standard TeX Live install also works:

```bash
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

Packages used: `article`, `natbib`, `booktabs`, `array`, `seqsplit`, `graphicx`, `hyperref`,
`microtype`.

## Files

- `main.tex`: the paper.
- `references.bib`: bibliography.
