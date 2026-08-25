## ------------------------------------------------------------------
## Builds BvChen-manual.pdf : a journal-style package manual
##   - title page with authors / emails / ORCID / abstract
##   - table of contents, theory section
##   - one documentation section per exported function, including
##     LIVE evaluated example output
## Run:  Rscript manual/build_manual.R
## ------------------------------------------------------------------

pkg    <- "C:/Users/Admin/.cline/data/workspaces/chat/BvChen"
lib    <- "C:/Users/Admin/.cline/data/workspaces/chat/_bvchenlib"
outdir <- file.path(pkg, "manual")
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

suppressMessages(library(BvChen, lib.loc = lib))
suppressMessages(library(evaluate))

topics <- c(
  "bvch",        # dbvch pbvch sbvch rbvch
  "Chen",        # univariate Chen d/p/q/r/h
  "fitchen",
  "Marginals",
  "Conditional",
  "Dependence",
  "Moments",
  "Likelihood",
  "fitbvch",
  "gofbvch",
  "plots"
)

rd_parse <- function(topic) {
  rd.file <- file.path(pkg, "man", paste0(topic, ".Rd"))
  if (!file.exists(rd.file)) return(NULL)
  txt <- paste(readLines(rd.file, warn = FALSE), collapse = "\n")
  tmp <- tempfile(fileext = ".Rd")
  con <- file(tmp, open = "wb")
  writeBin(charToRaw(txt), con)
  close(con)
  r <- try(tools::parse_Rd(tmp), silent = TRUE)
  if (inherits(r, "try-error"))
    stop("parse_Rd failed for topic '", topic, "': ",
         as.character(r), call. = FALSE)
  r
}

rd2latex_frag <- function(topic) {
  rd <- rd_parse(topic)
  if (is.null(rd)) return(NULL)
  tmp <- tempfile(fileext = ".tex")
  tools::Rd2latex(rd, tmp)   # full document; we strip preamble below
  tex <- paste(readLines(tmp, warn = FALSE), collapse = "\n")
  body <- sub("(?s).*\\\\begin\\{document\\}", "", tex, perl = TRUE)
  body <- sub("(?s)\\\\end\\{document\\}.*", "", body, perl = TRUE)
  body
}

get_examples <- function(topic) {
  rd <- rd_parse(topic)
  if (is.null(rd)) return(NULL)
  tmp <- tempfile(fileext = ".R")
  tools::Rd2ex(rd, tmp, silent = TRUE)
  if (!file.exists(tmp)) return("")
  code <- readLines(tmp, warn = FALSE)
  keep <- !grepl("^###", code)          # drop Rd2ex header/footer marks
  paste(code[keep], collapse = "\n")
}

run_example <- function(code) {
  if (is.null(code) || nzchar(trimws(paste(code, collapse = ""))) == 0)
    return("")
  ev <- evaluate::evaluate(code)
  lines <- character(0)
  for (piece in ev) {
    cls <- class(piece)
    if ("source" %in% cls) {
      # the code text is the element itself (one line per element)
      src <- sub("\n$", "", as.character(piece))
      lines <- c(lines, paste0("> ", src))
    } else if ("warning" %in% cls) {
      out <- paste0("Warning: ",
                    sub("\n$", "", attr(piece, "message")))
      if (nzchar(out)) lines <- c(lines, out)
    } else if (is.character(piece)) {
      # printed output comes back as plain "character" in evaluate
      out <- sub("\n$", "", as.character(piece))
      out <- out[nzchar(out)]
      if (length(out)) lines <- c(lines, out)
    }
  }
  paste(lines, collapse = "\n")
}

# ---- collect sections ---------------------------------------------
sections <- list()
for (tp in topics) {
  cat("processing:", tp, "\n")
  frag <- rd2latex_frag(tp)
  if (is.null(frag)) next
  ex_code <- get_examples(tp)
  cat("  got examples\n")
  ex_out  <- tryCatch(run_example(ex_code),
                      error = function(e) paste("example error:", e))
  # remove raw Examples environment; we re-render it executed
  frag_sub <- gsub("(?s)\\\\begin\\{Examples\\}.*?\\\\end\\{Examples\\}",
                   "", frag, perl = TRUE)
  ex_tex <- ""
  if (nzchar(trimws(ex_out))) {
    ex_tex <- paste0("\n\\vspace{0.8em}\n\\noindent{\\bfseries ",
                     "Examples} (live output):\n",
                     "\\begin{lstlisting}\n", ex_out,
                     "\n\\end{lstlisting}\n")
  }
  sections[[tp]] <- list(latex = frag_sub, examples = ex_tex)
}

frag_all <- paste(vapply(sections, function(s)
  paste0(s$latex, "\n", s$examples), character(1)),
  collapse = "\n\n\\newpage\n\n")

master <- paste(
  readLines(file.path(outdir, "preamble.tex"), warn = FALSE),
  collapse = "\n")


body <- paste0(master, "
\\begin{document}

% ======================= TITLE PAGE ===============================
\\begin{titlepage}
\\centering
\\vspace*{1.5cm}
{\\Huge \\bfseries The Bivariate Chen Distribution\\\\[0.6em]
The \\textsf{BvChen} R Package\\\\[1.2em]}
{\\Large A toolkit for density, distribution, survival, dependence
measures, EM-based estimation and goodness-of-fit analysis\\\\[2em]}
{\\large
\\textbf{Mukul Bijalwan}\\textsuperscript{1} \\quad and \\quad
\\textbf{Puneet Kumar Gupta}\\textsuperscript{2}\\\\[1em]}
{\\normalsize
\\textsuperscript{1}\\ \\texttt{mukulbijalwan555@gmail.com}\\quad
\\href{https://orcid.org/0009-0001-3040-6912}{%
\\orcidlink{0009-0001-3040-6912}}\\\\
\\textsuperscript{2}\\ \\texttt{puneetstat999@gmail.com}\\\\[2em]}
{\\normalsize \\today}
\\vspace{1.5em}
")
body <- paste0(body, "
\\begin{minipage}{0.88\\textwidth}
\\begin{abstract}
\\noindent
This manual documents the \\textsf{BvChen} package, an \\proglang{R}
implementation of the bivariate extension of the bathtub-shaped Chen
distribution proposed by Gupta, Pundir, Sharma and Mesfioui (2022,
\\emph{Life Cycle Reliability and Safety Engineering} 11:247--259),
constructed from three independent latent Chen (2000) components
through competing risks. The package provides the joint density, distribution and
survival functions (including the singular component on the diagonal),
random variate generation, marginal and conditional distributions,
a Marshall--Olkin type survival copula with Kendall's $\\tau$, joint
moments, maximum-likelihood estimation via the EM algorithm,
Kolmogorov--Smirnov goodness-of-fit tests and Q--Q diagnostics.
Complete univariate Chen building blocks are included. Every documented
function is illustrated with executable examples whose live output
appears directly beneath the code in this document.
\\end{abstract}
\\end{minipage}

\\vspace{1em}
{\\small \\textbf{Keywords:} Chen distribution; bivariate lifetime
model; Marshall--Olkin structure; competing risks; EM algorithm;
survival copula; \\proglang{R}.}

\\vspace{1em}
{\\footnotesize Generated automatically from the roxygen2 documentation
of version 0.1.0; all examples were executed while compiling this
manual.}
\\vspace*{\\fill}
\\end{titlepage}

% ======================= TOC =======================================
\\tableofcontents
\\newpage

% ======================= THEORY ====================================
\\section{The univariate Chen distribution}
The two-parameter Chen (2000) distribution has survival function
\\begin{equation}
S(x)=\\exp\\{-\\alpha(e^{x^{\\beta}}-1)\\}, \\qquad x>0,
\\; \\alpha,\\beta>0,
\\label{eq:chen}
\\end{equation}
hazard rate $h(x)=\\alpha\\beta x^{\\beta-1}e^{x^{\\beta}}$
(bathtub-shaped or increasing) and quantile function
$x_p=[\\ln\\{1-\\ln(1-p)/\\alpha\\}]^{1/\\beta}$.

\\section{The bivariate construction}
Let $X_0,X_1,X_2$ be independent with $X_i \\sim Ch(\\alpha_i,\\beta)$
and set $Z_1=\\min(X_0,X_1)$, $Z_2=\\min(X_0,X_2)$. Then
$Z_1 \\sim Ch(\\alpha_1+\\alpha_3,\\beta)$ and
$Z_2 \\sim Ch(\\alpha_2+\\alpha_3,\\beta)$ marginally. The joint law
is a mixture: an absolutely continuous part on $\\{z_1<z_2\\}$ and
$\\{z_1>z_2\\}$ carrying total mass
$(\\alpha_1+\\alpha_2)/(\\alpha_1+\\alpha_2+\\alpha_3)$, and a
singular component on the diagonal $z_1=z_2$ of mass
$\\alpha_3/(\\alpha_1+\\alpha_2+\\alpha_3)$.

\\subsection{Joint survival}
With $t(z)=e^{z^{\\beta}}-1$,
\\begin{equation}
S(z_1,z_2)=\\exp\\{-\\alpha_1 t(z_1)-\\alpha_2 t(z_2)
-\\alpha_3\\max(t(z_1),t(z_2))\\}.
\\label{eq:surv}
\\end{equation}

\\subsection{Survival copula}
In terms of marginal survival levels $u=S_1(z_1)$, $v=S_2(z_2)$,
\\begin{equation}
\\hat C(u,v)=\\min\\Bigl\\{u^{\\frac{\\alpha_1}{\\alpha_1+
\\alpha_3}}v,\\; uv^{\\frac{\\alpha_2}{\\alpha_2+\\alpha_3}}
\\Bigr\\},
\\label{eq:cop}
\\end{equation}
a Marshall--Olkin type copula whose dependence increases with the
relative size of the common-shock parameter $\\alpha_3$ and is free
of $\\beta$.

\\subsection{Estimation by the EM algorithm}
On the exponential scale $t_i=e^{z_i^{\\beta}}-1$ the latent model
reduces to a Marshall--Olkin competing-risks scheme. The package
alternates a numerical M-step maximising the observed log-likelihood
in $(\\alpha_1,\\alpha_2,\\alpha_3)$ at fixed $\\beta$ with a
one-dimensional update of $\\beta$; observed ties $z_{1i}=z_{2i}$
identify the common shock exactly.

\\newpage
% ==================== FUNCTION DOCUMENTATION =======================
\\addcontentsline{toc}{section}{Function documentation}
\\begin{center}{\\Large\\bfseries Function documentation}\\end{center}
\\vspace{0.5em}

", frag_all, "

\\newpage
% ==================== REFERENCES ====================================
\\newpage
\\section*{References}
\\addcontentsline{toc}{section}{References}
{\\small
\\noindent Ahmed EA (2014) Bayesian estimation based on progressive type-II censoring from two-parameter bathtub-shaped lifetime model: an Markov chain Monte Carlo approach. J Appl Stat 41(4):752--768\\\\[0.35em]
\\noindent Bhatti FA, Hamedani GG, Najibi SM, Ahmad M (2021) On the extended Chen distribution: development, properties, characterizations and applications. Ann Data Sci 8(1):159--180\\\\[0.35em]
\\noindent Block H, Basu AP (1974) A continuous bivariate exponential extension. J Am Stat Assoc 69:1031--1037\\\\[0.35em]
\\noindent Chen Z (2000) A new two-parameter lifetime distribution with bathtub shape or increasing failure rate function. Stat Probab Lett 49(2):155--161\\\\[0.35em]
\\noindent Dempster AP, Laird NM, Rubin DB (1977) Maximum likelihood from incomplete data via the EM algorithm. J R Stat Soc Ser B (Methodol) 39(1):1--38\\\\[0.35em]
\\noindent Dey S, Sharma VK, Mesfioui M (2017) A new extension of Weibull distribution with application to lifetime data. Ann Data Sci. https://doi.org/10.1007/s40745-016-0094-8\\\\[0.35em]
\\noindent Gupta PK, Pundir PS, Sharma VK, Mesfioui M (2022) Bivariate extension of bathtub-shaped distribution. Life Cycle Reliab Saf Eng 11:247--259. https://doi.org/10.1007/s41872-022-00193-4\\\\[0.35em]
\\noindent Johnson NL, Kotz S, Kemp AW (1992) Univariate discrete distributions, 2nd edn. Wiley, New York\\\\[0.35em]
\\noindent Johnson NL, Kotz S, Balakrishnan N (1995) Continuous univariate distributions, vol 2. Wiley, New York\\\\[0.35em]
\\noindent Johnson NL, Kotz S, Balakrishnan N (1997) Discrete multivariate distributions, vol 165. Wiley, New York\\\\[0.35em]
\\noindent Kayal T, Tripathi YM, Singh DP, Rastogi MK (2016) Estimation and prediction for Chen distribution with bathtub shape under progressive censoring. J Stat Comput Simul. https://doi.org/10.1080/00949655.2016.1209199\\\\[0.35em]
\\noindent Kocherlakota S, Kocherlakota K (1992) Bivariate discrete distributions. Wiley, New York\\\\[0.35em]
\\noindent Kotz S, Balakrishnan N, Johnson NL (2004) Continuous multivariate distributions, volume 1: models and applications. Wiley, New York\\\\[0.35em]
\\noindent Kundu D, Dey AK (2009) Estimating the parameters of the Marshall--Olkin bivariate Weibull distribution by EM algorithm. Comput Stat Data Anal 53(4):956--965\\\\[0.35em]
\\noindent Kundu D, Gupta RD (2009) Bivariate generalized exponential distribution. J Multivar Anal 100:581--593\\\\[0.35em]
\\noindent Kundu D, Gupta RD (2010) Modified Sarhan--Balakrishnan singular bivariate distribution. J Stat Plan Inference 140(2):526--538\\\\[0.35em]
\\noindent Marshall AW, Olkin I (1967) A multivariate exponential distribution. J Am Stat Assoc 62:30--44\\\\[0.35em]
\\noindent Meintanis SG (2007) Test of fit for Marshall--Olkin distributions with applications. J Stat Plan Inference 137(12):3954--3963\\\\[0.35em]
\\noindent Papageorgiou H (1997) Multivariate discrete distributions. In: Kotz S, Read CB, Banks DL (eds) Encyclopedia of statistical sciences, vol 1. Wiley, New York, pp 408--419\\\\[0.35em]
\\noindent Pundir PS, Gupta PK (2018a) Reliability estimation in load-sharing system model with application to real data. Ann Data Sci 5(1):69--91\\\\[0.35em]
\\noindent Pundir PS, Gupta PK (2018b) Stress-strength reliability of two-parameter bathtub-shaped lifetime model based on hybrid censored samples. J Stat Manag Syst 21(7):1229--1250\\\\[0.35em]
\\noindent Rajarshi S, Rajarshi MB (1988) Bathtub distributions: a review. Commun Stat Theory Methods 17(8):2597--2621\\\\[0.35em]
\\noindent Rastogi MK, Tripathi YM (2013) Estimation using hybrid censored data from a two parameter distribution with bathtub shape. Comput Stat Data Anal 67:268--281\\\\[0.35em]
\\noindent Sarhan AM, Hamilton DC, Smith B (2012) Parameter estimations for a two parameter bathtub-shaped lifetime distribution. Appl Math Model 36(11):5380--5392\\\\[0.35em]
\\noindent Tyagi A, Choudhary N, Singh B (2021) Reliability analysis of the dynamic system for the Chen model through sequential order statistics. Qual Reliab Eng Int 37(6):2514--2534\\\\[0.35em]
\\noindent Wang L, Wu K, Tripathi YM, Lodhi C (2022) Reliability analysis of multicomponent stress--strength reliability from a bathtub-shaped distribution. J Appl Stat 49(1):122--142\\\\[0.35em]
\\noindent Wu SJ (2008) Estimation of the two-parameter bathtub-shaped lifetime distribution with progressive censoring. J Appl Stat 35:1139--1150\\\\[0.35em]
\\noindent Wu JW, Lu HL, Chen CH, Wu CH (2004) Statistical inference about the shape parameter of the new two-parameter bathtub-shaped lifetime distribution. Qual Reliab Eng Int 20:607--616
}

\\end{document}
")
writeLines(body, file.path(outdir, "BvChen-manual.tex"))
cat("master tex written\n")

# ---- compile with pdflatex -----------------------------------------
# make R's Rd.sty available directly (avoids TEXINPUTS quirks)
rdsty <- file.path(R.home("share"), "texmf", "tex", "latex", "Rd.sty")
file.copy(rdsty, outdir, overwrite = TRUE)
oldwd <- setwd(outdir)
on.exit(setwd(oldwd), add = TRUE)
res <- 1
for (pass in 1:3) {
  res <- system2("pdflatex", c("-interaction=nonstopmode",
                               "BvChen-manual.tex"),
                 stdout = FALSE, stderr = FALSE)
}
cat("pdflatex exit:", res, "\n")
if (file.exists("BvChen-manual.pdf")) {
  file.copy("BvChen-manual.pdf", pkg, overwrite = TRUE)
  cat("PDF written to:", file.path(pkg, "BvChen-manual.pdf"), "\n")
} else {
  cat("PDF FAILED - inspect BvChen-manual.log in", outdir, "\n")
}