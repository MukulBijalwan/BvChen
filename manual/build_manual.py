## ---------------------------------------------------------------------
## Builds bvchen-manual.pdf : a journal-style package manual for the
## *Python* package `bvchen`, mirroring the layout of the reference
## BvChen-manual.pdf of the companion R package:
##   - title page with authors / e-mails / abstract / keywords
##   - table of contents and theory section
##   - one documentation block per exported function group, including
##     LIVE evaluated example output
##   - references
## Run:  python manual/build_manual.py
## ---------------------------------------------------------------------

from __future__ import annotations

import ast
import contextlib
import datetime
import io
import os
import re
import subprocess
import sys
import warnings

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTDIR = os.path.join(ROOT, "manual")
sys.path.insert(0, os.path.join(ROOT, "src"))
os.chdir(OUTDIR)

import numpy as np                      # noqa: E402
import matplotlib                       # noqa: E402
matplotlib.use("Agg")

import bvchen                           # noqa: E402
from bvchen import (BvChen, bvch_moment, dbvch, dchen, dcond1, dmarg1,  # noqa: E402
                    dmarg2, fit_bvchen, gof, hcond1, kendall_tau, loglik_bvch,
                    pbvch, pchen, pmarg1, pmarg2, profile_beta, qchen, rbvch,
                    rchen, sbvch, schen, survcop, qqplot)
from bvchen.plotting import contour_plot, surface_plot                   # noqa: E402


# ---------------------------------------------------------------------
# Live evaluation of example code: executes every top-level statement
# separately and interleaves the printed output beneath the source,
# exactly like the R manual does with the `evaluate` package.
# ---------------------------------------------------------------------
def run_example(code: str) -> str:
    buf = io.StringIO()
    glb = {"np": np, "bvchen": bvchen}
    segments = []
    tree = ast.parse(code)
    with contextlib.redirect_stdout(buf):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for node in tree.body:
                before = buf.getvalue()
                seg = ast.get_source_segment(code, node) or ""
                # mode="single" echoes bare expressions like a REPL
                mode = "single" if isinstance(node, ast.Expr) else "exec"
                mod = ast.Interactive(body=[node]) if mode == "single" \
                    else ast.Module(body=[node], type_ignores=[])
                exec(compile(mod, "<example>", mode), glb)
                emitted = buf.getvalue()[len(before):]
                src = "\n".join("> " + ln for ln in seg.splitlines())
                segments.append(src.rstrip())
                if emitted.strip():
                    segments.append(emitted.rstrip("\n"))
    text = "\n".join(s for s in segments)
    return re.sub(r"\n+>", "\n>", text)
# ---------------------------------------------------------------------
# LaTeX preamble: same look & feel as the reference BvChen manual, but
# fully self-contained (no Rd.sty needed).
# ---------------------------------------------------------------------
PREAMBLE = r"""\documentclass[11pt,a4paper]{article}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{mathptmx}
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{listings}
\usepackage{xcolor}

% ---- page geometry ---------------------------------------------------
\usepackage[margin=2.6cm]{geometry}
\usepackage{hyperref}
\hypersetup{hidelinks}
\usepackage{fancyhdr}

% ---- look & feel -----------------------------------------------------
\setlength{\parskip}{0.4em}
\setlength{\parindent}{0em}
\emergencystretch=2em
\sloppy

\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\small\itshape The bvchen Package}
\fancyhead[R]{\small\itshape Bivariate Chen Distribution in Python}
\fancyfoot[C]{\thepage}
\renewcommand{\headrulewidth}{0.4pt}

% ---- code listing style ----------------------------------------------
\lstdefinestyle{bvchex}{
  basicstyle=\ttfamily\footnotesize,
  breaklines=true,
  columns=fullflexible,
  frame=single,
  framerule=0.3pt,
  rulecolor=\color{gray!60},
  backgroundcolor=\color{gray!7},
  xleftmargin=1em
}
\lstset{style=bvchex}

% ---- Rd-style documentation macros -----------------------------------
\newcommand{\proglang}[1]{\textsf{#1}}
\newcommand{\pypkg}[1]{\textbf{#1}}
\newcommand{\code}[1]{\texttt{#1}}
\newcommand{\eqn}[2]{$#1$}

% topic header:  \HeaderA{function names}{title}
\newcommand{\HeaderA}[2]{%
  \vspace{1.1em}\phantomsection
  \addcontentsline{toc}{section}{\protect\numberline{}#2 \hfill{}%
    \normalfont\texttt{\small #1}}%
  \noindent{\Large\bfseries\ttfamily #1}\\[0.35em]
  {\large #2}\par
  \vspace{0.4em}\hrule\vspace{0.9em}}

\newcommand{\AliasLine}[1]{%
  \noindent{\small\bfseries Aliases: }{\small\ttfamily #1}\par\vspace{0.5em}}

\newcommand{\DocSec}[1]{%
  \vspace{0.7em}\noindent{\bfseries #1}\par\vspace{0.25em}}

\begin{document}"""
THEORY = r"""
% ======================= TITLE PAGE ===============================
\begin{titlepage}
\centering
\vspace*{1.5cm}
{\Huge \bfseries The Bivariate Chen Distribution\\[0.6em]
The \textsf{bvchen} \proglang{Python} Package\\[1.2em]}
{\Large A toolkit for density, distribution, survival, dependence
measures, EM-based estimation and goodness-of-fit analysis\\[2em]}
{\large
\textbf{Mukul Bijalwan}\textsuperscript{1} \quad and \quad
\textbf{Puneet Kumar Gupta}\textsuperscript{2}\\[1em]}
{\normalsize
\textsuperscript{1}\ \texttt{mukulbijalwan555@gmail.com}\\
\textsuperscript{2}\ \texttt{puneetstat999@gmail.com}\\[2em]}
{\normalsize \today}
\vspace{1.5em}

\begin{minipage}{0.88\textwidth}
\begin{abstract}
\noindent
This manual documents the \pypkg{bvchen} package, a \proglang{Python}
implementation of the bivariate extension of the bathtub-shaped Chen
distribution proposed by Gupta, Pundir, Sharma and Mesfioui (2022,
\emph{Life Cycle Reliability and Safety Engineering} 11:247--259),
constructed from three independent latent Chen (2000) components
through competing risks. The package provides the joint density,
distribution and survival functions (including the singular component
on the diagonal), random variate generation, marginal and conditional
distributions, a Marshall--Olkin type survival copula with Kendall's
$\tau$, joint moments, maximum-likelihood estimation via the EM
algorithm, Kolmogorov--Smirnov goodness-of-fit tests and Q--Q
diagnostics. Every documented function is illustrated with executable
examples whose live output appears directly beneath the code in this
document.
\end{abstract}
\end{minipage}

\vspace{1em}
{\small \textbf{Keywords:} Chen distribution; bivariate lifetime
model; Marshall--Olkin structure; competing risks; EM algorithm;
survival copula; \proglang{Python}.}

\vspace{1em}
{\footnotesize Generated automatically from the docstrings of version
0.1.0; all examples were executed while compiling this manual.}
\vspace*{\fill}
\end{titlepage}

% ======================= TOC =======================================
\tableofcontents
\newpage

% ======================= THEORY ====================================
\section{The univariate Chen distribution}
The two-parameter Chen (2000) distribution has survival function
\begin{equation}
S(x)=\exp\{-\alpha(e^{x^{\beta}}-1)\}, \qquad x>0,
\; \alpha,\beta>0,
\label{eq:chen}
\end{equation}
hazard rate $h(x)=\alpha\beta x^{\beta-1}e^{x^{\beta}}$
(bathtub-shaped or increasing) and quantile function
$x_p=[\ln\{1-\ln(1-p)/\alpha\}]^{1/\beta}$.

\section{The bivariate construction}
Let $X_0,X_1,X_2$ be independent with $X_i \sim Ch(\alpha_i,\beta)$
and set $Z_1=\min(X_0,X_1)$, $Z_2=\min(X_0,X_2)$. Then
$Z_1 \sim Ch(\alpha_1+\alpha_3,\beta)$ and
$Z_2 \sim Ch(\alpha_2+\alpha_3,\beta)$ marginally. The joint law is a
mixture: an absolutely continuous part on $\{z_1<z_2\}$ and
$\{z_1>z_2\}$ carrying total mass
$(\alpha_1+\alpha_2)/(\alpha_1+\alpha_2+\alpha_3)$, and a singular
component on the diagonal $z_1=z_2$ of mass
$\alpha_3/(\alpha_1+\alpha_2+\alpha_3)$.

\subsection{Joint survival}
With $t(z)=e^{z^{\beta}}-1$,
\begin{equation}
S(z_1,z_2)=\exp\{-(\alpha_1+\alpha_3)t(z_1)-(\alpha_2+\alpha_3)t(z_2)
+\alpha_3 t(\min(z_1,z_2))\}.
\label{eq:surv}
\end{equation}

\subsection{Survival copula}
In terms of marginal survival levels $u=S_1(z_1)$, $v=S_2(z_2)$,
\begin{equation}
\hat C(u,v)=\min\Bigl\{v\,u^{\frac{\alpha_1}{\alpha_1+\alpha_3}},\;
u\,v^{\frac{\alpha_2}{\alpha_2+\alpha_3}}\Bigr\},
\label{eq:cop}
\end{equation}
a Marshall--Olkin type copula whose dependence increases with the
relative size of the common-shock parameter $\alpha_3$ and is free
of $\beta$.

\subsection{Estimation by the EM algorithm}
On the exponential scale $t_i=e^{z_i^{\beta}}-1$ the latent model
reduces to a Marshall--Olkin competing-risks scheme. The package
alternates a numerical M-step maximising the observed log-likelihood
in $(\alpha_1,\alpha_2,\alpha_3)$ at fixed $\beta$ with a
one-dimensional update of $\beta$; observed ties $z_{1i}=z_{2i}$
identify the common shock exactly.

\newpage
"""
# ---------------------------------------------------------------------
# Documentation blocks -------------------------------------------------
# ---------------------------------------------------------------------
def _texfix(s: str) -> str:
    r"""Collapse accidental double backslashes (from raw strings) so
    that ``\\code`` reaches LaTeX as ``\code``."""
    return s.replace("\\\\", "\\")


def doc_block(s: dict) -> str:
    s = dict(s)
    for key in ("description", "details", "references"):
        if key in s:
            s[key] = _texfix(s[key])
    if "arguments" in s:
        s["arguments"] = [(a, _texfix(b)) for a, b in s["arguments"]]
    if "value" in s:
        s["value"] = [(a, _texfix(b)) for a, b in s["value"]]
    parts = [r"\HeaderA{%s}{%s}" % (_texfix(s["names"]), s["title"])]
    if s.get("aliases"):
        parts.append(r"\AliasLine{%s}" % s["aliases"])
    parts.append(r"\DocSec{Description}")
    parts.append(s["description"])
    parts.append(r"\DocSec{Usage}")
    parts.append("\\begin{verbatim}\n%s\n\\end{verbatim}" % s["usage"])
    if s.get("arguments"):
        parts.append(r"\DocSec{Arguments}")
        items = "\n".join(r"\item[\code{%s}] %s" % a
                          for a in s["arguments"])
        parts.append("\\begin{description}\n%s\n\\end{description}"
                     % items)
    if s.get("details"):
        parts.append(r"\DocSec{Details}")
        parts.append(s["details"])
    if s.get("value"):
        parts.append(r"\DocSec{Value}")
        items = "\n".join(r"\item[\code{%s}] %s" % v
                          for v in s["value"])
        parts.append("\\begin{description}\n%s\n\\end{description}"
                     % items)
    if s.get("references"):
        parts.append(r"\DocSec{References}")
        parts.append(s["references"])
    if s.get("example"):
        parts.append("")
        parts.append(r"\vspace{0.8em}"
                     r"\noindent{\bfseries Examples} (live output):")
        parts.append("\\begin{lstlisting}\n%s\n\\end{lstlisting}"
                     % run_example(s["example"]))
    return "\n".join(parts) + "\n"


SECTIONS = []


def add(**kw):
    SECTIONS.append(kw)
add(
    names="dchen pchen schen qchen rchen hchen fit\\_chen chen\\_moment",
    title="The Univariate Chen Distribution",
    aliases="dchen, pchen, schen, qchen, rchen, hchen, "
            "fit\\_chen, chen\\_moment",
    description=r"""Density (\code{dchen}), distribution function
(\code{pchen}), survival function (\code{schen}), quantile function
(\code{qchen}), random generation (\code{rchen}) and hazard rate
(\code{hchen}) for the Chen (2000) distribution with scale parameter
\eqn{\alpha}{} and power parameter \eqn{\beta}{};
maximum-likelihood fitting (\code{fit\_chen}) and raw moments
(\code{chen\_moment}).""",
    usage="""dchen(x, alpha, beta, log=False)
pchen(q, alpha, beta)
schen(q, alpha, beta)
qchen(p, alpha, beta)
rchen(n, alpha, beta, random_state=None)
hchen(x, alpha, beta)
fit_chen(x, start=None, tol=1e-08, maxit=100)
chen_moment(k, alpha, beta, tol=1e-09)""",
    arguments=[
        ("x, q", "array\\_like of non-negative quantiles."),
        ("p", "array\\_like of probabilities in $[0,1]$."),
        ("alpha, beta",
         "positive scale and power parameters ($\\alpha,\\beta>0$)."),
        ("log", "bool; if \\code{True} the log-density is returned."),
        ("n", "int; number of variates for \\code{rchen}."),
        ("random\\_state", "int or Generator; reproducibility seed."),
        ("start",
         "optional tuple/dict of starting values for \\code{fit\\_chen}."),
        ("k", "non-negative moment order for \\code{chen\\_moment}."),
    ],
    details=r"""The survival follows equation~(\ref{eq:chen});
the hazard is bathtub-shaped for $\beta<1$ and increasing for
$\beta\ge 1$. Random variates exploit that
$U=e^{X^{\beta}}-1\sim Exp(\alpha)$, i.e.\
$X=\{\ln(1+U)\}^{1/\beta}$.""",
    value=[
        ("dchen", "ndarray of density values (or log-density)."),
        ("pchen, schen", "ndarray of probabilities."),
        ("qchen", "ndarray of quantiles."),
        ("rchen", "ndarray of size \\code{n}."),
        ("hchen", "ndarray of hazard rates."),
        ("fit\\_chen", "dict with keys \\code{alpha}, \\code{beta}, "
                       "\\code{loglik}, \\code{converged}, \\code{se}."),
        ("chen\\_moment", "float, $E[X^k]$."),
    ],
    references=r"""Chen, Z. (2000). A new two-parameter lifetime
distribution with bathtub shape or increasing failure rate function.
\emph{Statistics \& Probability Letters} 49, 155--161.""",
    example="""from bvchen import (dchen, pchen, schen, qchen, rchen,
                    hchen, fit_chen)

dchen([0.5, 1.0, 2.0], alpha=2.0, beta=1.5)
pchen(1.0, alpha=2.0, beta=1.5)
schen(1.0, alpha=2.0, beta=1.5)
qchen([0.25, 0.5, 0.75], alpha=2.0, beta=1.5)
hchen(0.5, alpha=2.0, beta=1.5)

x = rchen(500, alpha=2.0, beta=1.5, random_state=42)
fit = fit_chen(x)
(round(fit["alpha"], 3), round(fit["beta"], 3),
 round(fit["loglik"], 2))""",
)
add(
    names="dbvch pbvch sbvch rbvch",
    title="The Bivariate Chen Distribution",
    aliases="dbvch, pbvch, sbvch, rbvch",
    description=r"""Density (\code{dbvch}), joint CDF (\code{pbvch}),
joint survival function (\code{sbvch}) and random generation
(\code{rbvch}) for the bivariate Chen distribution built from three
independent latent Chen components: $Z_1=\min(X_0,X_1)$,
$Z_2=\min(X_0,X_2)$ with $X_i \sim Ch(\alpha_i,\beta)$.""",
    usage="""dbvch(x, y, alpha1, alpha2, alpha3, beta,
     log=False, component="full")
sbvch(x, y, alpha1, alpha2, alpha3, beta)
pbvch(x, y, alpha1, alpha2, alpha3, beta)
rbvch(n, alpha1, alpha2, alpha3, beta, random_state=None)""",
    arguments=[
        ("x, y", "array\\_like of non-negative quantiles."),
        ("alpha1, alpha2, alpha3",
         "positive shape parameters; \\code{alpha3} is the common-shock "
         "parameter driving the diagonal ties."),
        ("beta", "positive common power parameter."),
        ("log", "bool; if \\code{True} the log-density is returned."),
        ("component",
         "\\code{\"full\"} (default), \\code{\"ac\"} (absolutely "
         "continuous part only) or \\code{\"sing\"} (diagonal singular "
         "density only)."),
        ("n", "int; number of observations for \\code{rbvch}."),
    ],
    details=r"""On $\{z_1<z_2\}$ the density factorises as
$f_{Ch(\alpha_1)}(z_1)\,f_{Ch(\alpha_2+\alpha_3)}(z_2)$ (symmetrically
on $\{z_1>z_2\}$); the diagonal carries the singular component of mass
$\alpha_3/(\alpha_1+\alpha_2+\alpha_3)$ with density
$g(z)=\alpha_3 t'(z)\exp\{-A\,t(z)\}$, $A=\alpha_1+\alpha_2+\alpha_3$.
The CDF uses the inclusion--exclusion identity $F=1-S_1-S_2+S$, valid
also in the presence of the singular part.""",
    value=[
        ("dbvch", "float or ndarray of joint density values."),
        ("pbvch", "$P(Z_1\\le x,\\, Z_2\\le y)$."),
        ("sbvch", "$P(Z_1>x,\\, Z_2>y)$."),
        ("rbvch", "ndarray of shape \\code{(n, 2)}, columns "
                  "\\code{z1}, \\code{z2}."),
    ],
    references=r"""Gupta, P.~K., Pundir, P.~S., Sharma, V.~K.,
Mesfioui, M. (2022). Bivariate extension of bathtub-shaped
distribution. \emph{Life Cycle Reliability and Safety Engineering}
11, 247--259.""",
    example="""from bvchen import dbvch, pbvch, sbvch, rbvch

dbvch(0.5, 1.5, 0.8, 1.2, 0.5, 1.5)   # region z1 < z2
dbvch(1.5, 0.5, 0.8, 1.2, 0.5, 1.5)   # region z1 > z2
dbvch(1.0, 1.0, 0.8, 1.2, 0.5, 1.5)   # singular diagonal part

pbvch(1.0, 2.0, 0.8, 1.2, 0.5, 1.5)
sbvch(1.0, 2.0, 0.8, 1.2, 0.5, 1.5)

sim = rbvch(400, 0.8, 1.2, 0.5, 1.5, random_state=2024)
np.mean(sim[:, 0] == sim[:, 1])   # tie rate about a3/(sum alphas)""",
)


add(
    names="dmarg1 dmarg2 pmarg1 pmarg2",
    title="Marginal Distributions of the Bivariate Chen Law",
    aliases="dmarg1, dmarg2, pmarg1, pmarg2",
    description=r"""Density (\code{dmarg1}, \code{dmarg2}) and
distribution function (\code{pmarg1}, \code{pmarg2}) of the marginal
laws $Z_1\sim Ch(\alpha_1+\alpha_3,\beta)$ and
$Z_2\sim Ch(\alpha_2+\alpha_3,\beta)$.""",
    usage="""dmarg1(x, alpha1, alpha2, alpha3, beta, log=False)
dmarg2(y, alpha1, alpha2, alpha3, beta, log=False)
pmarg1(q, alpha1, alpha2, alpha3, beta)
pmarg2(q, alpha1, alpha2, alpha3, beta)""",
    arguments=[
        ("x, q", "non-negative quantiles."),
        ("alpha1, alpha2, alpha3, beta",
         "parameters of the bivariate law; only the sums "
         "$\\alpha_1+\\alpha_3$ resp.\\ $\\alpha_2+\\alpha_3$ enter."),
        ("log", "bool; return the log-density instead."),
    ],
    details=r"""Identical by construction to the univariate Chen
functions evaluated at the summed scale parameters.""",
    value=[("dmarg1, dmarg2", "marginal densities."),
           ("pmarg1, pmarg2", "marginal CDFs.")],
    example="""from bvchen import dmarg1, dmarg2, pmarg1, pmarg2
from bvchen import dchen, pchen

dmarg1(1.0, 0.8, 1.2, 0.5, 1.5)
dchen(1.0, 1.3, 1.5)                    # identical by construction
pmarg2([0.5, 1.0, 2.0], 0.8, 1.2, 0.5, 1.5)
pchen([0.5, 1.0, 2.0], 1.7, 1.5)        # identical by construction""",
)
add(
    names="dcond1 dcond2 hcond1 hcond2",
    title="Conditional Distributions of the Bivariate Chen Law",
    aliases="dcond1, dcond2, hcond1, hcond2",
    description=r"""Conditional density of $Z_1$ given $Z_2=y$
(\code{dcond1}; suffix \code{2} symmetrically for $Z_2$ given
$Z_1=x$) and conditional hazard of the absolutely continuous part
(\code{hcond1}, \code{hcond2}). The conditional law carries an atom
at $z_i=y$ whose mass is exposed through the attribute
\code{atom}.""",
    usage="""dcond1(z1, y, alpha1, alpha2, alpha3, beta)
dcond2(z2, x, alpha1, alpha2, alpha3, beta)
hcond1(z1, y, alpha1, alpha2, alpha3, beta)
hcond2(z2, x, alpha1, alpha2, alpha3, beta)""",
    arguments=[
        ("z1, z2", "values at which the conditional density/hazard of "
                   "the response variable is evaluated."),
        ("y, x", "conditioning values; must be positive."),
        ("alpha1, alpha2, alpha3, beta", "parameters (all positive)."),
    ],
    details=r"""For $z_1<y$ the conditional density equals
$f_{Ch(\alpha_1)}(z_1)$, independent of $y$; for $z_1>y$ it equals
\linebreak
$f_{Ch(\alpha_2)}(y)\,f_{Ch(\alpha_1+\alpha_3)}(z_1)/
f_{Ch(\alpha_2+\alpha_3)}(y)$. The atom at $z_1=y$ has mass
$g(y)/f_{Ch(\alpha_2+\alpha_3)}(y)$. The conditional hazard is
\code{NaN} exactly at $z_i=y$, where only the point mass lives.""",
    value=[
        ("dcond1, dcond2",
         "ndarray subclass carrying the attribute \\code{atom} "
         "(point mass at the conditioning value)."),
        ("hcond1, hcond2", "conditional hazard rates."),
    ],
    example="""from bvchen import dcond1, hcond1

dc = dcond1([0.4, 0.9, 1.2], y=1.0,
            alpha1=1, alpha2=1, alpha3=1, beta=1.5)
dc
dc.atom                       # point mass at z1 = y

hcond1([0.5, 1.0, 1.5], y=1.0,
       alpha1=1, alpha2=1, alpha3=1, beta=1.5)""",
)


add(
    names="survcop kendall\\_tau",
    title="Dependence Measures of the Bivariate Chen Law",
    aliases="survcop, kendall\\_tau",
    description=r"""Survival copula $\hat C(u,v)$ (\code{survcop})
and Monte-Carlo Kendall's $\tau$ (\code{kendall\\_tau}, computed as
$\tau = 4E[F(Z_1,Z_2)]-1$). Dependence increases with the relative
size of the common-shock parameter $\alpha_3$ and does not depend on
$\beta$.""",
    usage="""survcop(u, v, alpha1, alpha2, alpha3)

kendall_tau(alpha1, alpha2, alpha3, beta,
            n=50000, random_state=None)""",
    arguments=[
        ("u, v", "marginal survival levels in $[0,1]$."),
        ("alpha1, alpha2, alpha3", "positive shape parameters."),
        ("beta", "power parameter (no influence on tau)."),
        ("n", "int; number of Monte-Carlo draws."),
        ("random\\_state", "seed or Generator for reproducibility."),
    ],
    value=[
        ("survcop",
         "$\\hat C(u,v)=\\min\\{v\\,u^{\\alpha_1/(\\alpha_1+\\alpha_3)},"
         "\\; u\\,v^{\\alpha_2/(\\alpha_2+\\alpha_3)}\\}$."),
        ("kendall\\_tau", "float estimate of Kendall's tau."),
    ],
    example="""from bvchen import survcop, kendall_tau

survcop([0.5, 0.8], [0.5, 0.8], alpha1=1, alpha2=1, alpha3=1)
survcop(0, 0.5, 1, 1, 1)              # boundary: 0

kendall_tau(1, 1, 0.05, 1.5, n=20000, random_state=9)
kendall_tau(1, 1, 10.0, 1.5, n=20000, random_state=9)""",
)
add(
    names="bvch\\_moment",
    title="Moments of the Bivariate Chen Distribution",
    aliases="bvch\\_moment; class methods BvChen.moment, .mean, "
            ".var, .cov, .min\\_dist",
    description=r"""Raw joint moment $E[Z_1^r Z_2^s]$
(\code{bvch\\_moment}). Following Proposition~4 of the paper, the
moment splits into integrals over $\{z_1<z_2\}$, $\{z_1>z_2\}$ and the
diagonal, all evaluated on the stable exponential scale
$t=e^{z^\beta}-1$ where $f_{Ch(\alpha)}(z)\,dz
=\alpha e^{-\alpha t}dt$. The frozen class offers \code{.moment},
\code{.mean}, \code{.var}, \code{.cov} and
\code{.min\_dist}.""",
    usage="""bvch_moment(r, s, alpha1, alpha2, alpha3, beta, tol=1e-08)

BvChen(r=...).moment(r, s)      # frozen-class equivalents
BvChen(...).mean()              # (E[Z1], E[Z2])
BvChen(...).var()               # variances
BvChen(...).cov()               # 2x2 covariance matrix
BvChen(...).min_dist()          # min(Z1, Z2) ~ Ch(a1+a2+a3, beta)""",
    arguments=[
        ("r, s", "non-negative orders of $Z_1$ and $Z_2$."),
        ("alpha1, alpha2, alpha3, beta", "parameters (all positive)."),
        ("tol", "absolute/relative tolerance for the quadratures."),
    ],
    details=r"""With $r=s=0$ the function returns one and reproduces
the mixture masses: $P(Z_1<Z_2)=\alpha_1/A$,
$P(Z_1>Z_2)=\alpha_2/A$, $P(Z_1=Z_2)=\alpha_3/A$,
$A=\alpha_1+\alpha_2+\alpha_3$.""",
    value=[("bvch\\_moment", "float, $E[Z_1^r Z_2^s]$."),
           ("min\\_dist", "dict with keys \\code{alpha}, \\code{beta} "
                          "and callables \\code{d}, \\code{p}, \\code{q}.")],
    example="""from bvchen import bvch_moment, BvChen

m00 = bvch_moment(0, 0, 0.8, 1.2, 0.5, 1.5)
round(m00, 12)                    # total probability = 1

dist = BvChen(0.8, 1.2, 0.5, 1.5)
dist.mean()
dist.cov()
dist.min_dist()["p"](1.0)         # P(min(Z1,Z2) <= 1)""",
)


add(
    names="loglik\\_bvch profile\\_beta",
    title="Log-Likelihood and Profile Likelihood of the Bivariate Chen Model",
    aliases="loglik\\_bvch, profile\\_beta",
    description=r"""Observed-data log-likelihood
(\code{loglik\\_bvch}) and profile likelihood over the power parameter
$\beta$ (\code{profile\\_beta}); at each grid point the three
$\alpha$'s are re-maximised with warm starts guarded by cold
restarts.""",
    usage="""loglik_bvch(z1, z2, alpha1, alpha2, alpha3, beta)

profile_beta(z1, z2, betas=None, start=None)""",
    arguments=[
        ("z1, z2", "numeric arrays of paired observations."),
        ("alpha1, alpha2, alpha3, beta", "parameters (all positive)."),
        ("betas",
         "optional candidate grid; default is a 25-point log-spaced "
         "grid over $(0.05, 20)$."),
        ("start", "optional length-3 start vector for the alphas."),
    ],
    details=r"""Ties $z_{1i}=z_{2i}$ contribute the log of the
singular diagonal density; off-diagonal points contribute the sum of
two Chen log-densities with summed scale parameters.""",
    value=[
        ("loglik\\_bvch", "float, the observed log-likelihood."),
        ("profile\\_beta",
         "dict with keys \\code{grid} (arrays of beta values and profile "
         "log-likelihoods) and \\code{beta\\_hat}."),
    ],
    example="""from bvchen import rbvch, loglik_bvch, profile_beta

sim = rbvch(300, 1, 1, 1, 1.5, random_state=7)
loglik_bvch(sim[:, 0], sim[:, 1], 1, 1, 1, 1.5)

pb = profile_beta(sim[:, 0], sim[:, 1],
                  betas=np.linspace(0.8, 2.4, 9))
round(pb["beta_hat"], 3)""",
)
add(
    names="fit\\_bvchen BvChenFit",
    title="EM Estimation of the Bivariate Chen Distribution",
    aliases="fit\\_bvchen, BvChenFit, BvChenFit.coefficients",
    description=r"""Fits the four-parameter bivariate Chen law
$(\alpha_1,\alpha_2,\alpha_3,\beta)$ to paired data with the EM
algorithm. Each iteration re-estimates the $\alpha$'s at the current
$\beta$ (numerical M-step) and then updates $\beta$ by
one-dimensional maximisation of the observed log-likelihood. An
observed tie $z_{1i}=z_{2i}$ identifies the common-shock component
exactly.""",
    usage="""fit_bvchen(z1, z2, start=None, tol=1e-08,
           maxit=200, verbose=False)""",
    arguments=[
        ("z1, z2", "paired non-negative observations."),
        ("start",
         "optional dict or sequence with starting values; defaults are "
         "derived from univariate marginal fits."),
        ("tol", "relative tolerance on the log-likelihood increase."),
        ("maxit", "maximum number of EM iterations."),
        ("verbose", "bool; print the iteration history."),
    ],
    details=r"""Returns a \\code{BvChenFit} object whose fields include
\\code{coefficients} $(\alpha_1,\alpha_2,\alpha_3,\beta)$,
\\code{loglik}, \\code{iterations}, \\code{converged}, a Monte-Carlo
Kendall's \\code{tau}, the sample size \\code{n}, the number of ties
\\code{n\\_ties} and the iteration \\code{history}. The estimator is
invariant to its starting values.""",
    value=[("fit\\_bvchen", "a fitted \\code{BvChenFit} instance.")],
    references=r"""Kundu, D., Dey, A.~K. (2009). Estimating the
parameters of the Marshall--Olkin bivariate Weibull distribution by EM
algorithm. \emph{Comput.\ Stat.\ Data Anal.}\ 53, 956--965.""",
    example="""from bvchen import rbvch, fit_bvchen

sim = rbvch(400, 0.8, 1.2, 0.5, 1.5, random_state=2024)
fit = fit_bvchen(sim[:, 0], sim[:, 1])
fit.coefficients.round(3)
(fit.converged, fit.iterations)
round(fit.tau, 4)
print(fit)""",
)


add(
    names="gof qqplot",
    title="Goodness-of-Fit Tests and Q-Q Diagnostics",
    aliases="gof, qqplot",
    description=r"""Kolmogorov-Smirnov goodness-of-fit tests for the
two marginals and for the minimum $\min(Z_1,Z_2)$, following the
testing strategy of Meintanis (2007), together with theoretical Q-Q
panels against their Chen quantiles.""",
    usage="""gof(z1, z2, alpha1, alpha2, alpha3, beta)

qqplot(z1, z2, alpha1, alpha2, alpha3, beta, ax=None)""",
    arguments=[
        ("z1, z2", "paired non-negative observations."),
        ("alpha1, alpha2, alpha3, beta", "fitted parameter values."),
        ("ax", "optional matplotlib axes (three panels)."),
    ],
    details=r"""\\code{gof} returns three scipy Kstest results under
the keys \\code{"ks\\_z1"}, \\code{"ks\\_z2"} and
\\code{"ks\\_min"}. \\code{qqplot} draws sample versus theoretical
quantiles for $Z_1$, $Z_2$ and the minimum.""",
    value=[
        ("gof", "dict of KstestResult objects."),
        ("qqplot", "array of matplotlib Axes."),
    ],
    references=r"""Meintanis, S.~G. (2007). Test of fit for
Marshall--Olkin distributions with applications.
\emph{J.~Statist.~Comput.~Simul.}\ 77, 171--179.""",
    example="""from bvchen import rbvch, gof, qqplot

sim = rbvch(400, 1, 1, 1, 1.5, random_state=5)
g = gof(sim[:, 0], sim[:, 1], 1, 1, 1, 1.5)
[(k, round(v.pvalue, 3)) for k, v in g.items()]

ax = qqplot(sim[:, 0], sim[:, 1], 1, 1, 1, 1.5)
ax[0].figure.savefig("fig-qq.png", dpi=110)
"figure written to fig-qq.png\"""",
)
add(
    names="surface\\_plot contour\\_plot",
    title="Plots for the Bivariate Chen Distribution",
    aliases="surface\\_plot, contour\\_plot",
    description=r"""Perspective surface and contour plot of the
(log) absolutely continuous part of the joint density on a grid,
reproducing Fig.~1 of the paper.""",
    usage="""surface_plot(alpha1, alpha2, alpha3, beta, xmax=3.0,
             n=120, log=True, ax=None)

contour_plot(alpha1, alpha2, alpha3, beta, xmax=3.0,
             n=120, log=True, levels=12, ax=None)""",
    arguments=[
        ("alpha1, alpha2, alpha3, beta", "parameters (all positive)."),
        ("xmax", "upper limit of both axes."),
        ("n", "grid size per axis."),
        ("log", "bool; plot $\\log f$ instead of $f$."),
        ("levels", "number of contour levels."),
        ("ax", "optional matplotlib axes."),
    ],
    details=r"""The diagonal carries the singular component; the
plotted surface shows only the absolutely continuous part
(\\code{component="ac"}). Requires the optional dependency matplotlib
(\\code{pip install bvchen[plot]}).""",
    value=[("surface\\_plot", "matplotlib 3-D axes object."),
           ("contour\\_plot", "matplotlib axes object.")],
    example="""from bvchen.plotting import surface_plot, contour_plot

ax = surface_plot(1.5, 2.5, 0.5, 0.8, xmax=2.0, n=80)
ax.figure.savefig("fig-surface.png", dpi=110)
axc = contour_plot(1.5, 2.5, 0.5, 0.8, xmax=2.0, n=80)
axc.figure.savefig("fig-contour.png", dpi=110)
"figures written to fig-surface.png and fig-contour.png\"""",
)

add(
    names="BvChen",
    title="The Frozen Bivariate Chen Distribution Object",
    aliases="bvchen.BvChen",
    description=r"""A frozen distribution holding fixed parameters
$(\alpha_1,\alpha_2,\alpha_3,\beta)$ and exposing every quantity of
the law as a method: \\code{pdf}/\\code{logpdf}, \\code{cdf},
\\code{sf}, \\code{rvs}, marginal densities/CDFs, conditional
densities/hazards, \\code{copula}, \\code{kendall\\_tau},
\\code{moment}/\\code{mean}/\\code{var}/\\code{cov} and
\\code{min\\_dist}.""",
    usage="""from bvchen import BvChen

dist = BvChen(alpha1, alpha2, alpha3, beta)
dist.pdf(x, y, log=False, component="full")
dist.cdf(x, y); dist.sf(x, y)
dist.rvs(n=1, random_state=None)
dist.marginal1_pdf(x); dist.marginal2_cdf(q)
dist.conditional1_pdf(z1, y); dist.conditional2_hazard(z2, x)
dist.copula(u, v); dist.kendall_tau(n=50000, random_state=None)
dist.mean(); dist.var(); dist.cov(); dist.min_dist()""",
    arguments=[
        ("alpha1, alpha2, alpha3, beta",
         "parameters of the BvCh law; all positive. \\code{alpha3} is "
         "the common-shock parameter driving the diagonal ties."),
    ],
    details=r"""Attribute \\code{alpha} aliases \\code{alpha3};
attribute \\code{params} returns the parameter tuple. The class is a
thin, stateless wrapper around the module-level functions shown in the
previous sections.""",
    value=[("BvChen", "frozen distribution object.")],
    example="""import bvchen

dist = bvchen.BvChen(alpha1=0.8, alpha2=1.2, alpha3=0.5, beta=1.5)
print(dist)
dist.pdf(0.5, 0.7)
dist.cdf(1.0, 1.5); dist.sf(1.0, 1.5)

sim = dist.rvs(200, random_state=42)
sim[:3]""",
)
# ---------------------------------------------------------------------
# Assemble the master .tex and compile with pdflatex -------------------
# ---------------------------------------------------------------------
FIGURES = r"""
\vspace{0.8em}
\noindent{\bfseries Resulting figures:}
\begin{center}
\includegraphics[width=0.98\textwidth]{fig-qq.png}

\vspace{0.6em}
\includegraphics[width=0.495\textwidth]{fig-surface.png}\hfill
\includegraphics[width=0.495\textwidth]{fig-contour.png}
\end{center}
"""

REFERENCES_TEX = r"""
\newpage
% ======================= REFERENCES ================================
\section{References}
\noindent Chen Z (2000) A new two-parameter lifetime distribution with
bathtub shape or increasing failure rate function. Stat Probab Lett
49(2):155--161\\[0.35em]
\noindent Gupta PK, Pundir PS, Sharma VK, Mesfioui M (2022) Bivariate
extension of bathtub-shaped distribution. Life Cycle Reliab Saf Eng
11:247--259\\[0.35em]
\noindent Kundu D, Dey AK (2009) Estimating the parameters of the
Marshall--Olkin bivariate Weibull distribution by EM algorithm. Comput
Stat Data Anal 53(4):956--965\\[0.35em]
\noindent Marshall AW, Olkin I (1967) A multivariate exponential
distribution. J Am Stat Assoc 62:30--44\\[0.35em]
\noindent Meintanis SG (2007) Test of fit for Marshall--Olkin
distributions with applications. J Stat Comput Simul 77:171--179
"""


def main() -> None:
    os.makedirs(OUTDIR, exist_ok=True)
    body = [PREAMBLE, THEORY]

    body.append(r"% ==================== FUNCTION DOCUMENTATION =======")
    body.append(r"\addcontentsline{toc}{section}{Function documentation}")
    body.append(r"\begin{center}{\Large\bfseries Function documentation}"
                r"\end{center}")
    body.append(r"\vspace{0.5em}")

    plot_idx = next(i for i, s in enumerate(SECTIONS)
                    if s["names"].startswith("surface"))
    class_idx = len(SECTIONS) - 1
    for i, s in enumerate(SECTIONS):
        print(f"processing section {i + 1}/{len(SECTIONS)}: {s['title']}")
        body.append(doc_block(s))
        if i == plot_idx:
            body.append(FIGURES)
            # keep the class topic on a fresh page after the figures
            body.append(r"\newpage")

    body.append(REFERENCES_TEX)
    body.append(r"\end{document}")

    tex_path = os.path.join(OUTDIR, "bvchen-manual.tex")
    with open(tex_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(body))
    print("master tex written:", tex_path)

    # ---- compile: three pdflatex passes for the ToC -------------------
    res = 1
    for _pass in range(3):
        res = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode",
             "bvchen-manual.tex"],
            cwd=OUTDIR,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        ).returncode
    print("pdflatex exit:", res)

    pdf_src = os.path.join(OUTDIR, "bvchen-manual.pdf")
    pdf_dst = os.path.join(ROOT, "bvchen-manual.pdf")
    if os.path.exists(pdf_src):
        import shutil
        shutil.copyfile(pdf_src, pdf_dst)
        print("PDF written to:", pdf_dst)
    else:
        log = os.path.join(OUTDIR, "bvchen-manual.log")
        print("PDF FAILED - inspect", log)


if __name__ == "__main__":
    main()
