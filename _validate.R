## Validation script for BvChen (run with Rscript)
pk <- "C:/Users/Admin/.cline/data/workspaces/chat/BvChen"
src <- list.files(file.path(pk, "R"), full.names = TRUE)
for (f in src) sys.source(f, envir = globalenv())
cat("all R files sourced OK\n")

ok <- function(label, cond) cat(if (isTRUE(cond)) "PASS " else "FAIL ", label, "\n", sep = "")

# 1) univariate consistency
ok("dchen integrates to 1",
   abs(integrate(function(z) dchen(z, 1.7, 1.3), 0, Inf)$value - 1) < 1e-6)
p <- c(.2, .5, .9)
ok("qchen inverts pchen", max(abs(pchen(qchen(p, 1.7, 1.3), 1.7, 1.3) - p)) < 1e-10)

# 2) mixture weights
a1 <- .8; a2 <- 1.1; a3 <- .6; b <- 1.4
m_diag <- integrate(function(z) dbvch(z, z, a1, a2, a3, b, component = "sing"), 0, Inf)$value
ok("diagonal mass = a3/(sum)", abs(m_diag - a3/(a1+a2+a3)) < 1e-4)

# 3) cdf/survival identity
z1 <- 1; z2 <- 2
lhs <- pbvch(z1, z2, a1, a2, a3, b)
rhs <- 1 - schen(z1, a1+a3, b) - schen(z2, a2+a3, b) + sbvch(z1, z2, a1, a2, a3, b)
ok("pbvch = 1-S1-S2+S", abs(lhs-rhs) < 1e-10)
ok("boundaries", sbvch(0,0,a1,a2,a3,b)==1 && sbvch(Inf,Inf,a1,a2,a3,b) < 1e-10 && pbvch(Inf,Inf,a1,a2,a3,b)==1)

# 4) conditional density normalisation
set.seed(1)
y <- 1
int_below <- integrate(function(x) dcond1(x, y, a1, a2, a3, b), 0, y)$value
int_above <- integrate(function(x) dcond1(x, y, a1, a2, a3, b), y, Inf)$value
atom <- attr(dcond1(y, y, a1, a2, a3, b), "atom")
tot <- int_below + int_above + atom
ok(paste0("dcond1 normalises (total = ", round(tot, 5), ")"), abs(tot - 1) < 1e-4)

# 5) simulation: tie rate + KS marginals
set.seed(42)
sim <- rbvch(3000, 1, 1.5, .7, 1.4)
tie <- mean(sim[, "z1"] == sim[, "z2"])
ok(paste0("tie rate ~ a3/sum (got ", round(tie, 4), ")"), abs(tie - .7/3.2) < .03)
ks <- suppressWarnings(ks.test(sim[, "z1"], function(q) pmarg1(q, 1, 1.5, .7, 1.4)))
ok(paste0("KS marginal z1 p = ", signif(ks$p.value, 3)), ks$p.value > .01)
ks2 <- suppressWarnings(ks.test(pmin(sim[,1], sim[,2]), min_bvch(1, 1.5, .7, 1.4)$p))
ok(paste0("KS min p = ", signif(ks2$p.value, 3)), ks2$p.value > .01)

# 6) copula
ok("copula corners", survcopbvch(1,1,1,1,1)==1 && survcopbvch(0,.5,1,1,1)==0)
tau <- tau_bvch(1, 1, .01, 1.4, n = 5000, seed = 9)
taub <- tau_bvch(1, 1, 10, 1.4, n = 5000, seed = 9)
ok(paste0("tau increasing in a3 (", round(tau,3), " < ", round(taub,3), ")"),
   taub > tau)

# 7) moments vs simulation
set.seed(3)
mm <- mean(rchen(200000, 2, 1.5))
ok(paste0("mchen mean vs MC (series ", round(mchen(1,2,1.5),4),
          " vs MC ", round(mm,4), ")"),
   abs(mchen(1, 2, 1.5) - mm)/mm < .02)

# 8) EM fit
set.seed(2024)
sim <- rbvch(600, .8, 1.2, .5, 1.5)
f1 <- fitbvch(sim[, "z1"], sim[, "z2"])
f2 <- fitbvch(sim[, "z1"], sim[, "z2"],
              start = list(alpha1=3, alpha2=.3, alpha3=2, beta=1))
print(f1)
cf <- coef(f1)
ok(paste0("beta recovered: ", round(cf["beta"], 3)), abs(cf["beta"]-1.5) < .25)
ok("start-invariant", max(abs(coef(f1)-coef(f2))) < 1e-3)
print(profile_beta(sim[, "z1"], sim[, "z2"])$beta_hat)

# 9) plots run headless
pdf(NULL)
g <- plot.bvch.bvch(sim, 1, 1.5, .7, 1.4, to = 2.5)
persp.bvch(sim, 1, 1.5, .7, 1.4, to = 2)
qqbvch(sim[, "z1"], sim[, "z2"], 1, 1.5, .7, 1.4)
dev.off()
ok("plots run headless", !is.null(g))

cat("\nVALIDATION DONE\n")
