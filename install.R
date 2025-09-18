# install.R
options(repos = c(CRAN = "https://cloud.r-project.org"))
pkgs <- c("jsonlite","dbarts","BART","pROC")
to_install <- setdiff(pkgs, rownames(installed.packages()))
if (length(to_install)) install.packages(to_install, Ncpus = 2)
