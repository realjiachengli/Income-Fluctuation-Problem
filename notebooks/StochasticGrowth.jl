# replicate QuantEcon Julia lecture: https://julia.quantecon.org/dynamic_programming/optgrowth.html
# solve a stochastic optimal growth model with value function iteration with linear interpolation


# ======================= Mathematical structure =======================
# The problem is to choose 0 ≤ ct ≤ yt to maximize 
#           E sum(β^t u(ct)) subject to yt+1 = f(yt - ct)ξt+1, with y0 as given.
# 
# ======================= Pseudo codes =========================
# 
# 
# 
# 

using LinearAlgebra, Statistics
using LaTeXStrings, Plots, Interpolations, NLsolve, Optim, Random, Parameters
using Optim: maximum, maximizer

