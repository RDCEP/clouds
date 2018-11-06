import numpy as np
from sporco.admm import bpdn
from sporco import util
from sporco import plot

plot.config_notebook_plotting()

N = 512  # Signal size
M = 4 * N  # Dictionary size
L = 32  # Number of non-zero coefficients in generator
sigma = 0.5  # Noise level

# Construct random dictionary
np.random.seed(12345)
true_dict = np.random.randn(N, M)  # Random initialized sparse dict
#  and random sparse coefficients
x0 = np.zeros((M, 1))
si = np.random.permutation(list(range(0, M - 1)))
x0[si[0:L]] = np.random.randn(L, 1)

# Construct reference and noisy signal
true_signal = true_dict.dot(x0)
noisy_signal = true_signal + sigma * np.random.randn(N, 1)


L_sc = 9.e2
opt = bpdn.BPDN.Options(
    {
        "Verbose": False,
        "MaxMainIter": 500,
        "RelStopTol": 1e-3,
        "AutoRho": {"RsdlTarget": 1.0},
    }
)

import IPython

IPython.embed()

# Function computing reconstruction error at lmbda
def evalerr(prm):
    print(prm)
    lmbda = prm[0]
    b = bpdn.BPDN(true_dict, noisy_signal, lmbda, opt)
    x = b.solve()
    return np.sum(np.abs(x - x0))


# Parallel evalution of error function on lmbda grid
lrng = np.logspace(1, 2, 20)
# sprm ~ Sparse representation vs
# fvmx ~ Error vs lambda
# QUESTION
sprm, sfvl, fvmx, sidx = util.grid_search(evalerr, (lrng,))
best_lambda = sprm[0]

print("Minimum ℓ1 error: %5.2f at 𝜆 = %.2e" % (sfvl, best_lambda))

# Initialise and run BPDN object for best L1 penalty
opt["Verbose"] = True
b = bpdn.BPDN(true_dict, noisy_signal, best_lambda, opt)
x = b.solve()

print("BPDN solve time: %.2fs" % b.timer.elapsed("solve"))

plot.plot(
    np.hstack((x0, x)),
    title="Sparse representation",
    lgnd=["Reference", "Reconstructed"],
)
plot.savefig("bar")

its = b.getitstat()
fig = plot.figure(figsize=(15, 10))
plot.subplot(2, 2, 1)
plot.plot(fvmx, x=lrng, ptyp="semilogx", xlbl="$\lambda$", ylbl="Error", fig=fig)
plot.subplot(2, 2, 2)
plot.plot(its.ObjFun, xlbl="Iterations", ylbl="Functional", fig=fig)
plot.subplot(2, 2, 3)
plot.plot(
    np.vstack((its.PrimalRsdl, its.DualRsdl)).T,
    ptyp="semilogy",
    xlbl="Iterations",
    ylbl="Residual",
    lgnd=["Primal", "Dual"],
    fig=fig,
)
plot.subplot(2, 2, 4)
plot.plot(its.Rho, xlbl="Iterations", ylbl="Penalty Parameter", fig=fig)
fig.savefig("foo.png")
