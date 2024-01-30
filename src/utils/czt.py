import torch
from torch.fft import fft, ifft


def variable_check(N, M, w, a):
    """check the validity of the input variables
    1. The CZT is performed on 1D signals. Images are processed by applying CZT along each dimension sequentially.
    Make sure to squeeze slices of images to remove empty dimensions.
    2. The number of points m for the output signal is set as the input one n if it's not specified.
    3. The complex ratio w between points and the complex starting point a are set to the ones for DFT if not specified.

    Parameters
    ----------
    N: int
        number of points for the input signal
    M: int
        number of points for the output signal
    w: torch tensor
        angle step on the complex exponential ratio
    a: complex torch Tensor
        starting point

    Returns
    -------
    N, M, a
    """
    if N is None:
        raise NotImplementedError('The size of the input signal need to be specified.')
    if M is None:
        M = N
        print('The number of points of the output m is not specified, setting it same as the input.')
    if w is None:
        w = torch.exp(torch.tensor(-1j * 2 * torch.pi / N))
        print(
            'The complex ratio w between sampling points is not specified, setting it to the DFT one: exp(-1j*2pi/N).')
    if a is None:
        a = torch.tensor(1 + 0j)
        print('The starting point a is not specified, setting it to 1.')
    return M, w, a


def number_fft_points(n, m):
    """calculate the optimal number for the ffts in the Z-transform
    The ffts in the CZT requires (n + m -1) points. For fastest performance of ffts, we need to find the closest number
    in the shape of 2-power to n + m - 1.

    Parameters
    ----------
    n: int
        number of sampling points of the input signal
    m: int
        number of sampling points of the output signal

    Returns
    -------
    n_fft: int
        optimal number of points
    """
    n_fft = int(2 ** torch.ceil(torch.log2(torch.tensor(n + m - 1))))
    return n_fft


class Czt1d:
    """function to perform the chirp Z transform (CZT) on a 1D signal
        The definition of CZT is:
        X_k = sum_{n=0}^{N-1} x_n * a^{-n} * w^{nk}. k = 0, 1, ..., M-1.
        By the relation nk = n^2/2 + k^2/2 - (k-n)^2/2, the above definition can be equivalently written as a convolution:
        X_k = w^{k**2/2} (x_n a^{-n} w^{n**2/2}) (*) (w^{-n**2/2}),
        which can be efficiently computed using (N + M -1)-point FFTs:
        X_k = w^{k**2/2} FFT^{-1} (FFT(x_n a^{-n} w^{n**2/2}) * FFT(w^{-n**2/2})).

        DFT is a special case of CZT with w=exp(-2pi*i/N), a=1 and M=N.
        iDFT is a special case too: w=exp(2pi*i/M), a=1 and M=N, with an extra normalization factor 1/M.

        The main advantages of replacing DFT with CZT is:
        1) the number of points for the input and output are allowed to be different;
        2) we can choose how to sample the points on the complex plane, e.g. on a spiral following exponential stepsizes
        starting from any point a, instead of just on the full unit circle.
        3) The cost of CZT is kept at O(Nlog(N)), N=max(N, M) through two FFTs and one iFFT.
        ----------
        Implementation details:
        1. our project only involves DFTs or iDFTs, so we restrict the CZT for these two cases by specifying 'mode' as 'dft'
        or 'idft'.
        2. we restrict the choice of the complex ratio w to only exponentials. w is the sampling stepsize on the spiral,
        e.g. w = exp(-2pi*i/N) => sampling on the unit circle => DFT.
        3. some implementation details are transferred from scipy/czt.py.
        (1) w^{k**2/2}, w^{n**2/2} and w^{-n**2/2} are based on one variable wk2, k = max(N, M) by slicing or taking reciprocal;
        (2) To assemble the (N+M-1)-point array for precomputing fft of w^{-n**2/2}, hstack is used:
        first n-1 entries are the second to (N-1)th entries of wk2 reversed;
        the remaining M entries are the first M entries of wk2.
        Then an n_fft point fft is used to get Fwk2 of length n_fft. In the end, the output of the iFFT is sliced to get an
        output of length M by taking the (N-1)th to (N+M-1)th entries represented by the variable idx.
        (3) AW represents FFT(a^{-n} w^{n**2/2}); W represents FFT(w^{-n**2/2}).

        Parameters
        ----------
        N: int
            number of points for the input signal
        M: int
            number of the points for the output
        w: torch.Tensor
            step size for the angle on the complex exponential (ratio between points)
        a: complex torch.Tensor
            starting point
        mode: str
            'dft': DFT
            'idft': iDFT

        -------
        Implementation based on scipy/czt.py:
        https://github.com/scipy/scipy/blob/v1.9.3/scipy/signal/_czt.py#L114-L270
        """

    def __init__(self, N, M, w, a, mode='dft', device='cpu'):
        M, w, a = variable_check(N, M, w, a)
        w, a = w.to(device), a.to(device)
        self.N = N
        self.mode = mode
        # calculate the optimal number of points: 2 power
        self._n_fft = number_fft_points(N, M)
        # define variable vector k that can represent two vectors n=torch.arange(N)=k[:N], k=torch.arange(M)=k[:M]
        k = torch.arange(max(N, M)).to(device)
        wk2 = w ** (k ** 2 / 2)
        # precompute a^{-n} * w^{n**2/2}
        self._AW = a ** (-k[:N]) * wk2[:N]
        # precompute FFT(1/w^{n**2/2}): n_fft-point fft on a (N+M-1)-length tensor
        self._W = fft(1 / torch.hstack((torch.flip(wk2[1:N], dims=(0,)), wk2[:M])), self._n_fft)
        # w^{k**2/2}
        self._wk2 = wk2[:M]
        # where the indices of the M-length output are
        self._idx = slice(N - 1, N + M - 1)

    def __call__(self, x):
        """Compute the CZT on a 1D signal x

        Parameters
        ----------
        x: torch.Tensor
            1D input signal of length N to be transformed

        Returns
        -------
        output: torch.Tensor
            transformed 1D signal of length M
        """
        if x.dim() != 1:
            raise ValueError(f'the dimension of the input signal {x} should be 1 but it is {x.dim()} instead.')

        output = ifft(fft(x * self._AW, n=self._n_fft) * self._W)
        output = self._wk2 * output[..., self._idx]

        if self.mode.lower() == 'idft':
            output /= self.N
        return output
