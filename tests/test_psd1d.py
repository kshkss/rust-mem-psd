import numpy as np
import mem_psd

def test_sgn():
    xs = np.linspace(-5, 4.99, 1000)
    ys = np.where(xs > 0, 1., -1.)

    model = mem_psd.psd1d(100, xs[1] -xs[0])
    model.fit(ys)

    q = 10**np.linspace(-5, 2, 100)
    f = model.predict(q)

