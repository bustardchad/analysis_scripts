import yt
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

def doit(ds,power):

    # a FFT operates on uniformly gridded data.  We'll use the yt
    # covering grid for this.

    nx, ny, nz = [512,512,512]

    nindex_rho = 1./2.

    Kk = np.zeros( (nx//2+1, ny//2+1, nz//2+1))

    Kk = fft_comp(ds,power)

    # wavenumbers
    L = np.array([2.,2.,2.])
    print(np.fft.rfftfreq(nx))
    kx = np.fft.rfftfreq(nx)*nx/L[0]
    ky = np.fft.rfftfreq(ny)*ny/L[1]
    kz = np.fft.rfftfreq(nz)*nz/L[2]

    dims = np.array([nx,ny,nz])

    # physical limits to the wavenumbers
    kmin = np.min(1.0/L)
    kmax = np.min(0.5*dims/L)

    kbins = np.arange(kmin, kmax, kmin)
    N = len(kbins)

    # bin the Fourier KE into radial kbins
    kx3d, ky3d, kz3d = np.meshgrid(kx, ky, kz, indexing="ij")
    k = np.sqrt(kx3d**2 + ky3d**2 + kz3d**2)

    whichbin = np.digitize(k.flat, kbins)
    ncount = np.bincount(whichbin)
    E_spectrum = np.zeros(len(ncount)-1)

    for n in range(0,len(ncount)-1):
        E_spectrum[n] = np.sum(Kk.flat[whichbin==n])

    k = kbins[0:N]
    print(k.shape)
    print(1/(k*0.001))
    E_spectrum = E_spectrum[0:N]
    print(E_spectrum.shape)


    index = np.argmax(E_spectrum)
    kmax = k[index]
    print("Wavelength with highest energy density (in kpc): ")
    print(1.0/(kmax))
    Emax = E_spectrum[index]
    print("Emax: ")
    print(Emax)


    plt.loglog((k), E_spectrum, 'bo',label='Simulation')
   # plt.ylim(1E-6,1E-3)
   # plt.xlim(1E0,3E1)
   # plt.xlim(3E-2,3E0)
    plt.xlabel(r"k",fontsize=18)
    plt.ylabel(r"E(k)",fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(loc="lower right")
    plt.tight_layout()
    #plt.title(r"$t/ \tau_{\rm eddy}$ = )
    plt.savefig('test_spectrum_gaussian_power1.png')
    plt.close()
    print("k in function:")
    print(k)

    print("Power spectrum: ")
    print(E_spectrum)
    return k, E_spectrum



# alternative way
def doit_alt(image,power):

    # a FFT operates on uniformly gridded data.  We'll use the yt
    # covering grid for this.
    npix = image.shape[0]

    fourier_image = np.fft.fftn(image**power)
    fourier_amplitudes = np.abs(fourier_image)**2

    kfreq = np.fft.fftfreq(npix) * npix
    kfreq3D = np.meshgrid(kfreq, kfreq, kfreq)
    knrm = np.sqrt(kfreq3D[0]**2 + kfreq3D[1]**2 + kfreq3D[2]**2)

    knrm = knrm.flatten()
    fourier_amplitudes = fourier_amplitudes.flatten()

    kbins = np.arange(0.5, npix//2+1, 1.)
    k = 0.5 * (kbins[1:] + kbins[:-1])
    Abins, _, _ = stats.binned_statistic(knrm, fourier_amplitudes,
                                     statistic = "mean",
                                     bins = kbins)
#    Abins *= 4.*np.pi/3. * (kbins[1:]**3 - kbins[:-1]**3)


    plt.loglog((k), Abins, 'bo',label='Simulation')
   # plt.ylim(1E-6,1E-3)
   # plt.xlim(1E0,3E1)
   # plt.xlim(3E-2,3E0)
    plt.xlabel(r"k",fontsize=18)
    plt.ylabel(r"P(k)",fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(loc="lower right")
    plt.tight_layout()
    #plt.title(r"$t/ \tau_{\rm eddy}$ = )
    plt.savefig('test_spectrum_alternative_gaussian_power1_noVolume.png')
    plt.close()
    print("k in function:")
    print(k)

    print("Power spectrum: ")
    print(Abins)
    return k, Abins


def fft_comp(u,power):

    nx, ny, nz = u.shape
    # the first half of the axes -- that's what we keep.  Our
    # normalization has an '8' to account for this clipping to one
    # octant.
    ru = np.fft.fftn(u**power)[0:nx//2+1,0:ny//2+1,0:nz//2+1]
    ru = 8.0*ru/(nx*ny*nz)


    return np.abs(ru)**2  # rho v^2

power = 1.0
nx, ny, nz = [512,512,512]
#ds = np.random.rand(nx,ny,nz)
ds = np.random.normal(size=[nx,ny,nz])
plt.imsave('whitenoise.png',ds[1,:])
doit(ds,power)
doit_alt(ds,power)
