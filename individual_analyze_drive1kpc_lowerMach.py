import numpy as np
import matplotlib.pyplot as plt
# Athena++ history data


#dir = './crbeta1/diff3e27_higherMach_res256/crbeta10000/'
#dir = './crbeta1/diff3e27_higherMach_res256/crbeta001/'
#dir = './diff3e27_streaming_higherMach_true/beta0125/Mach05/res128/crbeta1/lowerBetaAgain/noDiff/vm80/'
#dir = './diff3e27_streaming_higherMach_true/beta0125/Mach05/res128/crbeta1/lowerBetaAgain/beta1_sanityCheck/'
#dir = './diff3e27_higherMach_true/vm5_res256/'
#dir = './crbeta1/diff3e27_higherMach_res256/crbeta1/corrected/res128_beta10/'
#dir = './crbeta1/diff3e27_higherMach_res256/crbeta1/corrected/'
#dir = './crbeta001/sweetspot_lowerMach/'
#dir = './crbeta1/corrected/res128_lowerMach/'
#dir = './crbeta1/corrected/'
#dir = './crbeta1/corrected/streaming/Mach015/iso_justStream/beta100/'
#dir = './crbeta1/corrected/streaming/Mach015/res128/iso_justStream/beta100/'
dir = './crbeta1/corrected/streaming/Mach015/iso_justStream/beta100/'
#dir = './crbeta1/corrected/streaming/Mach015/iso_justStream/beta100/'

#dir = './crbeta1/diff3e27_higherMach_res256/crbeta001/res128_beta10/vm20/'
#dir = './crbeta1/diff3e27_higherMach_res256/crbeta001/res128_beta10/higherKappa/higherKappaAgain/'
#dir = './crbeta1/diff3e27_higherMach_res256/crbeta001/res128_beta10/vm20/'
#dir = './diff3e27_streaming_higherMach_true/beta0125/Mach05/res128/crbeta1/lowerBetaAgain/vm20_hlle/'
time, mass, ke1, ke2, ke3, me1, me2, me3, ec = np.loadtxt(dir+'cr.hst',skiprows = 2, usecols = (0,2,6,7,8,9,10,11,12),unpack=True)


vval = []
csval = []


edenstocgs = 6.54e-11
#cellVol = (250*3.0856e18)**3.0
cellVol1 = (2.0)**3.0
ketot = []
metot = []
thermaletot = []
ectot = []

for j in range(0,len(time),1):
  keval = (ke1[j] + ke2[j] + ke3[j])*edenstocgs/cellVol1
  ketot.append(keval)
  meval = (me1[j] + me2[j] + me3[j])*edenstocgs/cellVol1
  cs2 = (.11*1e8)**2.0
  csval.append(np.sqrt(cs2))
  ecval = ec[j]*edenstocgs/cellVol1
  vval.append((np.sqrt(2*(ke1[j] + ke2[j] + ke3[j])/(mass[j])) * 1.e8))
  metot.append(meval)
  ectot.append(ecval)

print(len(ectot))
# Change these to get slope between different start and end points
#start = 60000
#end = 100000
start = 200
end = 300


p0 = np.polyfit(time[start:end],np.log10(np.array(ectot[start:end])),1)
growthTime_log = (1/(p0[0]*np.log(10)))
predicted = 300


CRPresGasPres = np.array(ectot)*0.333/(1.67e-28*1e14)
#timeeddy = np.array(time)*3.155e13/(3.0856e21/7.e6)
timeeddy = np.array(time)*3.155e13/(0.667*3.0856e21/5.e6)


# v_ph ~ c_s = 1e7 for CGM
print("L*v_ph = {}".format(3.0856e21*1e7))

#print("Diff = LVph: Predicted growth time (Myrs) = {}".format(0.0))
print("Diff = Lvph: Simulation growth time (Myrs) = {}".format(growthTime_log) )


plt.semilogy(timeeddy,ketot,linewidth=1)
plt.xlabel(r'$t/ \tau_{eddy}$',fontsize=18)
plt.ylabel('KE Energy Density',fontsize=18)
plt.legend()
plt.tight_layout()
plt.savefig(dir+'KE_EnergyDensity_Compare.pdf')
plt.close()

plt.semilogy(timeeddy,vval,linewidth=1)
plt.xlabel(r'$t/ \tau_{eddy}$',fontsize=18)
plt.ylabel('Average Velocity (cm/s)',fontsize=18)
plt.legend()
plt.tight_layout()
plt.savefig(dir+'Velocity_Compare.pdf')
plt.close()

plt.semilogy(timeeddy,CRPresGasPres,linewidth=2)
plt.xlabel(r'$t/ \tau_{eddy}$',fontsize=18)
plt.ylabel(r'$P_{CR}/P_{g}$',fontsize=18)
#plt.legend()
plt.tight_layout()
plt.savefig(dir+'CRPresGasPres.pdf')
plt.close()


plt.semilogy(timeeddy,ectot,linewidth=1)
plt.xlabel(r'$t/ \tau_{eddy}$',fontsize=18)
plt.ylabel('CR Energy Density',fontsize=18)
#plt.legend()
plt.tight_layout()
plt.savefig(dir+'CR_EnergyDensity_Compare_semilog.pdf')
plt.close()

plt.semilogy(timeeddy,metot,linewidth=1)
plt.xlabel(r'$t/ \tau_{eddy}$',fontsize=18)
plt.ylabel('Magnetic Energy Density',fontsize=18)
plt.legend()
plt.tight_layout()
plt.savefig(dir+'Magnetic_EnergyDensity_Compare.pdf')
plt.close()

def get_deriv(ykey,xkey,smooth = None):
#def get_deriv(hst, ykey, xkey = 'time', smooth = None):
    """Return derivative of `ykey` wrt to `xkey`
     while keeping the lenght constant.
      Keywords:
     * hst           -- HST numpy array
       * ykey          -- other y values or key to take the derivate
       * xkey          -- key
       * smooth        -- smooth over length in units of `xkey`
     """
    y = ykey
    x = xkey
    # x = hst[xkey]
    # if isinstance(ykey, (str, basestring)):
    #     y = hst[ykey]
    # else:
    #     assert len(ykey) == len(x), "`key` has to be string or array of length `x`"
    #     y = ykey
    dx = (x[1:] - x[:-1])
    dydx = (y[1:] - y[:-1]) / dx
    if smooth is not None:
        if isinstance(smooth, int):
            n = smooth
        else:
            n = np.round(smooth / np.median(dx))
        dydx = np.convolve(dydx, np.ones(int(n)) / n, mode='same')

    r = np.interp(x, 0.5 * (x[1:] + x[:-1]), dydx)
    return r



smooth = 300

# new -- total energy
plt.semilogy(timeeddy,np.array(metot) + np.array(ectot) + np.array(ketot),linewidth=1)
plt.xlabel(r'$t/ \tau_{eddy}$',fontsize=18)
plt.ylabel('Total Energy Density',fontsize=18)
plt.legend()
plt.tight_layout()
plt.savefig(dir+'Total_EnergyDensity_Compare.pdf')
plt.close()

totE_codeunits = (np.array(metot) + np.array(ectot) + np.array(ketot))/edenstocgs

# new -- total energy
print("dedt: ")
print(get_deriv(totE_codeunits,time,smooth=smooth))
plt.semilogy(time,totE_codeunits,linewidth=1)
plt.xlabel(r't',fontsize=18)
plt.ylabel('Total Energy Density',fontsize=18)
plt.legend()
plt.tight_layout()
plt.savefig(dir+'Total_EnergyDensity_Compare_codeunits.pdf')
plt.close()


rhov3L = .03066*((np.array(vval)/1e8)**3.0)/(2.0*0.667)

#plt.plot(time,.03066*((np.array(vval)/1e8)**3.0)/0.667,'k-.',label=r"$\rho v^{3} / L$")
plt.plot(time,rhov3L,'k-.',label=r"$\rho v^{3} / 2L$")
plt.axhline(y=3e-6/cellVol1, color='r', linestyle='-',label="Driving dE/dt")
plt.semilogy(time,get_deriv(totE_codeunits,time,smooth=smooth),linewidth=1,label="Total dE/dt")
plt.semilogy(time,get_deriv(np.array(ectot)/edenstocgs,time,smooth=smooth),linewidth=1,label=r"dE$_{CR}$/dt")
plt.semilogy(time,get_deriv(np.array(ketot)/edenstocgs,time,smooth=smooth),linewidth=1,label=r"dE$_{k}$/dt")
plt.semilogy(time,get_deriv(np.array(metot)/edenstocgs,time,smooth=smooth),linewidth=1,label=r"dE$_{B}$/dt")
plt.ylim(1e-8,1e-6)
plt.xlabel(r't',fontsize=18)
plt.ylabel('dE/dt',fontsize=18)
plt.title(r"$P_{CR}/P_{g} \sim 1$",fontsize=22)
plt.legend()
plt.tight_layout()
plt.savefig(dir+'dEdt_codeunits.pdf')
plt.close()

plt.plot(time,rhov3L,'k-.',label=r"$\rho v^{3} / 2L$")
plt.axhline(y=3e-6/cellVol1, color='r', linestyle='-',label="Driving dE/dt")
plt.plot(time,get_deriv(totE_codeunits,time,smooth=smooth),linewidth=1,label="Total dE/dt")
plt.plot(time,get_deriv(np.array(ectot)/edenstocgs,time,smooth=smooth),linewidth=1,label=r"dE$_{CR}$/dt")
plt.semilogy(time,get_deriv(np.array(ketot)/edenstocgs,time,smooth=smooth),linewidth=1,label=r"dE$_{k}$/dt")
plt.plot(time,get_deriv(np.array(metot)/edenstocgs,time,smooth=smooth),linewidth=1,label=r"dE$_{B}$/dt")
plt.ylim(1e-8,1e-6)
plt.xlabel(r't',fontsize=18)
plt.ylabel('dE/dt',fontsize=18)
plt.title(r"$P_{CR}/P_{g} \sim 1$",fontsize=22)
plt.legend()
plt.tight_layout()
plt.savefig(dir+'dEdt_codeunits_nolog.pdf')
plt.close()

ecderiv = get_deriv(np.array(ectot)/edenstocgs,time,smooth=smooth)
totederiv = get_deriv(np.array(totE_codeunits),time,smooth=smooth)
print("f_cr : ")
print(np.mean(ecderiv[int(len(ecderiv)/2):len(ecderiv)-1])/(3e-6/cellVol1))
print("f_E : ")
print(np.mean(rhov3L[int(len(ecderiv)/2):len(ecderiv)-1])/(3e-6/cellVol1))
print("f_tot (B + CR + KE) : ")
print(np.mean(totederiv[int(len(totederiv)/2):len(totederiv)-1])/(3e-6/cellVol1))
