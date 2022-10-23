import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np


def add_legend(plot_args_lst, legend_loc = 'best', default_plot_args = {'color' : 'black'},
                 **kwargs):
      """Adds another legend to plot.
      Keyword Arguments:
      plot_args_lst      -- List of plotting arguments to show.
      legend_loc         -- Location of new legend (default 'best')
      default_plot_args  -- Arguments will be used for every item in new legend.
      kwargs             -- Will be passed to `plt.legend` of new legend.
      Example:
              add_legend([{'ls' : '-', 'label' : r'$\rho > \rho_{cl}/3$'}, {'ls' : '--', 'label' : r'$T   < 2 T_{cl}$'}],
                                  default_plot_args = {'c' : 'k'},
                                  title = r'')

      Will add a legend with two different lines (both black).
      """
      ax = plt.gca()
      leg = ax.get_legend()
      linelst = []
      for cargs in plot_args_lst:
          for k, v in default_plot_args.items():
              if k not in cargs:
                  cargs[k] = v
          l, = plt.plot(np.nan, **cargs)
          linelst.append(l)
      o = kwargs.copy()
      if 'loc' not in o:
          o['loc'] = legend_loc
      if 'fontsize' not in o:
          o['fontsize'] = 12
      if legend_loc == 'above':
          o['loc'] = 'lower center'
          o['bbox_to_anchor'] = (0.5, 1.01)
      plt.legend(handles = linelst, **o)
      if leg is not None:
          ax.add_artist(leg) # Add old legend





pcpg_iso_Mach05 = [1e-4,5e-2,0.2, 0.3, 1, 1.5, 10, 100]
pcpg_iso_Mach035 = [1.5,10,100]
pcpg_ad_Mach05 = [0.025,0.06,0.1,1/5,1/2,1,10,100]


fcr_iso_Mach05 = [9.6e-4,0.14, 0.58, 0.65, 0.8, 0.86, 0.89, 0.86]
fcr_iso_Mach035 = [0.83, 0.91, 0.85]
fcr_ad_Mach05 = [0.12,0.19,0.23,0.4,0.56, 0.68, 0.84, 0.79]
fg_ad_Mach05 = [0.78,0.7,0.62,0.47,0.3,0.21, 0.07, 0.14]

pcpg_advect_Mach05 = [1,10,100]
fcr_advect_Mach05 = [0.096,0.52,0.73]
fg_advect_Mach05 = [0.74,0.35,0.13]


pcpg_arr = np.arange(1e-4,2.0,0.0001)
fcr_expect = (2./3.)*(5e6/(1e7*np.sqrt(1.0 + pcpg_arr)))*(pcpg_arr*1e7**2.0)/(5e6**2.0)

fill_style='full'
ms = 12.0
#plt.semilogx(pcpg_iso_Mach05,fcr_iso_Mach05,'bo',label=r"$f_{CR}$, Isothermal, Diffusion Only",markersize=ms)
#plt.semilogx(pcpg_iso_Mach035,fcr_iso_Mach035,'bD',label=r"$f_{CR}$, Isothermal, $\mathcal{M} \sim 0.35$",markersize=ms)
plt.semilogx(pcpg_ad_Mach05,fcr_ad_Mach05,'ko-',label=r"$f_{CR}$",markersize=ms,fillstyle=fill_style)
plt.semilogx(pcpg_ad_Mach05,fg_ad_Mach05,'ro-',label=r"$f_{th}$",markersize=ms,fillstyle=fill_style)
plt.semilogx(pcpg_advect_Mach05,fcr_advect_Mach05,'kD-',markersize=ms,fillstyle=fill_style)
plt.semilogx(pcpg_advect_Mach05,fg_advect_Mach05,'rD-',markersize=ms,fillstyle=fill_style)
plt.semilogx(pcpg_arr,fcr_expect,'b-',label=r"$\frac{2}{3} \frac{M_{ph} P_{CR}}{\rho v^{2}}$",markersize=ms,fillstyle=fill_style)
plt.ylim(8e-3,1)
#plt.xlim(3E-2,2E0)
plt.xlim(1e-2,125)
#plt.xlabel(r"$\lambda,/,L$",fontsize=18)
#plt.ylabel(r"$f_{CR}$, $f_{th}$",fontsize=18)
plt.xlabel(r"$P_{CR}/P_{g}$",fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.title("Diffusion Only",fontsize=22)
plt.text(0.015,0.88,r"$\beta \sim 1$",fontsize=16,bbox=dict(edgecolor='black', alpha=0.1))
#plt.text(0.015,0.88,r"$\beta \sim 1$",fontsize=16,bbox)
#plt.legend(loc="upper left",ncol=1)
plt.legend(prop={'size': 12},ncol=2,bbox_to_anchor=(0.4, -0.12),title=r'$\bf{Color}$',frameon=False)
add_legend([{'marker' : 'o', 'label' : r'$\kappa \sim 0.15 L_{0}v_{ph}$'}, {'marker' : 'd', 'label' : r'$\kappa = 0$'}],
         default_plot_args = {'ls' : '', 'c' : 'k', 'markersize' : '12','fillstyle' : 'full'}, ncol=1,bbox_to_anchor=(1.02, -0.12),frameon=False,
                                   title = r'$\bf{Symbol}$')
plt.tight_layout()
plt.savefig("fcr_vs_pcpg_adiabatic.pdf")
plt.close()

# that plot is ugly...let's make it a stack plot instead
y_diff = np.vstack([fcr_ad_Mach05,fg_ad_Mach05])
y_nodiff = np.vstack([fcr_advect_Mach05,fg_advect_Mach05])

labels = [r"f$_{CR}$", r"f$_{th}$"]

fig, axs = plt.subplots(2,1,sharex=True,sharey=True)
axs[0].stackplot(pcpg_ad_Mach05,y_diff,labels=labels)
axs[0].plot(pcpg_arr,fcr_expect,'k--',label=r"$\frac{2}{3} \frac{M_{ph} P_{CR}}{\rho v^{2}}$")
plt.xscale('log')
axs[0].legend(loc='lower right',title=r"$\kappa_{||} \sim 0.15 L_0 c_s$")

axs[1].stackplot(pcpg_advect_Mach05,y_nodiff,labels=labels)
axs[1].set_xlim(min(pcpg_ad_Mach05),max(pcpg_ad_Mach05))
axs[1].set_ylim(0,1)
plt.xscale('log')
axs[1].legend(loc='lower right',title=r"$\kappa_{||} \sim 0$")
plt.tight_layout()
plt.savefig("fcr_vs_pcpg_adiabatic_stackplot.pdf")
plt.close()




# with streaming, changing beta
# pcpg = 1 in each case
#   "ad" denotes adiabatic sims
beta = [1.5,7,50]
#fcr_iso_stream_res128 = [0.06, 0.16, 0.29]
fcr_ad_stream_res128 = [0.04, 0.13, 0.26]
fg_ad_stream_res128 = [0.87, 0.76, 0.64]
Heps_res128 = [0.41,0.67,0.55]

# res = 512, isothermal, with streaming
Heps_iso_res512_lowerMach = [0.596, 0.696]  # for beta = 1, 10
Heps_iso_res512_Mach05 = [0.33] # for beta = 1

"""
fcr_iso_res512_lowerMach_stream = [..,0.0616] # beta = 1, 10
fcr_iso_res512_lowerMach_justDiff = [..,0.866] # beta = 1, 10
"""
Heps_iso_res256_lowerMach = [0.3788, 0.776, 0.723]

# are these adiabatic or isothermal?
Heps_res256 = [0.40, 0.63, 0.54] # the beta = 1 outcome is much lower than the elongated adiabatic runs, why??



#fcr_ad_stream_res128_elongated = [0.04,0.13,0.26]
fcr_ad_stream_res128_elongated = [0.052,0.17,0.268]
#fg_ad_stream_res128_elongated = [0.79,0.78,0.64]
fg_ad_stream_res128_elongated = [0.893,0.812,0.66]
Heps_res128_elongated = [0.71,0.67,0.55]

#vm = 10 instead of vm = 5
fcr_ad_stream_res128_elongated_vm10 = [0.06]
fg_ad_stream_res128_elongated_vm10 = [0.91]
Heps_res128_elongated_vm10 = [0.64]

fcr_ad_justStream_res128_elongated = [0.033, 0.094, 0.142]
fg_ad_justStream_res128_elongated = [0.908, 0.892,0.874]
Heps_justStream_res128_elongated = [0.7285,0.732,0.857]



plt.semilogx(beta,fcr_ad_stream_res128_elongated,'ko-',label=r"$f_{\rm CR}$",markersize=ms)
plt.semilogx(beta,fcr_ad_justStream_res128_elongated,'kd-',markersize=ms)
plt.semilogx(beta,np.array(fg_ad_stream_res128_elongated)-np.array(Heps_res128_elongated),'co-',label=r"$f_{\rm th} - f_{\rm CR,heating}$",markersize=ms)
plt.semilogx(beta,np.array(fg_ad_justStream_res128_elongated)-np.array(Heps_justStream_res128_elongated),'cd-',markersize=ms)
plt.semilogx(beta,fg_ad_stream_res128_elongated,'ro-',label=r"$f_{\rm th}$",markersize=ms)
plt.semilogx(beta,fg_ad_justStream_res128_elongated,'rd-',markersize=ms)
plt.ylim(0,1)
plt.xlim(1,60)
plt.xlabel(r"$\beta$",fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.text(1.5,0.5,r"$P_{CR}/P_{g} \sim 1$",fontsize=16,bbox=dict(edgecolor='black', alpha=0.1))
plt.title("With Streaming",fontsize=22)
#plt.legend(prop={'size': 12},ncol=3,bbox_to_anchor=(0.85, -0.2),frameon=False)
plt.legend(prop={'size': 12},ncol=2,bbox_to_anchor=(0.5, -0.12),title=r'$\bf{Color}$',   frameon=False)
add_legend([{'marker' : 'o', 'label' : r'$\kappa \sim 0.15 L_{0}v_{ph}$'}, {'marker' :   'd', 'label' : r'$\kappa = 0$'}],
          default_plot_args = {'ls' : '', 'c' : 'k', 'markersize' : '12','fillstyle' :    'full'}, ncol=1,bbox_to_anchor=(1.02, -0.12),frameon=False,
                                    title = r'$\bf{Symbol}$')
plt.tight_layout()
plt.savefig("fcr_vs_pcpg_stream.pdf")
plt.close()



# making a stacked and grouped bar chart plot with labels = no transport, diffusion only, streaming only, diffusion + streaming (isothermal), diffusion + streaming (adiabatic)

labels = ['$\beta \sim 1$', '$\beta \sim 10$', r'$\beta \sim 100$']
#arrays should be length 3, one for each beta

adiabatic_stream_diff_fheat = [1.27, 0.82, 0.4426]
adiabatic_stream_diff_fcr = [-0.05, -0.05, 0.05]
adiabatic_stream_diff_fth = [0.977, 0.774, 0.847]

adiabatic_streamOnly_fcr = [-0.2, -0.19, -0.115]
adiabatic_streamOnly_fth = [1.23, 0.98, 0.88]
adiabatic_streamOnly_fheat = [1.46, 0.885, 0.474]

diff_fcr = [0.86, 0.87, 0.90]
diff_fE = [0.359, 0.27, 0.18]
diff_fheat = [0,0,0]

stream_fcr = [0.03, 0.036, 0.03]
stream_fE = [1.15, 0.44, 0.07]
stream_fheat = [0.386, 0.834, 0.89]

stream_diff_fcr = [0.03, 0.07, 0.15]
stream_diff_fE = [1.20, 0.396, 0.15]
stream_diff_fheat = [0.38, 0, 0]

beta = [1,10,100]

df_diff = pd.DataFrame([diff_fcr,diff_fheat],
        index = [r'f$_{\rm CR}$', r'f$_{\rm CR, heating}$'],columns=[r'$\beta \sim 1$',r'$\beta \sim 10$', r'$\beta \sim   100$'])

df_stream = pd.DataFrame([stream_fcr,stream_fheat],
        index = [r'f$_{\rm CR}$', r'f$_{\rm CR, heating}$'],columns=[r'$\beta \sim 1$',r'$\beta \sim 10$', r'$\beta \sim   100$'])

df_stream_diff = pd.DataFrame([stream_diff_fcr,stream_diff_fheat],
        index = [r'f$_{\rm CR}$', r'f$_{\rm CR, heating}$'],columns=[r'$\beta \sim 1$',r'$\beta \sim 10$', r'$\beta \sim   100$'])


def plot_clustered_stacked(dfall, labels=None, title="multiple stacked bar plot",  H="/", **kwargs):
    """Given a list of dataframes, with identical columns and index, create a clustered stacked bar plot.
labels is a list of the names of the dataframe, used for the legend
title is a string for the title of the plot
H is the hatch used for identification of the different dataframe"""

    n_df = len(dfall)
    #n_col = len(dfall[0].columns)
    n_col = len(dfall[0].columns)
    n_ind = len(dfall[0].index)
    axe = plt.subplot(111)

    for df in dfall : # for each data frame
        axe = df.plot(kind="bar",
                      linewidth=1,
                      stacked=True,
                      ax=axe,
                      legend=False,
                      grid=False,
                      **kwargs)  # make bar plots

    h,l = axe.get_legend_handles_labels() # get the handles we want to modify
    for i in range(0, n_df * n_col, n_col): # len(h) = n_col * n_df
        for j, pa in enumerate(h[i:i+n_col]):
            for rect in pa.patches: # for each index
                rect.set_x(rect.get_x() + 1 / float(n_df + 1) * i / float(n_col))
                rect.set_hatch(H * int(i / n_col)) #edited part
                rect.set_width(1 / float(n_df + 1))

    axe.set_xticks((np.arange(0, 2 * n_ind, 2) + 1 / float(n_df + 1)) / 2.)
    axe.set_xticklabels(df.index, rotation = 0)
    axe.set_title(title)

    # Add invisible data to add another legend
    n=[]
    for i in range(n_df):
        n.append(axe.bar(0, 0, color="gray", hatch=H * i))

    l1 = axe.legend(h[:n_col], l[:n_col], loc=[1.01, 0.5])
    if labels is not None:
        l2 = plt.legend(n, labels, loc=[1.01, 0.1])
    axe.add_artist(l1)
    plt.tight_layout()
    plt.savefig('barchart_test.pdf')
    plt.close()

plot_clustered_stacked([df_diff.T, df_stream.T, df_stream_diff.T],["df1", "df2", "df3"])




"""
def prep_df(df, name):
    df = df.stack().reset_index()
    df.columns = ['c1', 'c2', 'values']
    df['DF'] = name
    return df

df1 = prep_df(df_diff, 'DF1')
df2 = prep_df(df_stream, 'DF2')
df3 = prep_df(df_stream_diff, 'DF3')

df = pd.concat([df1, df2, df3])


alt.Chart(df).mark_bar().encode(

    # tell Altair which field to group columns on
    x=alt.X('c2:N', title=None),

    # tell Altair which field to use as Y values and how to calculate
    y=alt.Y('sum(values):Q',
        axis=alt.Axis(
            grid=False,
            title=None)),

    # tell Altair which field to use to use as the set of columns to be  represented in each group
    column=alt.Column('c1:N', title=None),

    # tell Altair which field to use for color segmentation
    color=alt.Color('DF:N',
            scale=alt.Scale(
                # make it look pretty with an enjoyable color pallet
                range=['#96ceb4', '#ffcc5c','#ff6f69'],
            ),
        ))\
    .configure_view(
        # remove grid lines around column clusters
        strokeOpacity=0
    )

"""

"""
df_diff['Name'] = "df1"
df_stream['Name'] = "df2"
df_stream_diff['Name'] = "df3"

dfall = pd.concat([pd.melt(i.reset_index(),
                           id_vars=["Name", "index"]) # transform in tidy format each df
                   for i in [df_diff, df_stream, df_stream_diff]],
                   ignore_index=True)

dfall.set_index(["Name", "index", "beta"], inplace=1)
dfall["vcs"] = dfall.groupby(level=["Name", "index"]).cumsum()
dfall.head(2)
dfall.reset_index(inplace=True)


c = ["blue", "green"]
for i, g in enumerate(dfall.groupby("beta")):
    ax = sns.barplot(data=g[1],
                     x="index",
                     y="vcs",
                     hue="Name",
                     color=c[i],
                     zorder=-i, # so first bars stay on top
                     edgecolor="k")
ax.savefig('barchart_test.pdf')
"""

















# beta = 1, diff + streaming, heating rates:
"""
[0.40654389979691713 dimensionless, 0.439740956854437 dimensionless, 0.4131283512269335 dimensionless, 0.39947890847373946 dimensionless, 0.45478022920452776 dimensionless, 0.4033355493574192 dimensionless, 0.38445918463271006 dimensionless, 0.33320903702966853 dimensionless, 0.37942929381292434 dimensionless, 0.4208885913581928 dimensionless, 0.4258674750826827 dimensionless, 0.3828650814653422 dimensionless, 0.34961787810118095 dimensionless, 0.3542526392278516 dimensionless, 0.3392289140518748 dimensionless, 0.3669979319692487 dimensionless, 0.4224770927520684 dimensionless, 0.4324991957207392 dimensionless, 0.34814606085631755 dimensionless, 0.38176882368646503 dimensionless, 0.39592228421847625 dimensionless, 0.3575936935220483 dimensionless, 0.35411073817168853 dimensionless, 0.3893685669478144 dimensionless, 0.40464385190817875 dimensionless, 0.41145642868944565 dimensionless, 0.36386551440762194 dimensionless, 0.3573084224552761 dimensionless, 0.36714775541712397 dimensionless, 0.4029327701383428 dimensionless, 0.3677418597054848 dimensionless, 0.3419671284643319 dimensionless, 0.33756936825143785 dimensionless, 0.3897454834306467 dimensionless, 0.3694784161871997 dimensionless, 0.3725371146683567 dimensionless, 0.35929043054771104 dimensionless, 0.34758208044889427 dimensionless, 0.3717402931885796 dimensionless, 0.3587948432040534 dimensionless, 0.3677695831872022 dimensionless, 0.36616548657718295 dimensionless]
Times
[280.0003125015075, 285.0000000013135, 290.00015625111945, 295.0003125009254, 300.0000000007314, 305.0001562505374, 310.00031250034334, 315.0000000001493, 320.0001562499553, 325.00031249976126, 330.00046874956723, 335.0001562493732, 340.0003124991792, 345.00046874898516, 350.00015624879114, 355.0003124985971, 360.0004687484031, 365.00015624820907, 370.00031249801503, 375.000468747821, 380.000156247627, 385.00031249743296, 390.0004687472389, 395.0001562470449, 400.0, 405.00015624980597, 410.00031249961194, 415.0004687494179, 420.0001562492239, 425.00031249902986, 430.00046874883583, 435.0001562486418, 440.0003124984478, 445.00046874825375, 450.00015624805974, 455.0003124978657, 460.0004687476717, 465.00015624747766, 470.00031249728363, 475.0004687470896, 480.0001562468956, 485.00031249670155]
Mean heating rate
0.38079636210472234
"""

# beta = 10, streaming + diff, heating rates:
"""
[0.4162670790288492 dimensionless, 0.6437873971585063 dimensionless, 0.9522898337278627 dimensionless, 1.1657783339914216 dimensionless, 0.867834686348761 dimensionless, 0.6541572983977513 dimensionless, 0.5301329598821223 dimensionless, 0.7693696088221931 dimensionless, 0.8209712683144961 dimensionless, 1.0605372710001564 dimensionless, 0.8044497219480669 dimensionless, 0.6216101593284848 dimensionless, 0.6226752622130671 dimensionless, 0.7187618855438171 dimensionless, 1.9708145541504622 dimensionless, 0.534815203138722 dimensionless, 0.6417037863026771 dimensionless, 0.7682620302202727 dimensionless, 0.6704654456673499 dimensionless, 1.1130692557208601 dimensionless, 0.5971726566583393 dimensionless, 0.6913042169011718 dimensionless, 0.7020701602634363 dimensionless, 0.6507357596183287 dimensionless, 0.8177890269295103 dimensionless, 0.6679307906909437 dimensionless, 0.8595938895146882 dimensionless, 0.8743289722392008 dimensionless, 0.9333974373235313 dimensionless, 0.9727817630289741 dimensionless, 0.7390635225509847 dimensionless, 0.621242401518105 dimensionless, 0.8600077018492331 dimensionless, 1.138827455857281 dimensionless, 0.8311406066524712 dimensionless, 0.5651567956662238 dimensionless, 0.8294114113239273 dimensionless, 0.9878947602199254 dimensionless, 0.8535588655340964 dimensionless, 1.2250717180263586 dimensionless, 0.9276147796257211 dimensionless, 1.042805860345956 dimensionless]
Times
[250.00031250230785, 255.00000000241698, 260.0001562522836, 265.00031250208957, 270.00000000189556, 275.0001562517015, 280.0003125015075, 285.0000000013135, 290.00015625111945, 295.0003125009254, 300.0000000007314, 305.0001562505374, 310.00031250034334, 315.0000000001493, 320.0001562499553, 325.00031249976126, 330.00046874956723, 335.0001562493732, 340.0003124991792, 345.00046874898516, 350.00015624879114, 355.0003124985971, 360.0004687484031, 365.00015624820907, 370.00031249801503, 375.000468747821, 380.000156247627, 385.00031249743296, 390.0004687472389, 395.0001562470449, 400.0, 405.00015624980597, 410.00031249961194, 415.0004687494179, 420.0001562492239, 425.00031249902986, 430.00046874883583, 435.0001562486418, 440.0003124984478, 445.00046874825375, 450.00015624805974, 455.0003124978657]
Mean heating rate
0.8270631807915313
"""


# beta = 1, just streaming, heating rates
"""
[unyt_quantity(0.52840236, '(dimensionless)'), unyt_quantity(0.34203085, '(dimensionless)'), unyt_quantity(0.32523466, '(dimensionless)'), unyt_quantity(0.40892659, '(dimensionless)'), unyt_quantity(0.37336262, '(dimensionless)'), unyt_quantity(0.33004381, '(dimensionless)'), unyt_quantity(0.37741421, '(dimensionless)'), unyt_quantity(0.4321757, '(dimensionless)'), unyt_quantity(0.36713581, '(dimensionless)'), unyt_quantity(0.34379882, '(dimensionless)'), unyt_quantity(0.36759333, '(dimensionless)'), unyt_quantity(0.3088288, '(dimensionless)'), unyt_quantity(0.41992958, '(dimensionless)'), unyt_quantity(0.45921766, '(dimensionless)'), unyt_quantity(0.50141318, '(dimensionless)'), unyt_quantity(0.39754093, '(dimensionless)'), unyt_quantity(0.39785061, '(dimensionless)'), unyt_quantity(0.42118515, '(dimensionless)'), unyt_quantity(0.36839745, '(dimensionless)'), unyt_quantity(0.37744972, '(dimensionless)'), unyt_quantity(0.39597582, '(dimensionless)'), unyt_quantity(0.38490871, '(dimensionless)'), unyt_quantity(0.3691081, '(dimensionless)'), unyt_quantity(0.34055505, '(dimensionless)'), unyt_quantity(0.37665101, '(dimensionless)'), unyt_quantity(0.40866815, '(dimensionless)'), unyt_quantity(0.46695407, '(dimensionless)'), unyt_quantity(0.33137884, '(dimensionless)'), unyt_quantity(0.34083965, '(dimensionless)'), unyt_quantity(0.36409666, '(dimensionless)'), unyt_quantity(0.34272387, '(dimensionless)')]
Times
[250.00031250230785, 255.00000000241698, 260.0001562522836, 265.00031250208957, 270.00000000189556, 275.0001562517015, 280.0003125015075, 285.0000000013135, 290.00015625111945, 295.0003125009254, 300.0000000007314, 305.0001562505374, 310.00031250034334, 315.0000000001493, 320.0001562499553, 325.00031249976126, 330.00046874956723, 335.0001562493732, 340.0003124991792, 345.00046874898516, 350.00015624879114, 355.0003124985971, 360.0004687484031, 365.00015624820907, 370.00031249801503, 375.000468747821, 380.000156247627, 385.00031249743296, 390.0004687472389, 395.0001562470449, 400.0]
Mean heating rate
0.38612231496219007

"""


# beta = 10, just streaming, heating rates

"""
[unyt_quantity(0.99187332, '(dimensionless)'), unyt_quantity(0.68286717, '(dimensionless)'), unyt_quantity(0.56103309, '(dimensionless)'), unyt_quantity(0.83599449, '(dimensionless)'), unyt_quantity(0.63406874, '(dimensionless)'), unyt_quantity(0.80397873, '(dimensionless)'), unyt_quantity(0.99222289, '(dimensionless)'), unyt_quantity(0.57561627, '(dimensionless)'), unyt_quantity(0.98982519, '(dimensionless)'), unyt_quantity(0.83109149, '(dimensionless)'), unyt_quantity(0.83991018, '(dimensionless)'), unyt_quantity(0.60633054, '(dimensionless)'), unyt_quantity(0.86085742, '(dimensionless)'), unyt_quantity(0.70556452, '(dimensionless)'), unyt_quantity(1.28933714, '(dimensionless)'), unyt_quantity(1.02254055, '(dimensionless)'), unyt_quantity(0.78047357, '(dimensionless)'), unyt_quantity(0.57920655, '(dimensionless)'), unyt_quantity(0.56555536, '(dimensionless)'), unyt_quantity(2.39065348, '(dimensionless)'), unyt_quantity(0.82096969, '(dimensionless)'), unyt_quantity(0.68821171, '(dimensionless)'), unyt_quantity(0.68313639, '(dimensionless)'), unyt_quantity(0.70053849, '(dimensionless)'), unyt_quantity(0.93667052, '(dimensionless)'), unyt_quantity(0.80822268, '(dimensionless)'), unyt_quantity(0.66121611, '(dimensionless)'), unyt_quantity(0.70541156, '(dimensionless)'), unyt_quantity(0.73033954, '(dimensionless)'), unyt_quantity(0.72982352, '(dimensionless)'), unyt_quantity(0.64272396, '(dimensionless)'), unyt_quantity(0.82082795, '(dimensionless)'), unyt_quantity(0.836878, '(dimensionless)'), unyt_quantity(0.7752696, '(dimensionless)'), unyt_quantity(0.81013753, '(dimensionless)'), unyt_quantity(0.75964991, '(dimensionless)'), unyt_quantity(0.83441206, '(dimensionless)'), unyt_quantity(0.96643878, '(dimensionless)'), unyt_quantity(0.66434975, '(dimensionless)'), unyt_quantity(1.17805645, '(dimensionless)'), unyt_quantity(0.89588761, '(dimensionless)'), unyt_quantity(0.8697787, '(dimensionless)'), unyt_quantity(0.87309311, '(dimensionless)'), unyt_quantity(0.59372465, '(dimensionless)'), unyt_quantity(0.73489324, '(dimensionless)'), unyt_quantity(0.78788566, '(dimensionless)'), unyt_quantity(0.94622745, '(dimensionless)'), unyt_quantity(0.87541616, '(dimensionless)'), unyt_quantity(0.67247397, '(dimensionless)'), unyt_quantity(1.17114939, '(dimensionless)')]
Times
[250.00031250230785, 255.00000000241698, 260.0001562522836, 265.00031250208957, 270.00000000189556, 275.0001562517015, 280.0003125015075, 285.0000000013135, 290.00015625111945, 295.0003125009254, 300.0000000007314, 305.0001562505374, 310.00031250034334, 315.0000000001493, 320.0001562499553, 325.00031249976126, 330.00046874956723, 335.0001562493732, 340.0003124991792, 345.00046874898516, 350.00015624879114, 355.0003124985971, 360.0004687484031, 365.00015624820907, 370.00031249801503, 375.000468747821, 380.000156247627, 385.00031249743296, 390.0004687472389, 395.0001562470449, 400.0, 405.00015624980597, 410.00031249961194, 415.0004687494179, 420.0001562492239, 425.00031249902986, 430.00046874883583, 435.0001562486418, 440.0003124984478, 445.00046874825375, 450.00015624805974, 455.0003124978657, 460.0004687476717, 465.00015624747766, 470.00031249728363, 475.0004687470896, 480.0001562468956, 485.00031249670155, 490.0004687465075, 495.0001562463135]
Mean heating rate
0.8342562966454881

"""

# beta = 100, just streaming, heating rates

"""
[unyt_quantity(0.74234245, '(dimensionless)'), unyt_quantity(0.82815541, '(dimensionless)'), unyt_quantity(0.81230274, '(dimensionless)'), unyt_quantity(0.73211543, '(dimensionless)'), unyt_quantity(0.98692139, '(dimensionless)'), unyt_quantity(1.10845612, '(dimensionless)'), unyt_quantity(1.12451647, '(dimensionless)'), unyt_quantity(0.76129945, '(dimensionless)'), unyt_quantity(0.8364016, '(dimensionless)'), unyt_quantity(0.94029676, '(dimensionless)'), unyt_quantity(0.80051141, '(dimensionless)'), unyt_quantity(0.8428713, '(dimensionless)'), unyt_quantity(0.76676044, '(dimensionless)'), unyt_quantity(0.8531807, '(dimensionless)'), unyt_quantity(0.72529273, '(dimensionless)'), unyt_quantity(0.87318215, '(dimensionless)'), unyt_quantity(0.9213879, '(dimensionless)'), unyt_quantity(0.89751855, '(dimensionless)'), unyt_quantity(0.86226501, '(dimensionless)'), unyt_quantity(0.93181189, '(dimensionless)'), unyt_quantity(1.01208261, '(dimensionless)'), unyt_quantity(1.01208261, '(dimensionless)'), unyt_quantity(0.85530567, '(dimensionless)'), unyt_quantity(0.89897393, '(dimensionless)'), unyt_quantity(0.83531806, '(dimensionless)'), unyt_quantity(0.8023111, '(dimensionless)'), unyt_quantity(0.8103413, '(dimensionless)'), unyt_quantity(0.81894475, '(dimensionless)'), unyt_quantity(0.75707764, '(dimensionless)'), unyt_quantity(1.93167003, '(dimensionless)'), unyt_quantity(0.85755075, '(dimensionless)'), unyt_quantity(0.76993804, '(dimensionless)'), unyt_quantity(0.8346688, '(dimensionless)'), unyt_quantity(0.83292935, '(dimensionless)'), unyt_quantity(1.081655, '(dimensionless)'), unyt_quantity(0.82825315, '(dimensionless)'), unyt_quantity(0.86845394, '(dimensionless)'), unyt_quantity(0.77983413, '(dimensionless)'), unyt_quantity(0.74156867, '(dimensionless)'), unyt_quantity(0.93906976, '(dimensionless)')]
Times
[300.0000000007314, 305.0001562505374, 310.00031250034334, 315.0000000001493, 320.0001562499553, 325.00031249976126, 330.00046874956723, 335.0001562493732, 340.0003124991792, 345.00046874898516, 350.00015624879114, 355.0003124985971, 360.0004687484031, 365.00015624820907, 370.00031249801503, 375.000468747821, 380.000156247627, 385.00031249743296, 390.0004687472389, 395.0001562470449, 400.0, 400.0, 410.00031249961194, 415.0004687494179, 420.0001562492239, 425.00031249902986, 430.00046874883583, 435.0001562486418, 440.0003124984478, 445.00046874825375, 450.00015624805974, 455.0003124978657, 460.0004687476717, 465.00015624747766, 470.00031249728363, 475.0004687470896, 480.0001562468956, 485.00031249670155, 490.0004687465075, 495.0001562463135]
Mean heating rate
0.8903904800995338
"""


"""
x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width, diff_fcr, width, label=r'f$_{CR}$')
rects2 = ax.bar(x - width, diff_fheat, width, bottom=diff_fcr, label=r'f$_{CR, heating}$')
rects3 = ax.bar(x - width/2, stream_fcr, width, label=r'f$_{CR}$')
rects4 = ax.bar(x - width/2, stream_fheat, width, bottom=stream_fcr, label=r'f$_{CR, heating}$')
rects5 = ax.bar(x + width/2, stream_diff_fcr, width, label=r'f$_{CR}$')
#rects6 = ax.bar(x + width/2, stream_diff_fheat, width, bottom=stream_diff_fcr, label=r'f$_{CR, heating}$')
rects7 = ax.bar(x + width, adiabatic_stream_diff_fcr, width, label=r'f$_{CR}$')
rects8 = ax.bar(x + width, adiabatic_stream_diff_fheat, width, bottom=stream_diff_fcr, label=r'f$_{CR, heating}$')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Scores')
ax.set_title('Scores by group and gender')
ax.set_xticks(x, labels)
ax.legend()

ax.bar_label(rects1, padding=3)
ax.bar_label(rects2, padding=3)

fig.tight_layout()

plt.savefig("barchart_test.pdf")


"""




