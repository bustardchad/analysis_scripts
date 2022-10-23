import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import time
import seaborn as sns


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



#####################################################3
# Simulation details:
#   CR transport is diffusion only
#   Varying pc/pg, keeping beta ~ 1 constant
#   Run on 256^3 box, driving turbulence at k = 2 mode

# Isothermal equation of state
pcpg_iso_Mach05 = [1e-4,5e-2,0.2, 0.3, 1, 1.5, 10, 100]
pcpg_iso_Mach035 = [1.5,10,100]

fcr_iso_Mach05 = [9.6e-4,0.14, 0.58, 0.65, 0.8, 0.86, 0.89, 0.86]
fcr_iso_Mach035 = [0.83, 0.91, 0.85]


# Same but adiabatic equation of state instead of isothermal
pcpg_ad_Mach05 = [0.025,0.06,0.1,1/5,1/2,1,10,100]
fcr_ad_Mach05 = [0.12,0.19,0.23,0.4,0.56, 0.68, 0.84, 0.79]
fg_ad_Mach05 = [0.78,0.7,0.62,0.47,0.3,0.21, 0.07, 0.14]


# adiabatic eos, now with no CR diffusion (only advective transport)
pcpg_advect_Mach05 = [1e-2,1e-1,1,10,100]
fcr_advect_Mach05 = [0.0003,0.0003,0.096,0.52,0.73]
fg_advect_Mach05 = [0.969,0.969,0.74,0.35,0.13]

# expectation given diffusion-only CR reacceleration rates (see Bustard and Oh 2022b)
pcpg_arr = np.arange(1e-4,2.0,0.0001)
fcr_expect = (2./3.)*(5e6/(1e7*np.sqrt(1.0 + pcpg_arr)))*(pcpg_arr*1e7**2.0)/(5e6**2.0)


fill_style='full'
ms = 12.0
plt.semilogx(pcpg_ad_Mach05,fcr_ad_Mach05,'ko-',label=r"$f_{CR}$",markersize=ms,fillstyle=fill_style)
plt.semilogx(pcpg_ad_Mach05,fg_ad_Mach05,'ro-',label=r"$f_{th}$",markersize=ms,fillstyle=fill_style)
plt.semilogx(pcpg_advect_Mach05,fcr_advect_Mach05,'kD-',markersize=ms,fillstyle=fill_style)
plt.semilogx(pcpg_advect_Mach05,fg_advect_Mach05,'rD-',markersize=ms,fillstyle=fill_style)
plt.semilogx(pcpg_arr,fcr_expect,'b-',label=r"$\frac{2}{3} \frac{M_{ph} P_{CR}}{\rho v^{2}}$",markersize=ms,fillstyle=fill_style)
plt.ylim(8e-3,1)
plt.xlim(1e-2,125)
plt.xlabel(r"$P_{CR}/P_{g}$",fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.title("Diffusion Only",fontsize=22)
plt.text(0.015,0.88,r"$\beta \sim 1$",fontsize=16,bbox=dict(edgecolor='black', alpha=0.1))
plt.legend(prop={'size': 12},ncol=2,bbox_to_anchor=(0.4, -0.12),title=r'$\bf{Color}$',frameon=False)
# add another legend to distinguish diffusion vs no diffusion cases
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
color_map=["blue","orange"]

fig, axs = plt.subplots(2,1,sharex=True,sharey=True)
# top plot is with diffusive CR transport
axs[0].stackplot(pcpg_ad_Mach05,y_diff,labels=labels,colors=color_map)
axs[0].plot(pcpg_arr,fcr_expect,'k--',label=r"$\frac{2}{3} \frac{M_{ph} P_{CR}}{\rho v^{2}}$",linewidth=2)
plt.xscale('log')
axs[0].legend(loc='lower right',title=r"$\kappa_{||} \sim 0.15 L_0 c_s$")
#plt.ylabel(r"$\dot{E}/\epsilon$",fontsize=16)

# bottom plot is no diffusion
axs[1].stackplot(pcpg_advect_Mach05,y_nodiff,labels=labels,colors=color_map)
axs[1].set_xlim(min(pcpg_ad_Mach05),max(pcpg_ad_Mach05))
axs[1].set_ylim(0,1)
plt.xscale('log')
plt.xlabel(r"$P_{CR}/P_{g}$",fontsize=16)
#plt.ylabel(r"$\dot{E}/\epsilon$",fontsize=16)
axs[1].legend(loc='lower right',title=r"$\kappa_{||} \sim 0$")
plt.tight_layout()
plt.savefig("fcr_vs_pcpg_adiabatic_stackplot.pdf")
plt.close()



################################################
# Simulation details:
#   Lower Mach number: M ~ 0.15 instead of M ~ 0.5
#   CR transport now includes streaming
#   pc/pg = 1 = constant, now varying beta
#   "ad" denotes adiabatic eos instead of isothermal



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

adiabatic_stream_diff_fheat = [0.372, 0.812, 0.856]
adiabatic_stream_diff_fcr = [0.017, 0.054, 0.16]
adiabatic_stream_diff_fth = [0.72, 0.89 ,0.87]

adiabatic_streamOnly_fcr = [0.017, 0.013, 0.029]
adiabatic_streamOnly_fth = [0.84, 1.07, 0.88]
adiabatic_streamOnly_fheat = [0.377, 0.89, 1.016]

diff_fcr = [0.86, 0.87, 0.90]
diff_fE = [0.359, 0.27, 0.18]
diff_fheat = [0,0,0]
diff_fth = [0,0,0] # all zeros because no gas heating with isothermal eos

stream_fcr = [0.03, 0.036, 0.03]
stream_fE = [1.15, 0.44, 0.07]
stream_fheat = [0.386, 0.834, 0.89]
stream_fth = [0,0,0]

stream_diff_fcr = [0.03, 0.07, 0.15]
stream_diff_fE = [1.20, 0.396, 0.15]
stream_diff_fheat = [0.38, 0.827, 0.7627]
stream_diff_fth = [0,0,0]

# I've created many more vectors than I planned to...let's put them in a DataFrame and
# write to a CSV file. In the future, I'll just add to that.

df_diff = pd.DataFrame([diff_fcr,diff_fheat,diff_fth],
        index = [r'f$_{\rm CR}$', r'f$_{\rm CR, heating}$', r'f$_{\rm th}$'],columns=[r'$\beta \sim 1$',r'$\beta \sim 10$', r'$\beta \sim   100$'])

df_stream = pd.DataFrame([stream_fcr,stream_fheat, stream_fth],
        index = [r'f$_{\rm CR}$', r'f$_{\rm CR, heating}$', r'f$_{\rm th}$'],columns=[r'$\beta \sim 1$',r'$\beta \sim 10$', r'$\beta \sim   100$'])

df_stream_diff = pd.DataFrame([stream_diff_fcr,stream_diff_fheat,stream_diff_fth],
        index = [r'f$_{\rm CR}$', r'f$_{\rm CR, heating}$', r'f$_{\rm th}$'],columns=[r'$\beta \sim 1$',r'$\beta \sim 10$', r'$\beta \sim   100$'])

df_adiabatic_stream_diff = pd.DataFrame([adiabatic_stream_diff_fcr,adiabatic_stream_diff_fheat,adiabatic_stream_diff_fth],
        index = [r'f$_{\rm CR}$', r'f$_{\rm CR, heating}$', r'f$_{\rm th}$'],columns=[r'$\beta \sim 1$',r'$\beta \sim 10$', r'$\beta \sim   100$'])

df_adiabatic_streamOnly = pd.DataFrame([adiabatic_streamOnly_fcr,adiabatic_streamOnly_fheat,adiabatic_streamOnly_fth],
        index = [r'f$_{\rm CR}$', r'f$_{\rm CR, heating}$', r'f$_{\rm th}$'],columns=[r'$\beta \sim 1$',r'$\beta \sim 10$', r'$\beta \sim   100$'])



# flatten into the right shape
df_flat = pd.concat([df_diff,df_stream,df_stream_diff,df_adiabatic_streamOnly, df_adiabatic_stream_diff])
print(df_flat.head(8))

# Write this stuff to a CSV file, so I can add to it easily
df_flat.to_csv('turbulent_energy_partitions.csv')


# this data is in wide form, but it's easier to work with in long form
df_flat = df_flat.to_numpy()
df_flat = df_flat.flatten('F')
print(df_flat)


# this works for now, since I haven't added more data to my CSV file. Need to change in future
dfall = pd.DataFrame(dict(
    beta = [r'$\beta \sim 1$',r'$\beta \sim 1$',r'$\beta \sim 1$',r'$\beta \sim 1$',r'$\beta \sim 1$',r'$\beta \sim 1$',r'$\beta \sim 1$',r'$\beta \sim 1$',r'$\beta \sim 1$',r'$\beta \sim 1$',r'$\beta \sim 1$',r'$\beta \sim 1$',r'$\beta \sim 1$',r'$\beta \sim 1$',r'$\beta \sim 1$',r'$\beta \sim 10$',
    r'$\beta \sim 10$',r'$\beta \sim 10$',r'$\beta \sim 10$',r'$\beta \sim 10$',r'$\beta \sim 10$',r'$\beta \sim 10$',r'$\beta \sim 10$',r'$\beta \sim 10$',r'$\beta \sim 10$',r'$\beta \sim 10$',r'$\beta \sim 10$',r'$\beta \sim 10$',r'$\beta \sim 10$',r'$\beta \sim 10$',
    r'$\beta \sim 100$',r'$\beta \sim 100$',r'$\beta \sim 100$',r'$\beta \sim 100$',r'$\beta \sim 100$',r'$\beta \sim 100$',r'$\beta \sim 100$',r'$\beta \sim 100$',r'$\beta \sim 100$',r'$\beta \sim 100$',r'$\beta \sim 100$',r'$\beta \sim 100$',r'$\beta \sim 100$',r'$\beta \sim 100$',r'$\beta \sim 100$'],
    labels = ["Diffusion Only","Diffusion Only","Diffusion Only","Streaming Only","Streaming Only","Streaming Only","Diff + Stream","Diff + Stream","Diff + Stream","Adiab. Stream Only","Adiab. Stream Only","Adiab. Stream Only","Adiab. Diff + Stream","Adiab. Diff + Stream","Adiab. Diff + Stream"] * 3,
    flabel = [r'$f_{\rm CR} \text{, CR Energization}$', r'$f_{\rm CR, heating} \text{, Streaming Energy Loss}$', r'$f_{\rm th} - f_{\rm CR, heating} \text{, Grid-Scale Heating}$'] * 15,
    f = df_flat))

print(dfall.head(16))



##########################################################################3
# Now comes the fun part of plotting this as a stacked, grouped bar chart. After trying
# some methods in regular matplotlib, seaborn, etc., plotly emerged as the best option

# Without this ridiculous work-around, every PDF file I save has a box with "Loading MathJax" written in the lower left corner. Might be a problem with the Kaleido package used to write plotly images to files
figure="some_figure.pdf"
fig=px.scatter(x=[0, 1, 2, 3, 4], y=[0, 1, 4, 9, 16])
fig.write_image(figure, format="pdf")


time.sleep(2) # this delay gives some time for MathJax to load before the next figure (the real figure) is plotted and saved


fig = go.Figure()


# Note that all text that includes LaTeX anywhere near it has to be inside $ $
fig.update_layout(
    template="simple_white",
    xaxis=dict(ticklen=0),
    yaxis=dict(title_text=r"$\dot{E}/\epsilon$",range=(0,1)),
    font=dict(size=12),
    barmode="stack",
    legend=dict(
        font=dict(size=12),
        x=0.55,
        y=1.4,
        bgcolor='rgba(255, 255, 255, 0)',
        bordercolor='rgba(255, 255, 255, 0)'
    )
)

colors = ["blue", "red","orange"]

# Add traces for each bar. Grouped by $\beta \sim ...$, colored by f_{CR} or f_{CR, heating}
for r, c in zip(dfall.flabel.unique(), colors):
    plot_df = dfall[dfall.flabel == r]
    fig.add_trace(
        go.Bar(x=[plot_df.beta, plot_df.labels], y=plot_df.f, name=r, marker_color=c),
    )
fig.update_xaxes(
        tickangle = 90,
        tickson = "boundaries",
        ticks="inside",
        ticklen=0, # just get rid of the ticks
        dividerwidth=0,
        dividercolor='black',
        title_text = r"", # just get rid of the title
        title_font = {"size": 10},
        title_standoff = 20)

fig.write_image("barchart_plotly.pdf",engine="kaleido")


# Other variations on bar chart figures using Altair, seaborn. Keeping them here (commented out) in case they seem useful for the future.

"""

# Altair version -- mainly copied from stackexchange
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
# seaborn version -- mainly copied from stackexchange

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


# beta = 100, streaming + diff, heating rates
"""
[0.5751643964748747 dimensionless, 0.8812401820724147 dimensionless, 0.7743855209178467 dimensionless, 0.904050985830658 dimensionless, 0.7469830000176368 dimensionless, 0.8375819168550681 dimensionless, 0.8460678999433733 dimensionless, 0.7554840993508561 dimensionless, 0.8126117224696271 dimensionless, 0.782123792853733 dimensionless, 0.6495888708515523 dimensionless, 0.5671416656897461 dimensionless, 0.6719656569868332 dimensionless, 0.5781274893888504 dimensionless, 1.789374989498379 dimensionless, 0.5523189303305269 dimensionless, 0.5389342199646281 dimensionless, 0.6501179538646412 dimensionless, 0.7757406581675773 dimensionless, 0.9736545338336117 dimensionless, 0.6532879771700937 dimensionless, 0.7481055517663325 dimensionless, 0.7214939584733472 dimensionless, 0.6611935567816812 dimensionless, 0.7467904791209938 dimensionless, 0.6033208630811959 dimensionless, 0.8448285874700983 dimensionless, 0.7169277435486097 dimensionless, 0.7936201501203108 dimensionless, 0.7407620426807037 dimensionless, 0.6305844251297763 dimensionless, 0.6404846578368242 dimensionless, 0.6773214220104737 dimensionless, 0.9076464479698749 dimensionless, 0.7581704798138655 dimensionless, 0.6860855202840231 dimensionless, 0.6875675257178437 dimensionless, 0.6909917122035101 dimensionless, 0.665182093855881 dimensionless, 0.874057081186114 dimensionless, 0.6995557711698609 dimensionless, 0.8483122685688598 dimensionless, 0.9622222047772403 dimensionless, 0.564103091939513 dimensionless, 0.8349131542763308 dimensionless, 0.8480340697116481 dimensionless, 1.0657933831840467 dimensionless, 0.675680816111541 dimensionless]
Times
[250.00031250230785, 255.00000000241698, 260.0001562522836, 265.00031250208957, 270.00000000189556, 275.0001562517015, 280.0003125015075, 285.0000000013135, 290.00015625111945, 295.0003125009254, 300.0000000007314, 305.0001562505374, 310.00031250034334, 315.0000000001493, 320.0001562499553, 325.00031249976126, 330.00046874956723, 335.0001562493732, 340.0003124991792, 345.00046874898516, 350.00015624879114, 355.0003124985971, 360.0004687484031, 365.00015624820907, 370.00031249801503, 375.000468747821, 380.000156247627, 385.00031249743296, 390.0004687472389, 395.0001562470449, 400.0, 405.00015624980597, 410.00031249961194, 415.0004687494179, 420.0001562492239, 425.00031249902986, 430.00046874883583, 435.0001562486418, 440.0003124984478, 445.00046874825375, 450.00015624805974, 455.0003124978657, 460.0004687476717, 465.00015624747766, 470.00031249728363, 475.0004687470896, 480.0001562468956, 485.00031249670155]
Mean heating rate
0.762701990027563
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


