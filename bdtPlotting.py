# Original Author: Ishan Khurana
# Updated and Adapted by: Louis Heery (Including: Updating to Python3, Simplifying Codebase, Adapting Output Plots)

import sys
import numpy as np
import pandas as pd
import matplotlib as plt
from matplotlib.ticker import AutoMinorLocator
from mpl_toolkits.axes_grid.anchored_artists import AnchoredText
from matplotlib.text import OffsetFrom
from sensitivity import *

class_names_grouped = ['VH -> Vbb','Diboson','ttbar','Single top', 'W+(bb,bc,cc,bl)','W+cl','W+ll','Z+(bb,bc,cc,bl)','Z+cl','Z+ll']

class_names_map = {'VH -> Vbb':['ggZllH125','ggZvvH125','qqWlvH125', 'qqZllH125', 'qqZvvH125'],
    'Diboson':['WW','ZZ','WZ'],
    'ttbar':['ttbar'],
    'Single top':['stopWt','stops','stopt'],
    'W+(bb,bc,cc,bl)':['Wbb','Wbc','Wcc','Wbl'],
    'W+cl':['Wcl'],
    'W+ll':['Wl'],
    'Z+(bb,bc,cc,bl)':['Zbb','Zbc','Zcc','Zbl'],
    'Z+cl':['Zcl'],
    'Z+ll':['Zl']
}

colour_map = {'VH -> Vbb':'#FF0000',
    'Diboson':'#999999',
    'ttbar':'#FFCC00',
    'Single top':'#CC9900',
    'W+(bb,bc,cc,bl)':'#006600',
    'W+cl':'#66CC66',
    'W+ll':'#99FF99',
    'Z+(bb,bc,cc,bl)':'#0066CC',
    'Z+cl':'#6699CC',
    'Z+ll':'#99CCFF'
}

legend_names = [r'VH $\rightarrow$ Vbb','Diboson',r"t$\bar t$",'Single top', 'W+(bb,bc,cc,bl)','W+cl','W+ll','Z+(bb,bc,cc,bl)',
                'Z+cl','Z+ll'
                ]

def final_decision_plot(df, figureName, z_s = 10,z_b = 10,show=False, block=False, trafoD_bins = True, bin_number = 15):
    """Plots histogram decision score output of classifier"""

    nJets = df['nJ'].tolist()[1]

    if trafoD_bins == True:
        bins, arg2, arg3 = trafoD_with_error(df)
        print(len(bins))
    else:
         bins = np.linspace(-1,1,bin_number+1)

    # Initialise plot stuff
    plt.ion()
    plt.close("all")
    fig = plt.figure(figsize=(8.5,7))
    plot_range = (-1, 1)
    plot_data = []
    plot_weights = []
    plot_colors = []
    plt.rc('font', weight='bold')
    plt.rc('xtick.major', size=5, pad=7)
    plt.rc('xtick', labelsize=10)

    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    plt.rcParams["mathtext.default"] = "regular"

    df = setBinCategory(df,bins)

    bins = np.linspace(-1,1,len(bins))

    decision_value_list = df['bin_scaled'].tolist()
    post_fit_weight_list = df['post_fit_weight'].tolist()
    sample_list = df['sample'].tolist()

    # Get list of hists.
    for t in class_names_grouped[::-1]:
        class_names = class_names_map[t]
        class_decision_vals = []
        plot_weight_vals = []
        for c in class_names:
            for x in range(0,len(decision_value_list)):
                if sample_list[x] == c:
                    class_decision_vals.append(decision_value_list[x])
                    plot_weight_vals.append(post_fit_weight_list[x])

        plot_data.append(class_decision_vals)
        plot_weights.append(plot_weight_vals)
        plot_colors.append(colour_map[t])

    # Plot.
    if nJets == 2:

        multiplier = 20
    elif nJets == 3:
        multiplier = 100

    plt.plot([],[],color='#FF0000',label=r'VH $\rightarrow$ Vbb x '+str(multiplier))

    plt.hist(plot_data,
             bins=bins,
             weights=plot_weights,
             range=plot_range,
             rwidth=1,
             color=plot_colors,
             label=legend_names[::-1],
             stacked=True,
             edgecolor='none')

    df_sig = df.loc[df['Class']==1]

    plt.hist(df_sig['bin_scaled'].tolist(),
         bins=bins,
         weights=(df_sig['post_fit_weight']*multiplier).tolist(),
         range=plot_range,
         rwidth=1,
         histtype = 'step',
         linewidth=2,
         color='#FF0000',
         edgecolor='#FF0000')

    x1, x2, y1, y2 = plt.axis()
    plt.yscale('log', nonposy='clip')
    plt.axis((x1, x2, y1, y2 * 1.2))
    axes = plt.gca()
    axes.set_ylim([5,135000])
    axes.set_xlim([-1,1])
    x = [-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1]
    plt.xticks(x, x,fontweight = 'normal',fontsize = 20)
    y = [r"10",r"10$^{2}$",r"10$^{3}$",r"10$^{4}$",r"10$^{5}$"]
    yi = [10,100,1000,10000,100000]
    plt.yticks(yi, y,fontweight = 'normal',fontsize = 20)

    axes.yaxis.set_ticks_position('both')
    axes.yaxis.set_tick_params(which='major', direction='in', length=10, width=1)
    axes.yaxis.set_tick_params(which='minor', direction='in', length=5, width=1)

    axes.xaxis.set_ticks_position('both')
    axes.xaxis.set_tick_params(which='major', direction='in', length=10, width=1)
    axes.xaxis.set_tick_params(which='minor', direction='in', length=5, width=1)

    axes.xaxis.set_minor_locator(AutoMinorLocator(4))
    handles, labels = axes.get_legend_handles_labels()

    #Weird hack thing to get legend entries in correct order
    handles = handles[::-1]
    handles = handles+handles
    handles = handles[1:12]

    plt.legend(loc='upper right', ncol=1, prop={'size': 12},frameon=False,
               handles=handles)

    plt.ylabel("Events",fontsize = 20,fontweight='normal')
    axes.yaxis.set_label_coords(-0.07,0.93)
    plt.xlabel(r"BDT$_{VH}$ output",fontsize = 20,fontweight='normal')
    axes.xaxis.set_label_coords(0.89, -0.07)
    an1 = axes.annotate("ATLAS Internal", xy=(0.05, 0.91), xycoords=axes.transAxes,fontstyle = 'italic',fontsize = 16)

    offset_from = OffsetFrom(an1, (0, -1.4))
    an2 = axes.annotate(r'$\sqrt{s}$' + " = 13 TeV , 36.1 fb$^{-1}$", xy=(0.05,0.91), xycoords=axes.transAxes, textcoords=offset_from, fontweight='normal',fontsize = 12)

    offset_from = OffsetFrom(an2, (0, -1.4))
    an3 = axes.annotate("1 lepton, "+str(nJets)+" jets, 2 b-tags", xy=(0.05,0.91), xycoords=axes.transAxes, textcoords=offset_from,fontstyle = 'italic',fontsize = 12)

    offset_from = OffsetFrom(an3, (0, -1.6))
    an4 = axes.annotate("p$^V_T \geq$ 150 GeV", xy=(0.05,0.91), xycoords=axes.transAxes, textcoords=offset_from,fontstyle = 'italic',fontsize = 12)

    fig = plt.gcf()
    #fig.set_size_inches(10, 4)
    plt.savefig(figureName, bbox_inches='tight',dpi=300)  # should before plt.show method
    plt.show(block = block)

    return fig,axes


def variable_plot(df,z_s = 10,z_b = 5,show=False, block=False, trafoD_bins = True, bin_number = 15):
    """Plots histogram decision score output of classifier"""

    nJets = df['nJ'].tolist()[1]

    if trafoD_bins == True:
        bins, arg2, arg3 = trafoD_with_error(df,z_s = z_s,z_b = z_b)
    else:
         bins = np.linspace(-1,1,bin_number+1)

    # Initialise plot stuff
    plt.ion()
    plt.close("all")
    fig = plt.figure(figsize=(8.5,7))
    plot_range = (-1, 1)
    plot_data = []
    plot_weights = []
    plot_colors = []
    plt.rc('font', weight='bold')
    plt.rc('xtick.major', size=5, pad=7)
    plt.rc('xtick', labelsize=10)

    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    plt.rcParams["mathtext.default"] = "regular"

    df = setBinCategory(df,bins)

    bins = np.linspace(-1,1,len(bins))

    decision_value_list = df['bin_scaled'].tolist()
    post_fit_weight_list = df['post_fit_weight'].tolist()
    sample_list = df['sample'].tolist()

    # Get list of hists.
    for t in class_names_grouped[::-1]:
        class_names = class_names_map[t]
        class_decision_vals = []
        plot_weight_vals = []
        for c in class_names:
            for x in range(0,len(decision_value_list)):
                if sample_list[x] == c:
                    class_decision_vals.append(decision_value_list[x])
                    plot_weight_vals.append(post_fit_weight_list[x])

        plot_data.append(class_decision_vals)
        plot_weights.append(plot_weight_vals)
        plot_colors.append(colour_map[t])

    # Plot.
    if nJets == 2:

        multiplier = 20
    elif nJets == 3:
        multiplier = 100

    plt.plot([],[],color='#FF0000',label=r'VH $\rightarrow$ Vbb x '+str(multiplier))

    plt.hist(plot_data,
             bins=bins,
             weights=plot_weights,
             range=plot_range,
             rwidth=1,
             color=plot_colors,
             label=legend_names[::-1],
             stacked=True,
             edgecolor='none')

    df_sig = df.loc[df['Class']==1]



    plt.hist(df_sig['bin_scaled'].tolist(),
         bins=bins,
         weights=(df_sig['post_fit_weight']*multiplier).tolist(),
         range=plot_range,
         rwidth=1,
         histtype = 'step',
         linewidth=2,
         color='#FF0000',
         edgecolor='#FF0000')

    x1, x2, y1, y2 = plt.axis()
    plt.yscale('log', nonposy='clip')
    plt.axis((x1, x2, y1, y2 * 1.2))
    axes = plt.gca()
    axes.set_ylim([5,135000])
    axes.set_xlim([-1,1])
    x = [-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1]
    plt.xticks(x, x,fontweight = 'normal',fontsize = 20)
    y = [r"10",r"10$^{2}$",r"10$^{3}$",r"10$^{4}$",r"10$^{5}$"]
    yi = [10,100,1000,10000,100000]
    plt.yticks(yi, y,fontweight = 'normal',fontsize = 20)

    axes.yaxis.set_ticks_position('both')
    axes.yaxis.set_tick_params(which='major', direction='in', length=10, width=1)
    axes.yaxis.set_tick_params(which='minor', direction='in', length=5, width=1)

    axes.xaxis.set_ticks_position('both')
    axes.xaxis.set_tick_params(which='major', direction='in', length=10, width=1)
    axes.xaxis.set_tick_params(which='minor', direction='in', length=5, width=1)

    axes.xaxis.set_minor_locator(AutoMinorLocator(4))
    handles, labels = axes.get_legend_handles_labels()

    #Weird hack thing to get legend entries in correct order
    handles = handles[::-1]
    handles = handles+handles
    handles = handles[1:12]

    plt.legend(loc='upper right', ncol=1, prop={'size': 12},frameon=False,
               handles=handles)

    plt.ylabel("Events",fontsize = 20,fontweight='normal')
    axes.yaxis.set_label_coords(-0.07,0.93)
    plt.xlabel(r"BDT$_{VH}$ output",fontsize = 20,fontweight='normal')
    axes.xaxis.set_label_coords(0.89, -0.07)
    an1 = axes.annotate("ATLAS Internal", xy=(0.05, 0.91), xycoords=axes.transAxes,fontstyle = 'italic',fontsize = 16)

    offset_from = OffsetFrom(an1, (0, -1.4))
    an2 = axes.annotate(r'$\sqrt{s}$' + " = 13 TeV , 36.1 fb$^{-1}$", xy=(0.05,0.91), xycoords=axes.transAxes, textcoords=offset_from, fontweight='normal',fontsize = 12)

    offset_from = OffsetFrom(an2, (0, -1.4))
    an3 = axes.annotate("1 lepton, "+str(nJets)+" jets, 2 b-tags", xy=(0.05,0.91), xycoords=axes.transAxes, textcoords=offset_from,fontstyle = 'italic',fontsize = 12)

    offset_from = OffsetFrom(an3, (0, -1.6))
    an4 = axes.annotate("p$^V_T \geq$ 150 GeV", xy=(0.05,0.91), xycoords=axes.transAxes, textcoords=offset_from,fontstyle = 'italic',fontsize = 12)

    plt.show(block=block)


    return fig


def setBinCategory(df,bins):

    if len(bins)!=21:
        print ("ONLY SET FOR 20 BINS")

    df['bin_scaled'] = 999
    bin_scaled_list = df['bin_scaled'].tolist()

    step = 2/(len(bins)-1)  #step between midpoints
    midpoint = -1 + step/2.0   #Initial midpoint
    decision_value_list = df['decision_value'].tolist()

    for j in range(len(bins)-1):
        for i in range(len(decision_value_list)):
            if ((decision_value_list[i] >= bins[j]) & (decision_value_list[i] < bins[j+1])):
                bin_scaled_list[i] = midpoint
        midpoint = midpoint + step

    df['bin_scaled'] = bin_scaled_list

    return df
