
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
import pickle
import pylatex
import os
import logging
import numpy as np
import copy
import cmath


# import sigmond
import fvspectrum.sigmond_util as sigmond_util
import fvspectrum.spectrum_plotting_settings.settings as psettings
from matplotlib.lines import Line2D  # Import Line2D for custom legend handle
from scipy.interpolate import interp1d
from sigmond_scripts import util as utils
from sigmond_scripts import fit_info


#where source and sink labels go on plot
ctext_x = 0.3
ctext_y = 0.7

#transparent box
blank = patches.Rectangle((0, 0), 1, 1, fc="white", ec="white", 
                                 lw=0, alpha=0)

#for a set of spectrum levels, will shift overlapping levels slightly left or right so 
    #that they are more visible
def shift_levels( indexes, vals, errors, shifted_array=np.array([]), index=0 ):
    if len(indexes):
        if not shifted_array.any():
            shifted_array = np.array([0.0]*len(indexes))
            
        if index==len(indexes):
            return shifted_array
            
        these_indexes = indexes[index:]
        this_remaining_irrep_index = these_indexes[0]
        this_remaining_irrep_indexes = np.where( these_indexes==this_remaining_irrep_index )[0]+index
        
        if len(this_remaining_irrep_indexes)==1:
            return shift_levels( indexes, vals, errors, shifted_array, index=index+1 )
        else:
            shift = 1
            this_val_upper = vals[index]+errors[index]
            this_val_lower = vals[index]-errors[index]
            this_cluster = [index]
            for i in this_remaining_irrep_indexes[1:]:
                overlap = False  
                compare_upper = vals[i]+errors[i]
                compare_lower = vals[i]-errors[i]
                if shifted_array[i]!=0.0:
                    continue
                if compare_lower<=this_val_lower and compare_upper>=this_val_upper: #set new bounds on this_value
                    overlap=True
                elif compare_lower>=this_val_lower and compare_upper<=this_val_upper:
                    overlap=True
                elif compare_lower<=this_val_upper and compare_lower>=this_val_lower and compare_upper>=this_val_upper:
                    overlap=True
                elif compare_lower<=this_val_lower and compare_upper<=this_val_upper and compare_upper>=this_val_lower:
                    overlap=True
                if overlap:
                    shifted_array[i] = shift
                    this_cluster.append(i)
                    if shift>0:
                        shift = -shift
                    else:
                        shift = abs(shift)+1
            
            if len(this_cluster)%2==0 and len(this_cluster):
                for i in this_cluster:
                    shifted_array[i]-=0.5
            
            new_index = np.where(shifted_array[index+1:]==0.0)[0]
            if new_index.any():
                return shift_levels( indexes, vals, errors, shifted_array, index=new_index[0]+index+1 )
            else:
                return shifted_array
                        
    return 0.0

class PlottingHandler:
    # Flag to check if LaTeX is available for plot labels
    latex = True
    #checks if latex compler is present, if not sets this variable to false
            #use this for latex labels, and make an option for non-latex labels.
    latex = sigmond_util.set_latex_in_plots(plt.style)

    #does a quick test to see if necessary latex libraries are present
    x = np.linspace(0,10,10)
    plt.plot(x,x)
    try:
        plt.tight_layout()

    except RuntimeError as err:
        print("Not all libraries are available for matplotlib latex plots. Latex will not be used for plots.")
        print(f"\t message: {err}")
        latex = False
        matplotlib.rcParams['text.usetex'] = False

    #Fernando's settings
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Computer Modern Roman']
    plt.rcParams['legend.fancybox'] = False
    plt.rcParams['legend.shadow'] = False
    plt.rcParams['legend.framealpha'] = '0.8'
    plt.rcParams['legend.facecolor'] = 'white'
    plt.rcParams['legend.edgecolor'] = '0.5'
    plt.rcParams['legend.fontsize'] = 15.0
 
    plt.clf()

    def __init__(self):
        self.doc = []

    #############################
    ##### matplotlib actions ####
    #############################

    def create_fig(self, figwidth, figheight):
        """Create a new Matplotlib figure with specified width and height."""
        self.fig = plt.figure(figsize=(figwidth, figheight)) 
        self.figheight = figheight
        self.figwidth = figwidth
        return self.fig
    
    def clf(self):
        """Clear the current Matplotlib figure."""
        plt.clf()
        plt.gcf().set_size_inches(self.figwidth, self.figheight)

    def set_figsize(self, figwidth, figheight):
        """Set the figsize for both this class and matplotlib"""
        self.figheight = figheight
        self.figwidth = figwidth
        plt.gcf().set_size_inches(figwidth, figheight)

    def moving_textbox(self, labels):
        """Uses the legend object as a textbox instead so that the 
            text will move around the data accordingly"""
        plt.legend([blank]*len(labels), labels, frameon=False, fancybox=False, framealpha=0.0, shadow=None)

    def correlator_plot(self,df, ptype=0, op1=None, op2=None, color_index = 0, label=None):
        """Generate a correlator plot using Matplotlib."""
        plt.errorbar(x=df["aTime"],y=df["FullEstimate"],yerr=df["SymmetricError"], linewidth=0.0, elinewidth=1.5, capsize=5, color=psettings.colors[color_index], marker=psettings.markers[color_index] , label=label)
        if ptype==0:
            plt.ylabel(r"$C(t)$") #not all dollar signs require a latex compiler, for example this is okay.
        else:
            if self.latex:
                plt.ylabel(r"$a_tE_{\textup{lab}}$") #but the use of "\textup{}" command requires a latex compiler
            else: 
                plt.ylabel(r"$a_tE_{lab}$")

        plt.xlabel(r"$t/a_t$")

        # Annotate the plot with additional information if provided
        labels = []
        if op1:
            labels.append(f"snk: {str(op1)}") #double check that I'm not messing up sink and source
        if op2:
            labels.append(f"src: {str(op2)}")
        if labels:
            self.moving_textbox(labels)

        plt.tight_layout()

    def sigmond_corrfit_plot_and_save(self,df, fit_result_info, Nt, ptype=0, op1=None, save_pickle = "", save_pdf = "", 
                                      sh_index = 0, color_index = 0, new_trange = None, species = None):
        """runs sigmond_corrfit_plot and saves to pickle and/or pdf as desired, 
            if save_pdf or save_pickle are empty, then does not save"""
        self.sigmond_corrfit_plot(df, fit_result_info, Nt, ptype, op1, sh_index, color_index, new_trange, species)
        if save_pickle:
            self.save_pickle(save_pickle)
        if save_pdf:
            self.save_pdf(save_pdf)

    def sigmond_corrfit_plot(self,df, fit_result_info, Nt, ptype=0, op1=None, sh_index = 0, color_index = 0, new_trange = None, species = None):
        """Generate a correlator plot with fit using Matplotlib."""
        labels = []
        plt.errorbar(x=df["aTime"],y=df["FullEstimate"],yerr=df["SymmetricError"], linewidth=0.0, elinewidth=1.5, capsize=5, color=psettings.colors[color_index], marker=psettings.markers[color_index], zorder=1)

        tmin = fit_result_info["info"].tmin
        tmax = fit_result_info["info"].tmax
        if new_trange!=None:
            tmin = new_trange[0]
            tmax = new_trange[1]

        if ptype==0: #correlator plot
            if fit_result_info["success"]:
                x=np.linspace(tmin, tmax, 100)
                model = fit_result_info["info"].model.sigmond_object(Nt)
                params = []
                for estimate in fit_result_info["estimates"]:
                    params.append(estimate.getFullEstimate())
                y = [model.eval(params, xi) for xi in x]
                plt.plot(x,y, color="black", ls="--")
            if fit_result_info["info"].ratio:
                plt.ylabel(r"$R(t)$")
            else:
                plt.ylabel(r"$C(t)$") 
        else: #effective energy plot
            if fit_result_info["success"]:
                #for single -> use eval for double. 
                energy_index = fit_result_info["info"].energy_index
                #plot fit line
                if fit_result_info["info"].model.short_name!="1-exp":
                    x=np.linspace(tmin, tmax, tmax-tmin+1)
                    model = fit_result_info["info"].model.sigmond_object(Nt)
                    #if sim fit, grab the fit params that correspont to the correlator being plotted
                        #corrlator is indicated by sh_index => 0 - interacting correlator, 1,2 - single hadron correlators
                    if fit_result_info["info"].sim_fit: #make distinct between Deg/2-3exponential
                        if sh_index==0:
                            min_index = 0
                            max_index = fit_result_info["info"].num_params
                            params = [estimate.getFullEstimate() for estimate in fit_result_info["estimates"][min_index:max_index]] 
                        if sh_index==1:
                            model=fit_info.FitModel.TimeForwardTwoExponential.sigmond_object(Nt)
                            min_index = fit_result_info["info"].num_params
                            indexes = [min_index,min_index+1,2,min_index+2]
                            params = [fit_result_info["estimates"][i].getFullEstimate() for i in indexes]
                        if sh_index==2:
                            model=fit_info.FitModel.TimeForwardTwoExponentialForCons.sigmond_object(Nt)
                            min_index = fit_result_info["info"].num_params+3
                            indexes = [min_index,min_index+1,2,3,min_index+2]
                            params = [fit_result_info["estimates"][i].getFullEstimate() for i in indexes]
                        energy_index = min_index #correspond to fit model
                    else: #just plot fit line
                        params = []
                        for estimate in fit_result_info["estimates"]:
                            params.append(estimate.getFullEstimate())
                    y = [-np.log(model.eval(params, x[i+1])/model.eval(params, x[i])) for i in range(len(x)-1)]
                    plt.plot(x[:-1]+0.5,y, color="black", ls="--")
                energy_result = fit_result_info["estimates"][energy_index].getFullEstimate()
                energy_err = fit_result_info["estimates"][energy_index].getSymmetricError()
                plt.hlines(energy_result, tmin, tmax, color="black", zorder=2)
                
            yscale = "$a_t "
            if fit_result_info["info"].ratio:
                yscale+=r"\delta "
            if species is not None:
                yscale+=f"m_{{{psettings.latex_format[species].replace('$','')}}}"
            elif self.latex:
                yscale+=r"E_{\textup{lab}}"
            else:
                yscale+=r"E_{lab}"
            yscale+="$"

            plt.ylabel(yscale)
            if fit_result_info["success"]:
                nice_value = utils.nice_value(energy_result,energy_err)
                labels.append(f"{yscale}={nice_value}")

        plt.xlabel(r"$t/a_t$")

        # Annotate the plot with additional information if provided
        if op1:
            labels.insert(0,f"corr: {str(op1)}") 
        if fit_result_info["success"]:
            labels.append(f"$\chi^2/dof={format(float(fit_result_info['chisqrdof']),'.2f')}$")

        self.moving_textbox(labels)

        plt.tight_layout()

    def set_y_logscale(self):
        plt.yscale('log')
    def set_y_linscale(self):
        plt.yscale('linear')

    #tmin or tmax plots
    def add_fit_series(self, t,e,de,color_index, filled=True, model=None): #incorporate pvalue and chosen fit
        """add a set of fits to a tmin or tmax plot"""
        if filled:
            marker_color = psettings.colors[color_index]
        else:
            marker_color = "white"

        plt.errorbar(x=np.array(t)+0.05*color_index,y=e,yerr=de, linewidth=0.0, elinewidth=1.5, capsize=5, color=psettings.colors[color_index], 
                     marker=psettings.markers[color_index], label=model, mfc=marker_color)

    #add horizontal bar to plot corresponding to energy value of chosen fit
    def add_chosen_fit(self, energyval, energyerr, label="chosen"):
        """add chosen fit as a horizontal bar to a plot"""
        plt.axhspan(energyval-energyerr, energyval+energyerr, color="lightgray")
        plt.axhline(energyval,color="black",ls="--", label=label)

    #tmin or tmax or t_whatever plots
    def finalize_fit_series_plot(self, series_type="min", title=None, ratio=False, species = None):
        """finalize a tmin or tmax plot"""
        # if self.latex:
        #     if ratio:
        #         yscale = r"$a_t \delta E_{\textup{lab}}$" #but the use of "\textup{}" command requires a latex compiler
        #     else:
        #         yscale = r"$a_tE_{\textup{lab}}$" #but the use of "\textup{}" command requires a latex compiler
        # else: 
        #     if ratio:
        #         yscale = r"$a_t \delta E_{lab}$" #unsure if \delta needs latex
        #     else:
        #         yscale = r"$a_tE_{lab}$"

        yscale = "$a "
        if ratio:
            yscale+=r"\Delta "
        if species is not None:
            yscale+=f"M_{{{psettings.latex_format[species].replace('$','')}}}"
        elif self.latex:
            yscale+=r"E_{\textup{lab}}"
        else:
            yscale+=r"E_{lab}"
        yscale+="$"

        if self.latex:
            plt.xlabel(rf"$t_{{\textup{{{series_type}}}}}/a$")
        else:
            plt.xlabel(rf"$t_{{{series_type}}}/a$")
        plt.ylabel(yscale)
        plt.legend(title=title)
        plt.tight_layout()

    #simple labeled line plot
    def show_trend(self, x, y, ylabel=None, legend=False):
        plt.plot(x,y, label=ylabel)
        if legend:
            plt.legend()

    def legend(self):
        plt.legend()

    def ylim(self, ymin=None, ymax=None):
        if ymin==None or ymax==None:
            return plt.ylim()
        else:
            return plt.ylim((ymin, ymax))

    def summary_plot(self,indexes,levels,errs,xticks, reference=None, thresholds=[], label=None, index=0, ndatasets=1, shift=False, filled = True):
        """Summary of spectrum plot"""
        indexes = np.array(indexes)
        levels = np.array(levels)
        errs = np.array(errs)
        shifted_array = shift_levels(indexes,levels,errs)
        if type(shifted_array)==float:
            columns = 1.0
        else:
            columns = max(shifted_array)-min(shifted_array)+1.0
        shifted_array = shifted_array/columns/ndatasets
        shifted_array += index/ndatasets-0.5+0.5/ndatasets

        if filled:
            mfc = psettings.colors[index]
        else:
            mfc = 'white'
        
        plt.errorbar(x=indexes+shifted_array, y=levels, yerr=errs,linewidth=0.0, elinewidth=1.5, capsize=5, 
                     color=psettings.colors[index], marker=psettings.markers[index],mfc=mfc, label=label)

        if index==ndatasets-1:
            plt.xlim(min(indexes)-1.0,max(indexes)+1.0)
            dd = 0.005
            for line in thresholds:
                plt.axhline(line[1], color="black", ls="--")
                line_label = ""
                for particle in line[0]:
                    line_label+=psettings.latex_format[particle]
                minx,maxx = plt.xlim()
                miny,maxy = plt.ylim()
                plt.text(minx+dd*(maxx-minx), line[1]+dd*(maxy-miny), line_label )
            if len(xticks[0])==2:
                xticks = [f"{psettings.latex_format[irrep]}({mom})" for (irrep, mom) in xticks]
                rotation = 0
            elif len(xticks[0])==3:
                xticks = [f"{psettings.latex_format[irrep]}({mom}) {level}" for (irrep, mom, level) in xticks]
                rotation = 90

            plt.xticks(list(range(len(xticks))),xticks,size="small", rotation=rotation)
            if self.latex:
                if reference:
                    latex_rest_mass = psettings.latex_format[reference].replace('$',"")
                    plt.ylabel(rf"$E_{{\textup{{cm}}}}/E_{{{latex_rest_mass}}}$") #change to ref
                else:
                    if shift:
                        plt.ylabel(r"$a_t \delta E_{lab}$") #change to ref
                    else:
                        plt.ylabel(r"$a_t E_{\textup{cm}}$") #change to ref
            else:
                if reference:
                    latex_rest_mass = psettings.latex_format[reference].replace('$',"")
                    plt.ylabel(rf"$E_{{cm}}/E_{{{latex_rest_mass}}}$") #change to ref
                else:
                    if shift:
                        plt.ylabel(r"$a_t \delta E_{lab}$")
                    else:
                        plt.ylabel(r"$a_t E_{cm}$")
            if label:
                plt.legend(fontsize=15)

    def plot_operator_overlaps(self, values, errors, opname=""):
        """histogram of operator overlaps"""
        plt.bar(range(len(values)),values)
        plt.errorbar(x=range(len(values)), y=values, yerr=errors,linewidth=0.0, elinewidth=1.5, capsize=5, marker=None, color="black")
        plt.xlabel("rotate level")
        plt.ylabel("$|Z|^2$")
        if opname:
            self.moving_textbox([opname])

    #these pickle can only be opened on same system as it was generated 
    # (they are written with bytes because that was the only it would let me)
    def save_pickle( self, filename):
        """Save the current Matplotlib figure as a pickle file."""
        # fig = copy.deepcopy(self.fig)
        with open(filename, "wb") as f:
            pickle.dump(self.fig, f)

    def save_pdf( self, filename, transparent=True):
        """Save the current Matplotlib figure as a PDF file."""
        plt.savefig( filename, transparent=transparent ) 

    ###########################
    ##### Leuscher Plots ######
    ###########################
    #
    # compare chi2 results to input energies
    def chi2_energies_compare_plot(self, fig_params, channel, irreps, x_in,err_in):
        figwidth, figheight = fig_params
        # channel_1 = str(channel.split(',')[0])
        # channel_2 = str(channel.split(',')[1])
        plt.figure(figsize=(figwidth,figheight))
        # x axis is the irrep
        data_energies, chi2_energies = x_in
        ecm_bootstrap, chi2_energies_error = err_in 
        # y  is the data, one from data and one from fit
        x_range= [0,1,2,3]
        offset = 0.1
        irrep_list = []
        total_irrep = 0
        for psq in irreps:
                for irrep in irreps[psq][0]:     
                    irrep_list.append(irrep)
                    for level in irreps[psq][0][irrep]:
                        level_title = f"ecm_{level}_ref"
                        err = np.std(ecm_bootstrap[psq][irrep][level_title])
                        plt.errorbar(x_range[total_irrep]-offset,data_energies[psq][irrep][level_title], yerr=err,linewidth=0.0, elinewidth=1.5, capsize=5, marker='s',markersize=10, color="blue")
                        #plt.plot(x_range[total_irrep]-offset,data_energies[psq][irrep][level_title],marker='s',markersize=10,color='blue')
                        #plt.plot(x_range[total_irrep]+offset, chi2_energies[psq][irrep][level],marker='s',markersize=10,color='red')
                        plt.errorbar(x_range[total_irrep]+offset,chi2_energies[psq][irrep][level], yerr=chi2_energies_error[psq][irrep][level],linewidth=0.0, elinewidth=1.5, capsize=5, marker='s',markersize=10, color="red")
                        total_irrep += 1
        # Set the x-ticks using the collected irreps
        plt.xticks(x_range, irrep_list)
        plt.ylabel("$E^{*} / m_{\pi}$",fontsize=16)


    # input can be data files generated, and fit
    # input is shapes_dict, label_dict, 
    # energies is the average energies,
    def single_channel_plot( self,fig_params, channel, irreps, x_in, y_in):
        figwidth, figheight = fig_params
        channel_1 = str(channel.split(',')[0])
        channel_2 = str(channel.split(',')[1])
        plt.figure(figsize=(figwidth,figheight))
        # CONDITION FOR BOUND STATES 
        q2_values = np.linspace(-1, 0.0001, 175)
        virtual_state = []
        bound_state = []
        for q2 in q2_values:
            virtual_state.append(cmath.sqrt(-q2))
            bound_state.append(cmath.sqrt(q2))
        markers = ['o', 's', '^', 'D', 'v', '*']
        shapes = []
        marker_to_shape = 0
        for psq in irreps:
            for irrep in irreps[psq][0]:
                shapes.append(markers[marker_to_shape])
                marker_to_shape += 1

        #shapes = ['o','s','D','v',]
        colors = ['tab:blue','firebrick','green','purple']
        legend_handles = []
        x, x_range = x_in
        y, y_range = y_in
        shape_count = 0
        for psq in irreps:
                irrep_count = 0
                for irrep in irreps[psq][0]:
                    marker = shapes[shape_count]
                    shape_count += 1
                    irrep_count += 1
                    level_count = 0
                    for level in irreps[psq][0][irrep]:
                        if len(irrep) > 1:
                            label = f"${irrep[0]}_{irrep[1:]}$"
                        else:
                            label = f"${irrep[0]}$"
                        color = colors[level_count]                   
                        level_count += 1
                        plt.plot(x_range[psq][irrep][level], y_range[psq][irrep][level], color=color, alpha=0.75)
                        plt.plot(x[psq][irrep][level],y[psq][irrep][level] , marker=marker, color=color, label=label)
                        legend_handles.append(Line2D([0], [0], marker=marker,color=color, markerfacecolor=color, markersize=10,  label=label))
        plt.axhline(y=0,color='black')
        plt.axvline(x=0,color='black')
        legend = plt.legend(handles=legend_handles, loc='upper left', title='Legend', prop={'size': 10})
        plt.xlabel("$q^{*2} / m_{\pi}^2$",fontsize=16)
        plt.ylabel("$q^{*} / m_{\pi} \cot \delta $",fontsize=16)
        plt.title(f'{channel_1},{channel_2}  Scattering ',fontsize=16)       

    def parametrization_fit_plot( self,energies, fit_results, irreps, x_in, y_in):
        figwidth, figheight = fig_params
        channel_1 = str(channel.split(',')[0])
        channel_2 = str(channel.split(',')[1])
        plt.figure(figsize=(figwidth,figheight))
        # CONDITION FOR BOUND STATES 
        q2_values = np.linspace(-1, 0.0001, 120)
        virtual_state = []
        bound_state = []
        for q2 in q2_values:
            virtual_state.append(cmath.sqrt(-q2))
            bound_state.append(cmath.sqrt(q2))
        shapes = ['o','s','D','v']
        colors = ['tab:blue','firebrick','green','purple']
        legend_handles = []
        x, x_range = x_in
        y, y_range = y_in
        for psq in irreps:
                irrep_count = 0
                for irrep in irreps[psq][0]:
                    marker = shapes[irrep_count]
                    irrep_count += 1
                    level_count = 0
                    for level in irreps[psq][0][irrep]:
                        if len(irrep) > 1:
                            label = f"${irrep[0]}_{irrep[1:]}$"
                        else:
                            label = f"${irrep[0]}$"
                        color = colors[level_count]                   
                        level_count += 1
                        plt.plot(x_range[psq][irrep][level], y_range[psq][irrep][level], color=color, alpha=0.75)
                        plt.plot(x[psq][irrep][level], y[psq][irrep][level], marker=marker, color=color, label=label)
                        legend_handles.append(Line2D([0], [0], marker=marker,color=color, markerfacecolor=color, markersize=10,  label=label))
        plt.axhline(y=0,color='black')
        plt.axvline(x=0,color='black')
        legend = plt.legend(handles=legend_handles, loc='upper left', title='Legend', prop={'size': 10})
        plt.xlabel("$q^{*2} / m_{\pi}^2$",fontsize=16)
        plt.ylabel("$q^{*} / m_{\pi} \cot \delta $",fontsize=16)
        plt.title(f'{channel_1},{channel_2}  Scattering ',fontsize=16)         



    ###########################
    ##### pylatex actions #####
    ###########################
    
    def create_summary_doc(self, title):
        """Create a new LaTeX document with a specified title."""
        self.doc.append(utils.create_doc(title))

    def append_section(self, title, index = 0):
        """Append a new section to the LaTeX document."""
        self.doc[index].append(pylatex.Command("section",title))

    def append_subsection(self, title, index = 0):
        """Append a new section to the LaTeX document."""
        self.doc[index].append(pylatex.Command("subsection",title))

    def append_subsubsection(self, title, index = 0):
        """Append a new section to the LaTeX document."""
        self.doc[index].append(pylatex.Command("subsubsection",title))

    def add_correlator_subsection(self,corrname, leftplotfile, rightplotfile, index = 0): 
        """Add a subsection with two plots to the LaTeX document."""
        if not os.path.exists(leftplotfile) and not os.path.exists(rightplotfile):
            logging.warning(f"Unable to include {corrname} in summary pdf.")
            return

        with self.doc[index].create(pylatex.Subsubsection(corrname)):
            self.include_additional_plots(leftplotfile,rightplotfile, index)
    
    def include_additional_plots(self, leftplotfile, rightplotfile, index = 0): #add table of estimates?, list of correlators?
        """Add a subsection with two plots to the LaTeX document."""

        rlinewidth = r'\linewidth'
        llinewidth = r'\linewidth'
        if not os.path.exists(leftplotfile) and not os.path.exists(rightplotfile):
            return

        with self.doc[index].create(pylatex.Figure(position='H')) as thisfig:
            if os.path.exists(leftplotfile) and os.path.exists(rightplotfile):
                with self.doc[index].create(pylatex.SubFigure(
                    position='b', width=pylatex.NoEscape(r'0.5\linewidth'))) as left_fig:
                    left_fig.add_image(leftplotfile,width=pylatex.NoEscape(llinewidth), placement=pylatex.NoEscape("\centering") )
                with self.doc[index].create(pylatex.SubFigure(
                    position='b', width=pylatex.NoEscape(r'0.5\linewidth'))) as right_fig:
                    right_fig.add_image(rightplotfile,width=pylatex.NoEscape(rlinewidth), placement=pylatex.NoEscape("\centering") )
            elif os.path.exists(leftplotfile):
                thisfig.add_image(leftplotfile,width=pylatex.NoEscape(r'0.5\linewidth'), placement=pylatex.NoEscape("\centering") )
            elif os.path.exists(rightplotfile):
                thisfig.add_image(rightplotfile,width=pylatex.NoEscape(r'0.5\linewidth'), placement=pylatex.NoEscape("\centering") )

    def add_single_plot(self, plotfile, index=0):
        """add only one plot the textwidth"""
        if not os.path.exists(plotfile):
            logging.warning(f"Unable to include {plotfile} in summary pdf.")
            return
        
        with self.doc[index].create(pylatex.Figure(position='H')) as thisfig:
            thisfig.add_image(plotfile,width=pylatex.NoEscape(r'\linewidth'), placement=pylatex.NoEscape("\centering") )


    def summary_table(self, reference, headers, data, title = "", index = 0):
        """a table that includes the fit information of the correlators"""
        if reference:
            latex_rest_mass = psettings.latex_format[reference].replace('$',"")
            headers = [header.replace("latex_rest_mass",latex_rest_mass) for header in headers]
        headers = [pylatex.NoEscape(header) for header in headers]
        with self.doc[index].create(pylatex.Center()) as centered:
            centered.append(pylatex.utils.bold(pylatex.NoEscape(title)))
            with centered.create(pylatex.LongTable("|".join(['c']*len(headers)))) as current_table:
                current_table.add_row(headers)
                current_table.add_hline()
                for line in data:
                    line = [psettings.latex_format[col] if col in psettings.latex_format.keys() else col for col in line]
                    line = [pylatex.NoEscape(col) for col in line]
                    current_table.add_row(line)
        self.doc[index].append(pylatex.NoEscape("\n\n"))

    def add_operator_overlaps(self, files, index=0):
        """creates a section called 'Operator Overlaps' 
        and fills it in with the given file list"""
        with self.doc[index].create(pylatex.Subsubsection("Operator Overlaps")):
            self.add_plot_series(files,index)

    def add_plot_series(self, files, index=0):
        """adds a series of plots in a two columns"""
        for x,y in zip(files[::2],files[1::2]):
            self.include_additional_plots(x,y, index)
        if len(files)%2:
            self.include_additional_plots(files[-1],files[-1]+"2")

    def compile_pdf(self, filename, index = 0):
        """Compile the LaTeX document into a PDF file."""
        utils.compile_pdf(self.doc[index],filename) #detect compiler? -> if self.latex, check for pylatex?
