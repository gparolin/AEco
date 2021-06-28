import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import rc
import warnings
import squarify
import xarray as xr
import dask.dataframe as dd
import dask.array as da
#import bottleneck
import scipy


class Parameter():
    """Input parameter with a Beta-PERT distribution."""
    
    def __init__(self, expected=1, minimum=1, maximum=1, name='', unit='', description=''):
        """Initialization method with values, name and unit."""
        
        self.name = name
        self.unit = unit
        self.description = description
        self.minimum = minimum
        self.expected = expected
        self.maximum = maximum
        
    def __repr__(self):
        message = f"""Parameter '{self.name}':
        Minimum: {self.minimum}
        Expected: {self.expected}
        Maximum: {self.maximum}"""
        return message
 
    def PERT(self):
        """Calculates PERT values for the parameter."""
        
        if self.minimum == self.maximum:
            self.alpha = self.expected
            self.beta = self.expected
            self.mean = self.expected
        elif self.minimum > self.expected or self.maximum < self.expected:
            raise ValueError("Nominal must be equal to or between lower and upper bounds")
        else:
            self.mean = (self.minimum + 4*self.expected + self.maximum)/6
            self.alpha = 6*((self.mean - self.minimum)/(self.maximum - self.minimum))
            self.beta = 6*((self.maximum - self.mean)/(self.maximum - self.minimum))
    
    def resample(self, size=100, chunks='auto'):
        """ Samples the parameter based on the PERT values and given size."""
        
        sample = (da.random.beta(self.alpha, self.beta, size, chunks=chunks)
                  *(self.maximum - self.minimum)) + self.minimum  
        self.sample = xr.DataArray(sample, 
                            dims='i', 
                            coords=np.arange(size).reshape(1,size), 
                            name=self.name,
                            attrs={'unit': self.unit, 'description': self.description})
        return self.sample

class ParameterSet():
    """A set of Parameter objetcs."""

    def __init__(self, pandas, sample_size, chunks):
        """Converts inputs DataFrame to a set of Parameter objects (a ParameterSet)."""
    
        param_dict = {}
        for variable in pandas.index:
            p = pandas.loc[variable]
            param = Parameter(p["expected"], p["minimum"],p["maximum"],
                     name=variable, unit=p["unit"], description=p["description"])
            param.PERT()
            param_dict[variable] = param
        self.parameters = param_dict
        self.sample(sample_size, chunks)
        self.to_Dataset()
    
    def __repr__(self):
        return f"{self.data}"
    
    def __getitem__(self, variable):
        return self.data[variable]
    
    def __setitem__(self, variable, value):
        self.data[variable] = value
        
    def sample(self, sample_size, chunks):
        """Samples the initial self.parameters dict of Parameters with a given number of iterations."""
        for variable in self.parameters:
            self.parameters[variable].resample(sample_size, chunks)
    
    def to_Dataset(self):
        """Converts the ParameterSet object to a xr.Dataset of samples"""
        
        dataset = xr.Dataset()
        for variable in self.parameters:
            if np.isnan(self.parameters[variable].expected):
                pass
            else:
                dataset[variable] = self.parameters[variable].sample
                dataset[variable].attrs['max'] = self.parameters[variable].maximum
                dataset[variable].attrs['min'] = self.parameters[variable].minimum
                dataset[variable].attrs['exp'] = self.parameters[variable].expected
                dataset[variable].attrs['alpha'] = self.parameters[variable].alpha
                dataset[variable].attrs['beta'] = self.parameters[variable].beta
                dataset[variable].attrs['mean'] = self.parameters[variable].mean
        
        self.data = dataset
        
        return self.data

def read_inputs(input_path, input_sheet):
    """Reads input files. Outputs DataFrame of inputs."""
    
    with pd.ExcelFile(input_path) as xlsx:
        inputs = pd.read_excel(xlsx, input_sheet, header = 2, index_col = 0, na_values="", usecols="B:G")
    
    inputs = inputs.rename(columns = {"nominal": "expected", 
                                            "low":"minimum", 
                                            "high":"maximum"}) #renomear título das colunas

    return inputs

def func_unit(aircraft_type, p):
    """Calculates pkm (or tkm) for the aircraft."""
    if aircraft_type == "cargo":
        p["pkm_flight"] = p["payload"] * p["d_f"] * p["loadfactor"]

    elif aircraft_type == "pax":
        p["npax"] = p["seat_max"] * p["p_seat"] * p["loadfactor"]
        p["pkm_flight"] = p["npax"] * p["d_f"]

    else:
        raise Exception("Aircraft type must be 'pax' or 'cargo'")


    p["pkm_year"] = p["pkm_flight"] * p["flights_year"]
    p["pkm_life"] = p["pkm_year"] * p["lifetime"]
    p["pkm_fleet"] = p["pkm_life"] * p["fleet"]
    
    return p

def read_unit_processes(database_path):
    """Reads unit_processes excel file from database_path as a pd.DataFrame."""
    
    with pd.ExcelFile(database_path) as xlsx:
        unit_processes = pd.read_excel(xlsx, 'UP', header=5, index_col=0, na_values=0)
        unit_processes.set_index(['compartment','subcompartment'], 
                                 append=True, inplace=True)
        
    return unit_processes

def unit_process_dataset(pandas):
    """Transforms UPs in a pd.DataFrame to a xr.Dataset."""
    ds = xr.Dataset(pandas)
    ds = ds.rename({'dim_0':'Substances'})
    ds = ds.set_coords('unit')
    ds = ds.rename({'unit':'Units'})
    ds = ds.fillna(0)
    return ds

def read_CF(database_path):
    """ Reads the excel file containing CFs and returns midpoint and endpoint factors."""
    
    with pd.ExcelFile(database_path) as xlsx:
        MP = pd.read_excel(xlsx, 'MP', header = 0, index_col = 0, na_values=0)
        EP = pd.read_excel(xlsx, 'EP', header = 0, index_col = 0, na_values=0)
        return MP, EP

class Midpoint():
    """ Class representing midpoint characterization factors."""
    
    def __init__(self, raw, substances):
        """ Initialize the CF class with a raw CF dataframe."""
        
        self.categories = [['freshwater eutrophication', 'FE', '$yr \cdot kg / m^3$'],
                           ['metal depletion', 'MRD', '$kg^{-1}$'],
                           ['photochemical oxidant formation', 'POF', '$kg$'], 
                           ['marine ecotoxicity', 'MET', '$m^{2} \cdot yr$'],
                           ['terrestrial acidification', 'TA','$yr \cdot m^{2}$'],
                           ['urban land occupation', 'ULO', '$m^{2} \cdot yr$'],
                           ['particulate matter formation','PMF','$kg$'],
                           ['terrestrial ecotoxicity','TET','$m^{2} \cdot yr$'],
                           ['freshwater ecotoxicity','FET','$m^{2} \cdot yr$'],
                           ['natural land transformation','NLT','$m^2$'],
                           ['ozone depletion','OD','$ppt \cdot yr$'],
                           ['marine eutrophication','ME','$yr \cdot kg / m^3$'],
                           ['agricultural land occupation','ALO','$m^{2} \cdot yr$'],
                           ['human toxicity','HT','--'],
                           ['ionising radiation','IR','$man \cdot Sv$'],
                           ['fossil depletion','FD','$MJ$'],
                           ['water depletion','WD','$m^3$'],
                           ['climate change', 'CC', '$W \cdot yr / m^{2}$']]
        self.AOP = [['Damage to Human Health', 'HH', 'yr'],
                    ['Damage to Ecosystem Diversity', 'ED', '$'],
                    ['Damage to Resource Availability', 'RA', '$']]
        self.raw = raw
        self.substances = substances
        self.build()


    def build(self):
        self.abbrv(lst=self.categories)
        self.unit(lst=self.categories)
        self.to_dataset()
        self.to_array()
        
    def __repr__(self):
        return f"{self.array}"
    
    def __getitem__(self, item):
        return self.dataset[item]
    
    def abbrv(self, lst):
        """ Generates a category:abbreviation dictionary"""
        
        d = dict()
        for item in lst:
            d[item[0]] = item[1]
        self.abbrvs = d
        
        return self.abbrvs
    
    def unit(self, lst):
        """ Generates a abbreviation:unit dictionary."""
        
        d = dict()
        for item in lst:
            d[item[1]] = item[2]
        self.units = d
        
        return self.units
   
    def to_dataset(self):
        """Transforms the raw MP dataframe to a xr.Dataset."""

        # Adjusting the MP dataframe
        df = pd.DataFrame(index=self.substances.indexes["Substances"])
        for category in self.abbrvs:
            cf = self.raw.loc[category]
            cf = cf.set_index(['name','compartment','subcompartment'])
            cf = cf.squeeze()
            cf = cf.rename(category)
            df[category] = cf
        df = df.fillna(0)   

        # Creating dataset
        ds = xr.Dataset(df)

        # Renaming coords and data_vars
        ds = ds.rename({'dim_0':'Substances'})
        ds = ds.rename(self.abbrvs)
        #ds = ds.fillna(0)

        # Adding unit attributes
        for var in ds:
            ds[var].attrs['Unit'] = self.units[var]
            
        self.dataset = ds
        
        return self.dataset
    
    def to_array(self):
        """Turn the dataset to a xr.DataArray."""
        
        self.array = self.dataset.to_array()
        
        return self.array

class Endpoint(Midpoint):
    """ Class representing endpoint characterization factors."""

    def build(self):
        self.unit(lst=self.AOP)
        self.to_dataset()
        self.to_array()
    
    def to_dataset(self):
        """Transforms the raw EP dataframe to a xr.Dataset."""

        # Adjusting DataFrame
        df = self.raw.rename(index=self.abbrv(lst=self.categories), columns=self.abbrv(lst=self.AOP))
        
        # Creating dataset
        ds = xr.Dataset(df)

        # Adjusting dataset
        ds = ds.rename({'dim_0':'Categories'})
        ds = ds.fillna(0)

        # Adding unit attributes
        for var in ds:
            ds[var].attrs['Unit'] = self.units[var]
            
        self.dataset = ds
        
        return self.dataset

class CharactFactors():
    """Represents the Characterization Factors for both midpoint and endpoint."""
    
    def __init__(self, MP_raw, EP_raw, substances):
        self.MP = Midpoint(MP_raw, substances)
        self.EP = Endpoint(EP_raw, substances)

    def __repr__(self):
        return "Characterization factors. To access them, try 'CharactFactors.MP' or '~.EP'"

def covariance_gufunc(x, y):
    return ((x - x.mean(axis=-1, keepdims=True))
            * (y - y.mean(axis=-1, keepdims=True))).mean(axis=-1)

def pearson_correlation_gufunc(x, y):
    return covariance_gufunc(x, y) / (x.std(axis=-1) * y.std(axis=-1))

def spearman_correlation_gufunc(x, y):
    x_ranks = scipy.stats.rankdata(x, axis=-1)
    y_ranks = scipy.stats.rankdata(y, axis=-1)
    return pearson_correlation_gufunc(x_ranks, y_ranks)

def spearman_correlation(x, y, dim):
    return xr.apply_ufunc(
        spearman_correlation_gufunc, x, y,
        input_core_dims=[[dim], [dim]],
        dask='parallelized',
        output_dtypes=[float])

class LCIA():
    """Represent LCIA results."""

    def __init__(self, LCI=None, CF=None, CTV=None, MP=None, EP=None):
        """Initializes the LCIA object with a LCI and CF."""
        
        self.LCI = LCI
        self.CF = CF
        self.CTV = CTV
        self.MP = MP
        self.EP = EP
        self.phases = {'Development': ['Office','Infrastructure','Capital','Prototypes','Certification'],
                       'Manufacturing': ['Materials','Factory','Logistics','Sustaining'],
                       'Operation': ['LTO','CCD','Maintenance','Airport','Fuel'],
                       'End-of-Life': ['Recycling','Landfill','Incineration']
                      }
        
    def __repr__(self):
        return f"LCIA results of the {self.MP.attrs['Name']}"
    
    def __getitem__(self, item):
        return xr.concat((self.MP[item], self.EP[item]), dim='Pathway')
    
    @classmethod
    def build(cls, LCI, CF):
        """Assesses midpoint and endpoint impacts. Returns the LCI.MP and LCI.EP datasets."""
        
        MP = (LCI.data * CF.MP.array).sum('Substances')
        MP.attrs = LCI.data.attrs
        MP = MP.rename({'variable':'Categories'})
        
        EP = (MP * CF.EP.array).sum('Categories')
        EP.attrs = MP.attrs
        EP = EP.rename({'variable':'AOP'})

        return cls(LCI=LCI, CF=CF, MP=MP, EP=EP)
    
    def SS(self, by="phase"):
        """Returns the LCA Single Score as a xr.Dataset."""
        
        cf = np.array([1.35e-2*400,9.17e-4*400,2.45e-2*200])
        ss = xr.DataArray(cf, coords={'AOP':self.EP.AOP.data}, dims='AOP')
        
        return (self.groupby(by, "EP") * ss).sum('AOP')
                         
    def mean(self, pathway="MP", by="subphase"):
        """Returns a xr.Dataset with the mean result for the chosen pathway and groupby.
        
        By can be either "sum", which returns a dataset summed over all phases;
                         "phases", which returns a dataset grouped by phases; or
                         "subphases", which returns a dataset grouped by subphases. 
        """
        
        return self.groupby(by=by, pathway=pathway).mean('i')

    def groupby(self, by="phase", pathway="MP"):
        """Returns xr.Dataset of desired pathway with aggregated life cycle phases.
        
        By can be either "sum", which returns a dataset summed over all phases;
                         "phases", which returns a dataset grouped by phases; or
                         "subphases", which returns a dataset grouped by subphases. 
        """
        
        if pathway == "MP":
            data = self.MP
        elif pathway == "EP":
            data = self.EP
        
        if by == "phase":
            ds = xr.Dataset()
            for phase in self.phases:
                ds[phase] = data[self.phases[phase]].to_array().sum('variable')
            return ds
        elif by == "sum":
            return data.to_array('Phases').sum('Phases')
        else:
            return data
            
    def build_CTV(self, parameterset):
        """Calculates the CTV of the parameterset for MP and EP."""
        
        if self.CTV == None:
            og_params = xr.Dataset()
            for param in [*parameterset.parameters]:
                try:
                    og_params[param] = parameterset.data[param]
                except:
                    pass
            param_arr = og_params.to_array('Parameters')
            self.MP_array = self.MP.to_array().sum('variable').to_dataset('Categories')
            self.EP_array = self.EP.to_array().sum('variable').to_dataset('AOP')
            self.array = xr.merge([self.MP_array,self.EP_array])
            
            param_arr_nochunk = param_arr.loc[{'i':slice(999)}].chunk({'i':-1})
            array_nochunk = self.array.loc[{'i':slice(999)}].chunk({'i':-1})
            
            corr = spearman_correlation(param_arr_nochunk,array_nochunk,dim='i')
            corr_sq = corr ** 2
            corr_sum = corr_sq.sum(dim='Parameters')
            ctv = corr_sq / corr_sum
            self.CTV = ctv.fillna(0)*100
            
        return self.CTV
    
    def save(self, path, LCI=True):
        """Saves LCIA object to a NetCDF file."""
        
        
        self.MP.to_netcdf(path, group='MP', mode='w',engine='h5netcdf')
        self.EP.to_netcdf(path, group='EP', mode='a',engine='h5netcdf')
        
        if LCI:
            self.LCI.data.reset_index("Substances").to_netcdf(path, group='LCI', mode='a', engine='h5netcdf')
        
        if self.CTV != None:
            self.CTV.to_netcdf(path, group='CTV', mode='a', engine='h5netcdf')
        
        return print(f"LCIA saved at {path}")
    
    def to_excel(self, path, LCI=False):
        """Saves LCIA object to a .xlsx file."""
        
        if path[-5:] == '.xlsx':
            path = path
        else:
            path = path + '.xlsx'
        
        with pd.ExcelWriter(path) as writer:
            self.MP.to_dataframe().to_excel(writer, sheet_name='MP')
            self.EP.to_dataframe().to_excel(writer, sheet_name='EP')
            
            if self.CTV != None:
                self.CTV.to_dataframe().to_excel(writer, sheet_name='CTV')
                
            if LCI:
                self.LCI.data.reset_index("Substances").to_netcdf(path, group='LCI', mode='a')
        
        return print(f"LCIA saved at {path}")
    
    @classmethod
    def load(cls, path, chunks={}, CTV=False, LCI=False):
        """Loads NetCDF file from path.
        
        chunks: specify the chunks to pass the xr.open_dataset function.
        CTV: boolean that specifies if a CTV group if present on the NetCDF file.
        """
        
        ctv = None
        lci = None
        
        with xr.open_dataset(path, chunks=chunks, group='MP') as mp, \
             xr.open_dataset(path, chunks=chunks, group='EP') as ep:
            
            if LCI:
                with xr.open_dataset(path, chunks=chunks, group='LCI') as ds:
                    lci = ds.set_index({"Substances":["name","compartment","subcompartment"]})
        
            if CTV:
                with xr.open_dataset(path, chunks=chunks, group='CTV') as ds:
                    ctv = ds
            
            return cls(MP=mp, EP=ep, LCI=lci, CTV=ctv)
        
    
    def dist(self, pathway='MP', save=False, palette='GnBu'):
        """Plots the distribution of iterations for the pathway's categories."""
        
        fig = plt.figure(figsize=(13, 19), dpi=150)
        sns.despine(left=True)
        sns.set(style="ticks", palette=palette, color_codes=True)

        outer = gridspec.GridSpec(6, 3, wspace=0.2, hspace=0.35)
        
        if pathway == 'MP':
            da = self.MP.to_array().sum('variable')
            lst = self.MP['Categories'].data
            name = self.MP.attrs['Name'] + "_MP_"
            finder = 'Categories'
            
        elif pathway == 'EP':
            da = self.EP.to_array().sum('variable')
            lst = self.EP['AOP'].data
            name = self.EP.attrs['Name'] + "_EP_"
            finder = 'AOP'
        
        i = 0
        for cat in lst:

            inner = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[i], 
                                                     wspace=0.1, hspace=0.1, height_ratios=(.15, .85))
            
            data = da.loc[{finder:cat}]
            
            ax = plt.Subplot(fig, inner[0])
            ax.set_xticks([])
            sns.boxplot(data, ax=ax, color='seagreen', notch=True)
            ax.set_xlabel('')
            ax.set_yticks([])
            fig.add_subplot(ax)
            i += 1

            ax = plt.Subplot(fig, inner[1])
            ax.axvline(data.mean(), 0, 1, color='darkgreen', ls= '--')
            sns.histplot(data, kde=False, color='forestgreen', ax=ax)
            ax.set_yticks([])
            ax.set_xlabel(cat, fontsize=12)
            ax.ticklabel_format(axis='x', style='sci', scilimits=(0,0), useMathText=True)
            ax.set_xticks([data.min(), data.mean(), data.max()])
            fig.add_subplot(ax)
        
        if save:
            fig.savefig('.\\Outputs\\' + name + 'dist.pdf', bbox_inches='tight')    
        
        plt.show()
        
    
    def bar(self, pathway='MP', subphase=True, save=False, palette='deep'):
        """Plots the percentual contribution of each life cycle phase to the pathway's categories."""
        
        if pathway == 'MP':
            lst = self.MP['Categories'].data
            name = self.MP.attrs['Name'] + "_MP_"
            finder = 'Categories'
            xlabel = "Midpoint categories"
            ylabel = "Percent of midpoint impact'"
            xticks = 18
            ymin = -20
            
        elif pathway == 'EP':
            lst = self.EP['AOP'].data
            name = self.EP.attrs['Name'] + "_EP_"
            finder = 'AOP'
            xlabel = "Endpoint Areas of Protection"
            ylabel = "Percent of endpoint impact'"
            xticks = 3
            ymin = -5
            
        
        if subphase:
            data = self.mean(pathway, by="subphase")
            name = name + "subphase_"
        else:
            data = self.mean(pathway, by="phase")
            name = name + "phase_"
        
        ds = xr.Dataset()
        for cat in lst:
            num = data.loc[{finder:cat}]
            den = abs(num.to_array()).sum()
            pct = num / den * 100
            pct = pct.to_array('Phases').drop(finder)
            ds[cat] = pct
        df = ds.to_dataframe().T
                
        sns.set(style="white", palette=palette, color_codes=False)
        ax = df.plot.bar(stacked= True, figsize=(18,8), width=0.8)
        fig = ax.get_figure()
        fig.set_dpi(150)
        ax.axhline(lw=1, color='k')
        plt.title('')
        plt.ylabel(ylabel, fontsize=13)
        plt.xlabel(xlabel, fontsize=13)
        plt.xticks(ticks=np.arange(0,xticks), rotation=0, horizontalalignment='center')
        plt.yticks(ticks=[ymin,0,20,40,60,80,100], fontsize=13)
        plt.ylim(top=100)
        plt.legend(bbox_to_anchor=(1.01, 0), loc=3, borderaxespad=0.2, edgecolor='w', fontsize=11)
        
        if save:
            fig.savefig('.\\Outputs\\' + name + 'bar.pdf', bbox_inches='tight')
            
        plt.show()
        
    def square(self, save=False):
        """Plots the CTV results for each category."""
        
        name = self.MP.attrs['Name']
        np.set_printoptions(precision=2)
        with PdfPages('.\\Outputs\\' + name + '_CTV.pdf') as pdf:
            for cat in self.CTV:
                da = self.CTV[cat].compute()
                da = da.sortby(da, ascending=False)

                data = da.data[da.data>0]
                labels  = [f"{da[i].coords['Parameters'].data}\n{data[i].round(2)}%" for i in range(6)]
                colors = [plt.cm.Spectral(i/float(len(data))) for i in range(len(data))]

                plt.figure(figsize=(13,7), dpi=150)
                squarify.plot(data, label=labels, color=colors)
                plt.title('CTV: ' + cat)
                plt.axis('off')

                if save:
                    pdf.savefig(bbox_inches='tight')

                plt.show()
    
    def compare(self, other, pathway='MP'):
        """Compares two LCA results, yielding a ratio xr.Dataset"""

        self_sum = self.groupby(by='sum',pathway=pathway)
        other_sum = other.groupby(by='sum',pathway=pathway)

        ratio = self_sum/other_sum
        if pathway == "EP":
            ratio = ratio.to_dataset('AOP')
        elif pathway == "MP":
            ratio = ratio.to_dataset('Categories')

        ratio.attrs['Numerator'] = self.MP.attrs['Name']
        ratio.attrs['Denominator'] = other.MP.attrs['Name']
        
        return ratio

    def dist_compare(self, other, pathway="EP", save=False, palette='GnBu'):
        """Plots the paired comparison between two LCA results"""
        
        if pathway == "MP":
            nrow = 6
            size = 19

        elif pathway == "EP":
            nrow = 1
            size = 4

        ds = self.compare(other, pathway=pathway)
        num = ds.attrs['Numerator']
        den = ds.attrs['Denominator']
        name = num + '_' + den + '_' + pathway + '_comparison' 

        f, axes = plt.subplots(nrow, 3, figsize=(13, size))
        sns.set(style="ticks", palette=palette, color_codes=True)
        axes = axes.ravel()

        i=0
        for var in ds:

            data = ds[var]
            median = data.compute().median()
            sns.distplot(data, kde=True, hist=False, kde_kws={"shade": True}, \
                        ax=axes[i], color='forestgreen')
            axes[i].axvline(1, 0, 1, color='k', ls= '-', lw=0.8)
            axes[i].axvline(median, 0, 1, color='darkgreen', ls= '--')
            axes[i].set_xticks([0, 0.5, median, 1.5, 2])
            i += 1

        plt.setp(axes, yticks=[],  xlim=[0,2])
        plt.tight_layout()
        f.set_dpi(150)

        if save:
            f.savefig('.\\Outputs\\' + name + '.pdf', bbox_inches='tight', papertype='A4')

        plt.show()

def comparePercent(ds, threshold=1):
    """Returns a xr.Dataset with the percentage of iterations where the ratio is below the threshold."""
    
    PCT = xr.Dataset()
    for var in ds:
        arr = ds[var]
        PCT[var] = arr[arr<threshold].size/arr.i.size*100
    PCT.attrs = ds.attrs

    return PCT