import numpy as np
import pandas as pd
import warnings
import xarray as xr
import dask.dataframe as dd
import dask.array as da
import bottleneck

class LCI():
    """Defines a LCI class based on xr.DataArray."""
    
    def __init__(self, name, iterations, UP, parameters):
        """Initialization with the phases and substances of the LCI."""
        
        self.name = name
        self.UP = UP
        self.substances = UP.Substances
        self.p = parameters
        self.data = None
        self.build(iterations)
  
    def __repr__(self):
        return f"{self.data}"
    
    def __getitem__(self, phase):
        return self.data[phase]
    
    def __setitem__(self, phase, other):
        self.data[phase] = other
            
    def build(self, iterations):
        """Builds the xr.DataArray for the LCI."""
        
        if self.data == None:
            self.data = xr.Dataset(coords={'Substances': self.substances, 
                                           'i': np.arange(iterations)},
                                   attrs={'Name':self.name})
            self.data.coords['Units'] = self.substances.Units
            
        return self.data
    
    def substance(self, substance):
        """Locates the specified substance on the data."""
        return self.data.loc[{'Substances': substance}]
    
    def iteration(self, iteration):
        """Locates the specified iteration on the data."""
        
        return self.data.loc[{'i': iteration}]
    
    def find(self, phase, substance, iteration):
        """Locates the specified substance, phase and iteration on the data."""
            
        return self.data[phase].loc[{'Substances': substance, 'i':iteration}]
    
    def mean(self, phase):
        """Returns the mean for all iterations of a certain phase."""
        
        return self['Office'].mean('i').load()
    
    def median(self, phase):
        """Returns the median for all iterations of a certain phase."""
        
        return self['Office'].median('i').load()

    def office(self):
        LCI_E_office = self.electricity(self.p["E_office"])  #per month
        LCI_E_office = LCI_E_office * self.p["devmonths"]  #per development

        LCI_water_office = self.UP["Water"] * self.p["water_office"] \
                        + self.UP["Wastewater"] * self.p["wastewater_office"]  #per month
        LCI_water_office = LCI_water_office * self.p["devmonths"]  #per development

        self.p["travel"] = 18470 / 12 * self.p["developers"] * self.p["devmonths"]  #in km

        LCI_travel = self.UP["Car"]*self.p["travel"]*0.1 \
                    + self.UP["Airplane"]*self.p["travel"]*0.9  #per development

        LCI_paper = self.UP["Paper"]*self.p["developers"]*self.p["paper_use"]  #per year
        LCI_paper = LCI_paper * self.p["devmonths"] / 12  #per development

        LCI_office = (LCI_E_office + LCI_water_office + LCI_paper + LCI_travel)  #per development
        LCI_office = LCI_office / self.p["pkm_fleet"]  #per pkm

        self.data['Office'] = LCI_office

        return self.data

    def dev(self):
        self.office()
        
        
    def mfg(self):
        pass
    
    def ope(self):
        pass

    def eol(self):
        pass

    def run(self):
        self.dev()
        self.mfg()
        self.ope()
        self.eol()

    def electricity(self, E):
        """Calculates the LCI of electricity consumption based on a gas-wind-hydropower electricity grid."""
        
        E_wind = E * self.p['grid_wind']
        E_gas = E * self.p['grid_gas']
        E_hydro = E * self.p['grid_hydro']
        LCI_E = self.UP['Elec_wind']*E_wind \
                + self.UP['Elec_gas']*E_gas + self.UP['Elec_hydro']*E_hydro
        
        return LCI_E


