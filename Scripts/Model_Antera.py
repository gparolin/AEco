import numpy as np
import pandas as pd
import warnings
import xarray as xr
import dask.dataframe as dd
import dask.array as da

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
        
        return self[phase].mean('i').load()
    
    def median(self, phase):
        """Returns the median for all iterations of a certain phase."""
        
        return self[phase].median('i').load()

    def office(self):
        LCI_E_office = self.electricity(self.p["E_office"])  #per month
        LCI_E_office = LCI_E_office * self.p["devmonths"]  #per development

        LCI_water_office = self.UP["Water"] * self.p["water_office"] \
                        + self.UP["Wastewater"] * self.p["wastewater_office"]  #per month
        LCI_water_office = LCI_water_office * self.p["devmonths"]  #per development

        LCI_office = (LCI_E_office + LCI_water_office)  #per development
        LCI_office = LCI_office / self.p["ha_fleet"]  #per ha

        self.data['Office'] = LCI_office

    def infrastructure(self):
        LCI_construction = (self.UP["Facilities"]*self.p["new_factory"]/2.74e5) / self.p["ha_fleet"]
        self.data["Infrastructure"] = LCI_construction
    
    def capital(self):
        self.p["new_jigs"] = self.p["OEW"] * 500  # 50t of jigs per 100kg of product
        self.UP["Capital"] = self.UP["Steel"] + self.UP["Jigs"]  # material plus transformation
        LCI_capital = (self.UP["Capital"]*self.p["new_jigs"] + self.UP["Machine"]*self.p["new_machine"])/self.p["ha_fleet"]
        self.data["Capital"] = LCI_capital

    def dev(self):
        self.office()
        self.infrastructure()
        self.capital()     
        
    def materials(self):

        self.p["Al"] = self.p['p_Al'] * self.p['b2f_Al'] * self.p['OEW']
        self.p["steel"] = self.p['p_steel'] * self.p['b2f_steel'] * self.p['OEW']
        self.p["Ti"] = self.p['p_Ti'] * self.p['b2f_Ti'] * self.p['OEW']
        self.p["inconel"] = self.p['p_inconel'] * self.p['b2f_inconel'] * self.p['OEW']
        self.p["GFRP"] = self.p['p_GFRP'] * self.p['b2f_GFRP'] * self.p['OEW']
        self.p["CFRP"] = self.p['p_CFRP'] * self.p['b2f_CFRP'] * self.p['OEW']

        LCI_Al = self.UP["Aluminium"] * self.p["Al"]
        LCI_steel = self.UP["Steel"] * self.p["steel"]
        LCI_Ti = self.UP["Titanium"] * self.p["Ti"]
        LCI_inconel = self.UP["Inconel"] * self.p["inconel"]
        LCI_GFRP = self.UP["GFRP"] * self.p["GFRP"]
        LCI_CFRP = self.UP["CFRP"] * self.p["CFRP"]

        #LCI Material Extraction and Transformation
        LCI_material = (LCI_Al + LCI_steel + LCI_Ti + LCI_inconel + LCI_GFRP + LCI_CFRP) / self.p["ha_life"]
        self.data["Materials"] = LCI_material

    def factory(self):
        LCI_E_factory_alum = self.electricity(self.p["E_aluminum"])
        LCI_E_factory_comp = self.electricity(self.p["E_composite"])
        LCI_E_factory_assy = self.electricity(self.p["E_assy"])
        LCI_E_factory = (LCI_E_factory_alum+LCI_E_factory_comp+LCI_E_factory_assy) * self.p["takt"] / 30  # per aircraft

        LCI_water_factory = self.UP["Water"]*self.p["water_factory"] \
                    + self.UP["Wastewater"]*self.p["wastewater_factory"]  # per month
        LCI_water_factory = LCI_water_factory * self.p["takt"] / 30  # per aircraft

        LCI_lube = self.UP["Lubricant"] * self.p["lubricant"]  # per month
        LCI_lube = LCI_lube * self.p["takt"] / 30  # per aircraft

        LCI_matrix = self.UP["Polymer"] * self.p["consummable"] # per month
        LCI_matrix = LCI_matrix * self.p["takt"] / 30 #per aircraft

        self.p["facilities_maint"] = self.p["OEW"] * 4.58e-10  # use per kg of product

        LCI_facilities_maint = self.UP["Facilities"] * self.p["facilities_maint"] * 0.02  # per year
        LCI_facilities_maint = LCI_facilities_maint * self.p["takt"] / 365  # per aircraft

        LCI_factory = (LCI_E_factory + LCI_water_factory + LCI_lube + LCI_matrix + LCI_facilities_maint)/self.p["ha_life"]
        self.data["Factory"] = LCI_factory

    def logistics(self):
        lorry = self.p["d_lorry"] * self.p["m_lorry"] #tonne * km
        sea = self.p["d_sea"] * self.p["m_sea"] #tonne * km
        air = self.p["d_air"] * self.p["m_air"] #tonne * km

        LCI_logistics = (self.UP["Lorry"]*lorry + self.UP["Sea"]*sea \
                + self.UP["Air"]*air) / self.p["ha_life"]
        self.data['Logistics'] = LCI_logistics
    
    def sustaining(self):
        LCI_sustaining = self.data["Office"] * 0.01 / 30 #per day
        LCI_sustaining = (LCI_sustaining * self.p["takt"])/self.p["ha_life"]
        self.data["Sustaining"] = LCI_sustaining
        
    def mfg(self):
        self.materials()
        self.factory()
        self.logistics()
        self.sustaining()

    def flights(self):
        self.p["fuel_cruise"] = self.p["ff_cruise"] * self.p["t_cruise"]  # kg
        self.p["fuel_takeoff"] = self.p["ff_takeoff"] * self.p["t_takeoff"]  # kg
        self.p["fuel_landing"] = self.p["ff_landing"] * self.p["t_landing"]  # kg
        self.p["fuel_total"] = self.p["fuel_cruise"]+self.p["fuel_takeoff"]+self.p["fuel_landing"]
        self.data["Flight"] = self.UP["Engine"] * self.p["fuel_total"]/ self.p["ha_flight"]


    def spray(self):
        self.data["Pesticide"] = (self.p["pesticide_use"] / self.p["pesticide_eff"]) * self.UP["Pesticide"] #kg/ha


    def maintenance(self):
        LCI_maint = self.UP["Aluminium"]*self.p["maint_Al"] + self.UP["Steel"]*self.p["maint_steel"] \
            + self.UP["Polymer"]*self.p["maint_pol"] + self.UP["Battery"]*self.p['maint_battery'] #por ano

        LCI_maint = (LCI_maint / self.p["flights_year"]) / self.p["ha_flight"]
        self.data['Maintenance'] = LCI_maint

    def fuel(self):
            
        LCI_fuel = self.UP['Kerosene']*self.p["fuel_total"] / self.p["ha_flight"]
        self.data["Fuel"] = LCI_fuel

    def ope(self):
        self.flights()
        self.spray()
        self.maintenance()
        self.fuel()

    def eol(self):
    
        E_sort_constant = 0.4645 / 3.6  # kWh/kg of material, on average
        self.p["E_sort"] = E_sort_constant * self.p['OEW']
        LCI_sort = self.electricity(self.p["E_sort"])

        materials = ['Al','steel','Ti','inconel','GFRP','CFRP']
        scenarios = ['ldf', 'incin','recycl']
        #chunks = self.data.chunks['i'][0]
        iterations = self.data.i.size
        UP_eol = self.UP.rename_vars({'Landfill':'ldf','Incineration':'incin','Aluminium':'Al',
                                'Titanium':'Ti', 'Inconel':'inconel','Steel':'steel'})
        eol = xr.Dataset({scenario: (['Substances','i'],da.empty((1835,iterations), chunks="auto"))
                        for scenario in scenarios}, coords=self.data.coords)
        #chunks=(1835,chunks)

        for scenario in scenarios:
            for material in materials:
                self.p[scenario+"_"+material] = self.p["p_"+scenario+"_"+material]*self.p[material]
                if scenario == 'recycl':
                    eol[scenario] += UP_eol[material] * self.p[scenario + "_" + material]
                else:
                    eol[scenario] += UP_eol[scenario] * self.p[scenario + "_" + material]

        self.data["Recycling"] = (LCI_sort - eol['recycl']) / self.p["ha_life"]
        self.data["Incineration"] = eol["incin"] / self.p["ha_life"]
        self.data["Landfill"] = eol["ldf"] / self.p["ha_life"]

    def run(self):
        self.dev()
        self.mfg()
        self.ope()
        self.eol()

        MFG = self.data["Logistics"]+self.data["Sustaining"]+self.data["Factory"]+self.data["Materials"]
        LCI_prot = (MFG*self.p["prototypes"] + MFG*self.p["ironbirds"]*0.3)/self.p["ha_fleet"]
        self.data["Prototypes"] = LCI_prot

        self.p["cert_ha"] = self.p["test_FH"] * self.p["productivity"] / self.p["FH"]
        self.data["Certification"] = self.data["Flight"] * self.p["cert_ha"]/self.p["ha_fleet"]

        return self.data

    def electricity(self, E):
        """Calculates the LCI of electricity consumption based on a gas-wind-hydropower electricity grid."""
        
        E_wind = E * self.p['grid_wind']
        E_gas = E * self.p['grid_gas']
        E_hydro = E * self.p['grid_hydro']
        LCI_E = self.UP['Elec_wind']*E_wind \
                + self.UP['Elec_gas']*E_gas + self.UP['Elec_hydro']*E_hydro
        
        return LCI_E


