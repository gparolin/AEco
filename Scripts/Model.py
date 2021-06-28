from ast import Pass
import numpy as np
import pandas as pd
import warnings
import xarray as xr
import dask.dataframe as dd
import dask.array as da

class LCI():
    """Defines a LCI class based on xr.DataArray."""
    
    def __init__(self, name, type, iterations, UP, parameters):
        """Initialization with the phases and substances of the LCI."""
        
        self.name = name
        self.type = type
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

    def infrastructure(self):
        LCI_construction = (self.UP["Facilities"]*self.p["new_factory"]/2.74e5) / self.p["pkm_fleet"]
        self.data["Infrastructure"] = LCI_construction
    
    def capital(self):
        self.p["new_jigs"] = self.p["OEW"] * 500  # 50t of jigs per 100kg of product
        self.UP["Capital"] = self.UP["Steel"] + self.UP["Jigs"]  # material plus transformation
        LCI_capital = (self.UP["Capital"]*self.p["new_jigs"] + self.UP["Machine"]*self.p["new_machine"])/self.p["pkm_fleet"]
        self.data["Capital"] = LCI_capital

    def dev(self):
        self.office()
        self.infrastructure()
        self.capital()     
        
    def materials(self):
        try:
            reuse = self.p['reuse']
        except:
            reuse = 1

        self.p["Al"] = self.p['p_Al'] * self.p['b2f_Al'] * self.p['OEW'] * reuse
        self.p["steel"] = self.p['p_steel'] * self.p['b2f_steel'] * self.p['OEW'] * reuse
        self.p["Ti"] = self.p['p_Ti'] * self.p['b2f_Ti'] * self.p['OEW'] * reuse
        self.p["inconel"] = self.p['p_inconel'] * self.p['b2f_inconel'] * self.p['OEW'] * reuse
        self.p["GFRP"] = self.p['p_GFRP'] * self.p['b2f_GFRP'] * self.p['OEW'] * reuse
        self.p["CFRP"] = self.p['p_CFRP'] * self.p['b2f_CFRP'] * self.p['OEW'] * reuse

        LCI_Al = self.UP["Aluminium"] * self.p["Al"]
        LCI_steel = self.UP["Steel"] * self.p["steel"]
        LCI_Ti = self.UP["Titanium"] * self.p["Ti"]
        LCI_inconel = self.UP["Inconel"] * self.p["inconel"]
        LCI_GFRP = self.UP["GFRP"] * self.p["GFRP"]
        LCI_CFRP = self.UP["CFRP"] * self.p["CFRP"]

        #LCI Material Extraction and Transformation
        LCI_material = (LCI_Al + LCI_steel + LCI_Ti + LCI_inconel + LCI_GFRP + LCI_CFRP) / self.p["pkm_life"]
        self.data["Materials"] = LCI_material

    def factory(self):
        LCI_E_factory = self.electricity(self.p["E_factory"])
        LCI_E_factory = LCI_E_factory * self.p["takt"] / 30  # per aircraft

        LCI_water_factory = self.UP["Water"]*self.p["water_factory"] \
                    + self.UP["Wastewater"]*self.p["wastewater_factory"]  # per month
        LCI_water_factory = LCI_water_factory * self.p["takt"] / 30  # per aircraft

        LCI_lube = self.UP["Lubricant"] * self.p["lubricant"]  # per month
        LCI_lube = LCI_lube * self.p["takt"] / 30  # per aircraft

        self.p["facilities_maint"] = self.p["OEW"] * 4.58e-10  # use per kg of product

        LCI_facilities_maint = self.UP["Facilities"] * self.p["facilities_maint"] * 0.02  # per year
        LCI_facilities_maint = LCI_facilities_maint * self.p["takt"] / 365  # per aircraft

        LCI_factory = (LCI_E_factory + LCI_water_factory + LCI_lube + LCI_facilities_maint)/self.p["pkm_life"]
        self.data["Factory"] = LCI_factory

    def logistics(self):
        lorry = self.p["d_lorry"] * self.p["m_lorry"] #tonne * km
        sea = self.p["d_sea"] * self.p["m_sea"] #tonne * km
        air = self.p["d_air"] * self.p["m_air"] #tonne * km

        LCI_logistics = (self.UP["Lorry"]*lorry + self.UP["Sea"]*sea \
                + self.UP["Air"]*air) / self.p["pkm_life"]
        self.data['Logistics'] = LCI_logistics
    
    def sustaining(self):
        LCI_sustaining = self.data["Office"] * 0.01 / 30 #per day
        LCI_sustaining = (LCI_sustaining * self.p["takt"])/self.p["pkm_life"]
        self.data["Sustaining"] = LCI_sustaining
        
    def mfg(self):
        self.materials()
        self.factory()
        self.logistics()
        self.sustaining()

    def flights(self):
        try:
            self.p["t_ccd"] = self.p["FH"]*60 - (self.p["t_app"] + self.p["t_to"] + self.p["t_climb"])  # minutes
        except:
            self.p["t_ccd"] = self.p['FH']*60 - self.p['ff_lto']
        
        self.p["fuel_ccd"] = self.p["ff_ccd"] * self.p["t_ccd"] * 60  # kg
        self.data["LTO"] = self.UP["LTO"] / self.p["pkm_flight"]
        self.data["CCD"] = self.UP["CCD"] * self.p["fuel_ccd"] / self.p["pkm_flight"]

    def maintenance(self):
        LCI_maint = self.UP["Aluminium"]*self.p["maint_Al"] + self.UP["Steel"]*self.p["maint_steel"] \
            + self.UP["Polymer"]*self.p["maint_pol"] + self.UP["Battery"]*self.p['maint_battery'] #por ano

        LCI_maint = (LCI_maint / self.p["flights_year"]) / self.p["pkm_flight"]
        self.data['Maintenance'] = LCI_maint

    def airport(self):
        if self.type == "cargo":
            ap_impact = 0.132  # 13,2% of airport impacts are due to cargo
        elif self.type == "pax":
            ap_impact = 0.868
        else:
            ap_impact = 1

        self.p["f_pax_ap"] = self.p["pax_ap"] / 22500000  # fraction of pax relative to zurich in 2000
        LCI_ap = self.UP["Airport"] * self.p["f_pax_ap"]/100 / self.p["flights_ap"]  # 100 life years for building
        LCI_ap = LCI_ap * ap_impact / self.p["pkm_flight"]

        self.data["Airport"] = LCI_ap

    def fuel(self):
        try:
            self.p["fuel_lto"] = self.p['ff_lto'] * self.p['t_lto'] * 60
        except:
            self.p["fuel_lto"] = self.p['t_app']*60*self.p['ff_app'] + self.p['t_idle']*60*self.p['ff_idle'] \
                        + self.p['t_to']*60*self.p['ff_to'] + self.p['t_climb']*60*self.p['ff_climb']
            
        LCI_fuel = (self.UP['Kerosene']*(self.p["fuel_ccd"]+self.p["fuel_lto"]))/ self.p["pkm_flight"]
        self.data["Fuel"] = LCI_fuel

    def ope(self):
        self.flights()
        self.maintenance()
        self.airport()
        self.fuel()

    def eol(self):
        try:
            reuse_factor = (2 - p['reuse'])
        except:
            reuse_factor = 1
    
        E_sort_constant = 0.4645 / 3.6  # kWh/kg of material, on average
        self.p["E_sort"] = E_sort_constant * self.p['OEW'] * reuse_factor
        LCI_sort = self.electricity(self.p["E_sort"])

        materials = ['Al','steel','Ti','inconel','GFRP','CFRP']
        scenarios = ['ldf', 'incin','recycl']
        chunks = self.data.chunks['i'][0]
        iterations = self.data.i.size
        iterations
        UP_eol = self.UP.rename_vars({'Landfill':'ldf','Incineration':'incin','Aluminium':'Al',
                                'Titanium':'Ti', 'Inconel':'inconel','Steel':'steel'})
        eol = xr.Dataset({scenario: (['Substances','i'],da.empty((1835,iterations), chunks=(1835,chunks)))
                        for scenario in scenarios}, coords=self.data.coords)

        for scenario in scenarios:
            for material in materials:
                self.p[scenario+"_"+material] = self.p["p_"+scenario+"_"+material]*self.p[material]*reuse_factor
                if scenario == 'recycl':
                    eol[scenario] += UP_eol[material] * self.p[scenario + "_" + material]
                else:
                    eol[scenario] += UP_eol[scenario] * self.p[scenario + "_" + material]

        self.data["Recycling"] = (LCI_sort - eol['recycl']) / self.p["pkm_life"]
        self.data["Incineration"] = eol["incin"] / self.p["pkm_life"]
        self.data["Landfill"] = eol["ldf"] / self.p["pkm_life"]

    def run(self):
        self.dev()
        self.mfg()
        self.ope()
        self.eol()

        MFG = self.data["Logistics"]+self.data["Sustaining"]+self.data["Factory"]+self.data["Materials"]
        LCI_prot = (MFG*self.p["prototypes"] + MFG*self.p["ironbirds"]*0.3)/self.p["pkm_fleet"]
        self.data["Prototypes"] = LCI_prot

        self.p["cert_flights"] = self.p["test_FH"] / self.p["FH"]
        self.data["Certification"] = (self.data["LTO"]+self.data["CCD"])*self.p["cert_flights"]/self.p["pkm_fleet"]

        return self.data

    def electricity(self, E):
        """Calculates the LCI of electricity consumption based on a gas-wind-hydropower electricity grid."""
        
        E_wind = E * self.p['grid_wind']
        E_gas = E * self.p['grid_gas']
        E_hydro = E * self.p['grid_hydro']
        LCI_E = self.UP['Elec_wind']*E_wind \
                + self.UP['Elec_gas']*E_gas + self.UP['Elec_hydro']*E_hydro
        
        return LCI_E


