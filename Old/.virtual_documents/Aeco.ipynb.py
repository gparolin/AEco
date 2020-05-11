a = 1


aircraft_type = 'pax'  #pax or cargo

input_path = '.\\data\\A320_inputs.xlsx'
input_sheet = 'A320'
output_path = '.\\results\\'+ input_sheet + '_outputs.xlsx'
database_path = '.\\data\\database_A320.xlsx'

iterations = 2000


# packages

import numpy as np
import pandas as pd
from scipy.stats import spearmanr as spear
import warnings


#functions
def pert(row, it): 
    ''' Gera uma amostra de uma distribuição Beta-PERT
    
    nom : float
        valor nominal da variável
    pes: float
        valor pessimista (mínimo) da variável
    opt: float
        valor otimista (máximo) da variável
    sample: list
        quantidade e formato dos valores amostrados (output)
    '''
    
    sample = [1,it] #gerar array para sampling
    
    df = inputs.loc[[row]] #achar linha da variável em questão
    
    nom = df['nom'].iloc[0] #valor nominal
    pes = df['pes'].iloc[0] #valor pessimista
    opt = df['opt'].iloc[0] #valor otimista
    
    if pes == opt and pes == nom:
        alpha = nom
        beta = nom
        mean = nom
                
    elif pes > nom or opt < nom:
        raise ValueError("Valor 'nominal' deve ser igual ou estar entre 'pessimista' e 'otimista'")
    
    else:
        mean = (pes + 4*nom + opt)/6
        alpha = 6*((mean - pes)/(opt - pes))
        beta = 6*((opt - mean)/(opt - pes))

    
    sample_PERT = (np.random.beta(alpha, beta, sample) * (opt - pes)) + pes
    return sample_PERT.ravel()


def unitproc(col):
    ''' Obtem um UP da database em forma de Series com Multi Index. Col é um string.
    
    '''
    
    Col = Unit_Processes[['compartment','subcompartment', col]]
    Col.set_index(['compartment','subcompartment'], append=True, inplace=True)
    #Col.sort_index(inplace=True)
    Col = Col.squeeze()
    #Col.index = Col.index.set_levels(Col.index.levels[1].str.lower(), level=1)
    return Col.fillna(0)


def mtp(series, array):
    ''' Multiplica um pd.Series de UP por um array de PERT para gerar um pd.Series de LCI
    
    '''
    
    return series.apply(lambda x: x*array)

def charact_factors(category):
    ''' Retorna pd.Series de com os CF de uma category de impactos. Category é uma string.
    
    '''
    series = MP.loc[category]
    series.set_index(['name','compartment','subcompartment'], inplace=True)
    return series.squeeze()

def div(series, array):
    ''' Divide um pd.Series de UP por um array de PERT para gerar um pd.Series de LCI
    
    '''
    
    return series.apply(lambda x: x/array)

def avg(series):
    ''' Calcula a média de cada linha de um pd.Series composto de np.arrays
    
    '''
    
    return series.apply(lambda x: np.mean(x))

def electricity(E):
    ''' Calcula o LCI de consumo de eletricidade com a grade brasileira
    
    '''
    
    E_wind = E * inp['grid_wind']
    E_gas = E * inp['grid_gas']
    E_hydro = E * inp['grid_hydro']
 
    LCI_E = mtp(UP['Elec_wind'], E_wind) + mtp(UP['Elec_gas'], E_gas) \
            + mtp(UP['Elec_hydro'], E_hydro)
    return LCI_E

def database(series):
    ''' Transforma um pd.Series de listas em um pd. DataFrame
    
    '''
    
    db = pd.DataFrame.from_records(zip(*series.values))
    db.columns = series.index
    return db


#Leitura dos inputs da planilha inputs.xlsx
with pd.ExcelFile(input_path) as xlsx:
    inputs_full = pd.read_excel(xlsx, header = 2, index_col = 0, na_values="", usecols="B:G")
    
inputs_full = inputs_full.rename(columns = {"nominal": "nom", 
                                            "low":"pes", 
                                            "high":"opt"}) #renomear título das colunas

inputs_unit = inputs_full["unit"] #unidades dos inputs
inputs_description = inputs_full["description"] #descrição dos inputs

inputs = inputs_full.drop(columns = {"unit", "description"}) #valores dos inputs

inp = {}
for variable in inputs.index:
    inp[variable] = pert(variable, iterations)
    
pert_eye = np.linspace(1,1,iterations) #ones vector for monte carlo

inputs_full = None
inputs = None
variable = None


#Leitura dos UP e CF da planilha database.xlsx
with pd.ExcelFile(database_path) as xlsx:
    Unit_Processes = pd.read_excel(xlsx, 'UP', header = 5, index_col = 0, na_values=0)


UP = {}
for column in Unit_Processes.columns.values[3:]:
    UP[column] = unitproc(column)

Unit_Processes = None


#Definição do inventário

LCI_columns =  [['DEV','DEV','DEV','DEV','DEV','MFG','MFG','MFG','MFG','OP','OP','OP','OP','OP','EOL','EOL','EOL'],\
                ['Office','Prototype','Construction','Certification','Capital','Material','Factory','Logistics','Sustaining',\
                 'LTO','CCD','Maintenance','Airport','Fuel','Recycling','Landfill','Incineration']]
LCI_index = UP['Aluminium'].index
LCI = pd.DataFrame(index = LCI_index, columns = pd.MultiIndex.from_arrays(LCI_columns, names=['Phase', 'Subphase']))
LCI_index = None
LCI_columns = None


#definições iniciais de pkm ou tkm

if aircraft_type == "cargo":
    pkm_flight = inp['payload'] * inp['d_f'] * inp['loadfactor'] #tkm por voo

elif aircraft_type == "pax":
    npax = inp['seat_max'] * inp['p_seat'] * inp['loadfactor'] #número de passageiros por voo
    pkm_flight = npax * inp['d_f'] #pkm por voo

else:
    raise Exception("Aircraft type must be 'pax' or 'cargo'")
    
    
pkm_year = pkm_flight * inp['flights_year'] #pkm por ano
pkm_life = pkm_year * inp['lifetime'] #pkm por vida
pkm_fleet = pkm_life * inp['fleet'] #pkm da frota


LCI_E_office = electricity(inp['E_office']) #per month
LCI_E_office = mtp(LCI_E_office,inp['devmonths']) #per development


LCI_water_office = mtp(UP['Water'],inp['water_office']) \
                    + mtp(UP['Wastewater'],inp['wastewater_office']) #per month
LCI_water_office = mtp(LCI_water_office,inp['devmonths']) #per development


travel = 18470 / 12 * inp['developers'] * inp['devmonths'] #km

LCI_travel = mtp(UP['Car'],travel * 0.1) + mtp(UP['Airplane'],travel * 0.9) #per development


LCI_paper = mtp(UP['Paper'],(inp['developers'] * inp['paper_use'])) #per year
LCI_paper = mtp(LCI_paper,(inp['devmonths']/12)) #per development


LCI['DEV','Office'] = div((LCI_E_office + LCI_water_office + LCI_paper + \
                          + LCI_travel),pkm_fleet)


LCI['DEV','Construction'] = div(mtp(UP['Facilities'],inp['new_factory']/2.74e5),pkm_fleet)


new_jigs = inp['OEW'] * 500 #50t of jigs per 100kg of product
UP_capital = UP['Steel'].add(UP['Jigs'],fill_value=0) #material plus transformation

LCI['DEV','Capital'] = div(mtp(UP_capital,new_jigs)+mtp(UP['Machine'],inp['new_machine'])\
                           ,pkm_fleet)


Al = inp['p_Al'] * inp['b2f_Al'] * inp['OEW']
steel = inp['p_steel'] * inp['b2f_steel'] * inp['OEW']
Ti = inp['p_Ti'] * inp['b2f_Ti'] * inp['OEW']
inconel = inp['p_inconel'] * inp['b2f_inconel'] * inp['OEW']
GFRP = inp['p_GFRP'] * inp['b2f_GFRP'] * inp['OEW']
CFRP = inp['p_CFRP'] * inp['b2f_CFRP'] * inp['OEW']

#Aluminium
LCI_Al = mtp(UP['Aluminium'], Al)

#Steel
LCI_steel = mtp(UP['Steel'], steel)

#Titanium
LCI_Ti = mtp(UP['Titanium'], Ti)

#Inconel
LCI_inconel = mtp(UP['Inconel'], inconel)

#GFRP
LCI_GFRP = mtp(UP['GFRP'], GFRP)

#CFRP
LCI_CFRP = mtp(UP['CFRP'], CFRP)

#LCI Material Extraction and Transformation
LCI['MFG','Material'] = div(LCI_Al + LCI_steel + LCI_Ti + LCI_inconel + LCI_GFRP + LCI_CFRP,pkm_life)


LCI_E_factory = electricity(inp['E_factory'])
LCI_E_factory = mtp(LCI_E_factory,inp['takt']) / 30 #per aircraft


LCI_water_factory = (mtp(UP['Water'], inp['water_factory']) \
                     + mtp(UP['Wastewater'], inp['wastewater_factory'])) #per month
LCI_water_factory = mtp(LCI_water_factory,inp['takt']) / 30 #per aircraft


LCI_lube = mtp(UP['Lubricant'], inp['lubricant']) #per month
LCI_lube = mtp(LCI_lube,inp['takt']) / 30 #per aircraft


facilities_maint = inp['OEW'] * 4.58e-10 #use per kg of product

LCI_facilities_maint = mtp(UP['Facilities'], facilities_maint) * 0.02 #per year
LCI_facilities_maint = mtp(LCI_facilities_maint,inp['takt']) / 365 #per aircraft


LCI['MFG','Factory'] = div(LCI_E_factory + LCI_water_factory + LCI_lube + LCI_facilities_maint,pkm_life)


lorry = inp['d_lorry'] * inp['m_lorry'] #tonne * km
sea = inp['d_sea'] * inp['m_sea'] #tonne * km
air = inp['d_air'] * inp['m_air'] #tonne * km

LCI['MFG','Logistics'] = div(mtp(UP['Lorry'],lorry) + mtp(UP['Sea'],sea) + mtp(UP['Air'],air),pkm_life)


LCI_sustaining = LCI['DEV','Office'] * 0.01 / 30 #per day
LCI['MFG','Sustaining'] = div(mtp(LCI_sustaining,inp['takt']),pkm_life)


#tempo CCD
t_ccd = (inp['FH']*60) - (inp['t_app'] + inp['t_to'] + inp['t_climb']) #min
fuel_ccd = inp['ff_ccd'] * t_ccd*60 #kg

LCI['OP','LTO'] = div(UP['LTO'],pkm_flight)
LCI['OP','CCD'] = div(mtp(UP['CCD'],fuel_ccd),pkm_flight)


LCI_maint = mtp(UP['Aluminium'], inp['maint_Al']) + mtp(UP['Steel'],inp['maint_steel']) + \
            mtp(UP['Polymer'],inp['maint_pol']) + mtp(UP['Battery'],inp['maint_battery']) #por ano

LCI['OP','Maintenance'] = div(div(LCI_maint,inp['flights_year']),pkm_flight)


if aircraft_type == "cargo":
    ap_impact = 0.132 #13,2% of airport impacts are due to cargo
elif aircraft_type == "pax":
    ap_impact = 0.868
else:
    ap_impact = 1

f_pax_ap = inp['pax_ap']/22500000 #fração de passageiros em relação ao aeroporto de zurich em 2000
#f_flights = pert('flights_year').mean() / pert('flights_ap').mean() # fração de voos da aeronave em relação ao aeroporto   

#LCI_E_ap = electricity(inp['E_ap'])

#LCI_ap = div(mtp(UP['Heat'],inp['heat_ap']) + mtp(UP['Water'],inp['water_ap']) \
#             + mtp(UP['Wastewater'],inp['wastewater_ap']) + LCI_E_ap, inp['flights_ap'])

LCI_ap = div(mtp(UP['Airport'],f_pax_ap/100), inp['flights_ap']) #100 anos de vida para o prédio

LCI['OP','Airport'] = div(mtp(LCI_ap,ap_impact),pkm_flight)


fuel_lto = inp['t_app']*60*inp['ff_app'] + inp['t_idle']*60*inp['ff_idle'] \
            + inp['t_to']*60*inp['ff_to'] + inp['t_climb']*60*inp['ff_climb']
LCI['OP','Fuel']= div(mtp(UP['Kerosene'],fuel_ccd+fuel_lto),pkm_flight)


E_sort_constant = 0.4645/3.6 #kWh/kg of material, on average
E_sort = E_sort_constant * inp['OEW']
LCI_sort = electricity(E_sort)


#Aluminium
p_ldf_Al = inp['p_ldf_Al'] * Al
LCI_ldf_Al = mtp(UP['Landfill'],p_ldf_Al)
p_incin_Al = inp['p_incin_Al'] * Al
LCI_incin_Al = mtp(UP['Incineration'],p_incin_Al)
p_recycl_Al = inp['p_recycl_Al'] * Al
LCI_recycl_Al = mtp(UP['Aluminium'],p_recycl_Al)

#Steel
p_ldf_steel = inp['p_ldf_steel'] * steel
LCI_ldf_steel = mtp(UP['Landfill'],p_ldf_steel)
p_incin_steel = inp['p_incin_steel'] * steel
LCI_incin_steel = mtp(UP['Incineration'],p_incin_steel)
p_recycl_steel = inp['p_recycl_steel'] * steel
LCI_recycl_steel = mtp(UP['Steel'],p_recycl_steel)

#Titanium
p_ldf_Ti = inp['p_ldf_Ti'] * Ti
LCI_ldf_Ti = mtp(UP['Landfill'],p_ldf_Ti)
p_incin_Ti = inp['p_incin_Ti'] * Ti
LCI_incin_Ti = mtp(UP['Incineration'],p_incin_Ti)
p_recycl_Ti = inp['p_recycl_Ti'] * Ti
LCI_recycl_Ti = mtp(UP['Titanium'],p_recycl_Ti)

#Inconel
p_ldf_inconel = inp['p_ldf_inconel'] * inconel
LCI_ldf_inconel = mtp(UP['Landfill'],p_ldf_inconel)
p_incin_inconel = inp['p_incin_inconel'] * inconel
LCI_incin_inconel = mtp(UP['Incineration'],p_incin_inconel)
p_recycl_inconel = inp['p_recycl_inconel'] * inconel
LCI_recycl_inconel = mtp(UP['Inconel'],p_recycl_inconel)

#GFRP
p_ldf_GFRP = inp['p_ldf_GFRP'] * GFRP
LCI_ldf_GFRP = mtp(UP['Landfill'],p_ldf_GFRP)
p_incin_GFRP = inp['p_incin_GFRP'] * GFRP
LCI_incin_GFRP = mtp(UP['Incineration'],p_incin_GFRP)
p_recycl_GFRP = inp['p_recycl_GFRP'] * GFRP
LCI_recycl_GFRP = mtp(UP['GFRP'],p_recycl_GFRP)

#CFRP
p_ldf_CFRP = inp['p_ldf_CFRP'] * CFRP
LCI_ldf_CFRP = mtp(UP['Landfill'],p_ldf_CFRP)
p_incin_CFRP = inp['p_incin_CFRP'] * CFRP
LCI_incin_CFRP = mtp(UP['Incineration'],p_incin_CFRP)
p_recycl_CFRP = inp['p_recycl_CFRP'] * CFRP
LCI_recycl_CFRP = mtp(UP['CFRP'],p_recycl_CFRP)


LCI['EOL','Recycling'] = div(LCI_sort - (LCI_recycl_Al + LCI_recycl_Ti + LCI_recycl_steel \
                         + LCI_recycl_inconel + LCI_recycl_GFRP + LCI_recycl_CFRP),pkm_life)
LCI['EOL','Incineration'] = div(LCI_incin_Al + LCI_incin_Ti + LCI_incin_steel +\
                                LCI_incin_inconel + LCI_incin_GFRP + LCI_incin_CFRP, pkm_life)
LCI['EOL', 'Landfill'] = div(LCI_ldf_Al + LCI_ldf_Ti + LCI_ldf_steel + LCI_ldf_inconel \
                             + LCI_ldf_GFRP + LCI_ldf_CFRP, pkm_life)


#pending calculations from development section:
LCI['DEV','Prototype'] = div(mtp(LCI['MFG'].sum(axis=1),inp['prototypes']) + \
                             mtp(LCI['MFG'].sum(axis=1),(inp['ironbirds']*0.3)),pkm_fleet)

cert_flights = inp['test_FH'] / inp['FH']
LCI['DEV','Certification'] = div(mtp(LCI['OP','LTO'] + LCI['OP','CCD'],cert_flights),pkm_fleet) #per development


with pd.ExcelFile(database_path) as xlsx:
    MP = pd.read_excel(xlsx, 'MP', header = 0, index_col = 0, na_values=0)
    EP = pd.read_excel(xlsx, 'EP', header = 0, index_col = 0, na_values=0)


mp_categories = MP.index.unique()
mp_categories


mp_factors = pd.DataFrame(index = LCI.index)
for category in mp_categories:
    category_factor = charact_factors(category)
    category_factor = category_factor.rename(category)
    mp_factors[category] = category_factor
mp_factors.fillna(0, inplace=True)

MP = None


LCIA_MP = pd.DataFrame(columns=LCI.columns)

for column in LCI:
    LCIA_MP[column] = mp_factors.mul(LCI[column], axis=0).sum()


#correction of the natural land transformation values
LCIA_MP.loc['natural land transformation'] = -LCIA_MP.loc['natural land transformation']


ep_factors = EP.fillna(0)

LCIA_EP = pd.DataFrame(columns=LCIA_MP.columns, index=ep_factors.columns)

for column in LCIA_EP:
    LCIA_EP[column] = ep_factors.mul(LCIA_MP[column], axis=0).sum()


with warnings.catch_warnings():
    warnings.filterwarnings("ignore")

    corr = pd.DataFrame(index=inp.keys(), columns=LCIA_MP.index)
    for row in corr.index:
        for column in corr:
            corr[column][row] = spear(inp[row],LCIA_MP.loc[column].sum())[1]


    corr_sq = corr ** 2
    corr_sq_sum = corr_sq.sum(axis=0)

    CTV_MP = pd.DataFrame(index=inp.keys(), columns=LCIA_MP.index)
    CTV_MP = corr_sq / corr_sq_sum


with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
    
    corr = pd.DataFrame(index=inp.keys(), columns=LCIA_EP.index)
    for row in corr.index:
        for column in corr:
            corr[column][row] = spear(inp[row],LCIA_EP.loc[column].sum())[1]


    corr_sq = corr ** 2
    corr_sq_sum = corr_sq.sum(axis=0)

    CTV_EP = pd.DataFrame(index=inp.keys(), columns=LCIA_EP.index)
    CTV_EP = corr_sq / corr_sq_sum


float_formatter = "{:.8e}".format
np.set_printoptions(threshold=iterations, linewidth = 100, formatter={'float_kind':float_formatter})


with pd.ExcelWriter(output_path) as writer:
    LCIA_MP.to_excel(writer, sheet_name='MP')
    LCIA_EP.to_excel(writer, sheet_name='EP')
    CTV_MP.to_excel(writer, sheet_name='CTV_MP')
    CTV_EP.to_excel(writer, sheet_name='CTV_EP')
    
print(f"LCA complete! Check output file at {output_path}")
