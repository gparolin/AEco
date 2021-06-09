import os
os.chdir("C:/Users/giparoli/Documents/Projetos/AEco")

input_path = './Data/Antera_inputs_v2.xlsx'
input_sheet = 'Antera_original'
output_path = './Outputs/'+ input_sheet + '_outputs'
database_path = './Data/database_Antera.xlsx'

iterations = 20000
chunks = 100  #'auto'

from Tools_Antera import *
from Model_Antera import *

inputs = read_inputs(input_path, input_sheet)
p = ParameterSet(inputs, iterations, chunks)
p = func_unit(p)

UP_dataframe = read_unit_processes(database_path)
UP = unit_process_dataset(UP_dataframe)

inventory = LCI(name=input_sheet, iterations=iterations, UP=UP, parameters=p)
inventory.run()

MP_data, EP_data = read_CF(database_path)
CF = CharactFactors(MP_data, EP_data, UP.Substances)
CF.MP.dataset['NLT'].data = np.negative(CF.MP.dataset['NLT'].data)
CF.MP.to_array()

aeco = LCIA.build(inventory, CF)
aeco.build_CTV(parameterset=p)

aeco.save(output_path, LCI=False)