import os
os.chdir("C:/Users/Parolin/Documents/AEco")

aircraft_type = 'pax'  #pax or cargo

input_path = './Data/A320_inputs.xlsx'
input_sheet = 'A320'
output_path = './Outputs/'+ input_sheet + '_outputs'
database_path = './Data/database_A320.xlsx'

iterations = 1000000
chunks = 100  #'auto' or more than 500

from Tools import *
from Model import *

inputs = read_inputs(input_path, input_sheet)
p = ParameterSet(inputs, iterations, chunks)
p = pkm(aircraft_type, p)

UP_dataframe = read_unit_processes(database_path)
UP = unit_process_dataset(UP_dataframe)

inventory = LCI(name=input_sheet, type=aircraft_type, iterations=iterations, UP=UP, parameters=p)
inventory.run()

MP_data, EP_data = read_CF(database_path)
CF = CharactFactors(MP_data, EP_data, UP.Substances)
CF.MP.dataset['NLT'].data = np.negative(CF.MP.dataset['NLT'].data)
CF.MP.to_array()

aeco = LCIA.build(inventory, CF)
aeco.build_CTV(parameterset=p)

aeco.save(output_path, LCI=False)