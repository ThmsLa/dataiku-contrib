# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu
from dataiku.customrecipe import *
import fuzzy

# Recipe inputs
input_ds_name = get_input_names_for_role('input')[0]
input_ds = dataiku.Dataset(input_ds_name)

output_ds_name = get_output_names_for_role('output')[0]
output_ds = dataiku.Dataset(output_ds_name)

input_df = input_ds.get_dataframe()
algorithm = get_recipe_config().get('algorithm', 'soundex')
columns = get_recipe_config().get('text_columns')
soundex_length = float(get_recipe_config().get('soundex_length', 4))

phonetics = {
    'soundex': fuzzy.Soundex(soundex_length),
    'nysiis': fuzzy.nysiis,
    'dm': fuzzy.DMetaphone() 
}.get(algorithm)

for column in columns:
    data = input_df[column].apply(phonetics)
    location = input_df.columns.get_loc(column) + 1
    name = column + "_" + algorithm
    
    if algorithm == 'dm':
        first_codes, second_codes = zip(*data)
        input_df.insert(location, name + "_1", first_codes)
        input_df.insert(location + 1, name + "_2", second_codes)
    else:
        input_df.insert(location, name, data)

# Recipe outputs
output_ds.write_with_schema(input_df)
