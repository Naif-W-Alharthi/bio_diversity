import xlrd
import csv
import os
import pandas as pd
def csv_from_excel(path):
    
    if "xlsx" in file:
     df1 = pd.read_excel(
     os.path.join(path, "Data", "aug_latest.xlsm"),
     engine='openpyxl',
)
    

# runs the csv_from_excel function:
for file in os.listdir("Biodiversity"):
    
        csv_from_excel(file)