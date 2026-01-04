"""If data file is in different path then we will get error.
To copy path of the file from different directory path
then right click on file --> Copy path/reference --> copy the absolute path and use the path"""

"""import pandas
data = pandas.read_csv("diamonds.csv")
print(data)"""

"""Below lines are used to read csv file from given path"""
import pandas
data = pandas.read_csv(r"D:\Sairam\Data Science\Uptor116\November_2025\Dataset\diamonds_updated.csv")
#print(data)

"""Below lines are used to read column names"""
# column_data = data.columns
# print(column_data)

data_types = data.dtypes
# print(data_types)

"""Detail information on columns. It shows additional information if the column has null and 
how many int, float, etc. How much memory"""
data_info = data.info()
# print(data_info)

'''To get the int, float columns calulations like count, mean, std, min, etc.'''
data_describe = data.describe()
print(data_describe)
