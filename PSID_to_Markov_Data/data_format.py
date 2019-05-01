import pandas as pd
from numba import jit
import numpy as np
import time
# Import other files
import var_lists
import function_list
from var_lists import *
from function_list import *


time0 = time.time()

# Import data
data = pd.read_csv("data.csv")


# Define Unique ID
data["ID"] = 1000 * data["ER30001"] + data["ER30002"]
data.sort_values("ID")
# V3 is 68 family id

# drop extra vars
data = data.drop(drop_vars_parsed, axis=1)

# get array of IDs
all_people = np.array(data["ID"])


#Rename time series based on values (vs PSID codes)
print("begin rename", time.time() - time0)
renamed_data = rename_data(data, var_full_parsed, var_full_names)

# Reformat to long on year, ID
print("begin reformat", time.time() - time0)
long_data = pd.wide_to_long(renamed_data, stubnames=var_full_names,
                         i="ID", j="year", sep="__")


# restore ID, Year
indices = [list(i) for i in long_data.index]
indices = np.array(indices).astype(int)

long_data = long_data.assign(ID=indices[:, 0])
long_data = long_data.assign(year=indices[:, 1])

long_data.index = np.arange(len(long_data))


# get_heads defined here in original
print("begin get heads", time.time() - time0)
headdata = get_heads2(long_data)
headdata = headdata[headdata["to_drop"] == 0]
print("end get heads", time.time() - time0)

# Purge drop column
headdata = headdata.drop(columns=["to_drop"])

# Redo index
headdata.index = np.arange(len(headdata))

# Export data to get raw PSID data reformatted
print("begin export 1", time.time() - time0)
headdata.to_csv("out.csv")
print("end export 1", time.time() - time0)

# Begin setting up income series
print("begin income", time.time() - time0)
# setup yrs
old_inc_yrs = np.array((headdata.loc[:, "year"] < 94) & (
    headdata.loc[:, "year"] >= 60))
new_inc_yrs = np.array((headdata.loc[:, "year"] >= 94) | (
    headdata.loc[:, "year"] < 60))

# create variable np arrays
old_inc = np.array(headdata.loc[:, "total_labor_head"] + headdata.loc[:,
                                                                      "total_labor_spouse_68_93"] + headdata.loc[:, "head_wife_transfers_70_15"])
new_inc = np.array(headdata.loc[:, "total_labor_head"] + headdata.loc[:, "total_labor_spouse_94_15"] + headdata.loc[:,
                                                                                                                    "uninc_business_inc_94_15"] + headdata.loc[:, "farm_inc_94_15"] + headdata.loc[:, "head_wife_transfers_70_15"])
# Create array to hold income series
fullinc = np.empty_like(headdata.loc[:, "total_labor_head"])
# Put old and new income data together
fullinc[old_inc_yrs] = old_inc[old_inc_yrs]
fullinc[new_inc_yrs] = new_inc[new_inc_yrs]
# Put income data in data frame
headdata = headdata.assign(full_income = fullinc)
# Purge observations without income data

print("end income", time.time() - time0)

# Unify weights:
# get observations for 3 types of weights
weight_yrs_1 = np.array((headdata.loc[:, "year"] < 93) & (
    headdata.loc[:, "year"] >= 60))
weight_yrs_2 = np.array(
    (headdata.loc[:, "year"] >= 93) & (headdata.loc[:, "year"] < 97))
weight_yrs_3 = np.array(
    (headdata.loc[:, "year"] >= 97) | (headdata.loc[:, "year"] < 60))
# Put weights into numpy arrays
weights_68_92_np = np.array(headdata.loc[:, "weights_68_92"])
weights_93_96_np = np.array(headdata.loc[:, "weights_93_96"])
weights_97_15_np = np.array(headdata.loc[:, "weights_97_15"])

# Get array to store all weights
full_weights = np.empty_like(fullinc)
# Put weights in one array
full_weights[weight_yrs_1] = weights_68_92_np[weight_yrs_1]
full_weights[weight_yrs_2] = weights_93_96_np[weight_yrs_2]
full_weights[weight_yrs_3] = weights_97_15_np[weight_yrs_3]

headdata = headdata.assign(full_weights=full_weights)


# create normed weights
# generate numpy arrays of correct length
full_weights_norm = full_weights * np.nan
full_weights_norm_stable = full_weights * np.nan
#change_in_comp_np = np.array(headdata.loc[:,"change_in_fam_comp"])

# Get years to process
uni_years = np.unique(headdata["year"])

# Fore each year
for i, x in enumerate(uni_years):
    # find observations for both year and fixed families in a year
    working_data = np.array(headdata.loc[:, "year"] == x)
    working_data_s = np.array((headdata.loc[:, "year"] == x) & (
        headdata.loc[:, "change_in_fam_comp"] == 0))
    # get weights of active groups
    working_weights = full_weights[working_data]
    working_weights_s = full_weights[working_data_s]
    # normalize the weights to sum to one
    nworking_weights = working_weights / working_weights.sum()
    nworking_weights_s = working_weights_s / working_weights_s.sum()
    # Insert new weights into array
    full_weights_norm[working_data] = nworking_weights
    full_weights_norm_stable[working_data_s] = nworking_weights_s

# Put new weights in data frame
headdata = headdata.assign(full_weights_norm=full_weights_norm)
headdata = headdata.assign(full_weights_norm_s=full_weights_norm_stable)


# drop those with NaN incomes (or weights if neeeded, currently just income)
# & (headdata["full_weights_norm_s"].notnull())]
prefinaldata = headdata[(headdata["full_income"].notnull())]

# Export data to get raw PSID data reformatted
print("begin export 2", time.time() - time0)
headdata.to_csv("out2.csv")
print("end export 2", time.time() - time0)

# temporary adjustment to keep income reasonable used to incldue attempt to exclude  99%ile income
prefinaldata = prefinaldata[(prefinaldata["full_income"] > 0)]
# (prefinaldata["full_income"] < 200000) &

# create smaller new df
finaldata = prefinaldata.loc[:, [
    "ID", "year", "age", "change_in_fam_comp", "full_income", "full_weights", "full_weights_norm_s"]]

# create lags
finaldata.set_index(["ID", "year"], inplace=True)

# Create index to use for lags via fancy indexing
index1 = list(finaldata.index)
index2 = [(i[0], i[1] - 1) for i in index1]
for i, x in enumerate(index1):
    if x[1] == 1:
        index2[i] = (x[0], 99)
    elif x[1] > 98:
        index2[i] = (x[0], x[1] - 2)
    elif x[1] < 60:
        index2[i] = (x[0], x[1] - 2)
# Create lag with fancy values
finaldata = finaldata.assign(full_income_lag=np.array(
    finaldata.loc[index2, "full_income"]))

# create set for markov chain
markov_data = finaldata[(finaldata["full_income"].notnull()) & (finaldata["full_income_lag"].notnull()) & (
    finaldata["full_weights_norm_s"].notnull()) & (finaldata["change_in_fam_comp"] == 0)]

# restore index, id , year
indices = [list(i) for i in markov_data.index]
indices = np.array(indices).astype(int)

markov_data = markov_data.assign(ID=indices[:, 0])
markov_data = markov_data.assign(year=indices[:, 1])

markov_data.index = np.arange(len(markov_data))

# calc chang in income
markov_data = markov_data.assign(income_over_last=np.array(
    markov_data.loc[:, "full_income"]) / np.array(markov_data.loc[:, "full_income_lag"]))

# get rid of odd ones
markov_data = markov_data[(markov_data["income_over_last"] >= (
    1 / 20)) & (markov_data["income_over_last"] < 20)]


# redo weights
for i, x in enumerate(np.unique(markov_data["year"])):
    yeararray = (markov_data["year"] == x)
    markov_data.loc[yeararray, "full_weights_norm_s"] = markov_data.loc[yeararray,
                "full_weights_norm_s"] / (markov_data.loc[yeararray, "full_weights_norm_s"].sum())


# fix years
markov_data = fix_years(markov_data, "year")

# export data
print("begin export 3", time.time() - time0)
markov_data.to_csv("markov_data.csv",index=False)
print("end export 3", time.time() - time0)


# calc average change income by year

# avg_inc_change = np.empty_like(np.unique(markov_data["year"])).astype(float)
#
# for i, x in enumerate(np.unique(markov_data["year"])):
#     yeararray = (markov_data["year"] == x)
#     avg_inc_change[i] = (markov_data.loc[yeararray, "full_weights_norm_s"]
#                          * markov_data.loc[yeararray, "income_over_last"]).sum()
#
# print("total time", time.time() - time0)

# N = 5

# markov_gen_one(df,yr1,yr_name,var,weights,ids,N):
# example_markov = markov_gen_one(
#     markov_data, 2003, "year", "full_income", "full_weights_norm_s", "ID", N)

# markov_data_yrs = np.unique(np.asarray(markov_data["year"]))
#
# hold = []
# for i, x in enumerate(markov_data_yrs[3:-3]):
#     test_markov = markov_gen_one(
#         markov_data, x, "year", "full_income", "full_weights_norm_s", "ID", N)
#     hold.append(test_markov)
#
# hold_std = []
# for i in range(N):
#     for j in range(N):
#         hold_vals = []
#         for k, x in enumerate(hold):
#             hold_vals.append(x[i, j])
#         # print(hold_vals)
#         # print("var",np.asarray(hold_vals).var())
#         # print("std",np.asarray(hold_vals).std())
#         hold_std.append(np.asarray(hold_vals).std())
# print("avg std", np.asarray(hold_std).mean())
