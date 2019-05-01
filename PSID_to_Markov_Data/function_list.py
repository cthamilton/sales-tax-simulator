import pandas as pd
from numba import jit
import numpy as np


@jit
def rename_data(data, var_full_parsed, var_full_names):
    # data: pandas dataframe with columns named as PSID codes
    # var_full_parsed: list of strings which contain the PSID series
    #                  codes for the data series tagged with year
    # var_full_names: list of variable names as strings
    #                 aligning with var_full_parsed
    for i, x in enumerate(var_full_parsed):
        temp_dict = {}
        for j, y in enumerate(x[1]):
            newname = var_full_names[i] + "__" + str(x[0][j])
            temp_dict[y] = newname
        data = data.rename(index=str, columns=temp_dict)
    return data


# Deprecated function
#@jit
# def get_heads(nndata):
#    nndata["to_drop"] = 0
#    for i in range(len(nndata)):
#        yeararray = (nndata["year"] <= 82)
#        if yeararray[i]:
#            if nndata["relation_to_head"][i] != 1:
#                nndata.loc[i,"to_drop"] = 1
#        else:
#            if nndata["relation_to_head"][i] != 10:
#                nndata.loc[i,"to_drop"] = 1
#            elif nndata["sequence_number"][i] != 1:
#                nndata.loc[i,"to_drop"] = 1
#    return nndata


@jit
def get_heads2(nndata):
    nndata["to_drop"] = 0
    nndata.loc[(nndata["year"] >= 60) & (nndata["year"] <= 82) &
               (nndata["relation_to_head"] != 1), "to_drop"] = 1
    nndata.loc[((nndata["year"] > 82) | (nndata["year"] < 60)) & (
        (nndata["relation_to_head"] != 10) | (nndata["sequence_number"] != 1)), "to_drop"] = 1
    return nndata


@jit
def weighted_variance(df, var_name, weight_name):
    var = np.asarray(df.loc[:, var_name])

    weights = np.asarray(df.loc[:, weight_name])
    sumw = weights.sum()
    nweights = weights / sumw
    sumnw = nweights.sum()

    avg = (var * nweights).sum()
    # print(avg)
    # print(sumnw)
    wvar_array = (weights * (var - avg) ** 2)
    wvar = sum(wvar_array)
    return wvar


@jit
def wvar_cat(df, cat_name, var_name, weight_name):
    cats = np.asarray(df[cat_name])
    ucats = np.unique(cats)
    ncats = len(ucats)
    v_array = np.empty(ncats)
    for i, x in enumerate(ucats):
        tdf = df[df[cat_name] == x]
        v_array[i] = weighted_variance(tdf, var_name, weight_name)
    return ucats, v_array


@jit
def fix_years(df, yrs_name):
    df.loc[df[yrs_name] > 50, yrs_name] = df.loc[df[yrs_name] > 50, yrs_name] + 1900
    df.loc[df[yrs_name] < 50, yrs_name] = df.loc[df[yrs_name] < 50, yrs_name] + 2000
    return df


@jit
def wdivide(array, weights, N, witharrays=False, ids=np.array([0.]), witharrays_id=False):
    # numpify
    array = np.asarray(array)
    weights = np.asarray(weights)
    ids = np.asarray(ids)
    # order arrays
    order = np.argsort(array)
    array = array[order]
    weights = weights[order]
    # normalize weights
    weights = weights / weights.sum()
    # find breaks
    weight_breaks = np.arange(N + 1) / N
    cutoffs = np.empty(N + 1)
    cutoffs[0] = array[0]
    cutoffs[N] = array[-1] + 1

    argcutoffs = np.empty(N + 1)
    argcutoffs[0] = 0
    argcutoffs[N] = len(array)

    count = 0
    for i, x in enumerate(weight_breaks):
        if (i > 0) & (i < N):
            while (weights[:count].sum() < x):
                count = count + 1
            cutoffs[i] = (array[count] + array[count + 1]) / 2
            argcutoffs[i] = count + 1
    if witharrays == True:
        return array, weights, argcutoffs.astype(int), cutoffs
    elif witharrays_id == True:
        ids = ids[order]
        return ids, array, weights, argcutoffs.astype(int), cutoffs
    else:
        return argcutoffs, cutoffs


@jit
def markov_gen_one(df, yr1, yr_name, var, weights, ids, N, bin_avgs=False):
    # find next year
    yrs = np.asarray(df[yr_name])
    yrs = yrs[yrs < yr1]
    yr0 = yrs.max()

    # gen dfs
    df1 = df[df[yr_name] == yr1].loc[:, [ids, var, weights]]
    df1.index = df1[ids]
    ids1 = np.asarray(df1[ids])

    # Set up dataframes for each year and indices
    df0 = df[df[yr_name] == yr0].loc[:, [ids, var, weights]]
    df0.index = df0[ids]
    ids0 = np.asarray(df0[ids])

    ids_int = np.intersect1d(ids0, ids1)
    df1 = df1.loc[ids_int, :]
    df0 = df0.loc[ids_int, :]

    # get info
    ids0, array0, weights0, argcutoffs0, cutoffs0 = wdivide(
        df0[var], df0[weights], N, ids=df0[ids], witharrays_id=True)
    ids1, array1, weights1, argcutoffs1, cutoffs1 = wdivide(
        df1[var], df1[weights], N, ids=df1[ids], witharrays_id=True)

    # gen Markov matric
    markov = np.empty((N, N))

    # fill out matrix
    for i in range(N):
        # Set up np arrays for use
        init_ids = ids0[argcutoffs0[i]:argcutoffs0[i + 1]]
        init_wes = weights0[argcutoffs0[i]:argcutoffs0[i + 1]]
        init_wes = init_wes / init_wes.sum()
        post_var = np.asarray(df1.loc[init_ids, var])

        for j in range(N):
            # find bounds for this section
            l_bd = cutoffs1[j]
            u_bd = cutoffs1[j + 1]
            # determine which ones are in this category
            selection = (post_var >= l_bd) & (post_var < u_bd)
            value = init_wes[selection].sum()
            # fill in markov matrix
            markov[i, j] = value
            # account for error or upper bound loss
            if j == (N - 1):
                markov[i, j] = markov[i, j] + 1 - markov[i].sum()
    # transpose matrix to get in correct shape
    markov = markov.T
    if bin_avgs == True:
        old_bin_avgs = np.empty(N)
        new_bin_avgs = np.empty(N)
        for i in np.arange(N):
            # yr0 bin avgs
            init_wes = weights0[argcutoffs0[i]:argcutoffs0[i + 1]]
            init_wes = init_wes / init_wes.sum()
            init_var = array0[argcutoffs0[i]:argcutoffs0[i + 1]]
            old_bin_avgs[i] = np.dot(init_wes, init_var)

            # yr1 bin avgs
            post_wes = weights1[argcutoffs1[i]:argcutoffs1[i + 1]]
            post_wes = post_wes / post_wes.sum()
            post_var = array1[argcutoffs1[i]:argcutoffs1[i + 1]]
            new_bin_avgs[i] = np.dot(post_wes, post_var)
        return markov, old_bin_avgs, new_bin_avgs
    else:
        return markov


@jit
def markov_gen_enl(df, yr1, yr_name, var, weights, ids, N, bin_avgs=False):
    # find next year
    yrs = np.asarray(df[yr_name])
    yrs = yrs[yrs < yr1]
    yr0 = yrs.max()

    # gen dfs
    df1 = df[df[yr_name] == yr1].loc[:, [ids, var, weights]]
    df1.index = df1[ids]
    ids1 = np.asarray(df1[ids])

    # Set up dataframes for each year and indices
    df0 = df[df[yr_name] == yr0].loc[:, [ids, var, weights]]
    df0.index = df0[ids]
    ids0 = np.asarray(df0[ids])

    # get info
    ids0, array0, weights0, argcutoffs0, cutoffs0 = wdivide(
        df0[var], df0[weights], N, ids=df0[ids], witharrays_id=True)
    ids1, array1, weights1, argcutoffs1, cutoffs1 = wdivide(
        df1[var], df1[weights], N, ids=df1[ids], witharrays_id=True)

    # print(argcutoffs0)
    # print(argcutoffs1)

    ids_int = np.intersect1d(ids0, ids1)
    df1 = df1.loc[ids_int, :]
    df0 = df0.loc[ids_int, :]

    # sort arrays
    order0 = np.argsort(df0[var])
    order1 = np.argsort(df1[var])

    ids0 = np.asarray(df0[ids])[order0]
    ids1 = np.asarray(df1[ids])[order1]

    array0 = np.asarray(df0[var])[order0]
    array1 = np.asarray(df1[var])[order1]

    weights0 = np.asarray(df0[weights])[order0]
    weights1 = np.asarray(df1[weights])[order1]

    argcutoffs0 = np.empty_like(argcutoffs0).astype(int)
    argcutoffs1 = np.empty_like(argcutoffs1).astype(int)

    argcutoffs0[0] = 0
    argcutoffs1[0] = 0
    argcutoffs0[-1] = len(df0)
    argcutoffs1[-1] = len(df1)

    for i in np.arange(1, N):
        argcutoffs0[i] = (array0 <= cutoffs0[i]).sum()
        argcutoffs1[i] = (array1 <= cutoffs1[i]).sum()

    # print(len(df),len(df0),len(df1))
    # print(argcutoffs0)
    # print(argcutoffs1)

    # gen Markov matric
    markov = np.empty((N, N))

    # fill out matrix
    for i in range(N):
        # Set up np arrays for use
        init_ids = ids0[argcutoffs0[i]:argcutoffs0[i + 1]]
        init_wes = weights0[argcutoffs0[i]:argcutoffs0[i + 1]]
        if init_wes.sum() == 0:
            raise ValueError('Zero sum of weights for quantile', i)
        init_wes = init_wes / init_wes.sum()
        post_var = np.asarray(df1.loc[init_ids, var])

        for j in range(N):
            # find bounds for this section
            l_bd = cutoffs1[j]
            u_bd = cutoffs1[j + 1]
            # determine which ones are in this category
            selection = (post_var >= l_bd) & (post_var < u_bd)
            value = init_wes[selection].sum()
            # fill in markov matrix
            markov[i, j] = value
            # account for error or upper bound loss
            # if j == (N-1):
            #     markov[i,j] = markov[i,j] + 1 - markov[i].sum()
    # transpose matrix to get in correct shape
    markov = markov.T
    if bin_avgs == True:
        old_bin_avgs = np.empty(N)
        new_bin_avgs = np.empty(N)
        for i in np.arange(N):
            # yr0 bin avgs
            init_wes = weights0[argcutoffs0[i]:argcutoffs0[i + 1]]
            init_wes = init_wes / init_wes.sum()
            init_var = array0[argcutoffs0[i]:argcutoffs0[i + 1]]
            old_bin_avgs[i] = np.dot(init_wes, init_var)

            # yr1 bin avgs
            post_wes = weights1[argcutoffs1[i]:argcutoffs1[i + 1]]
            post_wes = post_wes / post_wes.sum()
            post_var = array1[argcutoffs1[i]:argcutoffs1[i + 1]]
            new_bin_avgs[i] = np.dot(post_wes, post_var)
        return markov, old_bin_avgs, new_bin_avgs
    else:
        return markov


# @jit
def markov_gen_enl_age(df, yr1, age, yr_name, age_name, var, weights, ids, N, bin_avgs=False,by_age_bins=False):
    # find next year
    yrs = np.asarray(df[yr_name])
    yrs = yrs[yrs < yr1]
    yr0 = yrs.max()

    # gen dfs
    df1 = df[df[yr_name] == yr1].loc[:, [ids, var, weights, age_name]]
    df1.index = df1[ids]
    ids1 = np.asarray(df1[ids])

    # Set up dataframes for each year and indices
    df0 = df[df[yr_name] == yr0].loc[:, [ids, var, weights, age_name]]
    df0.index = df0[ids]
    ids0 = np.asarray(df0[ids])

    if by_age_bins == True:
        df1 = df1[df1[age_name] == age]

        ids0 = np.asarray(df0[ids])
        ids1 = np.asarray(df1[ids])

        ids_int = np.intersect1d(ids0, ids1)
        ages0 = df0[age_name][df0[ids].isin(ids_int)]
        max_age0 = ages0.max()
        min_age0 = ages0.min()

        df0 = df0[(df0[age_name] <= max_age0) & (df0[age_name] >= min_age0)]


    # get info
    ids0, array0, weights0, argcutoffs0, cutoffs0 = wdivide(
        df0[var], df0[weights], N, ids=df0[ids], witharrays_id=True)
    ids1, array1, weights1, argcutoffs1, cutoffs1 = wdivide(
        df1[var], df1[weights], N, ids=df1[ids], witharrays_id=True)

    df1 = df1[df1[age_name] == age]
    df1 = df1.loc[:, [ids, var, weights]]
    df0 = df0.loc[:, [ids, var, weights]]

    ids0 = np.asarray(df0[ids])
    ids1 = np.asarray(df1[ids])

    ids_int = np.intersect1d(ids0, ids1)
    n = len(ids_int)
    # print(n)
    df1 = df1.loc[ids_int, :]
    df0 = df0.loc[ids_int, :]

    # sort arrays
    order0 = np.argsort(df0[var])
    order1 = np.argsort(df1[var])

    ids0 = np.asarray(df0[ids])[order0]
    ids1 = np.asarray(df1[ids])[order1]

    array0 = np.asarray(df0[var])[order0]
    array1 = np.asarray(df1[var])[order1]

    weights0 = np.asarray(df0[weights])[order0]
    weights1 = np.asarray(df1[weights])[order1]


    argcutoffs0 = np.empty_like(argcutoffs0).astype(int)
    argcutoffs1 = np.empty_like(argcutoffs1).astype(int)

    argcutoffs0[0] = 0
    argcutoffs1[0] = 0
    argcutoffs0[-1] = len(df0)
    argcutoffs1[-1] = len(df1)

    for i in np.arange(1, N):
        argcutoffs0[i] = (array0 <= cutoffs0[i]).sum()
        argcutoffs1[i] = (array1 <= cutoffs1[i]).sum()

    # print(cutoffs0[9])
    # print(array0)
    # print(argcutoffs0)
    # print(argcutoffs1)

    # gen Markov matric
    markov = np.empty((N, N))

    # fill out matrix
    for i in range(N):
        # Set up np arrays for use
        init_ids = ids0[argcutoffs0[i]:argcutoffs0[i + 1]]
        # init_wes = weights0[argcutoffs0[i]:argcutoffs0[i + 1]]
        # change180701
        init_wes = weights1[argcutoffs0[i]:argcutoffs0[i + 1]]
        if (age == 23):
            print (n, init_wes / weights1.sum())
        # print(echo)
        # print(init_wes)
        if init_wes.sum() == 0:
            # to receive age in error message first stop jit comp of function, ' for age ',age)
            # , ' for age ',age)
            raise ValueError('Zero sum of weights for quantile ', i)
        init_wes = init_wes / init_wes.sum()
        post_var = np.asarray(df1.loc[init_ids, var])
        for j in range(N):
            # find bounds for this section
            l_bd = cutoffs1[j]
            u_bd = cutoffs1[j + 1]
            # determine which ones are in this category
            selection = (post_var >= l_bd) & (post_var < u_bd)
            value = init_wes[selection].sum()
            # fill in markov matrix
            markov[i, j] = value
            # account for error or upper bound loss
            # if j == (N-1):
            #     markov[i,j] = markov[i,j] + 1 - markov[i].sum()
    # transpose matrix to get in correct shape
    markov = markov.T
    if bin_avgs == True:
        old_bin_avgs = np.empty(N)
        new_bin_avgs = np.empty(N)
        for i in np.arange(N):
            # yr0 bin avgs
            init_wes = weights0[argcutoffs0[i]:argcutoffs0[i + 1]]
            init_wes = init_wes / init_wes.sum()
            init_var = array0[argcutoffs0[i]:argcutoffs0[i + 1]]
            old_bin_avgs[i] = np.dot(init_wes, init_var)

            # yr1 bin avgs
            post_wes = weights1[argcutoffs1[i]:argcutoffs1[i + 1]]
            post_wes = post_wes / post_wes.sum()
            post_var = array1[argcutoffs1[i]:argcutoffs1[i + 1]]
            new_bin_avgs[i] = np.dot(post_wes, post_var)
        return markov, old_bin_avgs, new_bin_avgs
    else:
        return markov




#@jit
def markov_gen_enl_age_bins(df, yr1, age_low, age_high, yr_name, age_name, var, weights, ids, N, bin_avgs=False, cutoffs="lump",init_dist=False):
    # find next year
    yrs = np.asarray(df[yr_name])
    yrs = yrs[yrs < yr1]
    yr0 = yrs.max()

    # gen dfs
    df1 = df[df[yr_name] == yr1].loc[:, [ids, var, weights, age_name]]
    df1.index = df1[ids]
    ids1 = np.asarray(df1[ids])

    # Set up dataframes for each year and indices
    df0 = df[df[yr_name] == yr0].loc[:, [ids, var, weights, age_name]]
    df0.index = df0[ids]
    ids0 = np.asarray(df0[ids])

    # get info
    if cutoffs == "lump":
        df_all = pd.concat([df0, df1], ignore_index=True)

        ids0, array0, weights0, argcutoffs0, cutoffs0 = wdivide(
            df_all[var], df_all[weights], N, ids=df_all[ids], witharrays_id=True)

        ids1, array1, weights1 = ids0, array0, weights0
        argcutoffs1, cutoffs1 = argcutoffs0, cutoffs0


        if bin_avgs == True:
        # generate things needed for bin avg
            df_all = df_all.loc[:, [ids, var, weights]]
            ids_all = np.asarray(df1[ids])
            n_all = 0
            n_all = len(ids_all)
            order_all = np.argsort(df_all[var])
            ids_all = np.asarray(df_all[ids])[order_all]
            array_all = np.asarray(df_all[var])[order_all]
            weights_all = np.asarray(df_all[weights])[order_all]
            argcutoffs_all = np.empty_like(argcutoffs1).astype(int)
            argcutoffs_all[0] = 0
            argcutoffs_all[-1] = len(df_all)
            for i in np.arange(1, N):
                argcutoffs_all[i] = (array_all <= cutoffs0[i]).sum()


    elif cutoffs == "year":
        ids0, array0, weights0, argcutoffs0, cutoffs0 = wdivide(
            df0[var], df0[weights], N, ids=df0[ids], witharrays_id=True)
        ids1, array1, weights1, argcutoffs1, cutoffs1 = wdivide(
            df1[var], df1[weights], N, ids=df1[ids], witharrays_id=True)
    else:
        raise ValueError('cutoffs type must be lump or year')

    df1 = df1[df1[age_name] >= age_low]
    df1 = df1[df1[age_name] < age_high]
    df1 = df1.loc[:, [ids, var, weights]]
    df0 = df0.loc[:, [ids, var, weights]]

    ids0 = np.asarray(df0[ids])
    ids1 = np.asarray(df1[ids])

    ids_int = np.intersect1d(ids0, ids1)
    n = len(ids_int)

    df1 = df1.loc[ids_int, :]
    df0 = df0.loc[ids_int, :]

    # sort arrays
    order0 = np.argsort(df0[var])
    order1 = np.argsort(df1[var])

    ids0 = np.asarray(df0[ids])[order0]
    ids1 = np.asarray(df1[ids])[order1]

    array0 = np.asarray(df0[var])[order0]
    array1 = np.asarray(df1[var])[order1]

    weights0 = np.asarray(df0[weights])[order0]
    weights1 = np.asarray(df1[weights])[order1]

    argcutoffs0 = np.empty_like(argcutoffs0).astype(int)
    argcutoffs1 = np.empty_like(argcutoffs1).astype(int)

    argcutoffs0[0] = 0
    argcutoffs1[0] = 0
    argcutoffs0[-1] = len(df0)
    argcutoffs1[-1] = len(df1)

    for i in np.arange(1, N):
        argcutoffs0[i] = (array0 <= cutoffs0[i]).sum()
        argcutoffs1[i] = (array1 <= cutoffs1[i]).sum()

    # print(cutoffs0[9])
    # print(array0)
    # print(argcutoffs0)
    # print(argcutoffs1)

    # gen Markov matric
    markov = np.empty((N, N))

    dist = np.empty(N)

    # fill out matrix
    for i in range(N):
        # Set up np arrays for use
        init_ids = ids0[argcutoffs0[i]:argcutoffs0[i + 1]]
        init_wes = weights0[argcutoffs0[i]:argcutoffs0[i + 1]]
        dist[i] = init_wes.sum()
        if init_wes.sum() == 0:
            # to receive age in error message first stop jit comp of function, ' for age ',age)
            # , ' for age ',age)
            raise ValueError('Zero sum of weights for quantile ', i)
        init_wes = init_wes / init_wes.sum()
        post_var = np.asarray(df1.loc[init_ids, var])
        for j in range(N):
            # find bounds for this section
            l_bd = cutoffs1[j]
            u_bd = cutoffs1[j + 1]
            # determine which ones are in this category
            selection = (post_var >= l_bd) & (post_var < u_bd)
            value = init_wes[selection].sum()
            # fill in markov matrix
            markov[i, j] = value
            # account for error or upper bound loss
            # if j == (N-1):
            #     markov[i,j] = markov[i,j] + 1 - markov[i].sum()
    # transpose matrix to get in correct shape
    markov = markov.T
    dist = dist / dist.sum()
    if bin_avgs == True:
        if cutoffs == "year":
            old_bin_avgs = np.empty(N)
            new_bin_avgs = np.empty(N)
            for i in np.arange(N):
                # yr0 bin avgs
                init_wes = weights0[argcutoffs0[i]:argcutoffs0[i + 1]]
                init_wes = init_wes / init_wes.sum()
                init_var = array0[argcutoffs0[i]:argcutoffs0[i + 1]]
                old_bin_avgs[i] = np.dot(init_wes, init_var)

                # yr1 bin avgs
                post_wes = weights1[argcutoffs1[i]:argcutoffs1[i + 1]]
                post_wes = post_wes / post_wes.sum()
                post_var = array1[argcutoffs1[i]:argcutoffs1[i + 1]]
                new_bin_avgs[i] = np.dot(post_wes, post_var)
            if init_dist == True:
                return markov, old_bin_avgs, new_bin_avgs, dist
            else:
                return markov, old_bin_avgs, new_bin_avgs
        elif cutoffs == "lump":
            bin_avgs = np.empty(N)
            for i in np.arange(N):
                # yr0 bin avgs
                weights_all = weights_all / weights_all.sum()
                wes = weights_all[argcutoffs_all[i]:argcutoffs_all[i + 1]]
                wes = wes / wes.sum()
                var = array_all[argcutoffs_all[i]:argcutoffs_all[i + 1]]
                bin_avgs[i] = np.dot(wes, var)
            if init_dist == True:
                return markov, bin_avgs, dist
            else:
                return markov, bin_avgs
    else:
        if init_dist == True:
            return markov, dist
        else:
            return markov


@jit
def markov_gen_age(df, yr1, yr_name, var, weights, ids, N, age_name, low_age, high_age):
    ages = np.arange(low_age, high_age + 1)
    age_count = len(ages)
    out_matrix = np.empty((age_count, N, N)).astype(float)

    for i, age in enumerate(ages):
        out_matrix[i, :, :] = markov_gen_enl_age(
            df, yr1, age, yr_name, age_name, var, weights, ids, N, bin_avgs=False)

    return out_matrix

# @jit
def markov_gen_age_abins(df, yr1, yr_name, var, weights, ids, N, age_name, low_age, high_age):
    ages = np.arange(low_age, high_age + 1)
    age_count = len(ages)
    out_matrix1 = np.empty((age_count, N, N)).astype(float)
    out_matrix2 = np.empty((age_count, N)).astype(float)

    for i, age in enumerate(ages):
        markov, old_bin_avgs, new_bin_avgs = markov_gen_enl_age(
            df, yr1, age, yr_name, age_name, var, weights, ids, N, bin_avgs=True, by_age_bins=True)
        out_matrix1[i, :, :] = markov
        out_matrix2[i,:]     = new_bin_avgs
    return out_matrix1, out_matrix2


#@jit
def markov_gen_age_bin(df, yr1, yr_name, var, weights, ids, N, age_name, low_age, high_age, bin_size,bin_avgs=False,cutoffs="lump",init_dist=False):
    ages = np.arange(low_age, high_age + 1, bin_size)
    bin_count = len(ages)
    out_matrix = np.empty((bin_count - 1, N, N)).astype(float)
    avgs = np.empty(N).astype(float)
    for i, age in enumerate(ages):
        if (i < (bin_count - 1)):
            if (i == 0)&(bin_avgs == True)&(cutoffs == "lump"):
                if init_dist == True:
                    out_matrix[i, :, :], avgs, dist  = markov_gen_enl_age_bins(
                        df, yr1, ages[i], ages[i + 1], yr_name, age_name, var, weights, ids, N,bin_avgs=bin_avgs,cutoffs=cutoffs,init_dist=True)
                else:
                    out_matrix[i, :, :], avgs  = markov_gen_enl_age_bins(
                        df, yr1, ages[i], ages[i + 1], yr_name, age_name, var, weights, ids, N,bin_avgs=bin_avgs,cutoffs=cutoffs)

            else:
                out_matrix[i, :, :] = markov_gen_enl_age_bins(
                    df, yr1, ages[i], ages[i + 1], yr_name, age_name, var, weights, ids, N,cutoffs=cutoffs)
        else:
            pass
    if bin_avgs == True:
        if init_dist == True:
            return out_matrix, avgs, dist
        else:
            return out_matrix, avgs
    else:
        return out_matrix
