#Convincence functions for NeuralABC


def linreg(iv, dv, cov=None, verbose=1,save_r= True, save_r2=True, save_betas=True, save_pvals=True, save_tstats=True, save_res=True):
    """
    Run multiple regression with covariates.
    Not yet parallized.
    zklsmr

    :param iv: (arr, dict, df), required: Independent variable with the shape participant X features.
    :param dv: (arr, dict, df), required: Dependent variable with the shape participant X features.
    :param cov: (arr, dict, df), optional: Covariates with the shape participant X features. default=None
    :param verbose: (int), optional: Show progress (0=silent, 1=major steps, 2=I want to know everything). default=1.
    :param save_r: (Bool), optional: Save the correlation coefficient of the simple regression (type float). default=True.
    :param save_r2: (Bool), optional: Save R2 of the model (type float). default=True.
    :param save_betas: (Bool), optional: Save Regression Coefficients (beta) of the model (type float). default=True.
    :param save_pvals: (Bool), optional: Save p-values of the model (type float). default=True.
    :param save_tstats: (Bool), optional: Save t-values of the model (beta / SE) (type float). default=True.
    :param save_res: (Bool), optional: Save residuals of the model (participant X residual) (type float). default=True.

    :returns dict: Everything you said you want in one dictionary. It is up to you to parse it after based on dict keys.
        dict.keys() = ['corr_coefs', 'r_squared', 'reg_coefs', 'p_values', 't_values', 'residuals']

    Example:
    
    >>> import pandas as pd
    >>> import numpy as np
    >>> import conv_functions as cnvf


    >>> x = pd.DataFrame([[1,6],[3,3],[6,9],[5,8],[4,9],[5,6]], columns=["var1", "var2"])

    >>> y = [4,9,6,7,7,8]

    >>> age = [22,24,25,22,26,21]

    >>> results = cnvf.linreg(x,y,age, verbose=2, save_r = False, save_r2=False, save_betas=False, save_pvals=True, save_tstats=True, save_res=False)
        __________________________________
        Converting your data to pandas dataframes
        __________________________________
        Starting Regression
        __________________________________
        working on ---> var1
        Finished ---> var1
        __________________________________
        working on ---> var2
        Finished ---> var2
        __________________________________
        Done regression, saving output to dict
        _______________________________________
        Your regression analysis took 0.0 minutes
        =========================================================
        
    >>> pvalues = pd.DataFrame(results["p-values"])
    >>> tvalues = pd.DataFrame(results["t-values"])

    >>> print("pvalues are")
    >>> print(pvalues)

        pvalues are
            var1      var2
        0  0.524761  0.421867

    >>> print("tvalues are")
    >>> print(tvalues)
    
        tvalues are
            var1      var2
        0  0.717737 -0.927979
    """    

    import pandas as pd
    import numpy as np
    import pingouin as pg
    import time

    start = time.time()

    #initialize output dictionaries
    model_r = {}
    model_r2 = {}
    pvals = {}
    coefs = {}
    tstats = {}
    res = {}

    #convert to pandas df
    if verbose >=1:
        print("__________________________________")
        print("Converting your data to pandas dataframes")

    IV = pd.DataFrame(iv)
    DV = pd.DataFrame(dv)
    
    if cov is not None:
        Cov = pd.DataFrame(cov)

    if verbose >=1:
        print("__________________________________")
        print("Starting Regression")
        print("__________________________________")


    for col1 in IV.columns:
        IV_df = IV[col1]
        if cov is not None:
            x_vals = pd.concat([IV_df,Cov], axis=1)
        else:
            x_vals = IV_df
            
        rdata = []
        rsquared = []
        bdata = []
        pdata = []
        tdata = []
        resdata = {}
        
        if verbose >= 2:
            print(f"working on ---> {col1}")
        
        for col2 in DV.columns:
            y_vals = DV[col2]
            lm = pg.linear_regression(X = x_vals, y = y_vals)

         
            if cov is None:
                if save_r is not False:
                    co  = lm["coef"][1] * (x_vals.std() / y_vals.std())
                    rdata.append(co)
        
            if save_r2 is not False:
                r_p2 = lm["r2"][1]
                rsquared.append(r_p2)
            if save_betas is not False:
                bval = lm["coef"][1]
                bdata.append(bval)
            if save_pvals is not False:
                pval = lm["pval"][1]
                pdata.append(pval)
            if save_tstats is not False:
                tval = lm['T'][1]
                tdata.append(tval)
            if save_res is not False:
                resids = lm.residuals_   
                resdata[col2] = resids

        if verbose >= 2:
            print(f"Finished ---> {col1}")
            print("__________________________________")

        if cov is None:
            if save_r is not False:
                model_r[col1] = rdata  

        if save_r2 is not False:
            model_r2[col1] = rsquared      
        if save_betas is not False:
            coefs[col1] = bdata
        if save_pvals is not False:
            pvals[col1] = pdata
        if save_tstats is not False:
            tstats[col1] = tdata
        if save_res is not False:
            res[col1] = resdata  

    if verbose >=1:
            print("Done regression, saving output to dict")
            print("_______________________________________")

        
    if verbose >=2:
        end = time.time()
        ittook = (end - start)/60
        print(f"Your regression analysis took {np.round(ittook, 2)} minutes")
        print("=========================================================")
    
    out_put = {"corr_coefs": model_r, "r_squared": model_r2, "reg_coefs": coefs, "p_values": pvals, "t_values": tstats, "residuals": res}
    return out_put


