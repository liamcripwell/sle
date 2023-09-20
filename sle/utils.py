import re

import numpy as np
from easse.fkgl import corpus_fkgl


def smooth_labels(df, label_col="reading_level", num_labels=5):
    fkgls = []
    for _, row in df.iterrows():
        fkgl = corpus_fkgl([re.sub(r'^(\.+|\!|\?|")+ *', '', row.text)])
        fkgl = -fkgl # negative so that more complex == lower scores
        fkgls.append(fkgl)
        
    df["fkgl"] = fkgls

    # remove outliers
    df = df[(df["fkgl"] < df["fkgl"].quantile(0.99)) & (df["fkgl"] > df["fkgl"].quantile(0.01))]
    
    # scale fkgls
    scaled_fkgls = []
    mean_fkgls = []
    for lvl in range(num_labels):
        fkgls = df[df[label_col] == lvl]["fkgl"].to_numpy()
        scaled_fkgls_i = ( (fkgls - min(fkgls)) / (max(fkgls) - min(fkgls)) ) * 2 # *2 is for +-1 variance
        scaled_fkgls.append(scaled_fkgls_i)
        mean_fkgls.append(np.mean(scaled_fkgls_i))
        
    # adjust to be +- original reading level
    full_scaled_fkgls = []
    for _, row in df.iterrows():
        lvl = row[label_col]
        
        # pop the first item (ordered by label value)
        fkgl = scaled_fkgls[lvl][0]
        scaled_fkgls[lvl] = scaled_fkgls[lvl][1:]
        
        full_scaled_fkgls.append(fkgl - mean_fkgls[lvl] + lvl)
    df[f"{label_col}_smooth"] = full_scaled_fkgls

    return df