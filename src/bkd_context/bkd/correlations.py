import pandas as pd

pro_path = 'UCEC_discovery_Proteomics_PNNL_ratio_median_polishing_log2.csv'
muta_path = 'UCEC_discovery_somatic_mutation_gene_level.csv'
df_sommut, df_proteo = pd.read_csv(muta_path), pd.read_csv(pro_path)
df_sommut.columns = [df_sommut.columns[0]] + [col + '_mut' for col in df_sommut.columns[1:]]
df_sommut = df_sommut[['id', 'CTNNB1_mut']]
df_proteo.columns = [df_proteo.columns[0]] + [col + '_proteo' for col in df_proteo.columns[1:]]

merged_df = df_proteo.merge(df_sommut, on='id')
merged_df.drop('id', axis=1, inplace=True)
nan_all = merged_df.columns[merged_df.isnull().all()]
merged_df = merged_df.drop(columns=nan_all)
