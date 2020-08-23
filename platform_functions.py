#!/usr/bin/env python
# -*- coding: utf-8 -*-
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

# Pre-processing functions
def create_new_localization_feature(col_sg_uf, col_sg_uf_matriz):
    '''
    Create a new feature developed during pre-processing step. For more info read see pre-processing notebooks.
    Args:
        - col_sg_uf: self explanatory
        - col_sg_uf_matriz: self explanatory
    Returns:
        pd.Series
    '''
    return (col_sg_uf != col_sg_uf_matriz).astype(int)

def apply_mask_for_cats_bool(data):
    '''
    Apply mask for boolean categorical features and turn them into numbers.
    Args:
        - data: pd.DataFrame to transform
    Returns:
        pd.DataFrame transformed
    '''
    return data.replace({False:0,True:1,'NAO':0,'SIM':1})

def apply_mask_for_cats_ordinal(data):
    '''
    Apply mask for ordinal categorical features and turn them into numbers.
    Args:
        - data: pd.DataFrame to transform
    Returns:
        pd.DataFrame transformed
    '''
    mask_ord = {
                '<= 1':0,'1 a 5':1,'5 a 10':2,'10 a 15':3,'15 a 20':4,'> 20':5, # idade_emp_cat
                'VERMELHO':1,'LARANJA':2,'AMARELO':3,'CINZA':4,'AZUL':5,'VERDE':6, # de_saude_tributaria
                'MUITO BAIXA':1,'BAIXA':2,'MEDIA':3, 'ALTA':4, # de_nivel_atividade
                'SEM INFORMACAO':0,'ATE R$ 81.000,00':1,'DE R$ 81.000,01 A R$ 360.000,00':2,  
                'DE R$ 360.000,01 A R$ 1.500.000,00':3,'DE R$ 1.500.000,01 A R$ 4.800.000,00':4,
                'DE R$ 4.800.000,01 A R$ 10.000.000,00':5,'DE R$ 10.000.000,01 A R$ 30.000.000,00':6,
                'DE R$ 30.000.000,01 A R$ 100.000.000,00':7,'DE R$ 100.000.000,01 A R$ 300.000.000,00':8,
                'DE R$ 300.000.000,01 A R$ 500.000.000,00':9,'DE R$ 500.000.000,01 A 1 BILHAO DE REAIS':10,
                'ACIMA DE 1 BILHAO DE REAIS':11 # de_faixa_faturamento_estimado & estimado_grupo               
               }
    # Return
    return data.replace(mask_ord)

def transform_test_with_market_procedure(test, original_columns):
    '''
    Pre-processing pipeline function. This function will transformed test set based in pre-processing developed for training set.
    Args:
        - test: pd.DataFrame object to be transformed.
        - original_columns: columns used in training set.
    Returns:
        pd.DataFrame transformed
    '''
    # Create new feature
    test['nf_expanded_companies'] = create_new_localization_feature(test['sg_uf'],test['sg_uf_matriz'])
    # Use only remaining columns
    test = test[original_columns]
    # Apply Masks
    test = apply_mask_for_cats_bool(test)
    test = apply_mask_for_cats_ordinal(test)    
    # Transform to numeric features
    feats = sorted([x for x in original_columns if x not in ['de_faixa_faturamento_estimado', 'id']])
    cbEncoder = joblib.load('models/catBoostEncoder.sav')
    test[feats] = cbEncoder.transform(test[feats])   
    # Retrun
    return test

def impute_mean_for_numeric_columns(test, train, columns_to_impute):
    '''
    Function to realize mean imputation
    Args:
        - test: pd.DataFrame object to be transformed.
        - train: training set to get the means to impute
        - columns_to_impute: columns to be imputed.
    Returns:
        pd.DataFrame imputed
    '''
    for col in columns_to_impute:
        test[col] = test[col].fillna(train[col].mean())
    return test

def apply_standard_scaler(test):
    '''
    Function to apply standard scaler transformation with pre-loaded StandardScalerModel
    Args:
        - test: pd.DataFrame object to be transformed.
        - train: training set to get the means to impute
        - columns_to_impute: columns to be imputed.
    Returns:
        pd.DataFrame transformed
    '''
    cols_to_transform = ['qt_filiais','idade_empresa_anos','vl_total_veiculos_pesados_grupo','vl_total_veiculos_leves_grupo',
                         'de_faixa_faturamento_estimado','de_faixa_faturamento_estimado_grupo','de_natureza_juridica',
                         'de_nivel_atividade','nm_divisao']
    transformer = joblib.load('models/standardScalerReduced.sav')
    test[cols_to_transform] = transformer.transform(test[cols_to_transform])
    return test
    
def pipeline_for_test_with_imputation(test, training_cols):
    '''
    Function to realize all pre-processing phases: transforming, imputation and standardization.
    Args:
        - test: pd.DataFrame object to be transformed.
        - training_cols: columns used in training set.
    Returns:
        pd.DataFrame ready to evaluation
    '''
    # Import necessary dataset
    train_original = pd.read_csv('processed_data/market_transformed_processed.csv')
    # Apply transforms
    test = transform_test_with_market_procedure(test, train_original.columns)
    test = impute_mean_for_numeric_columns(test,train_original,test.columns[(pd.isnull(test).sum() > 0).values])
    test = apply_standard_scaler(test)
    test.set_index('id', drop=True, inplace=True)
    test = test[training_cols]
    return test

# Prediction functions
def get_leads(test,model, n_leads_per_client):
    '''
    Function to make predictions with pre-loaded NN based in test set.
    Args:
        - test: pd.DataFrame object to be predicted
        - n_leads_per_client: int value. How many leads you want to predict for each client.
        - model: recommender model from sklearn type
    Returns:
        predicted leads indexes
    '''    
    port_leads = model.kneighbors(test, n_neighbors=n_leads_per_client, return_distance=False)
    return port_leads.reshape(-1)

def get_leads_info_in_market(market_info_df, leads_idx):
    '''
    Function to get original informations about the leads.
    '''   
    return market_info_df.iloc[leads_idx,:]

def get_leads_info_in_training(training, leads_idx):
    '''
    Function to get informations about the leads.
    '''  
    return training.iloc[leads_idx,:]

# Load necessary info
@st.cache(allow_output_mutation=True)
def load_datasets():
    # Market Dataset with original categories - for plots
    market_info = pd.read_csv('processed_data/market_analyzed.csv',sep=',',encoding='latin-1').set_index('id',drop=True)
    # Market Processed - To process (2) Similarity
    training = pd.read_csv('processed_data/market_transformed_processed_featSelected_std.csv', index_col='id')
    return market_info, training

@st.cache(allow_output_mutation=True)
def load_recommender_model():
    rm = joblib.load('models/NN-5-l2DistanceNormalizedPlataform.sav')
    return rm

# Plot functions
def plot_portfolio_market_pie_similarity(X,Y,threshold=0.70):
    '''
    Plot Cosine Similarity for portfolio (X) and leads (Y).
    Args:
        - X: Portfolio
        - Y: Leads
        - threshold: value from 1 to 0 to define how powerful are the similarity.
    Returns:
        similarity score and plot
    '''
    # Calculate similarity
    cs = cosine_similarity(X,Y)
    # Get infos
    hs = (cs >= threshold).sum()
    dims = (cs.shape[0]*cs.shape[1])
    # Get similarities
    hs = (100*hs/dims).round(2)
    ot = 100 - hs
    # pie chart
    labels = ['Taxa de Similaridade alta', '']
    sizes = [hs, ot]    
    fig1, ax1 = plt.subplots()
    _, _, autopcts = ax1.pie(sizes, labels=labels, explode=(0.5, 0), autopct='%1.1f%%', shadow=True, startangle=30,colors = ('tab:red', 'tab:blue'))
    plt.setp(autopcts, **{'color':'black', 'weight':'bold', 'fontsize':15})
    ax1.set_title('', fontdict={'fontsize': 17})
    ax1.axis('equal')
    #plt.show()
    st.pyplot()
    return hs

def fazer_anotacao_frequencia_relativa(eixo, coord_texto_xy, total_amostras):
    '''
    Auxiliar function to to annotations.
    '''
    # Loop em cada barra que existe no gráfico
    for barra in eixo.patches:
        tamanho_barra = barra.get_width()  # Pegar o tamanho da barra
        texto_barra = str(round(100*tamanho_barra/total_amostras, 2)) + ' %' # Texto que sera anotado referente a barra
        coord_anotacao_xy = (barra.get_x() + tamanho_barra, barra.get_y())
        eixo.annotate(texto_barra, coord_anotacao_xy, xytext=coord_texto_xy, fontsize=25, color='black', 
                      textcoords='offset points', horizontalalignment='right',fontweight='bold')

def plot_bars(data, column, fig_title='Leads'):
    '''
    Plot Barplot for a desired column.
    '''
    fig, ax = plt.subplots(1,1, figsize=(20,30))
    vc_data = data[column].value_counts()
    # Leads
    if len(vc_data) >= 5:
        limit = 5
    else:
        limit = len(vc_data)
    sns.countplot(y=column,data=data,order=vc_data.iloc[:limit].index,ax=ax)
    ax.set_title(fig_title, fontsize=30,fontweight='bold')
    ax.set_ylabel('')
    ax.set_xlabel('')
    ax.set_xticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.set_tick_params(labelsize=24)
    fazer_anotacao_frequencia_relativa(ax,(140, -100),len(data))
    #plt.show()
    st.pyplot()