#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import streamlit as st
import platform_functions as platFunc

st.set_option('deprecation.showfileUploaderEncoding', False)

# Pages
def main_page():
    st.title("Point Leads")
    st.image("images/icon.PNG", width=None)
    st.write(
        """
        ## Sistema de recomendação de Leads para empresas

        Por: *Kaike Wesley Reis*

        ## **Sobre**
        O sistema de recomendação desenvolvido aqui busca encontrar novos leads para a sua empresa a partir do portfólio disponibilizado!

        ***Point Leads*** alcança seus objetivos utilizando um algoritmo de aprendizagem de máquina chamado *Nearest Neighbours* em sua versão não supervisionada.
        
        Porém você pode estar se perguntando: **Como realmente funciona este sistema?**

        Seu funcionamento se baseia na análise de distâncias em um espaço N-dimensional. A figura abaixo apresenta um exemplo do que realmente ocorre por 
        trás da cortina em ***Point Leads***:
        """
        )
    st.image("images/nearestExample.png", width=None)
    st.write(
        """
        Para cada cliente presente em seu portfólio, nosso algoritmo irá buscar encontrar **N** novos possíveis *leads* em nossa base de dados com mais de 400 mil 
        empresas no mercado, onde quanto menor for a distância, maior serão as chances de encontrarmos um *lead* para complementar seu portfólio!
        
        Essa abordagem apesar de simples, traz inúmeras vantagens:
        - **Customização na quantidade de leads**: Nosso algoritmo é flexível o suficiente para encontrar grandes grupos de leads por cliente, alterando o valor de **N**.
        - Funcionamento **rápido**, **eficaz** e **intuitivo**.
        - **Leads personalizados**: Caso sua empresa trabalhe com um portfólio bem diversificado com empresas de diferentes ramos e setores de atuação nosso algoritmo será sua
        melhor opção, pois ele é capaz de identificar essa demanda recomendando leads personalizados para cada cliente disponível em seu portfólio.

        Ao final de cada recomendação, nosso sistema retorna uma série de *reports* para você conhecer o novo mercado que lhe espera!

        Acesse a página **Como utilizar** para entender como utilizar esta plataforma de recomendação!
        """
        )   
    st.write(
        """
        ## **Outras informações**
        Caso queira conhecer o passo a passo para a construção deste sistema e o **tutorial de uso** acesse o [repositório explicativo no github](https://github.com/KaikeWesleyReis/point-leads-system).
        Este repositório contém além do tutorial todo o código desde o pré-processamento até as etapas finais com a criação do modelo e *reports*.

        Caso queira entrar em contato com criador deste sistema:
        - [LinkedIn](https://www.linkedin.com/in/kaike-wesley-reis/)
        - [Github](https://github.com/KaikeWesleyReis)
        

        """
        ) 

def report_page():
    st.title("Point Leads")
    st.image("images/icon.PNG", width=None)
    st.write(
        """
        ## Sistema de recomendação de Leads para empresas
        Por: *Kaike Wesley Reis*

        **Importante**: Acesse o [tutorial de uso hospedado no github](https://github.com/KaikeWesleyReis/point-leads-system). Além do tutorial, lá você encontra
        o link de download para os portfólios de teste.

        # **Primeira etapa**
        """
        )
    # 1 - Insira seu portfolio
    test_file = st.file_uploader("Insira aqui seu portfólio (suporta apenas formato csv):", type=['csv'])
    if test_file:
        test = pd.read_csv(test_file)
    
    # 2 - Pedir quantos leads por cliente
    st.write("""
             # **Segunda etapa**
             Insira a quantidade **N** de que deseja por cliente em seu portfólio:
             """
             )
    n_leads = st.slider("", 2, 10)

    # 3 - Aguardar o modelo retornar os leads
    st.write("""# **Terceira etapa**""")
    button_leads = st.button('Conseguir Leads!')
    process_complete = False
    if button_leads:
        st.write("""Aguarde nosso sistema retornar os possíveis leads para sua empresa ...""")
        # (*) - Carregar bases de dados e modelos & transformar dataset
        recommender_model = platFunc.load_recommender_model()
        market_info, training = platFunc.load_datasets()
        # Transformar portfolio
        test = platFunc.pipeline_for_test_with_imputation(test, training.columns)
        # Pegar leads
        leads_idx = platFunc.get_leads(test,recommender_model,n_leads)
        # Get predictions info
        leads_market = platFunc.get_leads_info_in_market(market_info, leads_idx.reshape(-1))
        leads_training = platFunc.get_leads_info_in_training(training, leads_idx.reshape(-1))
        process_complete = True
        if process_complete:
            # Mostrar report
            st.write("""
            # **Reports**
            Analise os resultados alcançados pelo nosso sistema:

            ## Veja a similaridade dos *leads* com o portfólio ...
            """
             )
            similaridade = platFunc.plot_portfolio_market_pie_similarity(test, leads_training,threshold=0.80)
            if similaridade > 50.0:
                st.write("""**O sistema conseguiu encontrar uma alta similaridade (acima de 50%) entre os *leads* e o seu portfólio. 
                         Isso indica que o seu mercado é bem homogêneo, ou seja, seu portfólio apresenta empresas de ramos/faturamento/regiões semelhantes!**""")
            else:
                st.write("""**O sistema conseguiu encontrar uma alta similaridade (abaixo de 50%) entre os *leads* e o seu portfólio. 
                         Isso indica que o seu mercado é bem heterogêneo, ou seja, seu portfólio apresenta empresas de ramos/faturamento/regiões distintas!**""")
            st.write("""## A concentração dos principais escritórios dos *leads* estão nos estados ...""")
            platFunc.plot_bars(leads_market, 'sg_uf_matriz', fig_title='')
            st.write("""## O faturamento esperado dos *leads* encontrados ...""")
            platFunc.plot_bars(leads_market, 'de_faixa_faturamento_estimado', fig_title='')
            st.write("""## Os setores de atuação das empresas encontradas ...""")
            platFunc.plot_bars(leads_market, 'nm_divisao', fig_title='')
            st.write("""## Veja abaixo alguns dos *leads* encontrados no link abaixo:""")
            st.dataframe(leads_market)
            

# Parte principal
nome_paginas = ("Sobre a plataforma","Conseguir leads!")
pagina = st.sidebar.selectbox('Selecione sua página', nome_paginas)

if pagina == nome_paginas[0]:
    main_page()
else:
    report_page()