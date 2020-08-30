# *Point Leads*
![iconREADME](https://user-images.githubusercontent.com/32513366/90978551-1e1d2a80-e525-11ea-9e37-ad4b710fa5b8.PNG)
Um sistema de recomendações para portfólios de empresas!

## Conhecendo a plataforma
Para acessar a plataforma acesse o [link da plataforma no Heroku](https://point-leads.herokuapp.com/). Lá você irá encontrar duas páginas:
- **Sobre a plataforma** - Página com explicação mais técnica a respeito do sistema desenvolvido e como usa-lo.
- **Conseguir leads!** - Página para adquirir recomendações para o portfólio da empresa. 

A seção abaixo trás um tutorial de como utilizar o sistema.

## Como utilizar
### Consiga os portfólio para recomendações
Para entender como a plataforma funciona, é recomendado baixar os portfólios de teste. Para isso, siga os passos abaixo:
- Acesse o [link](processed_data/portfolios.rar) e clique em `Download`
- Em seu computador, utilizando qualquer programa de descompactação como `Winrar` descompacte o arquivo baixo
Você terá em mãos nossos portfólios de teste
### Gere resultados
Abaixo, apresentamos como é intuitiva a plataforma *Point Leads* para gerar os resultados:
![tutorial](https://user-images.githubusercontent.com/32513366/90979592-83c0e500-e52c-11ea-8a32-2868f3308ad7.gif)

:exclamation: **A plataforma demora cerca de 2~5 minutos para gerar resultados. Esse tempo dependente do tamanho do seu portfólio e da quantidade de *leads* desejados. Ao final será apresentado o report avaliado pelo sistema.**

## Tecnologias utilizadas
![tecnologiasUsada](https://user-images.githubusercontent.com/32513366/90978659-d21eb580-e525-11ea-924d-328dcf6dcbc9.png)

- [streamlit](https://www.streamlit.io/)
- [heroku](https://www.heroku.com/)

## Considerações
Não foi inserido os links originais das planilhas de mercado e ortfólios nesse repositório, mais precisamente na pasta `raw_data`, dado o tamanho desses arquivos. Abaixo segue os links de download:
- [estaticos_market.csv](https://codenation-challenges.s3-us-west-1.amazonaws.com/ml-leads/estaticos_market.csv.zip)
- [estaticos_portfolio1.csv](https://codenation-challenges.s3-us-west-1.amazonaws.com/ml-leads/estaticos_portfolio1.csv)
- [estaticos_portfolio2.csv](https://codenation-challenges.s3-us-west-1.amazonaws.com/ml-leads/estaticos_portfolio2.csv)
- [estaticos_portfolio3.csv](https://codenation-challenges.s3-us-west-1.amazonaws.com/ml-leads/estaticos_portfolio3.csv)
- [descrição das variáveis da base de dados](https://s3-us-west-1.amazonaws.com/codenation-challenges/ml-leads/features_dictionary.pdf)
