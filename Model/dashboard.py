# Libraries
import numpy             as np
import pandas            as pd
import folium
import seaborn           as sns
import geopandas
import streamlit         as st
import plotly.express    as px
import matplotlib.pyplot as plt

from datetime         import datetime
from folium.plugins   import MarkerCluster
from streamlit_folium import folium_static


# Setting Page Configuration
st.set_page_config( layout='wide' )
pd.set_option('display.float_format', lambda x: '%.2f' % x)


@st.cache( allow_output_mutation=True )
def get_data( path ):
    #load data from dataset
    file_path = '../Datasets/' + str( path )
    data = pd.read_csv( file_path )

    return data

@st.cache( allow_output_mutation=True )
def get_geofile( url ):
    geofile = geopandas.read_file( url )

    return geofile

def set_feature( data ):
    # add new features
    data['id'] = data['id'].astype( 'str' )
    data['date'] = pd.to_datetime( data['date'] ).dt.strftime( '%Y-%m-%d' )
    data['year'] = pd.to_datetime( data['date'] ).dt.year
    data['year_month'] = pd.to_datetime( data['date'] ).dt.strftime( '%Y-%m' )
    data['m2_lot'] = data['sqft_lot'] * 0.092903
    data['price_m2'] = data['price'] / data['m2_lot']
    data['is_waterfront'] = data['waterfront'].apply( lambda x: 'yes' if x == 1 else 'no' )
    data['basement'] = data['sqft_basement'].apply( lambda x: 'yes' if x != 0 else 'no' )

    return data

def overview_data( data ):
    st.title( 'Data Overview' )

    # widgets
    st.sidebar.header( ' Select Data to Overview ' )
    f_attributes = st.sidebar.multiselect( 'Enter columns', data.columns )
    f_zipcode = st.sidebar.multiselect( 'Enter zipcode', data['zipcode'].unique() )

    # select data by widgets
    if ( f_zipcode != [] ) & ( f_attributes != [] ):
        data_filter = data.loc[data['zipcode'].isin(f_zipcode), f_attributes]

    elif ( f_zipcode != [] ) & ( f_attributes == [] ):
        data_filter = data.loc[data['zipcode'].isin( f_zipcode ), :]

    elif ( f_zipcode == [] ) & ( f_attributes != [] ):
        data_filter = data.loc[:, f_attributes]

    else:
        data_filter = data.copy()

    # display data
    st.dataframe( data_filter )

    # break in two columns
    c1, c2 = st.columns( 2 )

    # average metrics
    df1 = data[['id', 'zipcode']].groupby( 'zipcode' ).count().reset_index()
    df2 = data[['price', 'zipcode']].groupby( 'zipcode' ).mean().reset_index()
    df3 = data[['sqft_living', 'zipcode']].groupby( 'zipcode' ).mean().reset_index()
    df4 = data[['price_m2', 'zipcode']].groupby( 'zipcode' ).mean().reset_index()

    # merge average metrics in to dataframe
    m1 = pd.merge( df1, df2, on='zipcode', how='inner' )
    m2 = pd.merge( m1, df3, on='zipcode', how='inner' )
    df = pd.merge( m2, df3, on='zipcode', how='inner' )

    # set columns name
    df.columns = ['ZIPCODE', 'TOTAL_HOUSES', 'PRICE', 'SQFT_LIVING', 'PRICE/M2']

    # display average metrics
    c1.header( 'Average Values' )
    c1.dataframe( df, height=600 )

    # select only numerical attributes
    num_attributes = data.select_dtypes( include=['int64', 'float64'] )

    # statistic descriptive - central tendency
    mean = pd.DataFrame( num_attributes.apply( np.mean ) )
    median = pd.DataFrame( num_attributes.apply( np.median ) )

    # statistic descriptive - dispersion
    std = pd.DataFrame( num_attributes.apply( np.std ) )
    min_ = pd.DataFrame( num_attributes.apply( np.min ) )
    max_ = pd.DataFrame( num_attributes.apply( np.max ) )

    # concatenate statistic descriptive metrics in to dataframe
    df1 = pd.concat( [max_, min_, mean, median, std], axis=1 ).reset_index()

    # set columns names
    df1.columns = ['attributes', 'max', 'min', 'mean', 'median', 'std']

    # display statistics descriptive dataframe
    c2.header( 'Descriptive Analysis' )
    c2.dataframe( df1, height=600 )

    return None

def portfolio_density( data, geofile ):
    st.title( 'Region Overview' )

    # break in two columns
    c1, c2 = st.columns( 2 )

    # portfolio density
    c1.header( 'Portfolio Density' )

    # select data sample to increase process time
    df = data.sample(10)

    # base map - folium
    density_map = folium.Map( location=[data['lat'].mean(),
                                        data['long'].mean()],
                              default_zoom_start=15 )

    # add pop up data information
    marker_cluster = MarkerCluster().add_to( density_map )
    popup_msg = 'Sold R${0} on {1}. Features: {2} sqft, {3} bedrooms, {4} bathrooms, year built: {5}'
    for name, row in df.iterrows():
        folium.Marker( [row['lat'], row['long']],
                       popup=popup_msg.format(
                           row['price'],
                           row['date'],
                           row['sqft_living'],
                           row['bedrooms'],
                           row['bathrooms'],
                           row['yr_built'])).add_to( marker_cluster )
    with c1:
        folium_static( density_map )

    # region price map
    c2.header( 'Price Density' )

    df = data[['price', 'zipcode']].groupby( 'zipcode' ).mean().reset_index()
    df.columns = ['ZIP', 'PRICE']

    geofile = geofile[geofile['ZIP'].isin( df['ZIP'].tolist() )]

    # base map - folium
    region_price_map = folium.Map( location=[data['lat'].mean(),
                                             data['long'].mean()],
                                   default_zoom_start=15 )

    # set choropleth by price
    region_price_map.choropleth( data=df,
                                 geo_data=geofile,
                                 columns=['ZIP', 'PRICE'],
                                 key_on='feature.properties.ZIP',
                                 fill_color='YlOrRd',
                                 fill_opacity=0.7,
                                 line_oppacity=0.2,
                                 legend_name='AVG PRICE' )
    with c2:
        folium_static( region_price_map )

    return None

def commercial_distribution( data ):
    st.title( 'Commercial Attributes' )

    # filters
    st.sidebar.title( 'Commercial Options' )

    min_year_built = int( data['yr_built'].min() )
    max_year_built = int( data['yr_built'].max() )

    st.sidebar.subheader( 'Select Max Year Built' )
    f_year_built = st.sidebar.slider( 'Year Built',
                                      min_year_built,
                                      max_year_built,
                                      min_year_built )

    # average price per year
    st.header( 'Average Price per Year Built' )

    # data selection
    df = data.loc[data['yr_built'] < f_year_built]
    df = df[['yr_built', 'price']].groupby( 'yr_built' ).mean().reset_index()

    # plot
    fig = px.line( df, x='yr_built', y='price' )
    st.plotly_chart( fig, use_container_width=True )

    # average price per day
    st.header( 'Average Price per Day' )
    st.sidebar.subheader( 'Select Max Date' )

    # filters
    min_date = datetime.strptime( data['date'].min(), '%Y-%m-%d' )
    max_date = datetime.strptime( data['date'].max(), '%Y-%m-%d' )

    f_date = st.sidebar.slider( 'Date', min_date, max_date, min_date )

    # data selection
    data['date'] = pd.to_datetime(data['date'])
    df = data.loc[data['date'] < f_date]
    df = df[['date', 'price']].groupby( 'date' ).mean().reset_index()

    # plot
    fig = px.line( df, x='date', y='price' )
    st.plotly_chart( fig, use_container_width=True )

    # histograms
    min_price = int( data['price'].min() )
    max_price = int( data['price'].max() )
    avg_price = int( data['price'].mean() )

    # data selection
    f_price = st.sidebar.slider( 'Price', min_price, max_price, avg_price )
    df = data.loc[data['price'] < f_price]

    # plot
    st.header( 'Houses per Price' )
    fig = px.histogram( df, x='price', nbins=50 )
    st.plotly_chart( fig, use_container_width=True )

    return None

def attributes_distribution( data ):
    st.title( 'House Attributes' )

    # filters
    st.sidebar.title( 'Attributes Options' )

    f_bedrooms = st.sidebar.selectbox( 'Max number of bedrooms', sorted( set( data['bedrooms'].unique() ) ) )
    f_bathrooms = st.sidebar.selectbox( 'Max number of bathrooms', sorted( set( data['bathrooms'].unique() ) ) )

    # data selection
    df_bedrooms = data[data['bedrooms'] < f_bedrooms]
    df_bathrooms = data[data['bathrooms'] < f_bathrooms]

    # break columns in two
    c1, c2 = st.columns( 2 )

    # house per bedrooms
    c1.header( 'Houses per bedrooms' )
    fig = px.histogram( df_bedrooms, x='bedrooms', nbins=19 )
    c1.plotly_chart( fig, use_container_width=True )

    # house per bathrooms
    c2.header( 'Houses per bathrooms' )
    fig = px.histogram( df_bathrooms, x='bathrooms', nbins=19 )
    c2.plotly_chart( fig, use_container_width=True )

    # filters from floors and waterview
    f_floors = st.sidebar.selectbox( 'Max number of floors', sorted( set( data['floors'].unique() ) ) )
    f_waterview = st.sidebar.checkbox( 'Only Houses with Water View' )

    # data selection
    df_floors = data[data['floors'] < f_floors]

    if f_waterview:
        df_waterview = data[data['waterfront'] == 1]
    else:
        df_waterview = data.copy()

    # break columns in two
    c1, c2 = st.columns( 2 )

    # plot
    c1.header( 'Houses per Floors' )
    fig = px.histogram( df_floors, x='floors', nbins=19 )
    c1.plotly_chart( fig, use_container_width=True )

    c2.header( 'Houses per Water View' )
    fig = px.histogram( df_waterview, x='waterfront', nbins=19 )
    c2.plotly_chart( fig, use_container_width=True )

    return None


def hypothesis():
    st.title( 'Main Hypothesis Tests' )

    st.header( 'Hypotheses and results about the business' )

    # First hypothesis
    with st.expander(
            'Hypothesis 1: Houses with water view are 30% more expensive, in average.'
    ):
        # result
        st.write("""
            **FALSE:** House with a water view are 300% more expensive
        """)

        # build grouped data
        aux1 = data[['price', 'is_waterfront']].groupby( 'is_waterfront' ).mean().reset_index()

        aux2 = data[['year_month', 'is_waterfront', 'price']].groupby( ['year_month', 'is_waterfront'] ).mean().reset_index()

        # presenting dataframe about the hypothesis
        st.table( aux1 )

        # plots

        fig = plt.figure(figsize=(10, 8))

        plt.subplot( 2, 1, 1 )
        sns.barplot( x='is_waterfront', y='price', data=aux1 )

        plt.subplot( 2, 1, 2 )
        sns.barplot( x='year_month', y='price', hue='is_waterfront', data=aux2 )
        plt.xticks( rotation=300 )

        st.pyplot( fig )

    # Second Hypothesis
    with st.expander(
        'Hypothesis 2: Houses with a construction date before than 1955 are 50% cheaper, in average'
    ):
        # result
        st.write("""
            **FALSE:** The construction date has low influence on the priec of houses
        """)

        # build grouped data
        aux1 = data[['yr_built', 'price']].groupby( 'yr_built' ).mean().reset_index()

        aux1['before_after'] = aux1['yr_built'].apply( lambda x: 'before_1955' if x <= 1955 else 'after_1955' )
        aux2 = aux1[['before_after', 'price']].groupby( 'before_after' ).mean().reset_index()

        bins = list( np.arange( 1900, 2030, 10) )
        aux1['yr_built_binned'] = pd.cut( aux1['yr_built'], bins=bins, right=False )
        aux3 = aux1[['yr_built_binned', 'price']].groupby( 'yr_built_binned' ).mean().reset_index()

        # break columns in two
        c1, c2 = st.columns( 2 )

        # presenting dataframe about the hypothesis
        c1.table( aux2 )
        c2.table( aux3 )

        # plots
        fig1 = plt.figure(figsize=(10, 4))

        plt.subplot( 1, 2, 1)
        sns.barplot( x='before_after', y='price', data=aux2 )

        plt.subplot( 1, 2, 2)
        sns.heatmap(aux1.corr(method='pearson'), annot=True)
        st.pyplot(fig1)

        fig2 = plt.figure(figsize=(10, 4))
        plt.subplot( 1, 1, 1)
        sns.barplot( x='yr_built_binned', y='price', data=aux3 )
        plt.xticks(rotation=300)
        st.pyplot( fig2 )

    # Third hypothesis
    with st.expander(
        'Hypothesis 3: Houses without a basement have a lot size 50% larger than thoses with a basement'
    ):
        # result
        st.write("""
            **FALSE:** Houses without a basement have a lot size around 20% smaller than with a basement
        """)

        # build grouped data
        aux1 = data[['basement', 'price']].groupby( 'basement' ).mean().reset_index()
        # presenting dataframe about the hypothesis
        st.table( aux1 )

        aux2 = data[['sqft_basement', 'price']].groupby( 'sqft_basement' ).mean().reset_index()

        aux2['sqft_basement_binned'] = pd.cut( aux2['sqft_basement'], bins=bins, right=False )
        aux3 = aux2[['sqft_basement_binned', 'price']].groupby( 'sqft_basement_binned' ).mean().reset_index()

        # plots
        fig = plt.figure(figsize=(10, 4))

        plt.subplot( 2, 2, 1 )
        sns.barplot( x='basement', y='price', data=aux1 )

        plt.subplot( 2, 2, 2 )
        sns.heatmap( aux2.corr( method='pearson' ), data=aux2 )

        plt.subplot( 2, 2, 3 )
        sns.regplot( x='sqft_basement', y='price', data=aux2 )

        plt.subplot( 2, 2, 4 )
        sns.barplot( x='sqft_basement_binned', y='price', data=aux3 )
        plt.xticks(rotation=300)

        st.pyplot( fig )

    with st.expander(
        'Hypothesis 4: Houses without a basement have a lot size 50% larger than with a basement'
    ):
        # result
        st.write("""
            **FALSE:** Houses without a basement have a lot size around 20% smaller than with a basement
        """)

        # build grouped data
        aux1 = data[['year', 'price']].groupby( 'year' ).mean().reset_index()
        # presenting dataframe about the hypothesis
        st.table(aux1)

        aux2 = data[['year_month', 'price']].groupby( 'year_month' ).mean().reset_index()

        # plots
        fig = plt.figure(figsize=(10,4))

        plt.subplot( 1, 2, 1)
        sns.barplot( x='year', y='price', data=aux1 )

        plt.subplot( 1, 2, 2)
        sns.barplot( x='year_month', y='price', data=aux2 )

        st.pyplot( fig )

    with st.expander(
        'Hypothesis 5: Houses with a three bathrooms have a MoM (Month over Month) growth of 15%'
    ):
        # result
        st.write("""
            **FALSE:** Number of bathrooms don't have relantionship with a time.
        """)

        # built grouped data
        bins = list( np.arange( 0, 9, 1 ) )
        data['bathrooms_binned'] = pd.cut( data['bathrooms'], bins=bins, right=False )

        aux = data[data['bathrooms'] == 3]
        aux1 = aux[['bathrooms', 'year_month']].groupby( 'year_month' ).count().reset_index()
        # presenting dataframe about the hypothesis
        st.table( aux1 )

        aux2 = data[['year_month', 'bathrooms_binned', 'bathrooms']].groupby( ['year_month', 'bathrooms_binned'] ).count().reset_index()
        aux3 = aux2.pivot( index='year_month', columns='bathrooms_binned', values='bathrooms')
        aux3.columns = ['Year', '1', '2', '3', '4', '5', '6', '7']

        # plots
        fig = plt.figure(figsize=(10,4))

        plt.subplot( 2, 1, 1 )
        sns.barplot( x='year_month', y='bathrooms', data=aux1 )
        plt.xticks( rotation=300 )

        plt.subplot( 2, 1, 2 )
        sns.lineplot( x='year_month', y='1', data=aux3 )
        sns.lineplot( x='year_month', y='2', data=aux3 )
        sns.lineplot( x='year_month', y='3', data=aux3 )
        sns.lineplot( x='year_month', y='4', data=aux3 )
        sns.lineplot( x='year_month', y='5', data=aux3 )
        sns.lineplot( x='year_month', y='6', data=aux3 )
        sns.lineplot( x='year_month', y='7', data=aux3 )
        plt.xticks( rotation=300 )

        st.pyplot(fig)

    return None

def insights():
    st.title( 'Insights from King County Analysis' )

    # built dataframe with results of hypothesis
    df = pd.DataFrame( {'Hypothesis': ['H1', 'H2', 'H3', 'H4', 'H5'],
                        'Results': ['False', 'False', 'False', 'False', 'False'],
                        'Relevance': ['High', 'Low', 'Medium', 'Low', 'Low']} )

    st.write("""
        ### Results about the hypothesis tests
    """)

    st.table( df )

    return None

def purchase_recommendations():
    st.title( 'Table with Purchase Recommendations' )

    # select data
    cols = ['id', 'date', 'price', 'waterfront', 'condition', 'zipcode']
    df = data[cols]

    # create season feature
    season = {1:'winter', 2:'winter', 3:'winter', 4:'summer', 5:'summer', 6:'summer',
              7:'summer', 8:'summer', 9:'summer', 10:'winter', 11:'winter', 12:'winter'}
    df['date'] = pd.to_datetime(df['date'])
    df['season'] = df['date'].dt.month.map( season )

    aux2 = df[['season', 'price', 'zipcode']].groupby( ['zipcode', 'season'] ).mean().reset_index()
    aux2 = aux2.rename( columns={'price':'season_price'} )
    df1 = df.merge( aux2, on=['zipcode', 'season'] )

    df1['price_suggest'] = df1[['price', 'season_price']].apply(
        lambda x: 1.1*x['price'] if  x['price'] >= x['season_price'] else 1.3*x['price'], axis=1 )

    aux = df[['price', 'zipcode']].groupby( 'zipcode' ).mean().reset_index()
    aux.rename( columns={'price':'mean_price'}, inplace=True )
    df = df.merge( aux, on='zipcode' )

    df['status'] = df[['mean_price', 'price', 'condition', 'waterfront']].apply(
        lambda x: 'accept' if ( x['price'] < x['mean_price'] ) &
                              ( x['condition'] > 3 ) &
                              ( x['waterfront'] == 1 ) else 'reject', axis=1 )

    df = df[df['status'] == 'accept']

    st.table( df )

    return df

def sell_recommendations(df):
    st.title('TAble with Sell Price and Date Recommendations')

    # select data
    df2 = df.copy()

    aux1 = df2[['price', 'zipcode', 'season']].groupby( ['zipcode', 'season'] ).mean().reset_index()
    aux1 = aux1.rename( columns={'price':'season_price'} )

    df2 = df2.merge( aux1, on=['zipcode', 'season'], how='left' )

    df2['price_suggest'] = df2[['price', 'season_price']].apply(
        lambda x: 1.3*x['price'] if x['price'] < x['season_price'] else 1.1*x['price'], axis=1 )

    st.table( df2 )

    return None

def refurbishment_suggests():
    st.title('Table with Renovations Suggestions and the Increase Price')

    # select data
    df3 = data.loc[data['yr_renovated'] == 0,
                   ['id', 'price', 'bedrooms', 'bathrooms', 'sqft_basement', 'yr_renovated']]

    # basement
    df3['basement'] = data['sqft_basement'].apply(lambda x: 'no' if x == 0 else 'yes')
    df3['increase_basement'] = df3[['basement', 'price']].apply(lambda x: 0.50 * x['price'] if x['basement'] else 0, axis=1)
    df3['suggest_basement'] = df3['basement'].apply(lambda x: 'yes' if x == 'no' else 'no')

    # bathrooms
    df3['increase_bathrooms'] = df3[['bathrooms', 'price']].apply( lambda x: 0.40*x['price'] if x['bathrooms'] < 4 else 0, axis=1 )
    df3['suggest_bathrooms'] = df3['bathrooms'].apply( lambda x: x+1 if x < 4 else x )

    # bedrooms
    df3['increase_bedrooms'] = df3[['bedrooms', 'price']].apply( lambda x: 0.25*x['price'] if x['bedrooms'] < 5 else 0, axis=1 )
    df3['suggest_bedrooms'] = df3['bedrooms'].apply( lambda x: x+1 if x < 5 else x )

    # final price
    df3['final_increase'] = df3['increase_basement'] + df3['increase_bathrooms'] + df3['increase_bedrooms']
    df3['final_price'] = df3['price'] + df3['final_increase']

    df4 = df3[['id', 'final_price', 'suggest_basement', 'bathrooms', 'suggest_bathrooms', 'bedrooms', 'suggest_bedrooms']]

    st.table( df4 )



    return None

def introduction():

    return None



if __name__ == '__main__':

    # extraction
    path = 'kc_house_data.csv'
    url = 'https://opendata.arcgis.com/datasets/83fc2e72903343aabff6de8cb445b81c_2.geojson'

    data = get_data( path )
    geofile = get_geofile( url )

    # transformation
    data = set_feature( data )

    # load
    df = purchase_recommendations()

    sell_recommendations(df)

    refurbishment_suggests()

    insights()

    hypothesis()

    overview_data( data )

    portfolio_density(data, geofile)

    commercial_distribution( data )

    attributes_distribution( data )





