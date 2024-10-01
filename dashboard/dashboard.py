import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
@st.cache
def load_data():
    rent_df = pd.read_csv('data/hour.csv')
    rent_df.drop('instant', axis=1, inplace=True)
    
    # Rename columns for better readability
    rent_df.rename(columns={
        'dteday':'date', 
        'yr':'year', 
        'mnth':'month', 
        'hr':'hour',
        'weekday':'day',
        'weathersit':'weather', 
        'cnt':'count'
    }, inplace=True)

    # Convert categorical values
    rent_df['season'] = rent_df['season'].replace({
        1:'Spring',
        2:'Summer',
        3:'Fall',
        4:'Winter'
    })
    rent_df['year'] = rent_df['year'].replace({0:'2011', 1:'2012'})
    rent_df['month'] = rent_df['month'].replace({
        1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun',
        7:'Jul', 8:'Aug', 9:'Sept', 10:'Oct', 11:'Nov', 12:'Dec'
    })
    rent_df['holiday'] = rent_df['holiday'].replace({0:'Non-holidays', 1:'Holidays'})
    rent_df['day'] = rent_df['day'].replace({
        0:'Sun', 1:'Mon', 2:'Tue', 3:'Wed', 4:'Thu', 5:'Fri', 6:'Sat'
    })
    rent_df['workingday'] = rent_df['workingday'].replace({0:'Weekends', 1:'Weekdays'})
    rent_df['weather'] = rent_df['weather'].replace({
        1:'Clear/Partly cloudy', 
        2:'Mist/Cloudy', 
        3:'Light Rain/Light Snow', 
        4:'Heavy Rain/Snow/Fog'
    })

    # Denormalize the 'temp', 'atemp', 'hum', 'windspeed' columns
    rent_df['temp'] = rent_df['temp']*41
    rent_df['atemp'] = rent_df['atemp']*50
    rent_df['hum'] = rent_df['hum']*100
    rent_df['windspeed'] = rent_df['windspeed']*67
    
    return rent_df

# Load data
rent_df = load_data()

# Dashboard title
st.title("Bike Sharing Data Analysis Dashboard")

# Sidebar for navigation
st.sidebar.title("Navigation")
options = st.sidebar.radio("Go to", ['Introduction', 'Data Overview', 'Visualizations'])

if options == 'Introduction':
    st.subheader("Introduction")
    st.write("""
    This dashboard is created to analyze the bike sharing dataset. 
    The analysis is divided into several sections to answer key business questions such as:
    - At what time are bike rentals at their highest?
    - How does the weather affect bicycle rentals?
    - What is the pattern of bicycle rentals by season and month?
    - How do holidays affect bicycle rentals?
    - How does the count of bicycle rentals differ between weekdays and weekends?
    - How does the count of casual and registered users compare in 2011 and 2012?
    - How is the daily traffic of bicycle rentals? [Clustering Analysis]
    """)
    
elif options == 'Data Overview':
    st.subheader("Data Overview")
    st.write("Here is a sample of the dataset:")
    st.dataframe(rent_df.sample(10))
    st.write("Statistical summary:")
    st.write(rent_df.describe())

elif options == 'Visualizations':
    st.subheader("Visualizations")
    
    # Make two columns
    col1, col2 = st.columns(2)
    
    # Question 1: At what time are bike rentals at their highest?
    with col1:
        st.markdown("### Peak Rental Times by Hour")
        plt.figure(figsize=(8, 5))  
        lineplot = sns.lineplot(
            x='hour',  
            y='count',  
            data=rent_df,  
            estimator=np.mean,  
            marker='o',  
            linestyle='-',  
            color='teal',  
            ci=None  
        )
        plt.title('Bike Rentals by Hour', fontsize=14, weight='bold')  
        plt.xlabel('Hour of the Day', fontsize=10)  
        plt.ylabel('Average Total Users', fontsize=10)  
        plt.xticks(ticks=range(0, 24), fontsize=10)  
        plt.yticks(fontsize=10)  
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)  
        for x, y in zip(rent_df['hour'].unique(), rent_df.groupby('hour')['count'].mean().values):
            lineplot.text(x, y, round(y, 2), color='black', ha='center', fontsize=8)  
        st.pyplot(plt)
    
    # Question 2: How does the weather affect bicycle rentals?
    with col2:
        st.markdown("### Impact of Weather on Bike Rentals")
        plt.figure(figsize=(8, 5))
        sorted_rent_df = rent_df.groupby('weather')['count'].sum().reset_index().sort_values(by='count', ascending=False)
        max_value = sorted_rent_df['count'].max()
        colors = ['#29AF7FFF' if count != max_value else '#39568CFF' for count in sorted_rent_df['count']]
        barplot = sns.barplot(x='weather', y='count', data=sorted_rent_df, palette=colors)
        plt.title('Bike Rentals by Weather Condition', fontsize=14, weight='bold')
        plt.xlabel('Weather Condition', fontsize=10)
        plt.ylabel('Total Users', fontsize=10)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.yscale('log')
        for p in barplot.patches:
            barplot.annotate(format(p.get_height(), '.0f'), 
                            (p.get_x() + p.get_width() / 2., p.get_height()), 
                            ha='center', va='center', 
                            xytext=(0, 9), 
                            textcoords='offset points',
                            fontsize=10, color='black', weight='bold')
        st.pyplot(plt)
    
    # Make two columns
    col3, col4 = st.columns(2)
    
    # Question 3: What is the pattern of bicycle rentals by season and month?
    # Bike Rentals by Season
    with col3:
        st.markdown("### Seasonal Trends in Bike Rentals")
        plt.figure(figsize=(8, 5))
        sorted_rent_df = rent_df.groupby('season')['count'].sum().reset_index().sort_values(by='count', ascending=False)
        max_value = sorted_rent_df['count'].max()
        colors = ['#55C667FF' if count != max_value else '#FDE725FF' for count in sorted_rent_df['count']]
        barplot = sns.barplot(x='season', y='count', data=sorted_rent_df, palette=colors)
        plt.title('Bike Rentals by Season', fontsize=14, weight='bold')
        plt.xlabel('Season', fontsize=10)
        plt.ylabel('Total Users', fontsize=10)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        for p in barplot.patches:
            barplot.annotate(format(p.get_height(), '.0f'), 
                            (p.get_x() + p.get_width() / 2., p.get_height()), 
                            ha='center', va='center', 
                            xytext=(0, 9), 
                            textcoords='offset points',
                            fontsize=10, color='black', weight='bold')
        st.pyplot(plt)

    # Bike Rentals by Months
    with col4:
        st.markdown("### Monthly Trends in Bike Rentals")
        sorted_rent_df = rent_df.groupby('month')['count'].sum().reset_index().sort_values(by='count', ascending=False)
        sorted_rent_df['month'] = sorted_rent_df['month'].astype(str)
        sorted_rent_df['month'] = pd.Categorical(sorted_rent_df['month'], categories=sorted_rent_df['month'], ordered=True)
        plt.figure(figsize=(8, 5))
        barplot = sns.barplot(x='count', y='month', data=sorted_rent_df, palette='viridis')
        plt.title('Total Bike Rentals by Month', fontsize=14, weight='bold')
        plt.xlabel('Total Rentals', fontsize=10)
        plt.ylabel('Month', fontsize=10)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        for p in barplot.patches:
            barplot.annotate(format(p.get_width(), '.0f'), 
                            (p.get_width(), p.get_y() + p.get_height() / 2.), 
                            ha='left', va='center', 
                            xytext=(5, 0), 
                            textcoords='offset points',
                            fontsize=10, color='black', weight='bold')
        st.pyplot(plt)
    
    # Make two columns
    col5, col6 = st.columns(2)
    
    # Question 4: How do holidays affect bicycle rentals?
    with col5:
        st.markdown("### Holiday Impact on Bike Rentals")
        holiday_counts = rent_df.groupby('holiday')['count'].sum().reset_index()
        plt.figure(figsize=(8, 5))
        colors = sns.color_palette('viridis', len(holiday_counts))
        plt.pie(holiday_counts['count'], labels=holiday_counts['holiday'], autopct='%1.1f%%', startangle=140, colors=colors, textprops={'fontsize': 10, 'weight': 'bold'})
        plt.title('Bike Rentals on Holidays vs Non-Holidays', fontsize=14, weight='bold')
        st.pyplot(plt)
    
    # Question 5: How does the count of bicycle rentals differ between weekdays and weekends?
    with col6:
        st.markdown("### Weekday vs. Weekend Usage of Bike Rentals")
        workingday_counts = rent_df.groupby('workingday')['count'].sum().reset_index()
        plt.figure(figsize=(8, 5))
        colors = sns.color_palette('viridis', len(workingday_counts))
        plt.pie(workingday_counts['count'], labels=workingday_counts['workingday'], autopct='%1.1f%%', startangle=140, colors=colors, textprops={'fontsize': 10, 'weight': 'bold'})
        plt.title('Bike Rentals on Weekends vs Weekdays', fontsize=14, weight='bold')
        st.pyplot(plt)
    
    # Make two columns
    col7, col8 = st.columns(2)
    
    # Question 6: How does the count of casual and registered users compare in 2011 and 2012?
    with col7:
        st.markdown("### User Type Growth (2011 vs 2012)")
        plt.figure(figsize=(8, 5))
        barplot_casual = sns.barplot(x='year', y='casual', data=rent_df, color='#440154FF', label='Casual', estimator=sum, errorbar=None)
        for p in barplot_casual.patches:
            barplot_casual.annotate(format(p.get_height(), '.0f'),
                                    (p.get_x() + p.get_width() / 2., p.get_height()),
                                    ha = 'center', va = 'center', 
                                    xytext = (0, 9), 
                                    textcoords = 'offset points')
        barplot_registered = sns.barplot(x='year', y='registered', data=rent_df, color='#453781FF', label='Registered', estimator=sum, alpha=0.7, errorbar=None)
        for p in barplot_registered.patches:
            barplot_registered.annotate(format(p.get_height(), '.0f'),
                                        (p.get_x() + p.get_width() / 2., p.get_height()),
                                        ha = 'center', va = 'center', 
                                        xytext = (0, 9), 
                                        textcoords = 'offset points')
        plt.title('Comparison of Casual and Registered Users (2011 vs 2012)', fontsize=14, weight='bold')
        plt.xlabel('Year', fontsize=10)
        plt.ylabel('Total Users', fontsize=10)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.legend(title='User Type', fontsize=10, title_fontsize=12)
        st.pyplot(plt)
    
    # 7. Additional Analysis: How is the daily traffic of bicycle rentals? [Clustering Analysis]
    with col8:
        st.markdown("### Daily Traffic Pattern of Bike Rentals")
        rent_df['date'] = pd.to_datetime(rent_df['date'])
        daily_rentals = rent_df.groupby('date')['count'].sum().reset_index()
        Q1 = daily_rentals['count'].quantile(0.25)  
        Q3 = daily_rentals['count'].quantile(0.75)  
        daily_rentals['category'] = 'Medium'
        daily_rentals.loc[daily_rentals['count'] <= Q1, 'category'] = 'Low'
        daily_rentals.loc[daily_rentals['count'] >= Q3, 'category'] = 'High'
        
        plt.figure(figsize=(8, 5))
        sns.scatterplot(x='date', y='count', hue='category', data=daily_rentals, palette='viridis')
        plt.title('Quartile-based Clustering of Bike Rentals by Date')
        plt.xlabel('Date')
        plt.ylabel('Number of Rentals')
        plt.legend(title='Cluster (Quartile)')
        plt.grid(True)

        date_range = pd.date_range(start='2011-01-01', end='2012-12-31', freq='M')
        plt.xticks(date_range, date_range.strftime('%b %Y'), rotation=45)

        st.pyplot(plt)

    # Display a sample of the daily rentals data
    st.subheader("Sample of Daily Rentals Data")
    st.table(daily_rentals[['date', 'count', 'category']].sample(10))