import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import calendar
import os

# Set page configuration
st.set_page_config(
    page_title="Fentanyl Testing Data Analysis",
    page_icon="💉",
    layout="wide",
)

# Add title and description
st.title("Fentanyl Positive Test Analysis")
st.markdown("""
This dashboard analyzes urine test results where patients tested positive for fentanyl.
Data provided by Millennium Health from Ohio healthcare providers.
""")

# Function to load and preprocess data
@st.cache_data
def load_data():
    # First try to load from the uploaded file
    uploaded_file = st.file_uploader("Upload the fentanyl test data (TSV format)", type=['txt', 'tsv'])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, sep='\t')
    # If no upload, check if the file exists locally
    elif os.path.exists('paste.txt'):
        df = pd.read_csv('paste.txt', sep='\t')
    else:
        st.warning("Please upload the data file to continue.")
        return None
    
    # Convert DATE_TESTED to datetime
    df['DATE_TESTED'] = pd.to_datetime(df['DATE_TESTED'])
    
    # Create a proper year-month column for time series
    df['YEAR_MONTH'] = df['DATE_TESTED'].dt.strftime('%Y-%m')
    
    # Fill NA values with 'Not Tested' for drug test columns
    drug_cols = ['FENTANYL', 'HEROIN', 'COCAINE', 'METHAMPHETAMINE', 'MARIJUANA', 
                'ALCOHOL', 'XYLAZINE', 'CARFENTANIL', 'BUPRENORPHINE', 
                'METHADONE', 'NALTREXONE', 'GABAPENTIN']
    
    for col in drug_cols:
        if col in df.columns:
            df[col] = df[col].fillna('NOT TESTED')
    
    # Create age groups
    bins = [0, 18, 25, 35, 45, 55, 65, 100]
    labels = ['<18', '18-25', '26-35', '36-45', '46-55', '56-65', '65+']
    df['AGE_GROUP'] = pd.cut(df['PATIENT_AGE'], bins=bins, labels=labels, right=False)
    
    # Create a binary matrix for co-occurring drugs with fentanyl
    for drug in drug_cols:
        if drug != 'FENTANYL' and drug in df.columns:
            df[f'{drug}_POSITIVE'] = (df[drug] == 'POSITIVE').astype(int)
    
    # Create a column for the count of co-occurring drugs
    co_occurring_cols = [f'{drug}_POSITIVE' for drug in drug_cols if drug != 'FENTANYL' and f'{drug}_POSITIVE' in df.columns]
    df['NUM_CO_DRUGS'] = df[co_occurring_cols].sum(axis=1)
    
    return df

# Load the data
df = load_data()

if df is not None:
    # Create tabs for different analyses
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Time Trends", 
        "Geographic Analysis", 
        "Demographics", 
        "Provider & Payer", 
        "Polysubstance Use"
    ])
    
    # Tab 1: Time Series Analysis
    with tab1:
        st.header("Fentanyl Positive Tests Over Time")
        
        # Time aggregation options
        time_agg = st.radio(
            "Time Aggregation",
            ["Monthly", "Quarterly", "Yearly"],
            horizontal=True
        )
        
        if time_agg == "Monthly":
            # Count tests by year-month
            time_series = df.groupby('YEAR_MONTH').size().reset_index(name='count')
            time_series.columns = ['Date', 'Test Count']
            
            # Convert back to datetime for better x-axis
            time_series['Date'] = pd.to_datetime(time_series['Date'])
            
            # Create time series plot
            fig = px.line(
                time_series, 
                x='Date', 
                y='Test Count',
                title='Monthly Fentanyl Positive Tests'
            )
            
        elif time_agg == "Quarterly":
            # Extract quarter
            df['Quarter'] = df['DATE_TESTED'].dt.to_period('Q').astype(str)
            quarterly = df.groupby('Quarter').size().reset_index(name='Test Count')
            
            fig = px.line(
                quarterly, 
                x='Quarter', 
                y='Test Count',
                title='Quarterly Fentanyl Positive Tests'
            )
            
        else:  # Yearly
            yearly = df.groupby('TEST_YEAR').size().reset_index(name='Test Count')
            
            fig = px.bar(
                yearly, 
                x='TEST_YEAR', 
                y='Test Count',
                title='Yearly Fentanyl Positive Tests'
            )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Co-occurring drugs over time
        st.subheader("Co-occurring Drugs with Fentanyl Over Time")
        
        # Select drugs for analysis
        drug_options = [col for col in df.columns if col.endswith('_POSITIVE')]
        drug_names = [col.replace('_POSITIVE', '') for col in drug_options]
        
        selected_drugs = st.multiselect(
            "Select drugs to analyze alongside fentanyl",
            drug_names,
            default=["HEROIN", "COCAINE", "METHAMPHETAMINE", "MARIJUANA"]
        )
        
        if selected_drugs:
            # Create time series for each selected drug
            if time_agg == "Monthly":
                time_data = []
                for drug in selected_drugs:
                    drug_col = f"{drug}_POSITIVE"
                    if drug_col in df.columns:
                        monthly_counts = df.groupby('YEAR_MONTH')[drug_col].sum().reset_index()
                        monthly_counts['Drug'] = drug
                        monthly_counts.columns = ['Date', 'Count', 'Drug']
                        monthly_counts['Date'] = pd.to_datetime(monthly_counts['Date'])
                        time_data.append(monthly_counts)
                
                if time_data:
                    time_df = pd.concat(time_data)
                    fig = px.line(
                        time_df, 
                        x='Date', 
                        y='Count', 
                        color='Drug',
                        title='Monthly Co-occurring Drug Positives with Fentanyl'
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            elif time_agg == "Quarterly":
                time_data = []
                for drug in selected_drugs:
                    drug_col = f"{drug}_POSITIVE"
                    if drug_col in df.columns:
                        quarterly_counts = df.groupby('Quarter')[drug_col].sum().reset_index()
                        quarterly_counts['Drug'] = drug
                        quarterly_counts.columns = ['Quarter', 'Count', 'Drug']
                        time_data.append(quarterly_counts)
                
                if time_data:
                    time_df = pd.concat(time_data)
                    fig = px.line(
                        time_df, 
                        x='Quarter', 
                        y='Count', 
                        color='Drug',
                        title='Quarterly Co-occurring Drug Positives with Fentanyl'
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            else:  # Yearly
                time_data = []
                for drug in selected_drugs:
                    drug_col = f"{drug}_POSITIVE"
                    if drug_col in df.columns:
                        yearly_counts = df.groupby('TEST_YEAR')[drug_col].sum().reset_index()
                        yearly_counts['Drug'] = drug
                        yearly_counts.columns = ['Year', 'Count', 'Drug']
                        time_data.append(yearly_counts)
                
                if time_data:
                    time_df = pd.concat(time_data)
                    fig = px.bar(
                        time_df, 
                        x='Year', 
                        y='Count', 
                        color='Drug',
                        barmode='group',
                        title='Yearly Co-occurring Drug Positives with Fentanyl'
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # Percentage of fentanyl tests with co-occurring drugs over time
            st.subheader("Percentage of Fentanyl Tests with Co-occurring Drugs")
            
            if time_agg == "Monthly":
                pct_data = []
                for drug in selected_drugs:
                    drug_col = f"{drug}_POSITIVE"
                    if drug_col in df.columns:
                        # Group by month and calculate total tests and positive tests
                        monthly_group = df.groupby('YEAR_MONTH')
                        monthly_total = monthly_group.size().reset_index(name='Total')
                        monthly_positive = monthly_group[drug_col].sum().reset_index(name='Positive')
                        
                        # Merge and calculate percentage
                        monthly_pct = pd.merge(monthly_total, monthly_positive, on='YEAR_MONTH')
                        monthly_pct['Percentage'] = (monthly_pct['Positive'] / monthly_pct['Total']) * 100
                        monthly_pct['Drug'] = drug
                        monthly_pct['Date'] = pd.to_datetime(monthly_pct['YEAR_MONTH'])
                        pct_data.append(monthly_pct[['Date', 'Percentage', 'Drug']])
                
                if pct_data:
                    pct_df = pd.concat(pct_data)
                    fig = px.line(
                        pct_df, 
                        x='Date', 
                        y='Percentage', 
                        color='Drug',
                        title='Monthly Percentage of Fentanyl Tests with Co-occurring Drugs'
                    )
                    fig.update_layout(yaxis_title='Percentage (%)')
                    st.plotly_chart(fig, use_container_width=True)
            
            elif time_agg == "Quarterly":
                pct_data = []
                for drug in selected_drugs:
                    drug_col = f"{drug}_POSITIVE"
                    if drug_col in df.columns:
                        quarterly_group = df.groupby('Quarter')
                        quarterly_total = quarterly_group.size().reset_index(name='Total')
                        quarterly_positive = quarterly_group[drug_col].sum().reset_index(name='Positive')
                        
                        quarterly_pct = pd.merge(quarterly_total, quarterly_positive, on='Quarter')
                        quarterly_pct['Percentage'] = (quarterly_pct['Positive'] / quarterly_pct['Total']) * 100
                        quarterly_pct['Drug'] = drug
                        pct_data.append(quarterly_pct[['Quarter', 'Percentage', 'Drug']])
                
                if pct_data:
                    pct_df = pd.concat(pct_data)
                    fig = px.line(
                        pct_df, 
                        x='Quarter', 
                        y='Percentage', 
                        color='Drug',
                        title='Quarterly Percentage of Fentanyl Tests with Co-occurring Drugs'
                    )
                    fig.update_layout(yaxis_title='Percentage (%)')
                    st.plotly_chart(fig, use_container_width=True)
            
            else:  # Yearly
                pct_data = []
                for drug in selected_drugs:
                    drug_col = f"{drug}_POSITIVE"
                    if drug_col in df.columns:
                        yearly_group = df.groupby('TEST_YEAR')
                        yearly_total = yearly_group.size().reset_index(name='Total')
                        yearly_positive = yearly_group[drug_col].sum().reset_index(name='Positive')
                        
                        yearly_pct = pd.merge(yearly_total, yearly_positive, on='TEST_YEAR')
                        yearly_pct['Percentage'] = (yearly_pct['Positive'] / yearly_pct['Total']) * 100
                        yearly_pct['Drug'] = drug
                        pct_data.append(yearly_pct[['TEST_YEAR', 'Percentage', 'Drug']])
                
                if pct_data:
                    pct_df = pd.concat(pct_data)
                    fig = px.line(
                        pct_df, 
                        x='TEST_YEAR', 
                        y='Percentage', 
                        color='Drug',
                        title='Yearly Percentage of Fentanyl Tests with Co-occurring Drugs'
                    )
                    fig.update_layout(yaxis_title='Percentage (%)')
                    st.plotly_chart(fig, use_container_width=True)
    
    # Tab 2: Geographic Analysis
    with tab2:
        st.header("Geographic Distribution of Fentanyl Tests")
        
        # County analysis
        st.subheader("Fentanyl Tests by County")
        
        # Get top counties by test count
        county_counts = df['PATIENT_COUNTY'].value_counts().reset_index()
        county_counts.columns = ['County', 'Test Count']
        
        # Filter for top counties
        top_n = st.slider("Show top N counties:", 5, 30, 10)
        top_counties = county_counts.head(top_n)
        
        fig = px.bar(
            top_counties, 
            x='County', 
            y='Test Count',
            title=f'Top {top_n} Counties by Fentanyl Positive Test Count'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # County co-occurring drug analysis
        st.subheader("Co-occurring Drug Rates by County")
        
        # Select counties and drugs for analysis
        selected_counties = st.multiselect(
            "Select counties to analyze",
            sorted(df['PATIENT_COUNTY'].unique()),
            default=top_counties['County'].head(5).tolist()
        )
        
        county_drug = st.selectbox(
            "Select co-occurring drug to analyze",
            ["HEROIN", "COCAINE", "METHAMPHETAMINE", "MARIJUANA", "ALCOHOL", "BUPRENORPHINE"]
        )
        
        if selected_counties and county_drug:
            county_drug_col = f"{county_drug}_POSITIVE"
            if county_drug_col in df.columns:
                # Filter data for selected counties
                county_df = df[df['PATIENT_COUNTY'].isin(selected_counties)]
                
                # Calculate co-occurrence rate by county
                county_data = []
                for county in selected_counties:
                    county_subset = county_df[county_df['PATIENT_COUNTY'] == county]
                    total = len(county_subset)
                    positive = county_subset[county_drug_col].sum()
                    county_data.append({
                        'County': county,
                        'Total Tests': total,
                        'Positive Tests': positive,
                        'Percentage': (positive / total) * 100 if total > 0 else 0
                    })
                
                county_drug_df = pd.DataFrame(county_data)
                county_drug_df = county_drug_df.sort_values('Percentage', ascending=False)
                
                fig = px.bar(
                    county_drug_df,
                    x='County',
                    y='Percentage',
                    title=f'Percentage of Fentanyl Tests also Positive for {county_drug} by County',
                    hover_data=['Total Tests', 'Positive Tests']
                )
                fig.update_layout(yaxis_title='Percentage (%)')
                st.plotly_chart(fig, use_container_width=True)
        
        # County heat map (if interested in adding geographic visualization)
        st.subheader("County Test Count Map")
        st.info("A county map visualization would be available here with appropriate shapefile data")
        
    # Tab 3: Demographics
    with tab3:
        st.header("Demographic Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Age distribution
            st.subheader("Age Distribution")
            
            # Age histogram
            fig = px.histogram(
                df, 
                x='PATIENT_AGE',
                nbins=20,
                title='Age Distribution of Fentanyl Positive Tests'
            )
            fig.update_layout(xaxis_title='Age', yaxis_title='Count')
            st.plotly_chart(fig, use_container_width=True)
            
            # Age group distribution
            age_group_counts = df['AGE_GROUP'].value_counts().reset_index()
            age_group_counts.columns = ['Age Group', 'Count']
            age_group_counts = age_group_counts.sort_values('Age Group')
            
            fig = px.bar(
                age_group_counts,
                x='Age Group',
                y='Count',
                title='Fentanyl Positive Tests by Age Group'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Sex distribution
            st.subheader("Sex Distribution")
            
            sex_counts = df['SEX'].value_counts().reset_index()
            sex_counts.columns = ['Sex', 'Count']
            
            fig = px.pie(
                sex_counts,
                values='Count',
                names='Sex',
                title='Sex Distribution of Fentanyl Positive Tests'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Sex distribution over time
            st.subheader("Sex Distribution Over Time")
            
            # Group by year and sex
            sex_time = df.groupby(['TEST_YEAR', 'SEX']).size().reset_index(name='Count')
            
            fig = px.line(
                sex_time,
                x='TEST_YEAR',
                y='Count',
                color='SEX',
                title='Fentanyl Positive Tests by Sex Over Time'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Drug co-occurrence by demographic
        st.subheader("Drug Co-occurrence by Demographic")
        
        demo_col1, demo_col2 = st.columns(2)
        
        with demo_col1:
            # By age group
            drug_by_age = st.selectbox(
                "Select drug to analyze by age group",
                ["HEROIN", "COCAINE", "METHAMPHETAMINE", "MARIJUANA", "ALCOHOL", "BUPRENORPHINE"]
            )
            
            if drug_by_age:
                drug_age_col = f"{drug_by_age}_POSITIVE"
                if drug_age_col in df.columns:
                    # Calculate rates by age group
                    age_data = []
                    for age_group in sorted(df['AGE_GROUP'].unique()):
                        age_subset = df[df['AGE_GROUP'] == age_group]
                        total = len(age_subset)
                        positive = age_subset[drug_age_col].sum()
                        age_data.append({
                            'Age Group': age_group,
                            'Total': total,
                            'Positive': positive,
                            'Percentage': (positive / total) * 100 if total > 0 else 0
                        })
                    
                    age_drug_df = pd.DataFrame(age_data)
                    
                    fig = px.bar(
                        age_drug_df,
                        x='Age Group',
                        y='Percentage',
                        title=f'Percentage of Fentanyl Tests also Positive for {drug_by_age} by Age Group',
                        hover_data=['Total', 'Positive']
                    )
                    fig.update_layout(yaxis_title='Percentage (%)')
                    st.plotly_chart(fig, use_container_width=True)
        
        with demo_col2:
            # By sex
            drug_by_sex = st.selectbox(
                "Select drug to analyze by sex",
                ["HEROIN", "COCAINE", "METHAMPHETAMINE", "MARIJUANA", "ALCOHOL", "BUPRENORPHINE"]
            )
            
            if drug_by_sex:
                drug_sex_col = f"{drug_by_sex}_POSITIVE"
                if drug_sex_col in df.columns:
                    # Calculate rates by sex
                    sex_data = []
                    for sex in df['SEX'].unique():
                        sex_subset = df[df['SEX'] == sex]
                        total = len(sex_subset)
                        positive = sex_subset[drug_sex_col].sum()
                        sex_data.append({
                            'Sex': sex,
                            'Total': total,
                            'Positive': positive,
                            'Percentage': (positive / total) * 100 if total > 0 else 0
                        })
                    
                    sex_drug_df = pd.DataFrame(sex_data)
                    
                    fig = px.bar(
                        sex_drug_df,
                        x='Sex',
                        y='Percentage',
                        title=f'Percentage of Fentanyl Tests also Positive for {drug_by_sex} by Sex',
                        hover_data=['Total', 'Positive']
                    )
                    fig.update_layout(yaxis_title='Percentage (%)')
                    st.plotly_chart(fig, use_container_width=True)
    
    # Tab 4: Provider & Payer Analysis
    with tab4:
        st.header("Provider & Payer Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Provider specialty analysis
            st.subheader("Provider Specialty Analysis")
            
            specialty_counts = df['CLIENT_SPECIALTY_GROUP'].value_counts().reset_index()
            specialty_counts.columns = ['Specialty', 'Count']
            
            fig = px.pie(
                specialty_counts,
                values='Count',
                names='Specialty',
                title='Fentanyl Positive Tests by Provider Specialty'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Payer group analysis
            st.subheader("Payer Group Analysis")
            
            payer_counts = df['PAYER_GROUP'].value_counts().reset_index()
            payer_counts.columns = ['Payer', 'Count']
            
            fig = px.pie(
                payer_counts,
                values='Count',
                names='Payer',
                title='Fentanyl Positive Tests by Payer Group'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Drug co-occurrence by provider specialty
        st.subheader("Drug Co-occurrence by Provider Specialty")
        
        provider_drug = st.selectbox(
            "Select drug to analyze by provider specialty",
            ["HEROIN", "COCAINE", "METHAMPHETAMINE", "MARIJUANA", "ALCOHOL", "BUPRENORPHINE"]
        )
        
        if provider_drug:
            provider_drug_col = f"{provider_drug}_POSITIVE"
            if provider_drug_col in df.columns:
                # Calculate rates by provider specialty
                specialty_data = []
                for specialty in df['CLIENT_SPECIALTY_GROUP'].unique():
                    specialty_subset = df[df['CLIENT_SPECIALTY_GROUP'] == specialty]
                    total = len(specialty_subset)
                    positive = specialty_subset[provider_drug_col].sum()
                    specialty_data.append({
                        'Specialty': specialty,
                        'Total': total,
                        'Positive': positive,
                        'Percentage': (positive / total) * 100 if total > 0 else 0
                    })
                
                specialty_drug_df = pd.DataFrame(specialty_data)
                specialty_drug_df = specialty_drug_df.sort_values('Percentage', ascending=False)
                
                fig = px.bar(
                    specialty_drug_df,
                    x='Specialty',
                    y='Percentage',
                    title=f'Percentage of Fentanyl Tests also Positive for {provider_drug} by Provider Specialty',
                    hover_data=['Total', 'Positive']
                )
                fig.update_layout(
                    yaxis_title='Percentage (%)',
                    xaxis_title='Provider Specialty',
                    xaxis={'categoryorder': 'total descending'}
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Drug co-occurrence by payer group
        st.subheader("Drug Co-occurrence by Payer Group")
        
        payer_drug = st.selectbox(
            "Select drug to analyze by payer group",
            ["HEROIN", "COCAINE", "METHAMPHETAMINE", "MARIJUANA", "ALCOHOL", "BUPRENORPHINE"]
        )
        
        if payer_drug:
            payer_drug_col = f"{payer_drug}_POSITIVE"
            if payer_drug_col in df.columns:
                # Calculate rates by payer group
                payer_data = []
                for payer in df['PAYER_GROUP'].unique():
                    payer_subset = df[df['PAYER_GROUP'] == payer]
                    total = len(payer_subset)
                    positive = payer_subset[payer_drug_col].sum()
                    payer_data.append({
                        'Payer': payer,
                        'Total': total,
                        'Positive': positive,
                        'Percentage': (positive / total) * 100 if total > 0 else 0
                    })
                
                payer_drug_df = pd.DataFrame(payer_data)
                payer_drug_df = payer_drug_df.sort_values('Percentage', ascending=False)
                
                fig = px.bar(
                    payer_drug_df,
                    x='Payer',
                    y='Percentage',
                    title=f'Percentage of Fentanyl Tests also Positive for {payer_drug} by Payer Group',
                    hover_data=['Total', 'Positive']
                )
                fig.update_layout(
                    yaxis_title='Percentage (%)',
                    xaxis_title='Payer Group',
                    xaxis={'categoryorder': 'total descending'}
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # Tab 5: Polysubstance Use
    with tab5:
        st.header("Polysubstance Use Analysis")
        
        # Count of co-occurring substances
        st.subheader("Number of Co-occurring Substances with Fentanyl")
        
        num_drugs_count = df['NUM_CO_DRUGS'].value_counts().reset_index()
        num_drugs_count.columns = ['Number of Drugs', 'Count']
        num_drugs_count = num_drugs_count.sort_values('Number of Drugs')
        
        fig = px.bar(
            num_drugs_count,
            x='Number of Drugs',
            y='Count',
            title='Number of Co-occurring Substances with Fentanyl'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Co-occurring substance rates
        st.subheader("Co-occurring Substance Rates")
        
        # Calculate percentage positive for each drug
        drug_positives = []
        for drug in [col for col in df.columns if col.endswith('_POSITIVE')]:
            drug_name = drug.replace('_POSITIVE', '')
            if drug_name != 'FENTANYL':
                total = len(df)
                positive = df[drug].sum()
                drug_positives.append({
                    'Drug': drug_name,
                    'Positive Count': positive,
                    'Percentage': (positive / total) * 100
                })
        
        drug_pos_df = pd.DataFrame(drug_positives)
        drug_pos_df = drug_pos_df.sort_values('Percentage', ascending=False)
        
        fig = px.bar(
            drug_pos_df,
            x='Drug',
            y='Percentage',
            title='Percentage of Fentanyl Tests Positive for Other Substances',
            hover_data=['Positive Count']
        )
        fig.update_layout(
            yaxis_title='Percentage (%)',
            xaxis_title='Substance',
            xaxis={'categoryorder': 'total descending'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Common drug combinations
        st.subheader("Common Drug Combinations with Fentanyl")
        
        # Get most common 2-drug combinations with fentanyl
        st.write("Most Common 2-Drug Combinations with Fentanyl")
        
        # Create combinations
        drug_cols = [col for col in df.columns if col.endswith('_POSITIVE')]
        drug_names = [col.replace('_POSITIVE', '') for col in drug_cols if 'FENTANYL' not in col]
        
        # Calculate combination counts
        combo_data = []
        
        for i, drug1 in enumerate(drug_names):
            for drug2 in drug_names[i+1:]:
                drug1_col = f"{drug1}_POSITIVE"
                drug2_col = f"{drug2}_POSITIVE"
                
                if drug1_col in df.columns and drug2_col in df.columns:
                    combo_count = ((df[drug1_col] == 1) & (df[drug2_col] == 1)).sum()
                    total = len(df)
                    combo_data.append({
                        'Combination': f"{drug1} + {drug2}",
                        'Count': combo_count,
                        'Percentage': (combo_count / total) * 100
                    })
        
        combo_df = pd.DataFrame(combo