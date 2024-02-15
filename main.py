from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import io
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import base64
import urllib.parse
import math
import seaborn as sns
import scipy.stats as stats
import numpy as np
import csv
from io import StringIO

app = Flask(__name__)

dataframe = None
categorical_columns=None
categorical_columns_options=None
selected_categories=None
category_data=None


# Function to replace 'Z' or 'W' with 0 in a column
def replace_zw_with_zero(value):
    if isinstance(value, str) and ('Z' in value or 'W' in value):
        return 0
    return value


# Function to get unique values for selected categorical columns
def get_category_options(columns):
    options = {}
    for col in columns:
        options[col] = dataframe[col].unique().tolist()
    return options

def read_csv_data(file):
    # Read the CSV file content
    content = file.read().decode('utf-8')

    # Initialize an empty list to store rows of data
    data = []
    csvreader = csv.reader(StringIO(content), delimiter=',')
    for row in csvreader:
        data.append([element.strip('"') for element in row])

    # Create a DataFrame
    df = pd.DataFrame(data)
    
    # Set the first row as column headers
    df.columns = df.iloc[0]

    # Exclude the first row (header row) from the DataFrame
    df = df[1:]

    return df


@app.route('/', methods=['GET', 'POST'])
def index():
    global dataframe
    if request.method == 'POST':
        file = request.files['file']
        if file:
            # Read the CSV file into a Pandas DataFrame
            content = file.read().decode('utf-8')
            dataframe = pd.read_csv(io.StringIO(content))
            #Preprocessing
            dataframe = dataframe.iloc[:-2, :]
            # Renaming the columns
            new_columns = ['Etudiants', 'Promotion', 'Moyennes'] + [f'Exam{i+1}' for i in range(len(dataframe.columns) - 3)]
            dataframe.columns = new_columns
            dataframe = dataframe.drop(0)
            # Apply the function to all columns
            for column in dataframe.columns[2:]:
                dataframe[column] = dataframe[column].apply(replace_zw_with_zero)

            return redirect(url_for('process_form'))

    return render_template('index.html')




@app.route('/process_form', methods=['GET','POST'])
def process_form():
    global dataframe,categorical_columns, categorical_columns_options,selected_categories,category_data,selected_value,last_categorical_index
    
    selected_categories = {}
    # Identify categorical columns based on the absence of numerical values
    categorical_columns = [
            col for col in dataframe.columns
            if pd.to_numeric(dataframe[col], errors='coerce').isnull().all()
        ]
    
    # Find the index of the last categorical column
    last_categorical_index = dataframe.columns.get_loc(categorical_columns[-1])


    categorical_columns_options= get_category_options(categorical_columns)
    for col in categorical_columns:
        selected_value = request.form.get(col)
        if selected_value == 'all':
            # Handle "All" option
            selected_categories[col] = categorical_columns_options[col]
        else:
            selected_categories[col] = [selected_value]

    dataframe.iloc[:, last_categorical_index + 1:] = dataframe.iloc[:, last_categorical_index + 1:].apply(pd.to_numeric, errors='coerce')

    # Filter rows based on selected categories
    category_data = dataframe
    for col, values in selected_categories.items():
        category_data = category_data[category_data[col].isin(values)]
    
    if request.method == 'POST':

        action = request.form.get('action')
        if action == 'Display Histograms':
            # Histograms
            plot_url_1=show_histograms(category_data)
            return render_template('show_histograms.html', plot_url_1=plot_url_1)
            
        elif action == 'Display Boxplots':
            # Boxplots
            plot_box_1=show_boxplots(category_data)
            return render_template('show_boxplots.html',plot_box_1=plot_box_1)
            
        elif action == 'Display Heatmap':
            # Heatmaps
            plot_heatmap_1=show_heatmap(category_data)
            return render_template('show_heatmap.html', plot_heatmap_1=plot_heatmap_1)
                
            
        elif action == 'Show Statistics':
            # Statistics
            stats_1 = show_stats(category_data)
            return render_template('show_stats.html', stats_1=stats_1)
        
        elif action == 'Correlation':
            # Correkation
            return render_template('upload_csv.html')
        
        else:
            
            return "Unknown action"
    else : 
        if dataframe is not None and not dataframe.empty:
            # Identify categorical columns based on the absence of numerical values
            categorical_columns = [
                col for col in dataframe.columns
                if pd.to_numeric(dataframe[col], errors='coerce').isnull().all()
            ]

            categorical_columns_options= get_category_options(categorical_columns)
        return render_template('process_form.html', dataframe=dataframe, categorical_columns=categorical_columns, categorical_columns_options=categorical_columns_options)



@app.route('/subpopulations', methods=['GET','POST'])
def compare_subpopulations():
    global dataframe, categorical_columns, categorical_columns_options
    if request.method == 'POST':
        selected_categories_1 = {}
        selected_categories_2 = {}
    
        for col in categorical_columns:
                selected_value_1 = request.form.get(f'{col}_1')
                selected_value_2 = request.form.get(f'{col}_2')
               
                if selected_value_1 == 'all':
                    selected_categories_1[col] = categorical_columns_options[col]
                else:
                    selected_categories_1[col] = [selected_value_1]

                if selected_value_2 == 'all':
                    selected_categories_2[col] = categorical_columns_options[col]
                else:
                    selected_categories_2[col] = [selected_value_2]         

    # Filter rows based on selected categories for each subpopulation
        subpopulation_1 = dataframe.copy()
        subpopulation_2 = dataframe.copy()

        for col, values in selected_categories_1.items():
            subpopulation_1 = subpopulation_1[subpopulation_1[col].isin(values)]

        for col, values in selected_categories_2.items():
            subpopulation_2 = subpopulation_2[subpopulation_2[col].isin(values)]

        if request.method == 'POST':
            action = request.form.get('action')

            if action == 'Display Histograms':
                # Logique pour afficher les histogrammes
                plot_url_1=show_histograms(subpopulation_1)
                plot_url_2=show_histograms(subpopulation_2)
                return render_template('show_histograms.html', plot_url_1=plot_url_1,plot_url_2=plot_url_2)
            
            elif action == 'Display Boxplots':
                # Logique pour afficher les boxplots
                plot_box_1=show_boxplots(subpopulation_1)
                plot_box_2=show_boxplots(subpopulation_2)
                return render_template('show_boxplots.html',plot_box_1=plot_box_1,plot_box_2=plot_box_2)
            
            elif action == 'Display Heatmap':
                # Logique pour afficher les heatmap
                plot_heatmap_1=show_heatmap(subpopulation_1)
                plot_heatmap_2=show_heatmap(subpopulation_2)
                return render_template('show_heatmap.html', plot_heatmap_1=plot_heatmap_1,plot_heatmap_2=plot_heatmap_2)
                
            
            elif action == 'Show Statistics':
                # Compute statistics for each subpopulation
                stats_1 = show_stats(subpopulation_1)
                stats_2 = show_stats(subpopulation_2)
                return render_template('show_stats.html', stats_1=stats_1,stats_2=stats_2)
            
    else : 
        if dataframe is not None and not dataframe.empty:
            # Identify categorical columns based on the absence of numerical values
            categorical_columns = [
                col for col in dataframe.columns
                if pd.to_numeric(dataframe[col], errors='coerce').isnull().all()
            ]

            categorical_columns_options= get_category_options(categorical_columns)
        return render_template('subpopulations.html', dataframe=dataframe, categorical_columns=categorical_columns, categorical_columns_options=categorical_columns_options)




@app.route('/display_data', methods=['GET'])
def display_data():
    global dataframe,categorical_columns, categorical_columns_options,selected_categories,category_data
    if dataframe is not None and not dataframe.empty:
        # Identify categorical columns based on the absence of numerical values
        categorical_columns = [
            col for col in dataframe.columns
            if pd.to_numeric(dataframe[col], errors='coerce').isnull().all()
        ]

        categorical_columns_options= get_category_options(categorical_columns)
        
        
        
        return render_template('display_data.html', dataframe=dataframe, categorical_columns=categorical_columns, categorical_columns_options=categorical_columns_options)
    else:
        return "No data available or invalid CSV file uploaded."


@app.route('/show_stats', methods=['POST'])
def show_stats(dataframe):
    if request.method == 'POST':
        if dataframe is not None and not dataframe.empty:
            dataframe.iloc[:, last_categorical_index + 1:] = dataframe.iloc[:, last_categorical_index + 1:].apply(pd.to_numeric, errors='coerce')
            stats = dataframe.describe()
            return stats
        else:
            return "No statistics available."


    
@app.route('/show_histograms', methods=['POST'])
def show_histograms(dataframe):
    if request.method == 'POST':
        dataframe.iloc[:, last_categorical_index + 1:] = dataframe.iloc[:, last_categorical_index + 1:].apply(pd.to_numeric, errors='coerce')
        num_exams = len(dataframe.columns[last_categorical_index + 1:])
        num_rows = math.ceil(num_exams / 3)


        plt.figure(figsize=(14, 6))
        plt.suptitle(f'Histograms for Selected Categories', fontsize=16)

        for i in range(num_exams):
            plt.subplot(num_rows, 3, i+1)
            if i == 0:
                dataframe['Moyennes'].plot(kind='hist')
                plt.title('Moyennes')
                plt.xlabel('Moyennes')
            else:
                column_name = f'Exam{i}'
                dataframe[column_name].plot(kind='hist')
                plt.title(column_name)
                plt.xlabel(column_name)
            
        
        plt.tight_layout()

        # Convert the Matplotlib figure to a base64 encoded image
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = urllib.parse.quote(base64.b64encode(img.getvalue()))

        return plot_url


@app.route('/show_boxplots', methods=['POST'])
def show_boxplots(dataframe):
    if request.method == 'POST':
        boxplot_columns = dataframe.columns[last_categorical_index + 1:]  # Columns for boxplots excluding 'Promotion' and 'Moyennes'

        plt.figure(figsize=(12, 6))
        plt.suptitle(f'Boxplots for Selected Categories', fontsize=16)

        for i, column_name in enumerate(boxplot_columns, 1):
            plt.subplot(1, len(boxplot_columns), i)
            sns.boxplot(y=dataframe[column_name])
            plt.title(column_name)
            plt.xlabel(column_name)
        
        plt.tight_layout()

        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_box = urllib.parse.quote(base64.b64encode(img.getvalue()))
        return plot_box

       
    

@app.route('/show_heatmap', methods=['GET'])
def show_heatmap(dataframe):
    dataframe.iloc[:, last_categorical_index + 1:] = dataframe.iloc[:, last_categorical_index + 1:].apply(pd.to_numeric, errors='coerce')
    plt.figure(figsize=(7, 7))
    plt.title('Score Heatmap', color='Black', fontsize=20, pad=40)
    sns.heatmap(dataframe.corr(), annot=True, linewidths=.5)

    # Save the plot to a bytes buffer
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)

    # Encode the image to base64 to embed in HTML
    plot_heatmap = urllib.parse.quote(base64.b64encode(img.getvalue()))
    return plot_heatmap
    



@app.route('/upload_csv', methods=['GET', 'POST'])
def upload_csv():
    global dataframe
    if request.method == 'POST':
        file = request.files['file']
        if file:
            # Read and process the uploaded CSV file
            df_logs = read_csv_data(file)
            
            # Merge DataFrames on 'Etudiant' and 'Nom complet de l'utilisateur'
            merged_df = pd.merge(dataframe, df_logs, left_on='Etudiants', right_on='Nom complet de l\'utilisateur', how='inner')
            # Group by 'Etudiant' and count interactions
            interactions_count = merged_df.groupby('Etudiants').size().reset_index(name='interactions_count')
            # Aggregate average grade for each student
            average_grades = merged_df.groupby('Etudiants')['Moyennes'].mean().reset_index()
            # Merge counts and average grades into a single DataFrame
            merged_data = pd.merge(average_grades, interactions_count, on='Etudiants', how='inner')
            # Calculate correlation between interactions count and average grade
            correlation = merged_data['interactions_count'].corr(merged_data['Moyennes'])

            # Pass the correlation value to the HTML template
            return render_template('logs.html', correlation=correlation, dataframe=dataframe,df_logs=df_logs)


    return render_template('upload_csv.html')



if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
