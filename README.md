# Protein-Data-sorting-Protein-Anomaly-Detector
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

print("Attempting to load the Excel file...")


file_name = 'Data_Cortex_Nuclear.xls' 

try:
  
    df = pd.read_excel(file_name)
    
    print("\nSuccess! The data has been imported.")
    
    print("\nHere are the first 5 rows of your dataset:")
    print(df.head())
    
    print("\n--- Separating Protein Expressions ---")
    
    protein_data = df.iloc[:, 1:78]
    
    metadata = df[['MouseID', 'Genotype', 'Treatment', 'Behavior', 'class']]
    
    print(f"Successfully isolated {protein_data.shape[1]} protein columns!")
    print("\nHere are the first 5 rows of JUST the protein data:")
    print(protein_data.head())
    
    print("\n--- 1. Cleaning the Data (Filling Blanks) ---")
    
    imputer = SimpleImputer(strategy='mean')
    clean_protein_data = imputer.fit_transform(protein_data)
    
    print("--- 2. Scaling the Data ---")
    scaler = StandardScaler()
    scaled_protein_data = scaler.fit_transform(clean_protein_data)
    
    print("--- 3. Training the AI (Isolation Forest) ---")
    ai_model = IsolationForest(contamination=0.05, random_state=42)
    
    ai_model.fit(scaled_protein_data)
    
    predictions = ai_model.predict(scaled_protein_data)
    
    df['Is_Anomaly'] = predictions
    
    wrong_codes = df[df['Is_Anomaly'] == -1]
    
    print("\n--- RESULTS ---")
    print(f"The AI has finished reading {df.shape[0]} rows of data.")
    print(f"It found {len(wrong_codes)} 'wrong codes' (anomalies).")
    
    print("\nHere are the first 5 rows of the 'wrong' (anomalous) protein expressions:")
    print(wrong_codes[['MouseID', 'Is_Anomaly']].head()) 
    
    output_filename = 'wrong_protein_expressions.xlsx'
    wrong_codes.to_excel(output_filename, index=False)
    print(f"\nSuccess! All wrong codes have been saved to a new file: {output_filename}")
    
    rows, columns = df.shape
    print(f"\nYour dataset has {rows} rows and {columns} columns.")

except FileNotFoundError:
    print(f"\nERROR: Python could not find a file named '{file_name}'.")
    print("Please check that the Excel file is inside the exact same folder as this Python file!")
    
