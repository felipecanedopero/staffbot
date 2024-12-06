from langchain.agents import AgentType
from langchain_openai import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.callbacks import StreamlitCallbackHandler
import pandas as pd
import unicodedata
#from datetime import datetime, timedelta
from botocore.exceptions import ClientError
import os

# Obtiene la ruta absoluta del directorio donde está preprocessing.py
base_dir = os.path.dirname(os.path.abspath(__file__))

# Construye las rutas completas de cada archivo CSV
contacts_path = os.path.join(base_dir, "all_contacts.csv")
accounts_path = os.path.join(base_dir, "all_accounts.csv")
campaigns_path = os.path.join(base_dir, "all_campaigns.csv")
registrations_path = os.path.join(base_dir, "all_registrations.csv")

# Carga directa de cada DataFrame usando las rutas absolutas
df_contacts = pd.read_csv(contacts_path, sep=';')
df_accounts = pd.read_csv(accounts_path, sep=';')
df_campaigns = pd.read_csv(campaigns_path, sep=';')
df_registrations = pd.read_csv(registrations_path, sep=';')

# Función para preprocesar el DataFrame y sacarle los acentos..sacarle las ""
def preprocess_dataframe(df):
    # Convertir a minúsculas, eliminar acentos y quitar comillas
    df = df.applymap(lambda x: ''.join(
                            c for c in unicodedata.normalize('NFKD', x.lower()) 
                            if not unicodedata.combining(c)
                        ).replace("'", "").replace('"', '') 
                     if isinstance(x, str) else x)
    return df

# Aplicar preprocesamiento a todos los DataFrames
df_contacts = preprocess_dataframe(df_contacts)
df_accounts = preprocess_dataframe(df_accounts)
df_campaigns = preprocess_dataframe(df_campaigns)
df_registrations = preprocess_dataframe(df_registrations)


# Realizar los joins al inicio
df_accounts_contacts = df_accounts.merge(df_contacts, on='ID de la empresa', how='left')
df_campaigns_registrations = df_campaigns.merge(df_registrations, on='ID de la campaña', how='left')
df_combined = df_accounts_contacts.merge(df_campaigns_registrations, on='ID del contacto', how='outer')
#df_combined.to_csv("datos_combined.csv", index=False)

# Convertir la columna 'Fecha de campaña' a tipo datetime
df_combined['Fecha de campaña'] = pd.to_datetime(df_combined['Fecha de campaña'], errors='coerce')

# Crear una nueva columna 'Hora de campaña' que contenga solo la hora
df_combined['Hora de campaña'] = df_combined['Fecha de campaña'].dt.time

# Extraer año, mes y día en nuevas columnas numéricas
df_combined['Año de campaña'] = df_combined['Fecha de campaña'].dt.year
df_combined['Mes de campaña'] = df_combined['Fecha de campaña'].dt.month
df_combined['Dia de campaña'] = df_combined['Fecha de campaña'].dt.day

df_combined['Fecha de campaña'] = df_combined['Fecha de campaña'].dt.strftime('%d-%m-%Y')

df_combined['Asistentes de la campaña'] = df_combined['Asistentes de la campaña'].fillna(0)

df_combined = df_combined.drop_duplicates()
df_combined['ID del contacto'] = df_combined['ID del contacto'].dropna()

# Cambiar el valor específico en la columna 'Jerarquia' (al tener una coma generaba conflicto)
df_combined['Jerarquia del contacto'] = df_combined['Jerarquia del contacto'].replace(
    'asistente de ceo, presidente o gerente general', 'asistente de ceo'
)

# Construye la ruta completa para guardar el archivo de salida
output_path = os.path.join(base_dir, "data_combined.csv")

# Guarda el DataFrame combinado en la ruta especificada
df_combined.to_csv(output_path, index=False, sep=';')
