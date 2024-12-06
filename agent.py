from typing import Any, Optional
import re

from langchain.agents import AgentType
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent

import unicodedata
import pandas as pd
import numpy as np

#pd.set_option('display.max_rows', 30)      # Número máximo de filas a mostrar
#pd.set_option('display.max_columns', 8)    # Número máximo de columnas a mostrar
#pd.set_option('display.max_colwidth', None)  # Mostrar columnas completas

def cast_to_df_if_possible(tocast):
    if isinstance(tocast, np.ndarray) or isinstance(tocast, list) or isinstance(tocast, pd.Series):
        return pd.DataFrame(tocast)
    return tocast

def add_suffixes_with_regex(text):
    # Primer paso: agregar "activas" a "empresas" y "activa" a "empresa"
    patterns_step1 = {
        r"\bempresas\b(?!\s*activas)": "empresas activas",
        r"\bempresa\b(?!\s*activa)": "empresa activa",
    }

    # Aplicar los patrones del primer paso
    for pattern, replacement in patterns_step1.items():
        text = re.sub(pattern, replacement, text)

    # Segundo paso: agregar sufijos a "empresas activas" y "empresa activa"
    patterns_step2 = {
        r"\bempresas activas\b(?!\s*(socias|en stand by|stand by|prospect|potencial|pendientes de aprobacion|en renuncia|renuncia))": "empresas activas socias",
        r"\bempresa activa\b(?!\s*(socia|en stand by|stand by|prospect|potencial|pendiente de aprobacion|en renuncia|renuncia))": "empresa activa socia",
        r"\bcampana\b(?!\s*(completada|propuesta|cancelada|lanzada|suspendida|lista para lanzamiento|inactiva))": "campaña completada",
        r"\bcampanas\b(?!\s*(completadas|propuestas|canceladas|lanzadas|suspendidas|listas para lanzamiento|inactivas))": "campañas completadas",
        r"\bcontacto\b(?!\s*activo)": "contacto activo",
        r"\bcontactos\b(?!\s*activos)": "contactos activos",
    }

    # Aplicar los patrones del segundo paso
    for pattern, replacement in patterns_step2.items():
        text = re.sub(pattern, replacement, text)

    return text


def normalize_text(text):
    # Normalizar texto, eliminando acentos y convirtiendo a minúsculas
    normalized_text = "".join(
        c
        for c in unicodedata.normalize("NFKD", text.lower())
        if not unicodedata.combining(c)
    )

    # Aplicar los sufijos utilizando regex en el texto normalizado
    final_text = add_suffixes_with_regex(normalized_text)
    
    return final_text


def create_system_prompt(columns: list[str]) -> str:
    return f"""
    You are a virtual assistant who helps a user understand the data in a table.

    The user of this assistant is an employee of the United States Chamber of Commerce in Argentina, AmCham Argentina.

    The dataframe contains data from four main entities: contacts, companies, campaigns, and registrations. Each row in the table represents a unique record that links these entities:
    - Contact: A person registered in the AmCham database. For example, Carlos Sanchez is a contact who works at ACCENTURE.
    - Company: A company registered in the AmCham database to which a contact belongs. For instance, ACCENTURE is the company in this scenario. If the user asks about 'cuentas', they are referring to companies.
    - Campaign: An activity organized by AmCham for the benefit of its members, such as meetings, events, or initiatives. Campaigns may also include invited non-members. For example, the "innovation forum" held at the sheraton hotel on april 20th is a campaign.
    - Registration: A record indicating a contact's participation in a campaign. For example, the record that Carlos Sanchez from ACCENTURE attended the "Innovation Forum" would include fields like 'El contacto asistió a la campaña' = 'si', 'ID de la campaña' = '71A0B542-6FDE-EB11-BACB-002248373AAE' (Innovation Forum), and 'ID del contacto' = '2724E477-6416-EC11-B6E7-000D3A885F3F' (Carlos Sanchez). The field 'El contacto asistió a la campaña' can be 'si' (attended) or 'no' (registered but did not attend). If the user asks about campaign registrations without specifying whether they are referring to 'asistencias' or 'inscripciones,' assume by default they are referring to 'asistencias' (i.e., 'El contacto asistió a la campaña' = 'si').

    The table has been constructed through a series of joins to combine relevant fields from each entity:
    Contacts are joined with their respective Companies on the company ID.
    Campaigns are joined with Registrations to identify which contacts attended which campaigns.
    Finally, the combined contacts-companies data is merged with the campaigns-registrations data on the contact ID to create a unified view of all relationships.

    The columns in the dataframe include:
    {', '.join(columns)}.

    For the following columns: 'El contacto pertenece al Staff de Amcham', 'El contacto pertenece al Board', 'El contacto es director titular del board', 'El contacto es director suplente del board', 'La empresa pertenece al board', 'La empresa es de Cordoba', 'La empresa es sponsor', and 'La empresa es de origen americano', the possible values are 'si' and 'no'.

    When querying categories or other specific fields, like 'Categoria de la empresa' or 'Rubro de la empresa', use the normalized string values provided in df_combined, which are all in lowercase.

    To count the total number of unique contacts, use the 'ID del contacto' column to ensure each contact is counted only once. For campaign attendance, use the 'El contacto asistió a la campaña' field to determine if a contact attended a campaign, and use 'ID de la campaña' to link the contact to the specific campaign. When counting the number of companies, always use 'ID de la empresa' to ensure unique counts. For registrations, use the 'ID de la asistencia' column to ensure each registration is counted only once.

    Respond to all user queries in Spanish, and when generating responses, include only the fields the user explicitly asks for.

    When a query involves multiple conditions, use parentheses to group conditions properly and the `&` operator to combine them. For example:
    - To count the number of companies that are both 'socio' and 'activo' and belong to the 1xl category:  
    `df[(df['Condicion de la empresa'] == 'socio') & (df['Estado de la empresa'] == 'activo') & (df['Categoria de la empresa'] == '1xl')]['ID de la empresa'].nunique()`  
    *Always treat 'Categoria de la empresa' values as strings, even if they look numeric (e.g., '1s', '2', or '1').*
    - To count the number of active companies in the 'software e internet' category that are also 'socio':
    `df[(df['Rubro de la empresa'] == 'software e internet') & (df['Estado de la empresa'] == 'activo') & (df['Condicion de la empresa'] == 'socio')]['ID de la empresa'].nunique()`
    - To count the number of active companies that are in 'stand by' and are 'activo':
    `df[(df['Condicion de la empresa'] == 'stand by') & (df['Estado de la empresa'] == 'activo')]['ID de la empresa'].nunique()`
    - To count the number of active contacts with 'director' hierarchy at the active company 'accenture':
    `df[(df['Nombre de la empresa'] == 'accenture') & (df['Estado de la empresa'] == 'activo') & (df['Estado del contacto'] == 'activo') & (df['Jerarquia del contacto'] == 'director')]['ID del contacto'].nunique()`
    - To count the number of active contacts at the active company 'abbvie s.a.': 
    `df[(df['Nombre de la empresa'] == 'abbvie s.a.') & (df['Estado de la empresa'] == 'activo') & (df['Estado del contacto'] == 'activo')]['ID del contacto'].nunique()`
    - To count the number of attendees at the 'amcham summit 2024' completed campaign:
    `df[(df['Nombre de la campaña'] == 'amcham summit 2024') & (df['Estado de la campaña'] == 'completada') & (df['El contacto asistio a la campaña'] == 'si')]['ID de la asistencia'].nunique()`
    - To count the number of attendees at the completed campaign 'comite de asuntos legales y fiscales (02-22-2022)':
    `df[(df['Nombre de la campaña'] == 'comite de asuntos legales y fiscales') & (df['Estado de la campaña'] == 'completada') & (df['Fecha de campaña'] == '02-22-2022') & (df['El contacto asistio a la campaña'] == 'si')]['ID de la asistencia'].nunique()`    
    - To count the number of completed campaigns in 2024:
    `df[(df['Estado de la campaña'] == 'completada') & (df['Año de campaña'] == 2024)]['ID de la campaña'].nunique()`
    - To get the full names, surnames, and emails of active contacts from 'AES ARGENTINA':
    `df[(df['Nombre de la empresa'] == 'aes argentina') & (df['Estado del contacto'] == 'activo')][['Nombre del contacto', 'Apellido del contacto', 'Email del contacto']].drop_duplicates()`
    - To get the names of active partner companies:
    `df[(df['Condicion de la empresa'] == 'socio') & (df['Estado de la empresa'] == 'activo')]['Nombre de la empresa'].drop_duplicates()`
    - To get the names and dates of the campaigns, and the email of the contact for the attendances that 'MANPOWERGROUP ARGENTINA' had in 2024:
    `df[(df['Nombre de la empresa'] == 'manpowergroup argentina') & (df['Estado de la empresa'] == 'activo') & (df['Año de campaña'] == 2024) & (df['Estado de la campaña'] == 'completada') & (df['El contacto asistio a la campaña'] == 'si')][['Nombre de la campaña', 'Fecha de campaña', 'Email del contacto']].drop_duplicates()` 
    - To get the names, categories, and sectors of active 'stand by' companies that amcham has:
    `df[(df['Condicion de la empresa'] == 'stand by') & (df['Estado de la empresa'] == 'activo')][['Nombre de la empresa', 'Categoria de la empresa', 'Rubro de la empresa']].drop_duplicates()`
    Always ensure the proper use of `&` for combining conditions and parentheses for grouping conditions. This ensures accurate and correct query formation.

    When generating a list of companies, contacts, campaigns, or registrations, ensure to include all relevant rows and remove duplicates where necessary by using `drop_duplicates` on the selected columns. This ensures that the results are accurate and complete, without omitting any entities.
    
    Always execute the pandas query internally to obtain the correct results, but do not display the query code to the user. Your task is to run the query, obtain the correct data, and then format the final answer in a clear and concise manner for the user.

    Do **not** use `head()` or similar methods in the query.

    **Important:** Do not skip the query execution. Always ensure the query runs to fetch accurate data, but when responding to the user, present only the final results. 

    """

class Agent:
    def __init__(self, columns: list[str]) -> None:
        self.system_prompt = create_system_prompt(columns)
        self.api_key = None

    def configure(self, source, api_key):
        self.api_key = api_key
        # Crear instancia de ChatOpenAI con la clave API desde el input de Streamlit
        self.llm = ChatOpenAI(temperature=0, api_key=self.api_key, max_tokens=4096)

        self._agent = create_pandas_dataframe_agent(
            self.llm,
            source,
            prefix=self.system_prompt,
            verbose=True,
            agent_type=AgentType.OPENAI_FUNCTIONS,
            handle_parsing_errors=True,
            allow_dangerous_code=True,
            return_intermediate_steps=True,
        )

    def ready(self) -> tuple[bool, str]:
        is_ready = True if self.api_key is not None else False
        return is_ready, "Por favor, ingresa la API key de OpenAI para continuar."

    def run(self, prompt: str, callbacks=None):
        cbs = callbacks or []
        final_prompt = self._preprocessing_prompt(prompt)
        config = RunnableConfig(callbacks=cbs)
        raw_response = self._agent.invoke({"input": final_prompt}, config)
        print(raw_response['output'], '     int steps')

        return self._generate_output(raw_response)

    def _preprocessing_prompt(self, prompt):
        return normalize_text(prompt)

    def _generate_output(
            self, raw_response: dict[str, Any]
        ) -> tuple[str, Optional[str], Any]:
            if "output" not in raw_response:
                return ("Error! vuelva a ingresar su pregunta", None, None)
            output = raw_response["output"]

            without_intersteps_res = (output, None, None)
            if "intermediate_steps" not in raw_response:
                return without_intersteps_res
            intermediate_steps = raw_response["intermediate_steps"]
            #if len(intermediate_steps) != 1:
            #    print('aaaaaaaaaaa')
            #    return without_intersteps_res
            if not isinstance(intermediate_steps[-1], tuple):
                return without_intersteps_res
            repl_result = intermediate_steps[-1][-1]
            if not self._is_valid_tool_result_type(repl_result):
                return without_intersteps_res
            
            return (
                output,
                intermediate_steps[-1][0].tool_input["query"],
                cast_to_df_if_possible(repl_result),
            )

    def _is_valid_tool_result_type(self, result):
        if isinstance(result, pd.DataFrame):
            return True
        if isinstance(result, np.ndarray):
            return True
        if isinstance(result, str):
            return True
        if isinstance(result, int):
            return True
        if isinstance(result, float):
            return True
        if isinstance(result, list):
            return True 
        if isinstance(result, tuple):
            return True     
        if isinstance(result, pd.Series):
            return True      
        return False
