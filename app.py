from langchain.callbacks import StreamlitCallbackHandler
import streamlit as st
import pandas as pd
import numpy as np
import re
import random
import os
from component_lib import auto_prompt
from agent import Agent, cast_to_df_if_possible

from io import BytesIO

np.random.seed(42)
random.seed(42)

st.set_page_config(page_title="StaffBot", page_icon="🦜")
st.title("🦜 StaffBot")

@st.cache_data
def get_data():
    # Obtiene la ruta absoluta del directorio donde está app.py
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # Construye la ruta completa del archivo data_combined.csv
    file_path = os.path.join(base_dir, "data_combined.csv")
    # Lee el archivo CSV usando la ruta completa
    return pd.read_csv(file_path, sep=";")

@st.cache_data
def get_option_dictionary():
    unique_campaigns = df_combined[
        (df_combined["Año de campaña"] == 2024)
        & (df_combined["Estado de la campaña"] == "completada")
    ][["Nombre de la campaña", "Fecha de campaña"]].drop_duplicates()

    entities_dict = {
        "Empresas": list(
            sorted(
                    '"' + df_combined[
                    (df_combined["Condicion de la empresa"] == "socio")
                    & (df_combined["Estado de la empresa"] == "activo")
                ]["Nombre de la empresa"]
                .str.upper()
                .unique() + '"'
            )
        ),
        "Campañas": list(
            sorted(
                (
                    '"' + df_combined[
                        (df_combined["Año de campaña"] == 2024)
                        & (df_combined["Estado de la campaña"] == "completada")
                    ]["Nombre de la campaña"].str.upper()
                    + '" (' 
                    + df_combined[
                        (df_combined["Año de campaña"] == 2024)
                        & (df_combined["Estado de la campaña"] == "completada")
                    ]["Fecha de campaña"]
                    + ')'
                )
                .dropna()
                .unique()
            )
        ),
        "Contactos": list(
            sorted(
                (
                    '"' + df_combined[
                        (df_combined["Condicion de la empresa"] == "socio")
                        & (df_combined["Estado de la empresa"] == "activo")
                        & (df_combined["Estado del contacto"] == "activo")
                    ]["Nombre del contacto"].str.cat(
                        df_combined["Apellido del contacto"], sep=" "
                    )
                )
                .str.title()
                .dropna()
                .unique() + '"'
            )
        ),
    }

    return entities_dict


df_combined = get_data()

# ESTO ES SOLO porque solo el staff puede acceder al staffbot. En caso contrario, usariamos el codigo comentado.
# openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
openai_api_key = "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"

@st.cache_resource
def load_agent():
    agent = Agent(list(df_combined.columns))
    agent.configure(df_combined, openai_api_key)

    return agent


def block_prompt():
    st.session_state.ready = False

agent = load_agent()

if "ready" not in st.session_state:
    st.session_state.ready = True

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "assistant",
            "content": "¡Hola! Soy tu asistente virtual. Estoy aquí para responder tus consultas sobre los datos de AmCham. Puedes preguntarme sobre cualquier información específica de los datos cargados y te responderé lo mejor que pueda. ¿En qué puedo ayudarte hoy?",
        }
    ]

# Obtén la ruta absoluta del directorio donde está app.py
base_dir = os.path.dirname(os.path.abspath(__file__))

# Rutas absolutas de los archivos PDF
pdf_path_1 = os.path.join(base_dir, "Tutorial del StaffBot.pdf")
pdf_path_2 = os.path.join(base_dir, "Empresas Socias 02-09-2024.pdf")
pdf_path_3 = os.path.join(base_dir, "Listado de campos.pdf")


# Agregar un botón para descargar el archivo PDF
st.sidebar.title("Recursos Adicionales")

# Botón para descargar el archivo "Tutorial StaffBot.pdf"
with open(pdf_path_1, "rb") as pdf_file:
    pdf_bytes = pdf_file.read()
    st.sidebar.download_button(
        label="Descargar Tutorial",
        data=pdf_bytes,
        file_name="Tutorial del StaffBot.pdf",
        mime="application/pdf",
        on_click=block_prompt,
    )

# Botón para descargar el archivo "Empresas Socias 02-09-2024.pdf"
with open(pdf_path_2, "rb") as pdf_file:
    pdf_bytes = pdf_file.read()
    st.sidebar.download_button(
        label="Descargar Empresas socias",
        data=pdf_bytes,
        file_name="Empresas Socias 02-09-2024.pdf",
        mime="application/pdf",
        on_click=block_prompt,
    )

# Botón para descargar el archivo "Listado de campos.pdf"
with open(pdf_path_3, "rb") as pdf_file:
    pdf_bytes = pdf_file.read()
    st.sidebar.download_button(
        label="Descargar Campos y sus valores",
        data=pdf_bytes,
        file_name="Listado de campos.pdf",
        mime="application/pdf",
        on_click=block_prompt,
    )

messages_container = st.container()

for msg in st.session_state.messages:   
    messages_container.chat_message(msg["role"]).write(msg["content"])

execution_result = None

prompt = auto_prompt(
    entities=["Empresas", "Campañas", "Contactos"],
    entities_dict=get_option_dictionary(),
    key="prompt",
)

if prompt and st.session_state.ready:
    # Elimina mayusculas y acentos y agrega valores por default
    st.session_state.messages.append({"role": "user", "content": prompt})
    messages_container.chat_message("user").write(prompt)

    is_ready, err_msg = agent.ready()
    if not is_ready:
        st.info(err_msg)
        st.stop()

    with messages_container.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)

        # Ejecutar la consulta
        response, query ,execution_result = agent.run(
            st.session_state.messages[-1]["content"], callbacks=[st_cb]
        )

        if query:  
            # Saco funciones que limitan al df  
            query = re.sub(r'\.head\(\d*\)', '', query).replace('.tolist()', '') 
            #query = query.replace('.tolist()', '').replace('.head()', '').replace('.head(10)','').replace('.head(5)', '')    

            r = eval(query, {"df": df_combined})

            temp_df = cast_to_df_if_possible(r)

            if isinstance(temp_df, pd.DataFrame):
                # Eliminar la columna de índice si existe
                temp_df = temp_df.reset_index(drop=True)

                st.dataframe(temp_df, height=400) 

                st.session_state.messages.append({"role": "assistant", "content": temp_df})

                # Crear un buffer para el archivo .xlsx
                buffer = BytesIO()
                temp_df.to_excel(buffer, index=False, engine="openpyxl")
                buffer.seek(0)  # Mover el puntero al inicio del archivo
 
                st.download_button(
                    label="Descargar",
                    data=buffer,
                    file_name="data.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    on_click=block_prompt,
                )
            else:
                st.session_state.messages.append({"role": "assistant", "content": response})

                st.write(response)


st.session_state.ready = True
