import pandas as pd
import numpy as np
import streamlit as st
import io

### Aquest script serveix per obtenir un llistat dels trams de carretera per sota un valor llindar de CRT per així poder determinar quins trams seran subjecte de reparacions puntuals

# Ruta de l'arxiu amb els valors CRT
path = 'C:\\Users\\gullonav\\Desktop\\Ferms\\Input\\CRT_2.csv'

# Càrrega d'arxiu amb valors CRT
df = pd.read_csv(path, delimiter=";")

# Adaptació dels valors de l'arxiu per al seu tractament
df['PKI'] = pd.to_numeric(df['PKI'], errors='coerce')
df['CRT'] = pd.to_numeric(df['CRT'], errors='coerce')
df['Textura'] = pd.to_numeric(df['Textura'], errors='coerce')
df['Via'] = pd.to_numeric(df['Via'], errors='coerce')

# Funció per crear trams fixos per cada (Carretera, Via)
def crear_trams(df_grup, tram_length=1000, lim_crt=40):
    min_pki = df_grup['PKI'].min()
    max_pki = df_grup['PKI'].max()
    # Creem intervals de tram, començant pel mínim PKI i anant de tram_length en tram_length
    trams_bins = np.arange(min_pki, max_pki + tram_length, tram_length)
    
    # Assignem a cada fila el tram corresponent (interval on cau PKI)
    df_grup = df_grup.copy()
    df_grup['Tram_1000m'] = pd.cut(df_grup['PKI'], bins=trams_bins, right=False, include_lowest=True)
    
    # Resum per tram
    resum_trams = df_grup.groupby('Tram_1000m', observed=False).agg({
        'CRT': ['mean', 'std'],
        'Textura': ['mean', 'std']
    }).reset_index()
    
    # Afegim columnes per a descomposar l'interval del tram en valors numèrics (inici i fi)
    resum_trams['PKI'] = resum_trams['Tram_1000m'].apply(lambda x: x.left)
    resum_trams['PKF'] = resum_trams['Tram_1000m'].apply(lambda x: x.right)

    # Convertir Tram_fi a float per evitar errors de categoria
    resum_trams['PKF'] = resum_trams['PKF'].astype(float)

    # Ajustem l'últim tram per posar Tram_fi = max_pki real
    resum_trams.loc[resum_trams.index[-1], 'PKF'] = max_pki
    
    # Neteja columnes
    resum_trams.columns = ['Tram_1000m', 'CRT_mitjana', 'CRT_desviacio', 
                          'Textura_mitjana', 'Textura_desviacio', 'PKI', 'PKF']
    
    # Ordenem per tram
    resum_trams = resum_trams.sort_values('PKI').reset_index(drop=True)

    # Afegim columna per indicar si CRT_mitjana < límit fixat
    resum_trams['CRT_lim'] = resum_trams['CRT_mitjana'] < lim_crt
    resum_trams = resum_trams[resum_trams['CRT_lim'] == True]
    
    return resum_trams

def processar_dades(df, tram_length=1000, lim_crt=40):
    resultats = []
    # Agrupar per Carretera i Via
    grups = df.groupby(['Carretera', 'Via', 'Sector', 'Any'])
    
    for (carretera, via, sector, any), grup in grups:
        resum = crear_trams(grup, tram_length, lim_crt = lim_crt)
        resum['Carretera'] = carretera
        resum['Via'] = via
        resum['Sector'] = sector
        resum['Any'] = any
        resultats.append(resum)
        
    df_resultat = pd.concat(resultats, ignore_index=True)
    # Ordenar per carretera, via i inici tram
    df_resultat = df_resultat.sort_values(['Carretera', 'Via', 'PKI']).reset_index(drop=True)

    def ajuntar_trams_contigus(df):
        agrupacions = []
        tram_actual = df.iloc[0].copy()

        for idx in range(1, len(df)):
            fila = df.iloc[idx]
            mateixos = all([
                tram_actual['Carretera'] == fila['Carretera'],
                tram_actual['Via'] == fila['Via'],
                tram_actual['Sector'] == fila['Sector'],
                tram_actual['Any'] == fila['Any'],
                np.isclose(tram_actual['PKF'], fila['PKI'])  # contigu
            ])
            
            if mateixos:
                # Allarguem el tram actual
                tram_actual['PKF'] = fila['PKF']
                # Actualitzem mitjanes ponderades segons longituds (opcionalment)
                llarg1 = tram_actual['PKF'] - tram_actual['PKI']
                llarg2 = fila['PKF'] - fila['PKI']
                total = llarg1 + llarg2
                
                for col in ['CRT_mitjana', 'Textura_mitjana']:
                    tram_actual[col] = (tram_actual[col]*llarg1 + fila[col]*llarg2) / total

                for col in ['CRT_desviacio', 'Textura_desviacio']:
                    tram_actual[col] = np.nan  # Desviació ja no és representativa després d'ajuntar
            else:
                agrupacions.append(tram_actual)
                tram_actual = fila.copy()
        
        agrupacions.append(tram_actual)  # afegir l’últim
        return pd.DataFrame(agrupacions)


    # Reordenar les columnes segons l'ordre desitjat
    df_resultat = df_resultat[['Carretera', 'PKI', 'PKF', 'Via', 'Sector', 'Any',
                               'CRT_mitjana', 'CRT_desviacio', 
                               'Textura_mitjana', 'Textura_desviacio']]
    
    df_resultat = ajuntar_trams_contigus(df_resultat)

    df_resultat['Longitud'] = df_resultat['PKF'] - df_resultat['PKI']

    df_resultat = df_resultat.sort_values(by=['Carretera', 'Via', 'CRT_mitjana'], ascending=[True, True, True])

    df_resultat = df_resultat[['Carretera', 'PKI', 'PKF', 'Longitud', 'Via', 'Sector', 'Any',
                               'CRT_mitjana', 'Textura_mitjana']]

    return df_resultat

def convert_for_download(df):
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Resultats")
    # Get Excel binary data
    excel_data = buffer.getvalue()
    return excel_data

st.header('Càlcul trams sanejament')
with st.form("data_input"):
    long_tram = st.number_input("Longitud tram (m): ")
    limit_crt = st.number_input("Llindar CRT: ")
    submitted = st.form_submit_button("Calcular")
if submitted:
    df_result = processar_dades(df, tram_length=long_tram, lim_crt=limit_crt)
    excel = convert_for_download(df_result)
    st.download_button(label='Descarregar fitxer', data=excel, file_name='resultats.xlsx', mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                       icon=":material/download:")
