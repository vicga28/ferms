import pandas as pd
import numpy as np
import streamlit as st
import io
import matplotlib.pyplot as plt
import seaborn as sns
import math

st.set_page_config(layout='wide', page_title='Carreteres', page_icon='data\\road-icon.png')

### Aquest script serveix per obtenir un llistat dels trams de carretera per sota un valor llindar de CRT per així poder determinar quins trams seran subjecte de reparacions puntuals

# Ruta de l'arxiu amb els valors CRT
path_crt = 'data/CRT_2.csv'
path_imd = 'data/Última IMD.csv'
path_accidents = 'data/Accidents 2023.csv'
path_traçat_0 = 'data/Parametrització en planta_no_BV.csv'
path_traçat_1 = 'data/Parametrització en planta_BV_1.csv'
path_traçat_2 = 'data/Parametrització en planta_BV_2-4.csv'
path_traçat_3 = 'data/Parametrització en planta_BV_5-9.csv'
path_trams = 'data/Trams Catàleg Oficial.csv'


# Càrrega d'arxiu amb valors CRT
df_crt = pd.read_csv(path_crt, delimiter=";")
df = pd.read_csv(path_imd, delimiter=";")
df_acc = pd.read_csv(path_accidents, delimiter=";")

df_trams = pd.read_csv(path_trams, delimiter=";")

df_trams['INICI_TRAM'] = df_trams['INICI_TRAM'].str.replace("+", "", regex=False)
df_trams['FINAL_TRAM'] = df_trams['FINAL_TRAM'].str.replace("+", "", regex=False)
df_trams['INICI_TRAM'] = pd.to_numeric(df_trams['INICI_TRAM'], errors='coerce')
df_trams['FINAL_TRAM'] = pd.to_numeric(df_trams['FINAL_TRAM'], errors='coerce')
df_trams['Carretera'] = df_trams['CAR_CARRETERA']

df_traçat_0 = pd.read_csv(path_traçat_0, delimiter=";")
df_traçat_1 = pd.read_csv(path_traçat_1, delimiter=";")
df_traçat_2 = pd.read_csv(path_traçat_2, delimiter=";")
df_traçat_3 = pd.read_csv(path_traçat_3, delimiter=";")

df_traçat = pd.concat([df_traçat_0, df_traçat_1, df_traçat_2, df_traçat_3], ignore_index = True)

def es_contigua(df):
    df = df.sort_values('INICI_TRAM')
    return all(df.iloc[i]['FINAL_TRAM'] == df.iloc[i+1]['INICI_TRAM'] for i in range(len(df)-1))





df_traçat['PKF'] = df_traçat.groupby('Carretera')['Shape.STLength()'].cumsum()
df_traçat['PKI'] = df_traçat.groupby('Carretera')['PKF'].shift(1).fillna(0)

df_traçat = df_traçat[['Carretera', 'PKI', 'PKF', 'Tipus', 'Valor']]



resultats = []

for carretera, grup in df_traçat.groupby('Carretera'):
    trams = df_trams[df_trams['Carretera'] == carretera].sort_values('INICI_TRAM')
    if es_contigua(trams):
        PK_min = trams['INICI_TRAM'].min()
        sub = grup.copy()
        sub['PKI_real'] = PK_min + sub['PKI']
        sub['PKF_real'] = PK_min + sub['PKF']
        resultats.append(sub)
    else:
        sub=grup.copy()
        sub['PKI_real'] = None
        sub['PKF_real'] = None
        dist_inici = 0
        for _, tram in trams.iterrows():
            long_tram = (tram['FINAL_TRAM'] - tram['INICI_TRAM'])
            mask = (sub['PKI'] >= dist_inici) & (sub['PKI'] <= dist_inici + long_tram + 1e-6)
            sub.loc[mask, 'PKI_real'] = tram['INICI_TRAM'] + (sub.loc[mask, 'PKI'] - dist_inici)
            sub.loc[mask, 'PKF_real'] = tram['INICI_TRAM'] + (sub.loc[mask, 'PKF'] - dist_inici)
            dist_inici += long_tram
        resultats.append(sub)

df_traçat = pd.concat(resultats).sort_values(['Carretera', 'PKI'])

df_traçat['PKI'] = df_traçat['PKI_real'].astype('float').round()
df_traçat['PKF'] = df_traçat['PKF_real'].astype('float').round()
df_traçat['Valor'] = df_traçat['Valor'].round()

df_traçat = df_traçat[['Carretera', 'PKI', 'PKF', 'Tipus', 'Valor']]

df['PKI'] = df['PKI'].str.replace("+", "", regex=False)
df['PKF'] = df['PKF'].str.replace("+", "", regex=False)

df['PKI'] = pd.to_numeric(df['PKI'], errors='coerce')
df['PKF'] = pd.to_numeric(df['PKF'], errors='coerce')
df['Any càlcul'] = pd.to_numeric(df['Any càlcul'], errors='coerce')
df['IMD'] = pd.to_numeric(df['IMD'], errors='coerce')
df['% Pesants'] = pd.to_numeric(df['% Pesants'], errors='coerce')
df['% Motos'] = pd.to_numeric(df['% Motos'], errors='coerce')

df = df[['Carretera', 'PKI', 'PKF', 'Codi estació', 'Any càlcul', 'IMD', '% Pesants', '% Motos']]




df_imd = df.copy()


# Adaptació dels valors de l'arxiu per al seu tractament
df_crt['PKI'] = pd.to_numeric(df_crt['PKI'], errors='coerce')
df_crt['PKF'] = pd.to_numeric(df_crt['PKF'], errors='coerce')
df_crt['CRT'] = pd.to_numeric(df_crt['CRT'], errors='coerce')
df_crt['Textura'] = pd.to_numeric(df_crt['Textura'], errors='coerce')
df_crt['Via'] = pd.to_numeric(df_crt['Via'], errors='coerce')

# Funció per crear trams fixos per cada (Carretera, Via)
def crear_trams_saneig(df_grup, tram_length=1000, lim_crt=40):
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
    resum_trams.columns = ['Tram_1000m', 'CRT mig', 'CRT desviacio', 
                          'Textura mitjana', 'Textura desviacio', 'PKI', 'PKF']
    
    # Ordenem per tram
    resum_trams = resum_trams.sort_values('PKI').reset_index(drop=True)

    # Afegim columna per indicar si CRT_mitjana < límit fixat
    resum_trams['CRT_lim'] = resum_trams['CRT mig'] < lim_crt
    resum_trams = resum_trams[resum_trams['CRT_lim'] == True]
    
    return resum_trams

def processar_dades_saneig(df, tram_length=1000, lim_crt=40):
    resultats = []
    # Agrupar per Carretera i Via
    grups = df.groupby(['Carretera', 'Via', 'Sector', 'Any'])
    
    for (carretera, via, sector, any), grup in grups:
        resum = crear_trams_saneig(grup, tram_length, lim_crt = lim_crt)
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
                
                for col in ['CRT mig', 'Textura mitjana']:
                    tram_actual[col] = (tram_actual[col]*llarg1 + fila[col]*llarg2) / total

                for col in ['CRT desviacio', 'Textura desviacio']:
                    tram_actual[col] = np.nan  # Desviació ja no és representativa després d'ajuntar
            else:
                agrupacions.append(tram_actual)
                tram_actual = fila.copy()
        
        agrupacions.append(tram_actual)  # afegir l’últim
        return pd.DataFrame(agrupacions)


    # Reordenar les columnes segons l'ordre desitjat
    df_resultat = df_resultat[['Carretera', 'PKI', 'PKF', 'Via', 'Sector', 'Any',
                               'CRT mig', 'CRT desviacio', 
                               'Textura mitjana', 'Textura desviacio']]
    
    df_resultat = ajuntar_trams_contigus(df_resultat)

    df_resultat['Longitud'] = df_resultat['PKF'] - df_resultat['PKI']

    df_resultat = df_resultat.sort_values(by=['Carretera', 'Via', 'CRT mig'], ascending=[True, True, True])

    df_resultat = df_resultat[['Carretera', 'PKI', 'PKF', 'Longitud', 'Via', 'Sector', 'Any',
                               'CRT mig', 'Textura mitjana']]

    return df_resultat

def convert_for_download(df):
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Resultats")
        workbook = writer.book
        worksheet = writer.sheets['Resultats']

        num_format = workbook.add_format({'num_format': '0.0'})
        pk_format = workbook.add_format({'num_format':'0+000'})
        mil_format = workbook.add_format({'num_format': '#,##0'})

        crt_col = df.columns.get_loc("CRT mig")
        textura_col = df.columns.get_loc("Textura mitjana")
        pki_col = df.columns.get_loc("PKI")
        pkf_col = df.columns.get_loc("PKF")
        long_col = df.columns.get_loc("Longitud")

        if "Valor" in df.columns:
            valor_col = df.columns.get_loc("Valor")
            worksheet.set_column(valor_col, valor_col, None, mil_format)

        if "IMD" in df.columns:
            imd_col = df.columns.get_loc("IMD")
            worksheet.set_column(imd_col, imd_col, None, mil_format)

        worksheet.set_column(crt_col, crt_col, None, num_format)
        worksheet.set_column(textura_col, textura_col, None, num_format)
        worksheet.set_column(pki_col, pki_col, None, pk_format)
        worksheet.set_column(pkf_col, pkf_col, None, pk_format)
        worksheet.set_column(long_col, long_col, None, mil_format)
    # Get Excel binary data
    excel_data = buffer.getvalue()
    return excel_data

def add_accidents(df, df_acc):
    df = df.copy()
    df['Accidents'] = 0
    for _, row in df.iterrows():
        carretera = row['Carretera']
        pki = row['PKI']
        pkf = row['PKF']

        # Subtrams que se superposen
        subtrams = df[
            (df['Carretera'] == carretera) &
            (df['PKF'] >= pki) &
            (df['PKI'] <= pkf)
        ].copy()

        acc = df_acc[
            (df_acc['ACC_CARRETERA'] == carretera) &
            (df_acc['ACC_POS_'] >= pki) &
            (df_acc['ACC_POS_'] <= pkf)
        ].shape[0]

        df.at[_, 'Accidents'] += acc
    return df

def add_imd(df, df_imd):
    df = df.copy()
    for _, row in df.iterrows():
        carretera = row['Carretera']
        pki = row['PKI']
        pkf = row['PKF']

        # Subtrams que se superposen
        match = df_imd[
            (df_imd['Carretera'] == carretera) &
            (df_imd['PKF'] >= pki) &
            (df_imd['PKI'] <= pkf)
        ]

        if not match.empty:
            df.at[_, 'IMD'] = match['IMD'].iloc[0]
            df.at[_, '% Pesants'] = match['% Pesants'].iloc[0]
            df.at[_, '% Motos'] = match['% Motos'].iloc[0]
    return df

def add_traçat(df, df_traçat):
    df = df.copy()
    for _, row in df.iterrows():
        carretera = row['Carretera']
        pki = row['PKI']
        pkf = row['PKF']

        # Subtrams que se superposen
        match = df_traçat[
            (df_traçat['Carretera'] == carretera) &
            (df_traçat['PKF'] >= pki) &
            (df_traçat['PKI'] <= pkf)
        ]

        if not match.empty:
            df.at[_, 'Tipus'] = match['Tipus'].iloc[0]
            df.at[_, 'Valor'] = match['Valor'].iloc[0]
    return df

tab1, tab2, tab3 = st.tabs(['Sanejament', 'Capes de rodadura', 'Documentació tècnica'])

with tab1:
    st.header('Càlcul trams sanejament')
    with st.form("data_input"):
        long_tram = st.number_input("Longitud mínima tram (m): ", value=100, format="%d")
        st.write("La longitud mínima del tram indica quina longitud mínima tindran els trams seleccionats. En cas de no poder crear-se aquesta longitud mínima, es seleccionarà la major possible.")
        limit_crt = st.number_input("Llindar CRT (%): ", value=40, format="%d")
        st.write("Es seleccionaran tots els trams de carretera que tinguin un valor inferior al valor seleccionat.")
        st.write('Incloure:')
        compute_acc = st.checkbox('Accidents')
        compute_imd = st.checkbox('IMD')
        compute_traçat = st.checkbox('Traçat')
        submitted = st.form_submit_button("Calcular")
    if submitted:
        df_result = processar_dades_saneig(df_crt, tram_length=long_tram, lim_crt=limit_crt)
        if compute_acc:
            df_result = add_accidents(df_result, df_acc)
        if compute_imd:
            df_result = add_imd(df_result, df_imd)
        if compute_traçat:
            df_result = add_traçat(df_result, df_traçat)
        excel = convert_for_download(df_result)
        st.download_button(label='Descarregar fitxer', data=excel, file_name='resultats.xlsx', mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        icon=":material/download:")
    

### TRAMS HOMOGENIS CRT


resultats = []

df['Accidents'] = 0

for _, row in df.iterrows():
    carretera = row['Carretera']
    pki = row['PKI']
    pkf = row['PKF']

    # Subtrams que se superposen
    subtrams = df_crt[
        (df_crt['Carretera'] == carretera) &
        (df_crt['PKF'] >= pki) &
        (df_crt['PKI'] <= pkf)
    ].copy()

    acc = df_acc[
        (df_acc['ACC_CARRETERA'] == carretera) &
        (df_acc['ACC_POS_'] >= pki) &
        (df_acc['ACC_POS_'] <= pkf)
    ].shape[0]

    df.at[_, 'Accidents'] += acc

    if not subtrams.empty:
        # Longituds dels subtrams
        subtrams['Longitud'] = subtrams['PKF'] - subtrams['PKI']

        # Sèries útils
        crt_vals = subtrams['CRT'].dropna()

        # Mitjana
        crt_mig = crt_vals.mean()

        # Desviació estàndard
        crt_std = crt_vals.std()

        # Condició 1: 95% dins de [0.5 * mitjana, 1.5 * mitjana]
        if not crt_vals.empty:
            lower = 0.5 * crt_mig
            upper = 1.5 * crt_mig
            percent_inside = ((crt_vals >= lower) & (crt_vals <= upper)).sum() / len(crt_vals)

            cond1 = percent_inside >= 0.95
            cond2 = (crt_std / crt_mig) <= 0.4 if crt_mig != 0 else False
            homogeni = cond1 and cond2
        else:
            crt_std = None
            homogeni = None

        # Textura
        textura_mig = subtrams['Textura'].mean()
    else:
        crt_mig = None
        crt_std = None
        textura_mig = None
        homogeni = None

    # Afegim les dades
    resultats.append({
        **row,
        'Accidents': acc,
        'CRT mig': crt_mig,
        'Desviació estàndard CRT': crt_std,
        'Textura mitja': textura_mig,
        'Homogeni': homogeni
    })

# Nou DataFrame
df_resultats = pd.DataFrame(resultats)


def format_plus(x):
    s = str(x)
    if len(s) <= 3:
        return "0+" + s.zfill(3)
    else:
        return s[:-3] + "+" + s[-3:]

with tab2:
    with st.expander("Filtres"):
        with st.form('filter'):
            filter_crt = st.slider('Valors mig de CRT (%):', 0, 100, (0,70))
            filter_textura = st.slider('Valors mig de Textura: ', 0., 1.2, (0.,0.7))
            apply = st.form_submit_button("Aplicar")
        if apply:
            df_resultats = df_resultats[(df_resultats['CRT mig'] >= filter_crt[0]) & (df_resultats['CRT mig'] <= filter_crt[1])]
            df_resultats = df_resultats[(df_resultats['Textura mitja'] >= filter_textura[0]) & (df_resultats['Textura mitja'] <= filter_textura[1])]
    df_taula = df_resultats.copy()
    df_taula['PKI'] = df_taula['PKI'].apply(format_plus)
    df_taula['PKF'] = df_taula['PKF'].apply(format_plus)
    df_resultats['IMD Pesants'] = (df_resultats['IMD'] * df_resultats['% Pesants'] / 100).round()
    df_resultats['IMD Motos'] = (df_resultats['IMD'] * df_resultats['% Motos'] / 100).round()
    st.dataframe(df_taula)
    st.write("* **Un tram es defineix com a homogeni quan compleix que el quocient entre la desviació típica i el valor mig és inferior a 0,4 i que el 95% dels valors es troben entre 0,5 i 1,5 del valor mig.**")
    

# Eliminar files amb valors nuls en les columnes d'interès
    df_plot = df_resultats.dropna(subset=['CRT mig', '% Pesants', 'IMD', 'Accidents', 'IMD Pesants', 'IMD Motos'])

    # Estil de gràfics
    sns.set(style="whitegrid", font_scale=1.1)

    # Gràfica: CRT mitjà vs % Pesants
    fig_pesants = plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df_plot, x='IMD Pesants', y='CRT mig')
    sns.regplot(data=df_plot, x='IMD Pesants', y='CRT mig', scatter=False, color='red')
    plt.title('Relació entre IMD Pesants i CRT mig')
    plt.xlabel('IMD Pesants')
    plt.ylabel('CRT mig')
    plt.tight_layout()
    plt.show()
    # st.pyplot(fig_pesants)
    corr_pesants = df_plot['CRT mig'].corr(df_plot['IMD Pesants'])
    # st.write(f"Coeficient de correlació entre CRT mig i IMD Pesants: {corr_pesants:.3f}")


    # Gràfica: CRT mitjà vs IMD
    fig_imd = plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df_plot, x='IMD', y='CRT mig')
    sns.regplot(data=df_plot, x='IMD', y='CRT mig', scatter=False, color='red')
    plt.title('Relació entre IMD i CRT mig')
    plt.xlabel('IMD')
    plt.ylabel('CRT mig')
    plt.tight_layout()
    plt.show()
    # st.pyplot(fig_imd)
    corr_imd = df_plot['CRT mig'].corr(df_plot['IMD'])

    # st.write(f"Coeficient de correlació entre CRT mig i IMD: {corr_imd:.3f}")

    # Gràfica: CRT mitjà vs % Motos
    fig_motos = plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df_plot, x='IMD Motos', y='CRT mig')
    sns.regplot(data=df_plot, x='IMD Motos', y='CRT mig', scatter=False, color='red')
    plt.title('Relació entre IMD Motos i CRT mig')
    plt.xlabel('IMD Motos')
    plt.ylabel('CRT mig')
    plt.tight_layout()
    plt.show()
    # st.pyplot(fig_motos)
    corr_motos = df_plot['CRT mig'].corr(df_plot['IMD Motos'])
    # st.write(f"Coeficient de correlació entre CRT mig i IMD Motos: {corr_motos:.3f}")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.pyplot(fig_imd)
        st.write(f"Coeficient de correlació entre CRT mig i IMD: {corr_imd:.3f}")
    with col2:
        st.pyplot(fig_pesants)
        st.write(f"Coeficient de correlació entre CRT mig i IMD Pesants: {corr_pesants:.3f}")
    with col3:
        st.pyplot(fig_motos)
        st.write(f"Coeficient de correlació entre CRT mig i IMD Motos: {corr_motos:.3f}")


    df_corr_acc = []

    for n in range(0, df['Accidents'].max()):
        df_accidents = df_plot[df_plot['Accidents'] >= n]
        # Gràfica: Accidents vs CRT
        fig_acc = plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df_accidents, x='CRT mig', y='Accidents')
        sns.regplot(data=df_accidents, x='CRT mig', y='Accidents', scatter=False, color='red')
        plt.title('Relació entre Accidents i CRT mig')
        plt.xlabel('CRT mig')
        plt.ylabel('Accidents')
        plt.tight_layout()
        plt.show()
        # st.pyplot(fig_acc)
        # Calcular correlacions
        corr_acc = df_accidents['Accidents'].corr(df_accidents['CRT mig'])
        # st.write(f"Coeficient de correlació entre CRT mig i Accidents: {corr_acc:.3f}")
        df_corr_acc.append({'>= Accidents':n, 'Correlació amb CRT':corr_acc, 'Població':len(df_accidents)})
    
    st.dataframe(df_corr_acc,use_container_width=False , hide_index=True)



index_belgica = pd.DataFrame()

path_belgica = 'C:\\Users\\gullonav\\Desktop\\Ferms\\App\\data\\index_belgica.csv'
path_espanya_puntual = 'C:\\Users\\gullonav\\Desktop\\Ferms\\App\\data\\index_espanya_puntual.xlsx'
path_espanya_medio = 'C:\\Users\\gullonav\\Desktop\\Ferms\\App\\data\\index_espanya_medio.xlsx'


# Càrrega d'arxiu amb valors CRT
df_belgica = pd.read_csv(path_belgica, delimiter=";")
df_espanya_puntual = pd.read_excel(path_espanya_puntual)
df_espanya_medio = pd.read_excel(path_espanya_medio)


with tab3:
    st.markdown("# Índexs de qualitat")
    st.markdown("## Bèlgica")
    st.dataframe(df_belgica, use_container_width=False ,hide_index=True)
    st.markdown("## Espanya")
    col_1, col_2 = st.columns(2)
    with col_1:
        st.markdown("### Valores puntuales")
        st.dataframe(df_espanya_puntual,use_container_width=False , hide_index=True)
    with col_2:
        st.markdown("### Valores medios (1 km)")
        st.dataframe(df_espanya_medio,use_container_width=False , hide_index=True)

            


