import pandas as pd
import numpy as np
import joblib
import streamlit as st
from datetime import datetime
from database import Pacient, init_db, get_db

#initializare
init_db()

@st.cache_resource  # decorator; salveaza obiectele(modelele) in cache pentru a nu le incarca de fiecare data cand se face refresh la pagina
def load_model():
    model=joblib.load("model_random_forest.pkl")
    scaler=joblib.load("scaler.pkl")
    encoders=joblib.load("encoders.pkl")
    yeojohnson=joblib.load("yeojohnson.pkl")
    return model, scaler, encoders, yeojohnson

model, scaler, encoders, yeojohnson=load_model()

dict_stadii ={0:'0 - Fără Boală Cardiovasculară',
                                    1:'1 - Boală Cardiovasculară Ușoară',
                                    2:'2 - Boală Cardiovasculară Moderată',
                                    3:'3 - Boală Cardiovasculară Severă',
                                    4:'4 - Boală Cardiovasculară Critica'}

st.set_page_config(page_title="CardiologieAI", page_icon="🫀", layout="wide")

st.markdown("""
<div style="display:flex; align-items:center; gap:20px; padding:10px 0;">
    <img src="https://cdn-icons-png.flaticon.com/512/3004/3004458.png" width="70">
    <h1 style="color:#b30000; margin:0;">CardiologieAI – Sistem Inteligent de Diagnostic</h1>
</div>
""", unsafe_allow_html=True)


st.markdown("""
<style>

html, body, [data-testid="stAppViewContainer"] {
    font-family: 'Segoe UI', sans-serif;
}

/* ===== TITLURI ===== */
h1, h2, h3 {
    color: #b30000 !important;
    font-weight: 700 !important;
}

/* ===== SIDEBAR ===== */
[data-testid="stSidebar"] {
    background-color: #f7f7f7 !important;
    border-right: 1px solid #e0e0e0 !important;
}

[data-testid="stSidebar"] * {
    color: #333 !important;
    font-size: 18px !important;
    font-weight: 600 !important;
}

/* ===== CARDURI ===== */
.card {
    padding: 20px;
    border-radius: 12px;
    background-color: #ffffff;
    border: 1px solid #e6e6e6;
    box-shadow: 0px 2px 6px rgba(0,0,0,0.08);
    margin-bottom: 25px;
}

/* ===== BUTOANE ===== */
button[kind="primary"] {
    background-color: #b30000 !important;
    color: white !important;
    border-radius: 8px !important;
    padding: 10px 22px !important;
    font-size: 16px !important;
    font-weight: 600 !important;
    border: none !important;
}

button[kind="primary"]:hover {
    background-color: #8c0000 !important;
}

/* ===== INPUTURI ===== */
input, select, textarea {
    border-radius: 6px !important;
    border: 1px solid #cccccc !important;
}

/* ===== DATAFRAME ===== */
[data-testid="stDataFrame"] table tbody tr:hover {
    background-color: #f5f5f5 !important;
}

</style>
""", unsafe_allow_html=True)




menu = st.sidebar.selectbox(
    'Meniu Navigare', ['Evaluare Nouă Pacient', 'Istoric Pacienti']
)

if menu== "Evaluare Nouă Pacient":
    # st.title('🫀 Sistem Inteligent de Diagnostic Cardiovascular')
    st.write(
      'Introduceți parametrii clinici ai pacientului pentru a genera o'
      ' predicție și a salva automat datele.'
    )

    with st.form("pacient_form"):
        col1, col2=st.columns(2)

        with col1:
            name=st.text_input("Nume Pacient")
            age = st.number_input('Vârstă (age)', min_value=1, max_value=120, value=50)
            trestbps = st.number_input('Tensiune arterială de repaus (trestbps)', value=120.0)
            chol = st.number_input('Colesterol seric (chol)', value=200.0)
            thalch = st.number_input('Puls maxim atins (thalch)', value=150.0)
            oldpeak = st.number_input('Depresiune segment ST indusa dupa efort (oldpeak)', value=1.0)
            ca = st.number_input('Numar de vase colorate (ca)', min_value=0, max_value=3, value=0)

        with col2:
            sex = st.selectbox('Sex', ['Male', 'Female'])
            # dataset = st.selectbox('Dataset sursă', ['Cleveland', 'Hungary', 'Switzerland', 'Long Beach V'])
            cp = st.selectbox('Tip Dureri Toracice (cp)', ['typical angina', 'atypical angina', 'non-anginal', 'asymptomatic'])
            fbs = st.selectbox('Glicemie a jeun > 120 mg/dl (fbs)', [True, False])
            restecg = st.selectbox('Rezultate ECG repaus (restecg)', ['normal', 'stt abnormality', 'lv hypertrophy'])
            exang = st.selectbox('Angină indusă de efort (exang)', [True, False])
            slope = st.selectbox('Panta segmentului ST (slope)', ['upsloping', 'flat', 'downsloping'])
            thal = st.selectbox('Thalassemia', ['normal', 'fixed defect', 'reversable defect'])
            trestbps_bins = st.selectbox('Inverval Tensiune Arteriala de Repaus (trestbps_bins)', ['hipotensiune', 'normal', 'prehipertensiune', 'hipertensiune_stadiu1', 'hipertensiune_severa'])


        submitted = st.form_submit_button(
            'Rulează Predicția și Salvează în PostgreSQL'
        )

        if submitted:
            if not name.strip():
                st.error("Va rugam sa introduceti numele pacientului.")
            else:
                #construim DataFrame-ul cu atributele pacientului
                input_data=pd.DataFrame({
                    'age':[age],
                    'trestbps':[trestbps],
                    'chol':[chol],
                    'thalch':[thalch],
                    'oldpeak':[oldpeak],
                    'ca':[ca],
                    'sex':[sex],
                    'cp':[cp],
                    'fbs':[fbs],
                    'restecg':[restecg],
                    'exang':[exang],
                    'slope':[slope],
                    'thal':[thal],
                    'trestbps_bins':[trestbps_bins]

                })

                try:
                    cols_to_encode =['sex','cp','fbs','restecg','exang','slope','thal','trestbps_bins']
                    for col in cols_to_encode:
                        if col in input_data.columns:
                            le= encoders[col]
                            input_data[col]= le.transform(input_data[col])

                    numeric_cols= ['age', 'trestbps', 'chol', 'thalch', 'oldpeak', 'ca']

                    input_data[numeric_cols]=scaler.transform(input_data[numeric_cols])
                    input_data[numeric_cols]=yeojohnson.transform(input_data[numeric_cols])

                    feature_names = joblib.load('feature_names.pkl') #ne asiguram ca ordinea coloanelor esre ca cea din setul de date pe care am antrenat

                    input_data = input_data[feature_names]

                    predictie=int(model.predict(input_data)[0]) #modelul returneaza un array ce contine o singura valoare, pe care o accesam si o convertim la int

                    
                    #st.success(f'### Rezultat Diagnostic Estimat: Stadiul {dict_stadii[predictie]}')

                    # culoare în funcție de risc
                    risk_colors = {
                        0: "#d4f4dd",   # verde pastel
                        1: "#fff7cc",   # galben pastel
                        2: "#ffe5cc",   # portocaliu pastel
                        3: "#ffd6d6",   # roșu pastel
                        4: "#f2c2c2"    # vișiniu pastel
                    }

                    risk_icons = {
                        0: "🟢",
                        1: "🟡",
                        2: "🟠",
                        3: "🔴",
                        4: "🔴"
                    }



                    color = risk_colors[predictie]
                    icon=risk_icons[predictie]
                    text_color = "#333" if predictie < 3 else "white"


                    st.markdown(f"""
                    <div style="
                        padding:20px;
                        border-radius:12px;
                        background-color:{color};
                        color:{text_color};
                        box-shadow:0px 2px 6px rgba(0,0,0,0.2);
                        margin-bottom:25px;">
                        <h2 style="margin:0; color:#b30000;">
                            Rezultat Diagnostic {icon}
                        </h2>
                        <p style="font-size:22px; font-weight:600; margin-top:10px;">
                            {dict_stadii[predictie]}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)


                    # col_left, col_right = st.columns([2, 1])

                    # with col_left:
                    #     st.markdown(f"""
                    #     <div class="card">
                    #         <h2>Date Pacient</h2>
                    #         <p><b>Nume:</b> {name}</p>
                    #         <p><b>Vârstă:</b> {age}</p>
                    #         <p><b>Sex:</b> {sex}</p>
                    #         <p><b>Tensiune:</b> {trestbps}</p>
                    #         <p><b>Colesterol:</b> {chol}</p>
                    #         <p><b>Puls:</b> {thalch}</p>
                    #     </div>
                    #     """, unsafe_allow_html=True)

                    # with col_right:
                    #     st.markdown(f"""
                    #     <div class="card">
                    #         <h2>Rezultat Diagnostic</h2>
                    #         <p style="font-size:22px; font-weight:600;">{dict_stadii[predictie]}</p>
                    #     </div>
                    #     """, unsafe_allow_html=True)

                    # st.markdown(f"""
                    #     <div class="card">
                    #     <h2>Rezultat Diagnostic</h2>
                    #     <p style="font-size:22px; font-weight:600;">{dict_stadii[predictie]}</p>
                    #     </div>
                    #     """, unsafe_allow_html=True)


                    #salvare in baza de date
                    db=get_db()
                    inregistrare=Pacient(
                        nume=name,
                        varsta=age,
                        sex=sex,
                        tensiune=trestbps,
                        colesterol=chol,
                        puls=thalch,
                        diagnostic_prezis=predictie,
                        data_inregistrare=datetime.now()

                    )
                    db.add(inregistrare)
                    db.commit()
                    db.close()

                    st.info('Datele pacientului și rezultatul au fost salvate cu succes în'' PostgreSQL!')

                except Exception as e:
                    st.error(f'Eroare la procesarea datelor: {e}')

elif menu == 'Istoric Pacienti':
    st.write('Meniul selectat în acest moment este:', menu)
    st.title('📊 Istoric Pacienți')
    st.write(
      'Vizualizați istoricul pacienților și rezultatele diagnosticului estimate.'
    )

  # Conectare și interogare directă
    db = get_db()
    try:
        pacienti = db.query(Pacient).all()

        if pacienti:
            data = {
                'ID': [p.id for p in pacienti],
                'Nume': [p.nume for p in pacienti],
                'Vârstă': [p.varsta for p in pacienti],
                'Sex': [p.sex for p in pacienti],
                'Tensiune Arterială': [p.tensiune for p in pacienti],
                'Colesterol': [p.colesterol for p in pacienti],
                'Puls': [p.puls for p in pacienti],
                'Diagnostic Prezis': [dict_stadii[p.diagnostic_prezis] for p in pacienti],
                'Data Înregistrare': [p.data_inregistrare for p in pacienti],
            }

            df = pd.DataFrame(data)
            st.dataframe(df)
        else:
            st.info('Nu există înregistrări în baza de date.')

    except Exception as e:
        st.error(f'Eroare la citirea bazei de date: {e}')
    finally:
        db.close()