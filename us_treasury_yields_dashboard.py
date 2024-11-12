import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import logging
import requests
import io
import urllib3
from typing import Optional

# Configuración de la página
st.set_page_config(
    page_title="US Treasury Yields Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuración del logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('treasury_data.log')
    ]
)

# Suprimir advertencias SSL
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def standardize_date_format(df: pd.DataFrame) -> pd.DataFrame:
    """
    Estandarizar el formato de fecha en el dataframe.
    Maneja múltiples formatos de fecha comunes en los datos del Treasury.
    
    Args:
        df (pd.DataFrame): DataFrame con una columna 'Date'
    
    Returns:
        pd.DataFrame: DataFrame con fechas estandarizadas
    """
    df = df.copy()
    
    # Lista de formatos de fecha posibles
    date_formats = [
        '%m/%d/%Y',  # formato 01/31/2024
        '%Y-%m-%d',  # formato 2024-01-31
        '%d/%m/%Y',  # formato 31/01/2024
        '%m/%d/%y',  # formato 01/31/24
        '%d/%m/%y'   # formato 31/01/24
    ]
    
    def try_parsing_date(date_str):
        """Intenta parsear una fecha usando múltiples formatos"""
        if pd.isna(date_str):
            return None
            
        # Convertir a string si no lo es
        date_str = str(date_str).strip()
        
        for fmt in date_formats:
            try:
                return pd.to_datetime(date_str, format=fmt)
            except (ValueError, TypeError):
                continue
        return None

    # Aplicar la función de parseo a cada fecha
    df['Date'] = df['Date'].apply(try_parsing_date)
    
    # Eliminar filas donde no se pudo parsear la fecha
    invalid_dates = df['Date'].isna().sum()
    if invalid_dates > 0:
        logging.warning(f"Removed {invalid_dates} rows with invalid dates")
        df = df.dropna(subset=['Date'])
    
    return df

def fetch_url(url: str, timeout: int = 30) -> Optional[str]:
    """Obtener datos de una URL con manejo de errores."""
    try:
        with st.spinner(f'Fetching data from {url}...'):
            response = requests.get(url, verify=False, timeout=timeout)
            response.raise_for_status()
            return response.text
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching URL {url}: {e}")
        return None

def download_treasury_data() -> Optional[pd.DataFrame]:
    """Descargar y combinar datos del Treasury."""
    try:
        year_end = datetime.now().year - 1
        historic_url = f'https://home.treasury.gov/system/files/276/yield-curve-rates-1990-{year_end}.csv'
        current_url = f'https://home.treasury.gov/resource-center/data-chart-center/interest-rates/daily-treasury-rates.csv/{datetime.now().year}/all?type=daily_treasury_yield_curve&field_tdr_date_value={datetime.now().year}&page&_format=csv'
        
        st.info("Downloading historical data...")
        historic_data = fetch_url(historic_url)
        if historic_data:
            df_historic = pd.read_csv(io.StringIO(historic_data), on_bad_lines='skip')
            df_historic = standardize_date_format(df_historic)
            st.success(f"Historical data downloaded: {len(df_historic)} records")
        else:
            st.warning("Failed to download historical data")
            df_historic = pd.DataFrame()
        
        st.info("Downloading current year data...")
        current_data = fetch_url(current_url)
        if current_data:
            df_current = pd.read_csv(io.StringIO(current_data), on_bad_lines='skip')
            df_current = standardize_date_format(df_current)
            st.success(f"Current year data downloaded: {len(df_current)} records")
        else:
            st.warning("Failed to download current year data")
            df_current = pd.DataFrame()
        
        df_combined = pd.concat([df_historic, df_current], ignore_index=True)
        
        if df_combined.empty:
            st.error("No data was successfully downloaded")
            return None
        
        df_combined = df_combined.sort_values('Date', ascending=False)
        df_combined = df_combined.drop_duplicates(subset=['Date'], keep='first')
        df_combined = df_combined.reset_index(drop=True)
        
        return df_combined
        
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return None

def create_yield_chart(df: pd.DataFrame, start_date: datetime, selected_series: list) -> go.Figure:
    """Crear el gráfico de yields con Plotly."""
    filtered_df = df[df['Date'] >= start_date]
    
    fig = go.Figure()
    
    # Definir paleta de colores personalizada
    colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
        '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5'
    ]
    
    for i, series in enumerate(selected_series):
        color = colors[i % len(colors)]
        fig.add_trace(go.Scatter(
            x=filtered_df['Date'],
            y=filtered_df[series],
            name=series,
            mode='lines',
            line=dict(width=2, color=color),
            hovertemplate=f"{series}: %{{y:.2f}}%<br>"
        ))
    
    fig.update_layout(
        title={
            'text': 'US Treasury Yield Curves',
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=24)
        },
        xaxis=dict(
            title='Date',
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(211, 211, 211, 0.6)',
            showline=True,
            linewidth=2,
            linecolor='rgba(0, 0, 0, 0.3)',
            mirror=True,
            rangeslider=dict(visible=True),
            tickformat='%d/%m/%Y'
        ),
        yaxis=dict(
            title='Yield (%)',
            ticksuffix='%',
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(211, 211, 211, 0.6)',
            showline=True,
            linewidth=2,
            linecolor='rgba(0, 0, 0, 0.3)',
            mirror=True
        ),
        hovermode='x unified',
        hoverlabel=dict(font_size=12),
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=-0.1,
            xanchor="left",
            x=0,
            orientation="h"
        ),
        margin=dict(l=50, r=50, t=80, b=150),
        template='plotly_white',
        height=700
    )
    
    return fig

def create_date_selector(df, container):
    """Crear selector de fechas personalizado"""
    min_date = df['Date'].min()
    max_date = df['Date'].max()
    
    col1, col2, col3 = container.columns(3)
    
    years = sorted(df['Date'].dt.year.unique(), reverse=True)
    year = col1.selectbox(
        'Year',
        options=years,
        index=len(years)-1
    )
    
    months_df = df[df['Date'].dt.year == year]
    months = sorted(months_df['Date'].dt.month.unique())
    month = col2.selectbox(
        'Month',
        options=months,
        format_func=lambda x: datetime(2000, x, 1).strftime('%B'),
        index=0
    )
    
    days_df = months_df[months_df['Date'].dt.month == month]
    days = sorted(days_df['Date'].dt.day.unique())
    day = col3.selectbox(
        'Day',
        options=days,
        index=0
    )
    
    selected_date = datetime(year, month, day)
    container.text(f"Selected date: {selected_date.strftime('%d/%m/%Y')}")
    
    return selected_date

# Función para guardar el estado de selección de series
def get_selected_series_key():
    return "selected_series"

# Título principal
st.title("US Treasury Yields Dashboard 📈")

# Botón para actualizar datos
col1, col2 = st.columns([1, 5])
with col1:
    if st.button("🔄 Update Data"):
        df = download_treasury_data()
        if df is not None:
            st.session_state['data'] = df
            st.session_state['last_update'] = datetime.now()
            st.success("Data updated successfully!")
with col2:
    if 'last_update' in st.session_state:
        st.info(f"Last update: {st.session_state['last_update'].strftime('%d/%m/%Y %H:%M')}")

# Cargar datos iniciales si no existen
if 'data' not in st.session_state:
    df = download_treasury_data()
    if df is not None:
        st.session_state['data'] = df
        st.session_state['last_update'] = datetime.now()

# Usar los datos almacenados
if 'data' in st.session_state:
    df = st.session_state['data']
    
    # Obtener columnas de tasas
    rate_columns = [col for col in df.columns if col != 'Date']
    
    with st.sidebar:
        # Agregar sección del autor al principio de la barra lateral
        st.markdown("""
            <div class="author-box">
                <p class="author-text">
                    Autor <a href="https://github.com/tuusuario" target="_blank" class="author-name">MKTSignals</a>
                </p>
            </div>
        """, unsafe_allow_html=True)

    # Mostrar información del dataset
    with st.sidebar:
        st.header("Dataset Info")
        st.write(f"Date Range: {df['Date'].min().strftime('%d/%m/%Y')} to {df['Date'].max().strftime('%d/%m/%Y')}")
        st.write(f"Total Records: {len(df):,}")
        st.write(f"Available Series: {len(rate_columns)}")
        
        st.header("Controls")
        
        # Selector de fecha personalizado
        st.subheader("Select Start Date")
        start_date = create_date_selector(df, st)
        
        # Selector de series simplificado
        st.subheader("Series Selection")
        
        # Guardar estado de selección
        if get_selected_series_key() not in st.session_state:
            st.session_state[get_selected_series_key()] = rate_columns
        
        # Botones para seleccionar/deseleccionar todas las series
        col1, col2 = st.columns(2)
        if col1.button("Select All"):
            st.session_state[get_selected_series_key()] = rate_columns
        if col2.button("Deselect All"):
            st.session_state[get_selected_series_key()] = []
        
        selected_series = st.multiselect(
            "Choose yields to display",
            options=rate_columns,
            default=st.session_state[get_selected_series_key()]
        )
        
        # Actualizar estado
        st.session_state[get_selected_series_key()] = selected_series
    
    # Contenido principal
    if selected_series:
        # Crear y mostrar el gráfico
        fig = create_yield_chart(df, start_date, selected_series)
        st.plotly_chart(fig, use_container_width=True)
        
        # Sección de datos
        with st.expander("📊 Data Table", expanded=False):
            filtered_df = df[df['Date'] >= start_date].copy()
            filtered_df = filtered_df[['Date'] + selected_series]
            filtered_df['Date'] = filtered_df['Date'].dt.strftime('%d/%m/%Y')
            
            # Opciones de visualización
            col1, col2 = st.columns(2)
            rows_to_show = col1.slider("Rows to display", 5, 100, 10)
            sort_order = col2.radio("Sort order", ["Newest first", "Oldest first"])
            
            # Ordenar y filtrar datos
            if sort_order == "Newest first":
                filtered_df = filtered_df.iloc[:rows_to_show]
            else:
                filtered_df = filtered_df.iloc[-rows_to_show:].iloc[::-1]
            
            st.dataframe(filtered_df, width=None, height=None)
            
            # Botón de descarga
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="📥 Download data as CSV",
                data=csv,
                file_name=f'treasury_data_{start_date.strftime("%Y%m%d")}.csv',
                mime='text/csv',
            )
    else:
        st.warning("Please select at least one series to display")
else:
    st.error("No data available. Please try downloading the data again.")

# Actualizar los estilos CSS
st.markdown("""
    <style>
        .stSelectbox {
            margin-bottom: 1rem;
        }
        .stAlert {
            margin-top: 1rem;
        }
        .streamlit-expanderHeader {
            font-size: 1.2em;
            font-weight: 500;
        }
        div[data-testid="stSidebarNav"] {
            padding-top: 0;  /* Cambiado para que el autor aparezca arriba */
        }
        button[kind="secondary"] {
            padding: 0.25rem 0.75rem;
        }
        /* Estilos mejorados para la caja del autor */
        .author-box {
            background-color: #f0f2f6;
            border-radius: 10px;
            padding: 1.5rem;
            margin: -1rem -1rem 1.5rem -1rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .author-text {
            color: #262730;
            font-size: 1.2rem;
            font-weight: 500;
            margin: 0;
            line-height: 1.4;
        }
        .author-name {
            color: #1f77b4;
            font-weight: bold;
            text-decoration: none;
            transition: color 0.3s ease;
        }
        .author-name:hover {
            color: #155987;
            text-decoration: underline;
        }
    </style>
""", unsafe_allow_html=True)