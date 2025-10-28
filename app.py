import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os

# Konfigurasi halaman Streamlit
st.set_page_config(
    page_title="Pinpoint Map Restoran Jakarta - Full Data",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# KONFIGURASI & UTILITIES
# =============================================================================
class Config:
    JAKARTA_BOUNDS = {
        'min_lat': -6.45, 'max_lat': -6.08,
        'min_lon': 106.60, 'max_lon': 107.10
    }
    INITIAL_VIEW_STATE = pdk.ViewState(
        latitude=-6.2088,
        longitude=106.8456,
        zoom=10,
        pitch=0,
        bearing=0
    )

def validate_coordinates(lat, lon):
    """Validasi koordinat untuk memastikan dalam range Jakarta"""
    if pd.isna(lat) or pd.isna(lon):
        return False
    
    if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
        return False
    
    if not (Config.JAKARTA_BOUNDS['min_lat'] <= lat <= Config.JAKARTA_BOUNDS['max_lat']):
        return False
    if not (Config.JAKARTA_BOUNDS['min_lon'] <= lon <= Config.JAKARTA_BOUNDS['max_lon']):
        return False
    
    return True

def clean_coordinates(df, lat_col, lon_col, dataset_name):
    """Bersihkan dan validasi koordinat"""
    if df.empty:
        return df
        
    df_clean = df.dropna(subset=[lat_col, lon_col]).copy()
    
    if df_clean.empty:
        return df_clean
    
    valid_coords_mask = df_clean.apply(
        lambda row: validate_coordinates(row[lat_col], row[lon_col]), axis=1
    )
    df_valid = df_clean[valid_coords_mask].copy()
    
    return df_valid

# =============================================================================
# LOAD DATA DENGAN CACHING
# =============================================================================
@st.cache_data(show_spinner=False, ttl=3600)
def load_and_process_data(matched_file, esb_file, jakarta_file):
    """Load dan proses SEMUA data tanpa sampling"""
    
    try:
        df_matched = pd.read_csv(matched_file)
        df_esb_full = pd.read_csv(esb_file)
        df_jakarta_full = pd.read_csv(jakarta_file)
        
        st.info(f"📥 Data loaded - Matched: {len(df_matched):,}, ESB: {len(df_esb_full):,}, Jakarta: {len(df_jakarta_full):,}")
        
    except Exception as e:
        st.error(f"❌ Error loading files: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    # Standardize column names
    try:
        esb_columns = df_esb_full.columns.tolist()
        if 'longitude' in esb_columns and 'latitude' in esb_columns:
            df_esb_full = df_esb_full.rename(columns={'longitude': 'lon', 'latitude': 'lat'})
            
        jakarta_columns = df_jakarta_full.columns.tolist()
        rename_dict = {}
        if 'longitude' in jakarta_columns:
            rename_dict['longitude'] = 'lon'
        if 'latitude' in jakarta_columns:
            rename_dict['latitude'] = 'lat'
        if 'Nama Restoran' in jakarta_columns:
            rename_dict['Nama Restoran'] = 'nama_restoran'
            
        if rename_dict:
            df_jakarta_full = df_jakarta_full.rename(columns=rename_dict)
            
    except Exception as e:
        st.error(f"❌ Error standardizing columns: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    # Cleaning koordinat
    df_matched_clean = clean_coordinates(df_matched, 'latitude_esb', 'longitude_esb', "Matched Data")
    df_esb_clean = clean_coordinates(df_esb_full, 'lat', 'lon', "ESB Full Data")
    df_jakarta_clean = clean_coordinates(df_jakarta_full, 'lat', 'lon', "Jakarta Full Data")
    
    st.info(f"🧹 After cleaning - Matched: {len(df_matched_clean):,}, ESB: {len(df_esb_clean):,}, Jakarta: {len(df_jakarta_clean):,}")
    
    # Prepare data untuk visualisasi
    green_data = pd.DataFrame()
    orange_data = pd.DataFrame()
    blue_data = pd.DataFrame()
    
    # Data HIJAU (Matched)
    if not df_matched_clean.empty:
        try:
            green_data = df_matched_clean[[
                'brandName_esb', 'branchName_esb', 'latitude_esb', 'longitude_esb'
            ]].copy()
            green_data = green_data.rename(columns={
                'brandName_esb': 'nama_restoran',
                'branchName_esb': 'cabang',
                'latitude_esb': 'lat',
                'longitude_esb': 'lon'
            })
            
            if 'name_similarity' in df_matched_clean.columns:
                green_data['name_similarity'] = df_matched_clean['name_similarity']
            if 'match_confidence' in df_matched_clean.columns:
                green_data['match_confidence'] = df_matched_clean['match_confidence']
                
            green_data['kategori'] = 'Match'
            green_data['warna'] = [[0, 255, 0, 180] for _ in range(len(green_data))]
            
        except Exception as e:
            st.error(f"❌ Error processing green data: {e}")
    
    # Data ORANGE (Hanya ESB)
    if not df_esb_clean.empty:
        try:
            if not green_data.empty:
                # Cari brand yang ada di ESB tapi tidak di matched
                esb_brands = set(df_esb_clean['brandName'].unique())
                matched_brands = set(green_data['nama_restoran'].unique())
                unmatched_brands = esb_brands - matched_brands
                esb_unmatched = df_esb_clean[df_esb_clean['brandName'].isin(unmatched_brands)]
            else:
                esb_unmatched = df_esb_clean
                
            orange_data = esb_unmatched[['brandName', 'branchName', 'lat', 'lon']].copy()
            orange_data = orange_data.rename(columns={
                'brandName': 'nama_restoran',
                'branchName': 'cabang'
            })
            
            if 'cityName' in esb_unmatched.columns:
                orange_data['cityName'] = esb_unmatched['cityName']
                
            orange_data['kategori'] = 'Hanya ESB'
            orange_data['warna'] = [[255, 165, 0, 180] for _ in range(len(orange_data))]
            
        except Exception as e:
            st.error(f"❌ Error processing orange data: {e}")
    
    # Data BIRU (Hanya Jakarta)
    if not df_jakarta_clean.empty:
        try:
            if not green_data.empty:
                # Cari restoran yang ada di Jakarta tapi tidak di matched
                jakarta_restaurants = set(df_jakarta_clean['nama_restoran'].unique())
                matched_restaurants = set(green_data['nama_restoran'].unique())
                unmatched_restaurants = jakarta_restaurants - matched_restaurants
                jakarta_unmatched = df_jakarta_clean[df_jakarta_clean['nama_restoran'].isin(unmatched_restaurants)]
            else:
                jakarta_unmatched = df_jakarta_clean
                
            blue_data = jakarta_unmatched[['nama_restoran', 'lat', 'lon']].copy()
            
            if 'Pricing' in jakarta_unmatched.columns:
                blue_data['Pricing'] = jakarta_unmatched['Pricing']
                
            blue_data['cabang'] = ''
            blue_data['cityName'] = ''
            blue_data['kategori'] = 'Hanya Jakarta'
            blue_data['warna'] = [[0, 0, 255, 180] for _ in range(len(blue_data))]
            
        except Exception as e:
            st.error(f"❌ Error processing blue data: {e}")
    
    st.success(f"✅ Data processing complete - Green: {len(green_data):,}, Orange: {len(orange_data):,}, Blue: {len(blue_data):,}")
    return green_data, orange_data, blue_data

# =============================================================================
# VISUALISASI PETA DENGAN PYDECK - DIPERBAIKI
# =============================================================================
def create_deck_map(green_data, orange_data, blue_data, show_layers, map_style, performance_mode=False):
    """Buat peta interaktif dengan PyDeck - FIXED untuk peta hitam"""
    
    # Gabungkan data berdasarkan layer yang aktif
    layers_data = []
    
    if show_layers['match'] and not green_data.empty:
        green_data_copy = green_data.copy()
        green_data_copy['layer'] = 'match'
        layers_data.append(green_data_copy)
        
    if show_layers['esb'] and not orange_data.empty:
        orange_data_copy = orange_data.copy()
        orange_data_copy['layer'] = 'esb'
        layers_data.append(orange_data_copy)
        
    if show_layers['jakarta'] and not blue_data.empty:
        blue_data_copy = blue_data.copy()
        blue_data_copy['layer'] = 'jakarta'
        layers_data.append(blue_data_copy)
    
    if not layers_data:
        st.warning("⚠️ Tidak ada data yang ditampilkan. Silakan pilih layer di sidebar.")
        return None
    
    try:
        combined_data = pd.concat(layers_data, ignore_index=True)
        
        if combined_data.empty:
            st.warning("⚠️ Tidak ada data yang valid untuk ditampilkan.")
            return None
            
        # Pastikan kolom yang diperlukan ada
        required_columns = ['lon', 'lat', 'warna', 'nama_restoran', 'kategori']
        for col in required_columns:
            if col not in combined_data.columns:
                st.error(f"❌ Kolom {col} tidak ditemukan dalam data")
                return None
        
        total_points = len(combined_data)
        
        # OPTIMISASI: Sesuaikan parameter berdasarkan jumlah data dan performance mode
        if performance_mode or total_points > 5000:
            # Mode performa untuk data besar
            radius_min_pixels = 1
            radius_max_pixels = 4
            opacity = 0.6
            radius_scale = 3
        else:
            # Mode normal untuk data kecil
            radius_min_pixels = 2
            radius_max_pixels = 6
            opacity = 0.8
            radius_scale = 4
        
        # PERBAIKAN UTAMA: Gunakan ScatterplotLayer dengan konfigurasi yang benar
        layer = pdk.Layer(
            "ScatterplotLayer",
            combined_data,
            pickable=True,
            opacity=opacity,
            stroked=True,
            filled=True,
            radius_scale=radius_scale,
            radius_min_pixels=radius_min_pixels,
            radius_max_pixels=radius_max_pixels,
            line_width_min_pixels=0.5,
            get_position=['lon', 'lat'],
            get_fill_color='warna',
            get_line_color=[0, 0, 0, 128],
            auto_highlight=True,
        )
        
        # Tooltip yang informatif - DIPERBAIKI
        tooltip = {
            "html": """
            <div style="
                background: white; 
                color: black; 
                padding: 10px; 
                border-radius: 5px; 
                border: 1px solid #ccc;
                font-size: 12px;
                max-width: 300px;
            ">
                <b style="color: {kategori_color};">{nama_restoran}</b><br/>
                <hr style="margin: 5px 0;">
                <b>Kategori:</b> {kategori}<br/>
                <b>Koordinat:</b> [{lat:.4f}, {lon:.4f}]<br/>
                {% if cabang and cabang != "" %}
                <b>Cabang:</b> {cabang}<br/>
                {% endif %}
                {% if name_similarity %}
                <b>Similarity:</b> {name_similarity:.3f}<br/>
                {% endif %}
                {% if match_confidence %}
                <b>Confidence:</b> {match_confidence:.3f}<br/>
                {% endif %}
            </div>
            """,
            "style": {
                "backgroundColor": "steelblue",
                "color": "white"
            }
        }
        
        # PERBAIKAN PENTING: Konfigurasi view state dan map style
        view_state = pdk.ViewState(
            **Config.INITIAL_VIEW_STATE.to_json()
        )
        
        # PERBAIKAN: Gunakan map style yang kompatibel dengan PyDeck
        # PyDeck mendukung: 'light', 'dark', 'road', 'satellite', 'dark_no_labels', etc.
        compatible_map_styles = {
            "light": "light",
            "dark": "dark", 
            "satellite": "satellite",
            "road": "road",
            "outdoors": "light"  # Fallback untuk outdoors
        }
        
        selected_map_style = compatible_map_styles.get(map_style, "light")
        
        # Buat peta dengan konfigurasi yang diperbaiki
        deck = pdk.Deck(
            layers=[layer],
            initial_view_state=view_state,
            tooltip=tooltip,
            map_style=selected_map_style,
            parameters={
                "pickable": True,
                "doubleClickZoom": True,
                "scrollZoom": True,
                "dragRotate": False
            }
        )
        
        return deck
        
    except Exception as e:
        st.error(f"❌ Error creating map: {e}")
        import traceback
        st.error(f"Detail error: {traceback.format_exc()}")
        return None

# =============================================================================
# ANALISIS STATISTIK LENGKAP
# =============================================================================
def create_comprehensive_statistics(green_data, orange_data, blue_data):
    """Buat analisis statistik komprehensif dengan SEMUA data"""
    
    total_green = len(green_data) if not green_data.empty else 0
    total_orange = len(orange_data) if not orange_data.empty else 0
    total_blue = len(blue_data) if not blue_data.empty else 0
    total_all = total_green + total_orange + total_blue
    
    if total_all == 0:
        empty_fig = go.Figure()
        empty_fig.add_annotation(text="Tidak ada data untuk ditampilkan", 
                               xref="paper", yref="paper", x=0.5, y=0.5, 
                               showarrow=False, font_size=20)
        empty_fig.update_layout(width=400, height=300)
        return empty_fig, empty_fig, empty_fig, empty_fig, 0, 0, 0, 0
    
    # Pie chart distribusi
    sizes = [total_green, total_orange, total_blue]
    labels = [f'Match ({total_green:,})', f'Hanya ESB ({total_orange:,})', f'Hanya Jakarta ({total_blue:,})']
    
    fig_pie = px.pie(
        values=sizes,
        names=labels,
        title='Distribusi Data Restoran (Semua Data)',
        color=labels,
        color_discrete_map={
            labels[0]: '#00ff00',
            labels[1]: '#ffa500', 
            labels[2]: '#0000ff'
        }
    )
    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
    
    # Bar chart
    categories = ['Match', 'Hanya ESB', 'Hanya Jakarta']
    counts = [total_green, total_orange, total_blue]
    
    fig_bar = go.Figure(data=[
        go.Bar(
            x=categories, 
            y=counts,
            marker_color=['#00ff00', '#ffa500', '#0000ff'],
            text=[f'{x:,}' for x in counts],
            textposition='auto',
        )
    ])
    fig_bar.update_layout(
        title='Perbandingan Kategori Data (Semua Data)',
        xaxis_title='Kategori',
        yaxis_title='Jumlah Restoran'
    )
    
    # Similarity Distribution
    fig_similarity = go.Figure()
    if not green_data.empty and 'name_similarity' in green_data.columns:
        similarity_data = green_data['name_similarity'].dropna()
        if len(similarity_data) > 0:
            fig_similarity.add_trace(go.Histogram(
                x=similarity_data,
                nbinsx=20,
                name='Similarity Score',
                marker_color='#00ff00',
                opacity=0.7
            ))
            fig_similarity.update_layout(
                title='Distribusi Similarity Score (Data Match)',
                xaxis_title='Similarity Score',
                yaxis_title='Frequency'
            )
        else:
            fig_similarity.add_annotation(text="Tidak ada data similarity", 
                                       xref="paper", yref="paper", x=0.5, y=0.5, 
                                       showarrow=False, font_size=16)
            fig_similarity.update_layout(title='Distribusi Similarity Score')
    else:
        fig_similarity.add_annotation(text="Tidak ada data match", 
                                   xref="paper", yref="paper", x=0.5, y=0.5, 
                                   showarrow=False, font_size=16)
        fig_similarity.update_layout(title='Distribusi Similarity Score')
    
    # Top Restoran
    fig_top_restaurants = go.Figure()
    all_restaurants = pd.concat([green_data, orange_data, blue_data], ignore_index=True)
    top_restaurants = all_restaurants['nama_restoran'].value_counts().head(15)
    
    if len(top_restaurants) > 0:
        fig_top_restaurants.add_trace(go.Bar(
            y=top_restaurants.index,
            x=top_restaurants.values,
            orientation='h',
            marker_color='lightblue',
            text=top_restaurants.values,
            textposition='auto',
        ))
        fig_top_restaurants.update_layout(
            title='Top 15 Restoran Berdasarkan Jumlah Lokasi',
            xaxis_title='Jumlah Lokasi',
            yaxis_title='Nama Restoran'
        )
    else:
        fig_top_restaurants.add_annotation(text="Tidak ada data restoran", 
                                        xref="paper", yref="paper", x=0.5, y=0.5, 
                                        showarrow=False, font_size=16)
        fig_top_restaurants.update_layout(title='Top Restoran')
    
    return fig_pie, fig_bar, fig_similarity, fig_top_restaurants, total_green, total_orange, total_blue, total_all

# =============================================================================
# MAIN APP - DIPERBAIKI
# =============================================================================
def main():
    st.title("🗺️ Pinpoint Map Restoran Jakarta - Full Data Analysis")
    st.markdown("Visualisasi interaktif data restoran Jakarta dengan **SEMUA DATA** - tanpa batasan sampling")
    
    # Inisialisasi session state
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'green_data' not in st.session_state:
        st.session_state.green_data = pd.DataFrame()
    if 'orange_data' not in st.session_state:
        st.session_state.orange_data = pd.DataFrame()
    if 'blue_data' not in st.session_state:
        st.session_state.blue_data = pd.DataFrame()
    
    # Sidebar untuk kontrol
    st.sidebar.header("🎛️ Kontrol Visualisasi")
    
    # File uploader atau path input
    st.sidebar.subheader("📁 Input Data")
    data_source = st.sidebar.radio(
        "Sumber Data:",
        ["File Upload", "Path Local"]
    )
    
    matched_file = None
    esb_file = None
    jakarta_file = None
    
    if data_source == "File Upload":
        matched_file = st.sidebar.file_uploader("Data Matched", type=['csv'], key="matched")
        esb_file = st.sidebar.file_uploader("Data ESB", type=['csv'], key="esb")
        jakarta_file = st.sidebar.file_uploader("Data Jakarta", type=['csv'], key="jakarta")
    else:
        matched_file = st.sidebar.text_input("Path Data Matched", "/kaggle/input/final-dataset/esb_jakarta_matched_comprehensive.csv")
        esb_file = st.sidebar.text_input("Path Data ESB", "/kaggle/input/dataset-jakarta-compared/restaurant_esb_baru.csv")
        jakarta_file = st.sidebar.text_input("Path Data Jakarta", "/kaggle/input/dataset-jakarta-compared/restaurants_jakarta.csv")
    
    # Kontrol layer
    st.sidebar.subheader("👁️ Kontrol Layer")
    show_layers = {
        'match': st.sidebar.checkbox("✅ Data Match (Hijau)", True),
        'esb': st.sidebar.checkbox("🟠 Hanya ESB (Orange)", True),
        'jakarta': st.sidebar.checkbox("🔵 Hanya Jakarta (Biru)", True)
    }
    
    # Pengaturan peta - DIPERBAIKI
    st.sidebar.subheader("🗺️ Pengaturan Peta")
    map_style = st.sidebar.selectbox(
        "Style Peta:",
        ["light", "dark", "satellite", "road"]
    )
    
    # PERBAIKAN: Tambahkan toggle untuk mode performa
    performance_mode = st.sidebar.checkbox(
        "🚀 Mode Performa (untuk data besar)", 
        value=False,
        help="Mengurangi detail visual untuk meningkatkan performa saat menampilkan data besar"
    )
    
    # Load data
    if st.sidebar.button("🚀 Muat SEMUA Data & Generate Visualisasi", type="primary"):
        if not all([matched_file, esb_file, jakarta_file]):
            st.error("❌ Harap lengkapi semua file data!")
            return
            
        with st.spinner("🔄 Memuat dan memproses SEMUA data..."):
            try:
                green_data, orange_data, blue_data = load_and_process_data(
                    matched_file, esb_file, jakarta_file
                )
                
                st.session_state.green_data = green_data
                st.session_state.orange_data = orange_data
                st.session_state.blue_data = blue_data
                st.session_state.data_loaded = True
                
                st.success(f"✅ SEMUA data berhasil dimuat! Total: {len(green_data) + len(orange_data) + len(blue_data):,} records")
                
            except Exception as e:
                st.error(f"❌ Error memuat data: {str(e)}")
                return
    
    if not st.session_state.get('data_loaded', False):
        st.info("👆 Silakan upload file data dan klik 'Muat SEMUA Data & Generate Visualisasi' di sidebar untuk memulai.")
        return
    
    # Tampilkan statistik
    st.header("📊 Analisis Statistik Komprehensif (Semua Data)")
    
    fig_pie, fig_bar, fig_similarity, fig_top_restaurants, total_green, total_orange, total_blue, total_all = create_comprehensive_statistics(
        st.session_state.green_data, 
        st.session_state.orange_data, 
        st.session_state.blue_data
    )
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("✅ Data Match", f"{total_green:,}")
    with col2:
        st.metric("🟠 Hanya ESB", f"{total_orange:,}")
    with col3:
        st.metric("🔵 Hanya Jakarta", f"{total_blue:,}")
    with col4:
        st.metric("📊 Total Semua Data", f"{total_all:,}")
    
    # Charts
    st.subheader("📈 Visualisasi Distribusi")
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig_pie, use_container_width=True)
    with col2:
        st.plotly_chart(fig_bar, use_container_width=True)
    
    st.subheader("🔍 Analisis Detail")
    col3, col4 = st.columns(2)
    with col3:
        st.plotly_chart(fig_similarity, use_container_width=True)
    with col4:
        st.plotly_chart(fig_top_restaurants, use_container_width=True)
    
    # Tampilkan peta - DIPERBAIKI
    st.header("🗺️ Peta Interaktif (Semua Data)")
    
    total_points = len(st.session_state.green_data) + len(st.session_state.orange_data) + len(st.session_state.blue_data)
    
    # PERBAIKAN: Informasi yang lebih jelas tentang mode yang aktif
    if performance_mode:
        st.warning(f"🚀 **Mode Performa Aktif** - Menampilkan {total_points:,} titik data dengan optimasi. Klik marker untuk detail.")
    else:
        st.info(f"🎯 **Mode Normal** - Menampilkan {total_points:,} titik data. Klik marker untuk detail.")
    
    # Buat dan tampilkan peta - DIPERBAIKI
    deck_map = create_deck_map(
        st.session_state.green_data,
        st.session_state.orange_data, 
        st.session_state.blue_data,
        show_layers,
        map_style,  # Langsung gunakan string style
        performance_mode
    )
    
    if deck_map:
        st.pydeck_chart(deck_map, use_container_width=True)
        
        # Legenda yang lebih informatif
        st.markdown(f"""
        ### 🎯 Legenda & Cara Penggunaan
        
        **Kategori Data:**
        - ✅ **Hijau**: Restoran yang ada di kedua dataset (Match) - **{len(st.session_state.green_data):,}** data
        - 🟠 **Orange**: Restoran yang hanya ada di dataset ESB - **{len(st.session_state.orange_data):,}** data  
        - 🔵 **Biru**: Restoran yang hanya ada di dataset Jakarta - **{len(st.session_state.blue_data):,}** data
        
        **Cara Interaksi:**
        - 🖱️ **Klik marker** untuk melihat detail informasi restoran
        - 🔍 **Zoom** dengan scroll mouse
        - 🗺️ **Geser** dengan drag mouse
        - 👁️ **Sembunyikan/tampilkan layer** menggunakan kontrol di sidebar
        
        **Statistik:**
        - 📊 Total titik data: **{total_points:,}**
        - 🚀 Mode: **{'Performansi' if performance_mode else 'Normal'}**
        - 🗺️ Style peta: **{map_style}**
        """)
    else:
        st.error("❌ Gagal membuat peta. Pastikan data memiliki koordinat yang valid.")
    
    # Data tables
    st.header("📋 Detail Data (Semua Data)")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Summary", "✅ Data Match", "🟠 Hanya ESB", "🔵 Hanya Jakarta"])
    
    with tab1:
        st.subheader("Ringkasan Statistik")
        summary_data = {
            'Kategori': ['Match', 'Hanya ESB', 'Hanya Jakarta', 'Total'],
            'Jumlah': [total_green, total_orange, total_blue, total_all],
            'Persentase': [
                f"{(total_green/total_all)*100:.2f}%" if total_all > 0 else "0%",
                f"{(total_orange/total_all)*100:.2f}%" if total_all > 0 else "0%", 
                f"{(total_blue/total_all)*100:.2f}%" if total_all > 0 else "0%",
                '100%'
            ]
        }
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True)
    
    with tab2:
        if not st.session_state.green_data.empty:
            st.dataframe(st.session_state.green_data, use_container_width=True)
        else:
            st.info("Tidak ada data Match")
    
    with tab3:
        if not st.session_state.orange_data.empty:
            st.dataframe(st.session_state.orange_data, use_container_width=True)
        else:
            st.info("Tidak ada data Hanya ESB")
    
    with tab4:
        if not st.session_state.blue_data.empty:
            st.dataframe(st.session_state.blue_data, use_container_width=True)
        else:
            st.info("Tidak ada data Hanya Jakarta")
    
    # Informasi sistem
    st.sidebar.markdown("---")
    st.sidebar.subheader("ℹ️ Informasi Sistem")
    
    total_displayed = sum([
        len(st.session_state.green_data) if show_layers['match'] else 0,
        len(st.session_state.orange_data) if show_layers['esb'] else 0,
        len(st.session_state.blue_data) if show_layers['jakarta'] else 0
    ])
    
    st.sidebar.info(f"""
    **Status Data:**
    - ✅ Match: {len(st.session_state.green_data):,}
    - 🟠 ESB: {len(st.session_state.orange_data):,}  
    - 🔵 Jakarta: {len(st.session_state.blue_data):,}
    - 🎯 Total: {total_all:,}
    
    **Peta:**
    - 👁️ Ditampilkan: {total_displayed:,}
    - 🚀 Mode: {'Performansi' if performance_mode else 'Normal'}
    - 🗺️ Style: {map_style}
    """)

if __name__ == "__main__":
    main()
