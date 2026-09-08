import os
import io
import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import shapely
from shapely.geometry import box as shapely_box
from ecmwf.opendata import Client

import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
import matplotlib.patheffects as pe

# ==========================================
# 1. KONFIGURASI & VARIABEL GLOBAL
# ==========================================
LEVELS = [0.1, 1, 2, 5, 10, 15, 20, 30, 40, 50, 100, 300, 1000]
COLORS = [
    "#e6d7b6", "#a8e89a", "#72e467", "#addbe6", "#78addf", "#4e92df",
    "#2f6ee8", "#e9d96c", "#f7a600", "#ff0d0d", "#a32323", "#ef23ff"
]
CATEGORY_LABELS = [
    "0.1-1", "1-2", "2-5", "5-10", "10-15", "15-20",
    "20-30", "30-40", "40-50", "50-100", "100-300", "300-1000+"
]

# ==========================================
# 2. FUNGSI PEMROSESAN ECMWF (Di-Cache oleh Streamlit)
# ==========================================
@st.cache_data(show_spinner=False)
def get_ecmwf_dissolved(date_str, target_grib="temp_aifs.grib2"):
    """Mengunduh data ECMWF dan mengembalikan GeoDataFrame yang sudah di-dissolve"""
    client = Client(source="ecmwf", beta=False)
    
    # Download data
    request = {
        "date": date_str, "time": 0, "step": 162, 
        "stream": "oper", "type": "fc", "levtype": "sfc", 
        "model": "aifs-single", "param": ["tp"], "target": target_grib
    }
    client.retrieve(**request)

    # Baca GRIB dengan Xarray
    ds = xr.open_dataset(target_grib, engine="cfgrib", backend_kwargs={"indexpath": ""})
    tp = ds["tp"]
    
    if float(tp.longitude.max()) > 180:
        tp = tp.assign_coords(longitude=((tp.longitude + 180) % 360) - 180).sortby("longitude")
    if tp.latitude.values[0] < tp.latitude.values[-1]:
        tp = tp.sortby("latitude", ascending=False)

    units = str(tp.attrs.get("GRIB_units", tp.attrs.get("units", ""))).strip().lower()
    if units in {"m", "meter", "metre", "m of water equivalent"}:
        tp = tp * 1000.0

    # Ekstrak Subset Asia Tenggara / Indonesia
    lon_min, lon_max, lat_min, lat_max = 90, 141, -12, 24
    lat_slice = slice(lat_max, lat_min) if float(tp.latitude.values[0]) > float(tp.latitude.values[-1]) else slice(lat_min, lat_max)
    subset = tp.sel(longitude=slice(lon_min, lon_max), latitude=lat_slice)

    # Konversi ke DataFrame
    df = subset.to_dataframe(name='tp_mm').reset_index()
    df = df.dropna(subset=['tp_mm'])
    df = df[df['tp_mm'] >= 0.1] # Threshold 0.1mm

    if df.empty:
        return gpd.GeoDataFrame()

    # Buat geometri poligon pixel
    dlon = np.median(np.abs(np.diff(subset.longitude.values))) if len(subset.longitude.values) > 1 else 0.25
    dlat = np.median(np.abs(np.diff(subset.latitude.values))) if len(subset.latitude.values) > 1 else 0.25

    geoms = shapely.box(
        df['longitude'] - dlon / 2, df['latitude'] - dlat / 2,
        df['longitude'] + dlon / 2, df['latitude'] + dlat / 2
    )

    # Klasifikasi Warna
    bins = LEVELS[:-1] + [np.inf]
    cat = pd.cut(df['tp_mm'], bins=bins, labels=CATEGORY_LABELS, right=False, include_lowest=True)
    df["cat_lbl"] = cat.astype(str)
    
    lbl_to_color = {lbl: col for lbl, col in zip(CATEGORY_LABELS, COLORS)}
    df["color"] = df["cat_lbl"].map(lbl_to_color)

    gdf = gpd.GeoDataFrame(df, geometry=geoms, crs="EPSG:4326")
    gdf = gdf.clip(shapely_box(lon_min, lat_min, lon_max, lat_max))

    # Dissolve untuk efisiensi rendering Matplotlib
    dis = gdf.dissolve(by="cat_lbl", as_index=False, aggfunc={"color": "first"})
    return dis

# ==========================================
# 3. TAMPILAN WEB & LOGIKA UTAMA
# ==========================================
st.set_page_config(page_title="Peta Curah Hujan ECMWF", layout="wide")
st.title("AIFS Single: Total Accumulated Precipitation")

# Menyiapkan folder data simulasi (agar tidak error jika kosong)
Path("data/pt").mkdir(parents=True, exist_ok=True)

with st.sidebar:
    st.header("Pengaturan Peta")
    selected_date = st.date_input("Tanggal Model Run", datetime.date.today())
    date_str = selected_date.strftime("%Y%m%d")
    
    pt_files = [f.stem for f in Path("data/pt").glob("*.geojson")]
    if not pt_files:
        st.warning("Tambahkan file geojson (misal: THIP.geojson, Region_2.geojson) ke folder data/pt/")
        pt_files = ["Contoh_Area"] # Fallback
        
    selected_pt = st.selectbox("Pilih Area / Company", pt_files)
    generate_btn = st.button("Generate Peta", type="primary")

if generate_btn:
    with st.spinner("Mengunduh GRIB ECMWF dan merender layout..."):
        # 1. Load Data Hujan
        try:
            gdf_hujan = get_ecmwf_dissolved(date_str)
        except Exception as e:
            st.error(f"Gagal mengambil data ECMWF: {e}")
            st.stop()

        # 2. Load Batas Negara (Fallback ke Natural Earth jika file tidak ada)
        negara_path = Path("data/batas_negara.geojson")
        if negara_path.exists():
            negara_gdf = gpd.read_file(negara_path)
        else:
            negara_gdf = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
            
        # 3. Load Batas PT (Buat bounding box dummy jika file tidak ada)
        pt_path = Path(f"data/pt/{selected_pt}.geojson")
        if pt_path.exists():
            pt_gdf = gpd.read_file(pt_path)
        else:
            # Fallback dummy poligon di Riau/Sumatra
            dummy_poly = shapely_box(101.5, 0.5, 102.5, 1.5)
            pt_gdf = gpd.GeoDataFrame(geometry=[dummy_poly], crs="EPSG:4326")

        # ==========================================
        # 4. RENDERING MATPLOTLIB GRIDSPEC
        # ==========================================
        fig = plt.figure(figsize=(15, 9), facecolor='white')
        gs = GridSpec(2, 2, width_ratios=[2.5, 1], height_ratios=[1, 1], wspace=0.01, hspace=0.01)
        
        # Setup Axes
        ax_main = fig.add_subplot(gs[:, 0])    # Kiri full
        ax_reg = fig.add_subplot(gs[0, 1])     # Kanan atas
        ax_loc = fig.add_subplot(gs[1, 1])     # Kanan bawah
        
        axes = [ax_main, ax_reg, ax_loc]
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
            # Border tebal antar panel
            for spine in ax.spines.values():
                spine.set_linewidth(1.5)

        # Plot Data di Ketiga Panel
        if not gdf_hujan.empty:
            gdf_hujan.plot(ax=ax_main, color=gdf_hujan['color'], edgecolor='none')
            gdf_hujan.plot(ax=ax_reg, color=gdf_hujan['color'], edgecolor='none')
            gdf_hujan.plot(ax=ax_loc, color=gdf_hujan['color'], edgecolor='none', alpha=0.5)

        negara_gdf.plot(ax=ax_main, facecolor='none', edgecolor='black', linewidth=0.6)
        negara_gdf.plot(ax=ax_reg, facecolor='none', edgecolor='black', linewidth=0.8)
        
        # Highlight PT di semua panel
        pt_gdf.plot(ax=ax_main, facecolor='none', edgecolor='red', linewidth=1)
        pt_gdf.plot(ax=ax_reg, facecolor='none', edgecolor='red', linewidth=2)
        pt_gdf.plot(ax=ax_loc, facecolor='none', edgecolor='blue', linewidth=2.5)

        # Atur Extent (Bounding Box)
        # Main Map: SE Asia
        ax_main.set_xlim(90, 141)
        ax_main.set_ylim(-12, 24)

        # Menghitung extent dinamis berdasarkan PT
        minx, miny, maxx, maxy = pt_gdf.total_bounds
        
        # Regional Map: Buffer ~5 derajat
        ax_reg.set_xlim(minx - 5, maxx + 5)
        ax_reg.set_ylim(miny - 5, maxy + 5)
        
        # Local Map: Buffer ~0.2 derajat
        ax_loc.set_xlim(minx - 0.2, maxx + 0.2)
        ax_loc.set_ylim(miny - 0.2, maxy + 0.2)

        # Watermark/Label PT di panel lokal
        txt = ax_loc.text(0.95, 0.05, selected_pt.upper().replace("_", " "), 
                    transform=ax_loc.transAxes, fontsize=20, fontweight='bold', 
                    ha='right', va='bottom', bbox=dict(facecolor='white', edgecolor='black', pad=8))

        # Legenda di bawah (Fig Legend)
        legend_elements = [Patch(facecolor=c, edgecolor='none', label=l) for c, l in zip(COLORS, CATEGORY_LABELS)]
        fig.legend(handles=legend_elements, loc='lower center', ncol=12, frameon=False, 
                   title="AIFS Single: Total accumulated precipitation (kg/m2 (mm))",
                   bbox_to_anchor=(0.5, 0.02), title_fontsize=12, fontsize=10)

        # Judul Peta
        fig.suptitle(f"AIFS Single: Total accumulated precipitation\nBase time: {selected_date} 00 UTC Valid time: +162h Area: South East Asia & Indonesia", 
                     fontsize=14, fontweight='bold', y=0.96)

        plt.tight_layout(rect=[0, 0.10, 1, 0.93]) # Ruang untuk legenda dan judul

        # Render di Streamlit
        st.pyplot(fig)

        # Siapkan file untuk di-download
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
        
        st.download_button(
            label="Unduh Peta (PNG Resolusi Tinggi)", 
            data=buf.getvalue(), 
            file_name=f"Curah_Hujan_{selected_pt}_{date_str}.png", 
            mime="image/png"
        )
