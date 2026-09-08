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
# 2. FUNGSI PEMROSESAN ECMWF 
# ==========================================
# CATATAN: Cache Streamlit sengaja dimatikan agar tidak menyebabkan Out of Memory (OOM) 
def get_ecmwf_spatial(date_str, target_grib="temp_aifs.grib2"):
    """Mengunduh data ECMWF dan mengembalikan GeoDataFrame tanpa dissolve untuk menghemat RAM"""
    client = Client(source="ecmwf", beta=False)
    
    request = {
        "date": date_str, "time": 0, "step": 162, 
        "stream": "oper", "type": "fc", "levtype": "sfc", 
        "model": "aifs-single", "param": ["tp"], "target": target_grib
    }
    client.retrieve(**request)

    ds = xr.open_dataset(target_grib, engine="cfgrib", backend_kwargs={"indexpath": ""})
    tp = ds["tp"]
    
    # Standarisasi Longitude/Latitude
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

    # Konversi ke DataFrame dan Filter Threshold
    df = subset.to_dataframe(name='tp_mm').reset_index()
    df = df.dropna(subset=['tp_mm'])
    df = df[df['tp_mm'] >= 0.1]

    if df.empty:
        return gpd.GeoDataFrame()

    # Hitung ukuran grid dinamis dan buat geometri box
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

    # Mengembalikan GeoDataFrame mentah tanpa .dissolve() untuk mengurangi beban memori
    return gdf

# ==========================================
# 3. TAMPILAN WEB & LOGIKA UTAMA
# ==========================================
st.set_page_config(page_title="Peta Curah Hujan ECMWF", layout="wide")
st.title("AIFS Single: Total Accumulated Precipitation")

# Menyiapkan folder data agar tidak error jika dijalankan pertama kali
Path("data/pt").mkdir(parents=True, exist_ok=True)

with st.sidebar:
    st.header("Pengaturan Peta")
    selected_date = st.date_input("Tanggal Model Run", datetime.date.today())
    date_str = selected_date.strftime("%Y%m%d")
    
    pt_files = [f.stem for f in Path("data/pt").glob("*.geojson")]
    if not pt_files:
        st.warning("Tambahkan file geojson (misal: THIP.geojson) ke folder data/pt/")
        pt_files = ["Contoh_Area"] 
        
    selected_pt = st.selectbox("Pilih Area / Company", pt_files)
    generate_btn = st.button("Generate Peta", type="primary")

if generate_btn:
    st.info("🔄 Memulai proses... pantau indikator di bawah ini.")
    
    with st.spinner("Memproses data..."):
        
        # 1. Load Data Hujan
        try:
            st.write("📥 Mengunduh dan memproses GRIB ECMWF (Mungkin butuh waktu beberapa saat)...")
            gdf_hujan = get_ecmwf_spatial(date_str)
            st.write("✅ Data ECMWF berhasil diekstrak ke poligon spasial!")
        except Exception as e:
            st.error(f"Gagal memproses data ECMWF: {e}")
            st.stop()

        # 2. Load Batas Negara (Fallback ke Natural Earth jika file lokal tidak ada)
        st.write("🗺️ Menyiapkan peta dasar dan batas wilayah perusahaan...")
        negara_path = Path("data/batas_negara.geojson")
        if negara_path.exists():
            negara_gdf = gpd.read_file(negara_path)
        else:
            negara_gdf = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
            
        # 3. Load Batas PT & Konversi CRS (3857 -> 4326)
        pt_path = Path(f"data/pt/{selected_pt}.geojson")
        if pt_path.exists():
            pt_gdf = gpd.read_file(pt_path)
            # Pastikan CRS disamakan ke EPSG:4326 (WGS84 Lat/Lon)
            if pt_gdf.crs != "EPSG:4326":
                pt_gdf = pt_gdf.to_crs("EPSG:4326")
        else:
            # Fallback dummy poligon jika file tidak ditemukan
            dummy_poly = shapely_box(101.5, 0.5, 102.5, 1.5)
            pt_gdf = gpd.GeoDataFrame(geometry=[dummy_poly], crs="EPSG:4326")

        # ==========================================
        # 4. RENDERING MATPLOTLIB GRIDSPEC
        # ==========================================
        st.write("🎨 Merender layout gambar dengan Matplotlib...")
        
        fig = plt.figure(figsize=(15, 9), facecolor='white')
        gs = GridSpec(2, 2, width_ratios=[2.5, 1], height_ratios=[1, 1], wspace=0.01, hspace=0.01)
        
        ax_main = fig.add_subplot(gs[:, 0])    # Peta Kiri Utama
        ax_reg = fig.add_subplot(gs[0, 1])     # Peta Kanan Atas (Regional)
        ax_loc = fig.add_subplot(gs[1, 1])     # Peta Kanan Bawah (Lokal)
        
        # Format Frame Peta
        for ax in [ax_main, ax_reg, ax_loc]:
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(1.5)

        # 5. Eksekusi Plotting Layer
        if not gdf_hujan.empty:
            gdf_hujan.plot(ax=ax_main, color=gdf_hujan['color'], edgecolor='none')
            gdf_hujan.plot(ax=ax_reg, color=gdf_hujan['color'], edgecolor='none')
            gdf_hujan.plot(ax=ax_loc, color=gdf_hujan['color'], edgecolor='none', alpha=0.5)

        negara_gdf.plot(ax=ax_main, facecolor='none', edgecolor='black', linewidth=0.6)
        negara_gdf.plot(ax=ax_reg, facecolor='none', edgecolor='black', linewidth=0.8)
        
        # Highlight Wilayah PT
        pt_gdf.plot(ax=ax_main, facecolor='none', edgecolor='red', linewidth=1)
        pt_gdf.plot(ax=ax_reg, facecolor='none', edgecolor='red', linewidth=2)
        pt_gdf.plot(ax=ax_loc, facecolor='none', edgecolor='blue', linewidth=2.5)

        # 6. Pengaturan Bounding Box / Kamera Peta
        ax_main.set_xlim(90, 141) # Asia Tenggara
        ax_main.set_ylim(-12, 24)

        # Dapatkan batas bounding box PT
        minx, miny, maxx, maxy = pt_gdf.total_bounds
        
        ax_reg.set_xlim(minx - 5, maxx + 5)
        ax_reg.set_ylim(miny - 5, maxy + 5)
        
        ax_loc.set_xlim(minx - 0.2, maxx + 0.2)
        ax_loc.set_ylim(miny - 0.2, maxy + 0.2)

        # Label Identitas PT di sudut kanan bawah panel lokal
        ax_loc.text(0.95, 0.05, selected_pt.upper().replace("_", " "), 
                    transform=ax_loc.transAxes, fontsize=20, fontweight='bold', 
                    ha='right', va='bottom', bbox=dict(facecolor='white', edgecolor='black', pad=8))

        # 7. Menambahkan Elemen Layout (Legenda & Judul)
        legend_elements = [Patch(facecolor=c, edgecolor='none', label=l) for c, l in zip(COLORS, CATEGORY_LABELS)]
        fig.legend(handles=legend_elements, loc='lower center', ncol=12, frameon=False, 
                   title="AIFS Single: Total accumulated precipitation (kg/m2 (mm))",
                   bbox_to_anchor=(0.5, 0.02), title_fontsize=12, fontsize=10)

        fig.suptitle(f"AIFS Single: Total accumulated precipitation\nBase time: {selected_date} 00 UTC Valid time: +162h Area: South East Asia & Indonesia", 
                     fontsize=14, fontweight='bold', y=0.96)

        # Sisakan area di bawah dan atas untuk legenda/judul
        plt.tight_layout(rect=[0, 0.10, 1, 0.93]) 

        # 8. Render Gambar di Streamlit & Tombol Unduh
        st.write("🎉 Rendering gambar telah selesai!")
        st.pyplot(fig)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
        
        st.download_button(
            label="Unduh Peta (PNG)", 
            data=buf.getvalue(), 
            file_name=f"Curah_Hujan_{selected_pt}_{date_str}.png", 
            mime="image/png"
        )
