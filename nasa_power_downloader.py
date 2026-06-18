import streamlit as st
import requests
import csv
import io
import math
import statistics
import urllib3
from datetime import date, timedelta

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# === UPDATE THESE ===
API_HOURLY = "PUT_YOUR_HOURLY_URL_HERE"
API_DAILY = "PUT_YOUR_DAILY_URL_HERE"

PARAMETERS = {
    "Temperature @2m (°C)": ("T2M", True),
    "Relative Humidity @2m (%)": ("RH2M", True),
    "Wind Speed @2m (m/s)": ("WS2M", True),
    "Wind Direction @2m (°)": ("WD2M", True),
    "Precipitation (mm/day) [Daily Only]": ("PRECTOTCORR", False),
}

HEADER_MAP = {
    "T2M": "TEMP_C",
    "RH2M": "RELHUM_%",
    "WS2M": "WS_MPS",
    "WD2M": "WD_DEGREES",
    "WD2M_COMPASS": "WD_CARDINAL",
    "PRECTOTCORR": "PREC_MM"
}

COMMUNITIES = ["AG", "RE", "SB"]
TIME_STANDARDS = ["LST", "UTC"]

# --- UTILITIES ---
def deg_to_compass_16(deg):
    try:
        d = float(deg) % 360.0
    except:
        return ""
    dirs = ["N","NNE","NE","ENE","E","ESE","SE","SSE","S","SSW",
            "SW","WSW","W","WNW","NW","NNW"]
    return dirs[int((d + 11.25) // 22.5) % 16]

def vector_average_degrees(angles):
    if not angles:
        return None
    sin_sum = sum(math.sin(math.radians(a)) for a in angles)
    cos_sum = sum(math.cos(math.radians(a)) for a in angles)
    avg_rad = math.atan2(sin_sum / len(angles), cos_sum / len(angles))
    return math.degrees(avg_rad) % 360

def valid_lat_lon(lat, lon):
    try:
        lat = float(lat); lon = float(lon)
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            return None, None
        return lat, lon
    except:
        return None, None

def iter_clean_rows(csv_text):
    lines = csv_text.splitlines()
    clean = [l for l in lines if l.strip() and not l.startswith("#") and not l.startswith("-END")]

    header = next(csv.reader([clean[0]]))
    rows = [next(csv.reader([l])) for l in clean[1:]]
    return header, rows

def http_get(url, params):
    r = requests.get(url, params=params, verify=False)
    return r.text if r.status_code == 200 else None

# --- STREAMLIT UI ---
st.set_page_config(layout="wide")
st.title("🌦️ NASA POWER Downloader")

col1, col2 = st.columns(2)

with col1:
    lat = st.text_input("Latitude")

with col2:
    lon = st.text_input("Longitude")

today = date.today()
start_date = st.date_input("Start date", value=date(today.year,1,1))
end_date = st.date_input("End date", value=today - timedelta(days=1))

st.subheader("Parameters")
selected = {}
for label, (code, _) in PARAMETERS.items():
    selected[code] = st.checkbox(label, True)

col3, col4 = st.columns(2)
with col3:
    community = st.selectbox("Community", COMMUNITIES)
with col4:
    tstd = st.selectbox("Time Standard", TIME_STANDARDS)

generate_hourly = st.checkbox("Hourly Data", True)
generate_daily = st.checkbox("Daily Stats")

# --- RUN PROCESS ---
if st.button("🚀 Run"):
    lat, lon = valid_lat_lon(lat, lon)

    if lat is None:
        st.error("Invalid coordinates")
        st.stop()

    st.info("Processing...")

    hourly_params = []
    daily_params = []

    for code, val in selected.items():
        if val:
            is_hourly = next(v[1] for k,v in PARAMETERS.items() if v[0]==code)
            if is_hourly:
                hourly_params.append(code)
            else:
                daily_params.append(code)

    q = {
        "parameters": ",".join(hourly_params),
        "community": community,
        "longitude": lon,
        "latitude": lat,
        "start": start_date.strftime("%Y%m%d"),
        "end": end_date.strftime("%Y%m%d"),
        "format": "CSV",
        "time-standard": tstd
    }

    txt = http_get(API_HOURLY, q)

    if not txt:
        st.error("API error")
        st.stop()

    header, rows = iter_clean_rows(txt)

    output = io.StringIO()
    writer = csv.writer(output)

    writer.writerow(header)
    for r in rows:
        writer.writerow(r)

    csv_data = output.getvalue()

    st.success("Done ✅")

    st.download_button(
        "⬇️ Download Hourly Data",
        csv_data,
        "hourly_data.csv",
        "text/csv"
    )
