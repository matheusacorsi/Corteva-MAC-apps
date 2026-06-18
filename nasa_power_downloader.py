import streamlit as st
import requests
import csv
import io
import math
import statistics
import time
import urllib3
from datetime import date, timedelta, datetime
import pandas as pd

# Suppress SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# API Endpoints
API_HOURLY = "https://power.larc.nasa.gov/api/temporal/hourly/point"
API_DAILY  = "https://power.larc.nasa.gov/api/temporal/daily/point"

# === Metric Configuration ===
PARAMETERS = {
    "Temperature @2m (°C)": ("T2M", True),
    "Relative Humidity @2m (%)": ("RH2M", True),
    "Wind Speed @2m (m/s)": ("WS2M", True),
    "Wind Direction @2m (°)": ("WD2M", True),
    "Precipitation (mm/day) [Daily Only]": ("PRECTOTCORR", False), 
}

# === HEADER RENAMING MAP ===
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

# --- Utilities ---
def deg_to_compass_16(deg):
    try:
        d = float(deg) % 360.0
    except (TypeError, ValueError):
        return ""
    dirs = ["N","NNE","NE","ENE","E","ESE","SE","SSE","S","SSW","SW","WSW","W","WNW","NW","NNW"]
    return dirs[int((d + 11.25) // 22.5) % 16]

def vector_average_degrees(angles):
    if not angles: return None
    sin_sum = sum(math.sin(math.radians(a)) for a in angles)
    cos_sum = sum(math.cos(math.radians(a)) for a in angles)
    avg_rad = math.atan2(sin_sum / len(angles), cos_sum / len(angles))
    avg_deg = math.degrees(avg_rad)
    return avg_deg % 360.0

def valid_lat_lon(lat_str, lon_str):
    try:
        lat = float(lat_str); lon = float(lon_str)
        if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
            return None, None
        return lat, lon
    except ValueError:
        return None, None

def year_chunks(start_d: date, end_d: date):
    chunks = []
    cur_start = start_d
    while cur_start <= end_d:
        year_end = date(cur_start.year, 12, 31)
        cur_end = min(year_end, end_d)
        chunks.append((cur_start, cur_end))
        cur_start = cur_end + timedelta(days=1)
    return chunks

def http_get_with_retries(url, params, timeout=90, max_retries=4, backoff=2.0, verify=False):
    headers = {"User-Agent": "NASA-POWER-Downloader/1.0"}
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(url, params=params, timeout=timeout, headers=headers, verify=verify)
            if resp.status_code == 200:
                return True, resp
            msg = f"HTTP {resp.status_code}"
            if resp.status_code in (429, 500, 502, 503, 504) and attempt < max_retries:
                time.sleep(backoff ** attempt)
                continue
            return False, msg
        except Exception as e:
            msg = f"Error: {e}"
        if attempt < max_retries:
            time.sleep(backoff ** attempt)
            continue
        return False, msg

def iter_clean_rows(csv_text):
    lines = csv_text.splitlines()
    clean_lines = [line for line in lines if line.strip() and not line.strip().startswith("#") and not line.strip().startswith("-END")]
    if not clean_lines: return None, []
    header_idx = -1
    header_row = []
    for i, line in enumerate(clean_lines):
        row = next(csv.reader(io.StringIO(line)))
        row_upper = [c.upper().strip() for c in row]
        if any(x in row_upper for x in ["YEAR", "YYYY", "YR"]):
            header_idx = i; header_row = row_upper; break
    if header_idx == -1: return None, []
    data_rows = []
    for line in clean_lines[header_idx+1:]:
        row = next(csv.reader(io.StringIO(line)))
        if row: data_rows.append(row)
    return header_row, data_rows

def get_precip_sum(start_d, end_d, data_dict):
    total = 0.0
    cur_d = start_d
    while cur_d <= end_d:
        dt_str = cur_d.strftime("%Y%m%d")
        val = data_dict.get(dt_str, {}).get("PRECTOTCORR", 0.0)
        if val != -999.0 and val is not None:
            total += val
        cur_d += timedelta(days=1)
    return round(total, 2)

def to_arm_date(date_key):
    d = datetime.strptime(date_key, "%Y%m%d")
    return d.strftime("%d%b%y").lstrip("0")

# --- Streamlit UI Setup ---
st.set_page_config(page_title="NASA POWER Downloader", layout="wide", page_icon="🌦️")

# Custom CSS for Bottom-Right NASA Rights
st.markdown(
    """
    <style>
    .nasa-footer {
        position: fixed;
        bottom: 10px;
        right: 15px;
        font-size: 11px;
        color: #888888;
        background-color: rgba(255, 255, 255, 0.8);
        padding: 5px 10px;
        border-radius: 5px;
        text-align: right;
        z-index: 100;
        max-width: 350px;
        pointer-events: none;
    }
    @media (prefers-color-scheme: dark) {
        .nasa-footer {
            background-color: rgba(14, 17, 23, 0.8);
            color: #aaaaaa;
        }
    }
    </style>
    <div class="nasa-footer">
        <b>NASA POWER Project</b><br>
        These data were obtained from the NASA Langley Research Center (LaRC) POWER Project funded through the NASA Earth Science/Applied Science Program.
    </div>
    """,
    unsafe_allow_html=True
)

# Initialize Session State Variables
if "csv_hourly_str" not in st.session_state: st.session_state.csv_hourly_str = None
if "csv_daily_str" not in st.session_state: st.session_state.csv_daily_str = None
if "excel_hourly_arm" not in st.session_state: st.session_state.excel_hourly_arm = None
if "excel_daily_arm" not in st.session_state: st.session_state.excel_daily_arm = None
if "excel_app_format" not in st.session_state: st.session_state.excel_app_format = None
if "base_filename" not in st.session_state: st.session_state.base_filename = ""
if "is_arm" not in st.session_state: st.session_state.is_arm = False

# Main Layout
st.title("🌦️ NASA POWER Downloader")
st.markdown("Download and process high-resolution weather data with direct ARM software integration.")

# 1. Location
st.subheader("1. Location")
col1, col2 = st.columns(2)
with col1: lat_input = st.text_input("Latitude", value="")
with col2: lon_input = st.text_input("Longitude", value="")

# 2. Date Range
st.subheader("2. Date Range")
today = date.today()
col3, col4 = st.columns(2)
with col3: start_date = st.date_input("Start Date", value=date(today.year, 1, 1))
with col4: end_date = st.date_input("End Date", value=today - timedelta(days=1))

# 3. Parameters
st.subheader("3. Parameters")
selected_params = {}
cols = st.columns(3)
for idx, (label, (code, is_hourly)) in enumerate(PARAMETERS.items()):
    with cols[idx % 3]:
        selected_params[code] = st.checkbox(f"{label} [{'Hr' if is_hourly else 'Day'}]", value=True)

# 4. Options
st.subheader("4. Output Options")
col5, col6, col7 = st.columns(3)
with col5:
    community = st.selectbox("Community", COMMUNITIES, index=0)
    out_daily = st.checkbox("Generate Daily Stats", value=True)
with col6:
    tstd = st.selectbox("Time Standard", TIME_STANDARDS, index=0)
    out_hourly = st.checkbox("Generate Hourly Data", value=False)
with col7:
    output_format = st.selectbox("Output Layout", ["Standard Layout (CSV)", "ARM Software Layout (Excel)"])
    apply_precip_filter = st.checkbox("Filter Low Rainfall (Daily)", value=True)
    precip_threshold = st.number_input("Rainfall Threshold (mm)", value=0.5, step=0.1, disabled=not apply_precip_filter)

# Hidden Menus
with st.expander("🌱 Weather Application Export (ARM Format)"):
    st.caption("Generate a secondary structured Excel sheet configured for agronomic software input (Requires Daily Stats).")
    enable_app_format = st.checkbox("Enable Application Formatting", value=False)
    app_dates_input = []
    if enable_app_format:
        num_apps = st.number_input("Number of Applications", min_value=1, max_value=20, value=1)
        for i in range(num_apps):
            app_letter = chr(65 + i)
            d = st.date_input(f"Application {app_letter} Date", value=date.today() - timedelta(days=7), key=f"app_{app_letter}")
            app_dates_input.append((app_letter, d))
        st.caption("⚠️ Ensure your Date Range covers at least 14 days prior and 28 days after your application dates.")

with st.expander("⚙️ Advanced Settings"):
    ssl_verify = st.checkbox("Enable SSL Verification", value=False)
    debug_mode = st.checkbox("Debug Mode (Show internal logs)", value=False)

# --- Processing Engine ---
if st.button("🚀 DOWNLOAD & PROCESS", type="primary", use_container_width=True):
    lat, lon = valid_lat_lon(lat_input, lon_input)
    if lat is None:
        st.error("❌ Invalid coordinates. Please check your Latitude and Longitude.")
        st.stop()
    if start_date > end_date:
        st.error("❌ Start date must be before or equal to End date.")
        st.stop()
    if not out_hourly and not out_daily:
        st.error("❌ Please select at least one output format (Hourly or Daily).")
        st.stop()

    # Clear previous session state data
    st.session_state.csv_hourly_str = None
    st.session_state.csv_daily_str = None
    st.session_state.excel_hourly_arm = None
    st.session_state.excel_daily_arm = None
    st.session_state.excel_app_format = None
    
    st.session_state.is_arm = (output_format == "ARM Software Layout (Excel)")
    st.session_state.base_filename = f"POWER_{community}_{lat:.4f}_{lon:.4f}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"

    hourly_req = [code for code, sel in selected_params.items() if sel and PARAMETERS[[k for k, v in PARAMETERS.items() if v[0]==code][0]][1]]
    daily_req = [code for code, sel in selected_params.items() if sel and not PARAMETERS[[k for k, v in PARAMETERS.items() if v[0]==code][0]][1]]

    debug_log = []
    daily_storage = {} 
    hourly_records = []
    chunks = year_chunks(start_date, end_date)
    
    with st.status("Fetching Data from NASA POWER...", expanded=True) as status:
        
        # PHASE 1: HOURLY DATA
        if hourly_req:
            f_hourly = io.StringIO()
            writer_h = csv.writer(f_hourly)
            first_header_written = False

            for idx, (cstart, cend) in enumerate(chunks, 1):
                st.write(f"🔄 Fetching Hourly Data: {cstart.year}...")
                q = {
                    "parameters": ",".join(hourly_req), "community": community, "longitude": lon, "latitude": lat, 
                    "start": cstart.strftime("%Y%m%d"), "end": cend.strftime("%Y%m%d"), "format": "CSV", "time-standard": tstd
                }
                ok, res = http_get_with_retries(API_HOURLY, q, verify=ssl_verify)
                if not ok: continue

                header, rows = iter_clean_rows(res.text)
                if not header: continue

                map_idx = {h: i for i, h in enumerate(header)}
                ix_y = next((map_idx.get(k) for k in ["YEAR","YYYY","YR"] if k in map_idx), None)
                ix_m = next((map_idx.get(k) for k in ["MO","MM","MONTH"] if k in map_idx), None)
                ix_d = next((map_idx.get(k) for k in ["DY","DD","DAY"] if k in map_idx), None)
                ix_h = next((map_idx.get(k) for k in ["HR","HH","HOUR"] if k in map_idx), None)

                if out_hourly and not first_header_written and not st.session_state.is_arm:
                    out_h = ["DATE", "HR"]
                    for p in hourly_req: out_h.append(HEADER_MAP.get(p, p))
                    if "WD2M" in hourly_req: out_h.append(HEADER_MAP.get("WD2M_COMPASS", "WD_CARDINAL"))
                    writer_h.writerow(out_h)
                    first_header_written = True

                for r in rows:
                    try:
                        y, m, d = int(float(r[ix_y])), int(float(r[ix_m])), int(float(r[ix_d]))
                        date_key = f"{y:04d}{m:02d}{d:02d}"
                    except: continue

                    vals = []
                    for p in hourly_req:
                        try: vals.append(float(r[map_idx[p]]))
                        except: vals.append(None)

                    hr_val = r[ix_h] if ix_h is not None else ""
                    hr_record = {"date_key": date_key, "hr": hr_val}
                    for i, p in enumerate(hourly_req): hr_record[p] = vals[i]
                    hourly_records.append(hr_record)

                    if out_hourly and not st.session_state.is_arm:
                        row_out = [date_key, hr_val] + [v if v is not None else "" for v in vals]
                        if "WD2M" in hourly_req:
                            row_out.append(deg_to_compass_16(vals[hourly_req.index("WD2M")]) if vals[hourly_req.index("WD2M")] is not None else "")
                        writer_h.writerow(row_out)

                    if out_daily or enable_app_format:
                        if date_key not in daily_storage: daily_storage[date_key] = {}
                        for i, p in enumerate(hourly_req):
                            if p not in daily_storage[date_key]: daily_storage[date_key][p] = []
                            if vals[i] is not None and vals[i] != -999: daily_storage[date_key][p].append(vals[i])
            
            if out_hourly and not st.session_state.is_arm: 
                st.session_state.csv_hourly_str = f_hourly.getvalue()

        # PHASE 2: DAILY API
        if (out_daily or enable_app_format) and daily_req:
            for idx, (cstart, cend) in enumerate(chunks, 1):
                st.write(f"🔄 Fetching Daily Data: {cstart.year}...")
                q = {
                    "parameters": ",".join(daily_req), "community": community, "longitude": lon, "latitude": lat, 
                    "start": cstart.strftime("%Y%m%d"), "end": cend.strftime("%Y%m%d"), "format": "CSV", "time-standard": tstd
                }
                ok, res = http_get_with_retries(API_DAILY, q, verify=ssl_verify)
                if not ok: continue
                
                header, rows = iter_clean_rows(res.text)
                if not header: continue
                
                map_idx = {h: i for i, h in enumerate(header)}
                ix_y   = next((map_idx.get(k) for k in ["YEAR","YYYY","YR"] if k in map_idx), None)
                ix_m   = next((map_idx.get(k) for k in ["MO","MM","MONTH"] if k in map_idx), None)
                ix_d   = next((map_idx.get(k) for k in ["DY","DD","DAY"] if k in map_idx), None)
                ix_doy = next((map_idx.get(k) for k in ["DOY"] if k in map_idx), None)

                for r in rows:
                    date_key = None
                    try:
                        if ix_m is not None and ix_d is not None: date_key = f"{int(float(r[ix_y])):04d}{int(float(r[ix_m])):02d}{int(float(r[ix_d])):02d}"
                        elif ix_doy is not None: date_key = (date(int(float(r[ix_y])), 1, 1) + timedelta(days=int(float(r[ix_doy])) - 1)).strftime("%Y%m%d")
                    except: continue

                    if not date_key: continue
                    if date_key not in daily_storage: daily_storage[date_key] = {}

                    for p in daily_req:
                        col_idx = map_idx.get(p)
                        if col_idx is None:
                            for h_name, h_idx in map_idx.items():
                                if h_name.startswith(p): col_idx = h_idx; break
                        if col_idx is not None:
                            try:
                                val = float(r[col_idx])
                                if apply_precip_filter and p == "PRECTOTCORR" and val != -999.0: val = 0.0 if val < precip_threshold else val
                                daily_storage[date_key][p] = val
                            except: pass

        # PHASE 3: WRITE OUT DATA
        ARM_COLS = ["No.", "Date", "Time", "Moisture Total", "Unit_1", "Precip", "Unit_2", "Irrigation", "Unit_3", "Type", "Type Description", "Interval", "Unit_4", "Leaf Wetness Duration", "Unit_5", "Min Temp", "Max Temp", "Avg Temp", "Temp Unit", "Min % Relative Humidity", "Max % Relative Humidity", "Avg % Relative Humidity", "Min Wind", "Max Wind", "Avg Wind", "Unit_6", "% Cloud Cover", "Avg Shortwave Radiation", "Unit_7", "Avg Soil Temp", "Unit_8", "0-10 cm Scaled Soil Moisture", "0-200 cm Scaled Soil Moisture", "Source", "Additional Comments"]
        ARM_DISPLAY = [c.split("_")[0] for c in ARM_COLS]

        if out_daily and daily_storage:
            st.write("📊 Calculating Daily Statistics...")
            sorted_dates = sorted(daily_storage.keys())
            
            if st.session_state.is_arm:
                arm_data = []
                for idx, dt in enumerate(sorted_dates):
                    d_map = daily_storage[dt]
                    prec = d_map.get("PRECTOTCORR")
                    prec_v = f"{prec:.2f}" if prec is not None else ""
                    
                    t_vals, rh_vals, ws_vals = d_map.get("T2M", []), d_map.get("RH2M", []), d_map.get("WS2M", [])
                    t_min = f"{min(t_vals):.2f}" if t_vals else ""
                    t_max = f"{max(t_vals):.2f}" if t_vals else ""
                    t_avg = f"{statistics.mean(t_vals):.2f}" if t_vals else ""
                    
                    rh_min = f"{min(rh_vals):.2f}" if rh_vals else ""
                    rh_max = f"{max(rh_vals):.2f}" if rh_vals else ""
                    rh_avg = f"{statistics.mean(rh_vals):.2f}" if rh_vals else ""
                    
                    ws_min = f"{min(ws_vals):.2f}" if ws_vals else ""
                    ws_max = f"{max(ws_vals):.2f}" if ws_vals else ""
                    ws_avg = f"{statistics.mean(ws_vals):.2f}" if ws_vals else ""

                    arm_data.append([
                        idx + 1, to_arm_date(dt), "", prec_v, "mm" if prec_v else "", prec_v, "mm" if prec_v else "",
                        "", "", "RAIN" if prec and prec > 0 else "", "rain" if prec and prec > 0 else "", "", "", "", "",
                        t_min, t_max, t_avg, "C" if t_avg else "", rh_min, rh_max, rh_avg, ws_min, ws_max, ws_avg, "MPS" if ws_avg else "",
                        "", "", "", "", "", "", "", "ENTERED", ""
                    ])
                
                df = pd.DataFrame(arm_data, columns=ARM_DISPLAY)
                excel_daily_arm_buffer = io.BytesIO()
                with pd.ExcelWriter(excel_daily_arm_buffer, engine='openpyxl') as writer: df.to_excel(writer, index=False)
                st.session_state.excel_daily_arm = excel_daily_arm_buffer.getvalue()
            
            else:
                f_daily = io.StringIO()
                writer_d = csv.writer(f_daily)
                d_header = ["DATE"]
                for p in hourly_req:
                    base = HEADER_MAP.get(p, p)
                    if p == "WS2M": d_header.extend([f"{base}_AVG", f"{base}_MAX", f"{base}_MIN"])
                    elif p == "WD2M": d_header.extend([f"{base}_AVG", HEADER_MAP.get("WD2M_COMPASS", "WD_CARDINAL")])
                    else: d_header.extend([f"{base}_AVG", f"{base}_MAX", f"{base}_MIN"])
                for p in daily_req: d_header.append(HEADER_MAP.get(p, p))
                writer_d.writerow(d_header)

                for dt in sorted_dates:
                    row = [dt]
                    for p in hourly_req:
                        vals = daily_storage[dt].get(p, [])
                        if not vals: row.extend(["", ""] if p == "WD2M" else ["", "", ""]); continue
                        if p == "WS2M" or p not in ["WD2M", "WS2M"]:
                            row.extend([f"{statistics.mean(vals):.2f}", f"{max(vals):.2f}", f"{min(vals):.2f}"])
                        elif p == "WD2M":
                            row.extend([f"{vector_average_degrees(vals):.2f}", deg_to_compass_16(vector_average_degrees(vals))])
                    for p in daily_req:
                        val = daily_storage[dt].get(p)
                        row.append(f"{val:.2f}" if val is not None and val != -999 else "")
                    writer_d.writerow(row)
                st.session_state.csv_daily_str = f_daily.getvalue()

        if out_hourly and st.session_state.is_arm and hourly_records:
            arm_hr_data = []
            for idx, r in enumerate(hourly_records):
                t_val = f"{r['T2M']:.2f}" if r.get('T2M') is not None else ""
                rh_val = f"{r['RH2M']:.2f}" if r.get('RH2M') is not None else ""
                ws_val = f"{r['WS2M']:.2f}" if r.get('WS2M') is not None else ""
                time_str = f"{str(r['hr']).split('.')[0].zfill(2)}:00"
                arm_hr_data.append([
                    idx + 1, to_arm_date(r['date_key']), time_str, "", "", "", "", "", "", "", "", "", "", "", "",
                    "", "", t_val, "C" if t_val else "", "", "", rh_val, "", "", ws_val, "MPS" if ws_val else "",
                    "", "", "", "", "", "", "", "ENTERED", ""
                ])
            df_hr = pd.DataFrame(arm_hr_data, columns=ARM_DISPLAY)
            excel_hourly_arm_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_hourly_arm_buffer, engine='openpyxl') as writer: df_hr.to_excel(writer, index=False)
            st.session_state.excel_hourly_arm = excel_hourly_arm_buffer.getvalue()

        if enable_app_format and daily_storage:
            st.write("🌱 Generating Application Layout...")
            app_table_data = []
            for app_letter, app_date in app_dates_input:
                w2_before = get_precip_sum(app_date - timedelta(days=14), app_date - timedelta(days=8), daily_storage)
                w1_before = get_precip_sum(app_date - timedelta(days=7), app_date - timedelta(days=1), daily_storage)
                day_0 = get_precip_sum(app_date, app_date, daily_storage)
                h6_after = round(day_0 * 0.25, 2)
                h24_after = day_0
                w1_after = get_precip_sum(app_date + timedelta(days=1), app_date + timedelta(days=7), daily_storage)
                w2_after = get_precip_sum(app_date + timedelta(days=8), app_date + timedelta(days=14), daily_storage)
                w3_after = get_precip_sum(app_date + timedelta(days=15), app_date + timedelta(days=21), daily_storage)
                w4_after = get_precip_sum(app_date + timedelta(days=22), app_date + timedelta(days=28), daily_storage)

                app_table_data.extend([
                    [f"--- Application {app_letter} ---", "", ""],
                    ["Moisture 2 Weeks Before Appl.", w2_before, "mm"],
                    ["Moisture 1 Week Before Appl.", w1_before, "mm"],
                    ["Moisture 6 Hours After Appl.", h6_after, "mm"],
                    ["Moisture 24 Hours After Appl.", h24_after, "mm"],
                    ["Moisture 1 Week After Appl.", w1_after, "mm"],
                    ["Moisture 2 Weeks After Appl.", w2_after, "mm"],
                    ["Moisture 3 Weeks After Appl.", w3_after, "mm"],
                    ["Moisture 4 Weeks After Appl.", w4_after, "mm"],
                    ["", "", ""] 
                ])

            df_apps = pd.DataFrame(app_table_data, columns=["Interval", "Value", "Unit"])
            excel_app_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_app_buffer, engine='openpyxl') as writer: df_apps.to_excel(writer, index=False, header=False, sheet_name="Application_Moisture")
            st.session_state.excel_app_format = excel_app_buffer.getvalue()

        status.update(label="Data Processing Complete!", state="complete", expanded=False)
        if debug_mode and debug_log: st.session_state.debug_log = debug_log

# --- Rendering Persisted Download Buttons ---
# This block runs independently of the "DOWNLOAD & PROCESS" button so it won't disappear on click
if any([st.session_state.csv_hourly_str, st.session_state.csv_daily_str, st.session_state.excel_hourly_arm, st.session_state.excel_daily_arm, st.session_state.excel_app_format]):
    st.divider()
    st.success("✅ Downloads are ready!")
    
    cols_dl = st.columns(3)
    idx_dl = 0
    
    if st.session_state.excel_hourly_arm:
        with cols_dl[idx_dl % 3]: st.download_button("⬇️ Download Hourly Data (Excel)", data=st.session_state.excel_hourly_arm, file_name=f"{st.session_state.base_filename}_Hourly_ARM.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
        idx_dl += 1
    elif st.session_state.csv_hourly_str:
        with cols_dl[idx_dl % 3]: st.download_button("⬇️ Download Hourly Data (CSV)", data=st.session_state.csv_hourly_str, file_name=f"{st.session_state.base_filename}_Hourly.csv", mime="text/csv", use_container_width=True)
        idx_dl += 1

    if st.session_state.excel_daily_arm:
        with cols_dl[idx_dl % 3]: st.download_button("⬇️ Download Daily Stats (Excel)", data=st.session_state.excel_daily_arm, file_name=f"{st.session_state.base_filename}_DailyStats_ARM.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
        idx_dl += 1
    elif st.session_state.csv_daily_str:
        with cols_dl[idx_dl % 3]: st.download_button("⬇️ Download Daily Stats (CSV)", data=st.session_state.csv_daily_str, file_name=f"{st.session_state.base_filename}_DailyStats.csv", mime="text/csv", use_container_width=True)
        idx_dl += 1

    if st.session_state.excel_app_format:
        with cols_dl[idx_dl % 3]:
            st.download_button("⬇️ Download Application Format", data=st.session_state.excel_app_format, file_name=f"{st.session_state.base_filename}_AppFormat.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)

    if hasattr(st.session_state, "debug_log") and st.session_state.debug_log:
        with st.expander("Show Debug Logs"): st.code("\n".join(st.session_state.debug_log))
