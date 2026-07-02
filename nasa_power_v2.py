import streamlit as st
import csv
import io
import json
import math
import re
import statistics
import urllib3
from datetime import date, timedelta, datetime, time
import pandas as pd
from weather_sources import build_weather_dataset

# Suppress SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

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

DEFAULT_COMMUNITY = "AG"
DEFAULT_TIME_STANDARD = "LST"
SOURCE_STRATEGIES = ["Auto", "NASA only", "Prefer INMET"]

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

def dms_to_decimal(dms_str, is_lat=True):
    txt = str(dms_str or "").strip().upper()
    if not txt:
        return None

    hemi_match = re.search(r"[NSEW]", txt)
    hemi = hemi_match.group(0) if hemi_match else None
    nums = re.findall(r"[-+]?\d+(?:\.\d+)?", txt)
    if len(nums) == 0:
        return None

    try:
        deg = float(nums[0])
        minutes = float(nums[1]) if len(nums) > 1 else 0.0
        seconds = float(nums[2]) if len(nums) > 2 else 0.0
    except ValueError:
        return None

    if minutes < 0 or minutes >= 60 or seconds < 0 or seconds >= 60:
        return None

    value = abs(deg) + (minutes / 60.0) + (seconds / 3600.0)
    sign = -1 if deg < 0 else 1

    if hemi in ("S", "W"):
        sign = -1
    elif hemi in ("N", "E"):
        sign = 1

    value *= sign
    if is_lat and not (-90.0 <= value <= 90.0):
        return None
    if (not is_lat) and not (-180.0 <= value <= 180.0):
        return None
    return value

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
st.set_page_config(page_title="Weather2ARM", layout="wide", page_icon="🌦️")

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
        <b>NASA POWER + INMET Data References</b><br>
        These data were obtained from the NASA Langley Research Center (LaRC) POWER Project funded through the NASA Earth Science/Applied Science Program.
        <br><br>
        INMET station data are provided by Instituto Nacional de Meteorologia (INMET), Brazil.
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
if "output_metadata_json" not in st.session_state: st.session_state.output_metadata_json = None
if "base_filename" not in st.session_state: st.session_state.base_filename = ""
if "is_arm" not in st.session_state: st.session_state.is_arm = False

# Main Layout
st.title("🌦️ Weather2ARM")
st.markdown("Download and process weather data for ARM software using NASA POWER and INMET sources.")

# 1. Location
st.subheader("1. Location")
coord_mode = st.selectbox("Coordinate Input Format", ["Decimal Degrees", "GMS (Degrees Minutes Seconds)"], index=0)
col1, col2 = st.columns(2)
if coord_mode == "Decimal Degrees":
    with col1: lat_input = st.text_input("Latitude", value="", placeholder="-26.9386111")
    with col2: lon_input = st.text_input("Longitude", value="", placeholder="-52.39805555")
    lat_dms_input, lon_dms_input = "", ""
    st.caption("Decimal format note: use '.' as decimal separator (example: -26.9386111).")
else:
    with col1: lat_dms_input = st.text_input("Latitude (GMS)", value="", placeholder="26 56 19 S")
    with col2: lon_dms_input = st.text_input("Longitude (GMS)", value="", placeholder="52 23 53 W")
    lat_input, lon_input = "", ""
    st.caption("Accepted examples: 26 56 19 S, 26°56'19\"S, -26 56 19")

# 2. Date Range
st.subheader("2. Date Range")
today = date.today()
col3, col4 = st.columns(2)
with col3: start_date = st.date_input("Start Date", value=date(today.year, 1, 1))
with col4: end_date = st.date_input("End Date", value=today - timedelta(days=1))

selected_params = {code: True for code, _ in PARAMETERS.values()}

# 3. Output Options
st.subheader("3. Output Options")
col5, col6 = st.columns(2)
with col5:
    out_daily = st.checkbox("Generate Daily Stats", value=True)
    out_hourly = st.checkbox("Generate Hourly Data", value=False)
with col6:
    output_format = st.selectbox("Output Layout", ["Standard Layout (CSV)", "ARM Software Layout (Excel)"], index=1)
    apply_precip_filter = st.checkbox("Filter Low Rainfall (Daily)", value=True)
    precip_threshold = st.number_input("Rainfall Threshold (mm)", value=0.5, step=0.1, disabled=not apply_precip_filter)
st.caption("Rainfall filter applies to NASA POWER daily precipitation values. INMET-primary daily precipitation is not filtered.")
st.caption("Hourly precipitation is included in hourly CSV downloads when INMET is the source. NASA POWER provides precipitation at daily resolution only.")

st.subheader("4. Data Source")
col8, col9, col10 = st.columns(3)
with col8:
    source_strategy = st.selectbox("Source Selection", SOURCE_STRATEGIES, index=0)
    inmet_gap_fill = st.checkbox("Fill INMET gaps with NASA POWER", value=True)
with col9:
    inmet_radius_km = st.number_input("INMET Search Radius (km)", min_value=1.0, value=50.0, step=5.0)
    timezone_offset_hours = st.number_input("Local UTC Offset (Brasilia default: -3)", min_value=-12, max_value=14, value=-3, step=1)
with col10:
    inmet_data_dir = st.text_input("INMET Data Directory", value="INMET", help="Path relative to the app root. Example: INMET (supports nested folders and ZIP files).")
    preferred_inmet_station = st.text_input("Preferred INMET Station (optional)", value="", help="Optional station code or name filter, e.g. A858 or XANXERE.")
    force_nasa_timezone = st.checkbox("Keep NASA Time Standard Setting", value=True)
st.info("INMET dataset availability: monthly station files generally cover dates up to the end of the previous month and currently go back to 2025.")

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
            t = st.time_input(f"Application {app_letter} Time", value=time(9, 0), step=1800, key=f"app_t_{app_letter}")
            app_dates_input.append((app_letter, d, t))
        st.caption("⚠️ Ensure your Date Range covers at least 14 days prior and 28 days after your application dates.")

with st.expander("⚙️ Advanced Settings"):
    ssl_verify = st.checkbox("Enable SSL Verification", value=False)
    debug_mode = st.checkbox("Debug Mode (Show internal logs)", value=False)

# --- Processing Engine ---
if st.button("🚀 DOWNLOAD & PROCESS", type="primary", use_container_width=True):
    if coord_mode == "Decimal Degrees":
        lat, lon = valid_lat_lon(lat_input, lon_input)
    else:
        lat = dms_to_decimal(lat_dms_input, is_lat=True)
        lon = dms_to_decimal(lon_dms_input, is_lat=False)
    if lat is None:
        st.error("❌ Invalid coordinates. Please check your Latitude and Longitude format.")
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
    st.session_state.output_metadata_json = None
    
    st.session_state.is_arm = (output_format == "ARM Software Layout (Excel)")
    community = DEFAULT_COMMUNITY
    tstd = DEFAULT_TIME_STANDARD
    st.session_state.base_filename = f"POWER_{community}_{lat:.4f}_{lon:.4f}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"

    hourly_req = [code for code, sel in selected_params.items() if sel and PARAMETERS[[k for k, v in PARAMETERS.items() if v[0]==code][0]][1]]
    daily_req = [code for code, sel in selected_params.items() if sel and not PARAMETERS[[k for k, v in PARAMETERS.items() if v[0]==code][0]][1]]

    debug_log = []
    daily_storage = {}
    hourly_records = []
    
    with st.status("Fetching and processing weather data...", expanded=True) as status:
        weather_result = build_weather_dataset(
            lat=lat,
            lon=lon,
            start_date=start_date,
            end_date=end_date,
            selected_params=selected_params,
            community=community,
            tstd=tstd if force_nasa_timezone else "UTC",
            out_daily=out_daily,
            out_hourly=out_hourly,
            enable_app_format=enable_app_format,
            apply_precip_filter=apply_precip_filter,
            precip_threshold=precip_threshold,
            source_strategy=source_strategy,
            inmet_radius_km=float(inmet_radius_km),
            inmet_gap_fill=inmet_gap_fill,
            inmet_data_dir=inmet_data_dir,
            preferred_inmet_station=preferred_inmet_station,
            timezone_offset_hours=int(timezone_offset_hours),
            ssl_verify=ssl_verify,
        )

        daily_storage = weather_result.daily_storage
        hourly_records = weather_result.hourly_records
        output_metadata = weather_result.metadata
        st.session_state.output_metadata_json = json.dumps(output_metadata, indent=2, ensure_ascii=False)

        if output_metadata.get("primary_source") == "INMET":
            station_meta = output_metadata.get("station", {})
            st.info(
                f"Using INMET station {station_meta.get('station_code', 'N/A')} - "
                f"{station_meta.get('station_name', 'Unknown')} "
                f"({station_meta.get('distance_km', 'N/A')} km)."
            )
            if out_hourly and not st.session_state.is_arm:
                st.caption("INMET hourly output includes precipitation (PREC_MM_HR). NASA POWER hourly output does not include precipitation.")
        else:
            st.info(f"Using NASA POWER. Reason: {output_metadata.get('selection_reason', 'N/A')}")

        candidates = output_metadata.get("candidate_stations", [])
        if candidates:
            st.caption("INMET candidate ranking (best-first):")
            st.dataframe(pd.DataFrame(candidates), use_container_width=True)
            st.caption("Coverage ratio = fraction of expected hourly timestamps with records in requested period. Missing ratio = fraction of required variable cells that are missing over expected hourly grid.")
            st.caption("Missing days columns indicate dates with missing INMET values for required variables and/or daily precipitation.")

        if not daily_storage and not hourly_records:
            status.update(label="No data found for the selected inputs.", state="error", expanded=True)
            st.error(
                "No weather records were returned. Try increasing INMET radius, adjusting date range, "
                "or switching Source Selection to NASA only to test connectivity."
            )
            st.caption(f"Selection details: {output_metadata.get('selection_reason', 'N/A')}")
            st.stop()

        # PHASE 3: WRITE OUT DATA
        ARM_COLS = ["Date", "Time", "Moisture Total", "Unit_1", "Precip", "Unit_2", "Irrigation", "Unit_3", "Type", "Type Description", "Interval", "Unit_4", "Leaf Wetness Duration", "Unit_5", "Min Temp", "Max Temp", "Avg Temp", "Temp Unit", "Min % Relative Humidity", "Max % Relative Humidity", "Avg % Relative Humidity", "Min Wind", "Max Wind", "Avg Wind", "Unit_6", "% Cloud Cover", "Avg Shortwave Radiation", "Unit_7", "Avg Soil Temp", "Unit_8", "0-10 cm Scaled Soil Moisture", "0-200 cm Scaled Soil Moisture", "Source", "Additional Comments"]
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

                    # Convert wind speed from m/s to km/h for ARM output.
                    ws_min_kps = f"{float(ws_min) * 3.6:.2f}" if ws_min else ""
                    ws_max_kps = f"{float(ws_max) * 3.6:.2f}" if ws_max else ""
                    ws_avg_kps = f"{float(ws_avg) * 3.6:.2f}" if ws_avg else ""

                    arm_data.append([
                        to_arm_date(dt), "", prec_v, "mm" if prec_v else "", prec_v, "mm" if prec_v else "",
                        "", "", "RAIN" if prec and prec > 0 else "", "rain" if prec and prec > 0 else "", "", "", "", "",
                        t_min, t_max, t_avg, "C" if t_avg else "", rh_min, rh_max, rh_avg, ws_min_kps, ws_max_kps, ws_avg_kps, "KPS" if ws_avg_kps else "",
                        "", "", "", "", "", "", "", "ENTERED", ""
                    ])
                
                df = pd.DataFrame(arm_data, columns=ARM_DISPLAY)
                excel_daily_arm_buffer = io.BytesIO()
                with pd.ExcelWriter(excel_daily_arm_buffer, engine='openpyxl') as writer:
                    df.to_excel(writer, index=False, sheet_name="Meteorological_Data")

                    if enable_app_format:
                        # Build application worksheet in the same ARM workbook.
                        hourly_prec_by_dt = {}
                        for rec in hourly_records:
                            pval = rec.get("PRECTOTCORR")
                            if pval is None:
                                continue
                            try:
                                dt_local = datetime.strptime(f"{rec.get('date_key')} {int(float(rec.get('hr', 0))):02d}", "%Y%m%d %H")
                            except Exception:
                                continue
                            hourly_prec_by_dt[dt_local] = float(pval)

                        app_rows = []
                        for app_letter, app_date, app_time in app_dates_input:
                            app_dt = datetime.combine(app_date, app_time)

                            first_moisture_dt = None
                            for dtk in sorted(hourly_prec_by_dt.keys()):
                                if dtk >= app_dt and hourly_prec_by_dt[dtk] > 0:
                                    first_moisture_dt = dtk
                                    break

                            first_moisture_date = None
                            time_to_first = ""
                            time_unit = ""

                            if first_moisture_dt is not None:
                                first_moisture_date = first_moisture_dt.date()
                                delta_hrs = max(0, int(round((first_moisture_dt - app_dt).total_seconds() / 3600.0)))
                                if delta_hrs <= 24:
                                    time_to_first = str(delta_hrs)
                                    time_unit = "HR"
                                else:
                                    time_to_first = str(max(1, int(delta_hrs // 24)))
                                    time_unit = "DAY"
                            else:
                                cur_d = app_date
                                while cur_d <= end_date:
                                    dkey = cur_d.strftime("%Y%m%d")
                                    dprec = daily_storage.get(dkey, {}).get("PRECTOTCORR")
                                    if dprec is not None and dprec > 0:
                                        first_moisture_date = cur_d
                                        time_to_first = str((cur_d - app_date).days)
                                        time_unit = "DAY"
                                        break
                                    cur_d += timedelta(days=1)

                            first_moisture_arm = first_moisture_date.strftime("%d%b%y").lstrip("0") if first_moisture_date else ""
                            first_moisture_amt = ""
                            if first_moisture_date is not None:
                                dkey_fm = first_moisture_date.strftime("%Y%m%d")
                                dprec = daily_storage.get(dkey_fm, {}).get("PRECTOTCORR")
                                if dprec is not None:
                                    first_moisture_amt = f"{float(dprec):.1f}"

                            w2_before = get_precip_sum(app_date - timedelta(days=14), app_date - timedelta(days=8), daily_storage)
                            w1_before = get_precip_sum(app_date - timedelta(days=7), app_date - timedelta(days=1), daily_storage)
                            day_0 = get_precip_sum(app_date, app_date, daily_storage)
                            h6_after = round(day_0 * 0.25, 2)
                            h24_after = day_0
                            w1_after = get_precip_sum(app_date + timedelta(days=1), app_date + timedelta(days=7), daily_storage)
                            w2_after = get_precip_sum(app_date + timedelta(days=8), app_date + timedelta(days=14), daily_storage)
                            w3_after = get_precip_sum(app_date + timedelta(days=15), app_date + timedelta(days=21), daily_storage)
                            w4_after = get_precip_sum(app_date + timedelta(days=22), app_date + timedelta(days=28), daily_storage)

                            app_rows.extend([
                                [f"--- Application {app_letter} ---", ""],
                                ["First Moisture Occured On", first_moisture_arm],
                                ["Time to First Moisture", time_to_first],
                                ["", time_unit],
                                ["Amount of First Moisture", first_moisture_amt],
                                ["", "mm"],
                                ["Moisture 2 Weeks Before Appl.", w2_before],
                                ["", "mm"],
                                ["Moisture 1 Week Before Appl.", w1_before],
                                ["", "mm"],
                                ["Moisture 6 Hours After Appl.", h6_after],
                                ["", "mm"],
                                ["Moisture 24 Hours After Appl.", h24_after],
                                ["", "mm"],
                                ["Moisture 1 Week After Appl.", w1_after],
                                ["", "mm"],
                                ["Moisture 2 Weeks After Appl.", w2_after],
                                ["", "mm"],
                                ["Moisture 3 Weeks After Appl.", w3_after],
                                ["", "mm"],
                                ["Moisture 4 Weeks After Appl.", w4_after],
                                ["", "mm"],
                                ["", ""],
                            ])

                        df_apps = pd.DataFrame(app_rows, columns=["Field", "Value"])
                        df_apps.to_excel(writer, index=False, header=False, sheet_name="Weather_Application")

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

        if out_hourly and not st.session_state.is_arm and hourly_records:
            f_hourly = io.StringIO()
            writer_h = csv.writer(f_hourly)
            out_h = ["DATE", "HR"]
            for p in hourly_req:
                out_h.append(HEADER_MAP.get(p, p))
            include_hourly_precip = output_metadata.get("primary_source") == "INMET"
            if include_hourly_precip:
                out_h.append("PREC_MM_HR")
            if "WD2M" in hourly_req:
                out_h.append(HEADER_MAP.get("WD2M_COMPASS", "WD_CARDINAL"))
            writer_h.writerow(out_h)

            for rec in hourly_records:
                row_vals = [rec.get("date_key", ""), rec.get("hr", "")]
                for p in hourly_req:
                    v = rec.get(p)
                    row_vals.append(v if v is not None else "")
                if include_hourly_precip:
                    pval = rec.get("PRECTOTCORR")
                    row_vals.append(pval if pval is not None else "")
                if "WD2M" in hourly_req:
                    wd_val = rec.get("WD2M")
                    row_vals.append(deg_to_compass_16(wd_val) if wd_val is not None else "")
                writer_h.writerow(row_vals)

            st.session_state.csv_hourly_str = f_hourly.getvalue()

        if out_hourly and st.session_state.is_arm and hourly_records:
            arm_hr_data = []
            for idx, r in enumerate(hourly_records):
                t_val = f"{r['T2M']:.2f}" if r.get('T2M') is not None else ""
                rh_val = f"{r['RH2M']:.2f}" if r.get('RH2M') is not None else ""
                ws_val = f"{r['WS2M'] * 3.6:.2f}" if r.get('WS2M') is not None else ""
                time_str = f"{str(r['hr']).split('.')[0].zfill(2)}:00"
                arm_hr_data.append([
                    to_arm_date(r['date_key']), time_str, "", "", "", "", "", "", "", "", "", "", "", "",
                    "", t_val, "C" if t_val else "", "", "", rh_val, "", "", ws_val, "KPS" if ws_val else "",
                    "", "", "", "", "", "", "", "", "ENTERED", ""
                ])
            df_hr = pd.DataFrame(arm_hr_data, columns=ARM_DISPLAY)
            excel_hourly_arm_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_hourly_arm_buffer, engine='openpyxl') as writer: df_hr.to_excel(writer, index=False)
            st.session_state.excel_hourly_arm = excel_hourly_arm_buffer.getvalue()

        if enable_app_format and daily_storage and not st.session_state.is_arm:
            st.write("🌱 Generating Application Layout...")
            app_table_data = []
            for app_letter, app_date, _app_time in app_dates_input:
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
    if st.session_state.output_metadata_json:
        with cols_dl[idx_dl % 3]:
            st.download_button("⬇️ Download Source Metadata (JSON)", data=st.session_state.output_metadata_json, file_name=f"{st.session_state.base_filename}_metadata.json", mime="application/json", use_container_width=True)

    if hasattr(st.session_state, "debug_log") and st.session_state.debug_log:
        with st.expander("Show Debug Logs"): st.code("\n".join(st.session_state.debug_log))