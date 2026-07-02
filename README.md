# Weather2ARM - Quick Start for Field Scientists

Weather2ARM is designed to help field teams generate weather files for trial reports with consistent, traceable data.

The app combines:
- INMET local station data (preferred when available and complete)
- NASA POWER data (fallback, or optional gap fill)

This guide is text-only and focused on practical use in Streamlit.

## What This App Delivers

- Hourly weather output for trial-event context
- Daily weather summaries for trial report tables
- Optional ARM-oriented Excel layouts
- Source metadata for audit trail and QA checks

## 5-Minute Quick Start

1. Open Weather2ARM in Streamlit.
2. In Location, choose coordinate format and enter coordinates.
3. Set your Start Date and End Date.
4. In Output Options, choose if you need Hourly Data, Daily Stats, or both, and select output layout.
5. In Data Source, keep Source Selection = Auto (recommended), then click DOWNLOAD and PROCESS.
6. Download output files and Metadata JSON.

## Full Step-by-Step Workflow

### 1) Location

- Coordinate Input Format:
   - Decimal Degrees (default, recommended for most users)
   - GMS (Degrees Minutes Seconds) if that is what your GPS or field notes use
- Enter Latitude and Longitude for the trial site.
- Decimal input note: use "." as decimal separator (example: -26.9386111).

When to use GMS:
- Your notebook/device records coordinates as degrees/minutes/seconds
- You want to avoid manual conversion errors before input

### 2) Date Range

- Choose Start Date and End Date covering your trial period.
- Include pre-application and post-application windows when moisture/rain context matters.

Practical recommendation:
- For spray efficacy interpretation, include at least 2 weeks before and 4 weeks after application dates.

### 3) Output Options

- Generate Daily Stats:
   - Recommended for most trial reports
   - Supports day-level interpretation and summary tables
- Generate Hourly Data:
   - Use for event-level analysis (spray timing, rain timing, rapid humidity/temperature changes)
- Output Layout:
   - ARM Software Layout (Excel): recommended if data goes directly into ARM workflow
   - Standard Layout (CSV): use for external analytics/reporting systems
- Fixed defaults (not shown in UI):
   - Community = AG
   - Time Standard = LST
- Parameters:
   - All supported weather variables are always included automatically
- Filter Low Rainfall (Daily) + threshold:
   - Applies to NASA POWER daily precipitation values only
   - Does not alter INMET-primary daily precipitation

Important precipitation behavior:
- INMET source: hourly CSV includes hourly precipitation
- NASA source: hourly CSV has no precipitation column (NASA precipitation is daily)

### 4) Data Source

- Source Selection:
   - Auto:
      - Best default for most users
      - Uses INMET if a suitable station is found; otherwise falls back to NASA
   - NASA only:
      - Use when you need consistent satellite-derived baseline, or if INMET coverage is uncertain
   - Prefer INMET:
      - Use when local station representativeness is critical
      - Optionally fill missing records from NASA
- Fill INMET gaps with NASA POWER:
   - Enable when you need complete series and can accept mixed-source traceability
   - Disable when you need strict local-station-only records for sensitivity analysis
- INMET Search Radius (km):
   - Start at 50 km
   - Increase if no station is found
   - Decrease if you want stronger local representativeness
- Local UTC Offset:
   - Default is -3 (Brasilia)
   - Keep default unless your reporting standard requires another local offset
- INMET Data Directory:
   - Folder path where INMET files are stored (usually INMET)
   - Supports nested year folders and ZIP archives
- Preferred INMET Station (optional):
   - Enter station code or name to prioritize one station
   - Leave blank to let app auto-rank candidates

INMET availability note:
- Monthly INMET data is typically available up to the end of the previous month
- Current operational coverage starts from 2025 in this app dataset

### 5) Run and Download

1. Click DOWNLOAD and PROCESS.
2. Wait for completion message and source summary.
3. Download the files you need.

## Weather Application Export Module (Detailed)

This module is designed for trial-report moisture interpretation around application events.

How to enable:
- Open "Weather Application Export (ARM Format)"
- Enable Application Formatting
- Enter:
   - Number of Applications
   - Application Date for each event
   - Application Time for each event

### What the module computes

For each application, the module calculates:
- First Moisture Occured On
   - First date with rainfall after application datetime
   - Can be the same day as application
   - Date format is ARM style (example: 9Jul26)
- Time to First Moisture
   - Uses hourly timing when possible
   - Unit = HR when first rainfall happens within 24 hours
   - Unit = DAY when first rainfall happens after 24 hours
- Amount of First Moisture
   - Total rainfall on the first-moisture day
- Moisture windows:
   - 2 Weeks Before Application
   - 1 Week Before Application
   - 6 Hours After Application
   - 24 Hours After Application
   - 1 to 4 Weeks After Application

### Cross-year logic (important)

When Application Formatting is enabled, the app automatically expands internal data retrieval to cover all required moisture windows:
- up to 14 days before earliest application
- up to 28 days after latest application

This allows correct calculations across year boundaries (for example, early-Jan applications requiring Dec data from previous year).

### Output behavior by layout

- If Output Layout = ARM Software Layout (Excel):
   - A single workbook is generated
   - Sheet 1: Meteorological_Data
   - Sheet 2: Weather_Application
   - This supports direct copy/paste workflows into ARM
- If Output Layout = Standard Layout (CSV):
   - Weather application export is generated as a separate file

### Source behavior in application calculations

- The weather-application moisture summaries use the processed precipitation dataset for the run.
- If INMET is primary, calculations use INMET precipitation where available.
- If gap-fill is enabled, NASA can complete missing parts according to selected source strategy.

## Which Output Should You Use for Trial Reports?

- Daily Stats:
   - Primary file for report tables and interpretation sections
   - Best for rainfall totals, daily temperature/humidity context
- Hourly Data:
   - Use when trial conclusions depend on short time windows
   - Example: rainfall within hours after application, overnight humidity spikes
- Application Format:
   - Use when your ARM process expects predefined moisture-interval structure
   - In ARM layout, this is included as a second worksheet in the same Excel file
- Metadata JSON:
   - Always archive with trial outputs
   - Essential for traceability and review audits

## Understanding Candidate Station Metrics

When candidate stations are listed, the app helps you judge quality:

- coverage_ratio:
   - Fraction of expected hourly timestamps with station records
   - Higher is better
- missing_ratio:
   - Fraction of required variable cells missing over expected hourly grid
   - Lower is better
- missing_day_count:
   - Number of days with at least one required-value gap
- missing_days:
   - Exact dates where missingness was detected

Use these values to decide:
- strict local data vs mixed-source gap fill
- whether to increase radius or force NASA only

## Data Traceability and QA Practice

For each trial run, keep:
- downloaded weather file(s)
- Metadata JSON
- selected coordinates and date range
- Source Selection mode used

Recommended QA checks before publishing report:
- confirm primary source in metadata
- verify station name/code if INMET was used
- check whether gap_fill_applied is true/false
- if gap fill is true, review filled variables and timestamps

## Troubleshooting

### No results returned

- Increase INMET Search Radius
- Expand date range
- Try Source Selection = NASA only to confirm service path
- Verify INMET Data Directory

### INMET station not selected as expected

- Clear Preferred INMET Station and retry
- Check station spelling/code
- Review candidate table metrics (coverage/missing)

### App is slow

- Large INMET ZIP archives can increase startup and processing time
- Use only required years when possible

### Output looks different from expectations

- Check Output Layout selected (ARM Excel vs CSV)
- Confirm source and gap-fill details in metadata JSON
