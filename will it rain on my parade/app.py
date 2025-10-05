"""
Streamlit Weather Risk Dashboard (Prototype)
Single-file app that:
- Lets user pick a location (search by name using Nominatim or enter lat/lon)
- Lets user pick a target date (month + day)
- Lets user select variables and thresholds
- Fetches historical daily data from NASA POWER API
- Computes probability of exceeding thresholds for the chosen day-of-year across years
- Shows a probability bar + timeseries + trend and allows CSV download

Run: pip install streamlit pandas requests plotly scikit-learn
Then: streamlit run streamlit_weather_risk_app.py

Notes: This is a beginner-friendly prototype. Improve UI/UX, caching, error handling, map pin-drop, and add more variables later.
"""

import streamlit as st
import requests
import pandas as pd
import numpy as np
import datetime
import io
from sklearn.linear_model import LinearRegression
import plotly.graph_objs as go

st.set_page_config(page_title="Weather Risk Dashboard", layout="wide")

# ------------------ Helper functions ------------------
@st.cache_data(show_spinner=False)
def geocode_place(place_name: str):
    """Use Nominatim (OpenStreetMap) to geocode a place name to lat/lon."""
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": place_name, "format": "json", "limit": 1}
    resp = requests.get(url, params=params, headers={"User-Agent": "weather-risk-app/1.0"}, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    if not data:
        return None
    return float(data[0]["lat"]), float(data[0]["lon"]), data[0].get("display_name", place_name)

@st.cache_data(show_spinner=False)
def fetch_nasa_power_daily(lat: float, lon: float, start_year:int=1981, end_year:int=2024, params_list=None):
    """Fetch daily data for given lat/lon and years from NASA POWER.
    Returns a DataFrame with a 'date' index and requested parameters as columns.

    NASA POWER daily point API example:
    https://power.larc.nasa.gov/api/temporal/daily/point?parameters=T2M,PRECTOTCORR,WS2M&start=19810101&end=20241231&latitude=38&longitude=-77&format=JSON
    """
    if params_list is None:
        params_list = ["T2M","PRECTOTCORR","WS2M"]
    start = f"{start_year}0101"
    end = f"{end_year}1231"
    base = "https://power.larc.nasa.gov/api/temporal/daily/point"
    params = {
        "start": start,
        "end": end,
        "latitude": lat,
        "longitude": lon,
        "parameters": ",".join(params_list),
        "format": "JSON",
        "community": "AG"  # community can be 'AG' or others; doesn't affect basic variables
    }
    resp = requests.get(base, params=params, timeout=30)
    resp.raise_for_status()
    j = resp.json()

    # Parse returned JSON
    try:
        daily = j["properties"]["parameter"]
    except Exception as e:
        raise RuntimeError("Unexpected response from NASA POWER API") from e

    # daily is a dict: parameter -> {YYYYMMDD: value}
    df = pd.DataFrame()
    for p in params_list:
        if p not in daily:
            continue
        s = pd.Series(daily[p]).rename(p)
        df = pd.concat([df, s], axis=1)

    # convert index to datetime
    df.index = pd.to_datetime(df.index, format="%Y%m%d")
    df.index.name = 'date'

    # convert units if needed: T2M is °C, PRECTOTCORR is mm/day, WS2M is m/s
    return df.sort_index()


def day_of_year_stats(df: pd.DataFrame, month:int, day:int, variable:str, threshold:float, threshold_direction="gt"):
    """Compute probability that variable on given month/day across years exceeds threshold.
    threshold_direction: "gt" or "lt" (greater than or less than)
    Returns dict of results and series of historical values for that day-of-year across years.
    """
    # select rows matching month and day across all years
    sel = df[(df.index.month == month) & (df.index.day == day)]
    # drop NaNs
    vals = sel[variable].dropna()
    if vals.empty:
        return None
    if threshold_direction == "gt":
        exceed = (vals > threshold).sum()
    else:
        exceed = (vals < threshold).sum()
    prob = exceed / len(vals)

    # Trend: compute linear regression of values over years
    years = vals.index.year.values.reshape(-1,1)
    model = LinearRegression()
    try:
        model.fit(years, vals.values)
        trend_slope = model.coef_[0]
    except Exception:
        trend_slope = np.nan

    return {
        "n_years": len(vals),
        "probability": prob,
        "exceed_count": int(exceed),
        "values": vals,
        "trend_slope_per_year": float(trend_slope)
    }


# ------------------ UI ------------------
st.title("🌤️ Weather Risk Dashboard — NASA POWER Prototype")
st.write("Build a probability estimate (based on historical data) for 'very hot', 'very wet', or 'very windy' conditions on a given day and place.")

with st.sidebar:
    st.header("Query inputs")
    place = st.text_input("Search place name (city, landmark) or leave blank to enter lat/lon", value="New Delhi, India")
    lat_input = st.text_input("Latitude (optional)", value="")
    lon_input = st.text_input("Longitude (optional)", value="")

    month = st.selectbox("Month", list(range(1,13)), index=datetime.datetime.now().month-1)
    day = st.selectbox("Day", list(range(1,32)), index=datetime.datetime.now().day-1)

    st.markdown("---")
    st.subheader("Variables & thresholds")
    var_temp = st.checkbox("Temperature (daily mean) — T2M, °C", value=True)
    temp_threshold = st.slider("""Very hot threshold (°C)""", min_value=-20.0, max_value=50.0, value=30.0)

    var_precip = st.checkbox("Precipitation (daily total) — PRECTOTCORR, mm", value=True)
    precip_threshold = st.slider("Very wet threshold (mm/day)", min_value=0.0, max_value=500.0, value=10.0)

    var_wind = st.checkbox("Wind speed (2m) — WS2M, m/s", value=False)
    wind_threshold = st.slider("Very windy threshold (m/s)", min_value=0.0, max_value=40.0, value=8.0)

    years_range = st.slider("Historical years range (start, end)", 1981, datetime.datetime.now().year, (1981, datetime.datetime.now().year))

    compute_btn = st.button("Compute probabilities")

# Resolve lat/lon
lat = None
lon = None
place_display = None
if lat_input and lon_input:
    try:
        lat = float(lat_input)
        lon = float(lon_input)
        place_display = f"Lat {lat:.4f}, Lon {lon:.4f}"
    except ValueError:
        st.sidebar.error("Invalid lat/lon. Clear or enter numeric values.")
elif place:
    with st.spinner("Geocoding place name..."):
        try:
            g = geocode_place(place)
            if g:
                lat, lon, display = g
                place_display = display
            else:
                st.sidebar.error("Place not found. Try a different query or enter lat/lon.")
        except Exception as e:
            st.sidebar.error(f"Geocoding failed: {e}")

if not lat or not lon:
    st.warning("Please provide a valid location (place name or lat/lon) in the sidebar.")

# Main compute
if compute_btn and lat and lon:
    st.info(f"Fetching NASA POWER daily data for {place_display} — this may take ~10-20s (cached).")
    try:
        vars_to_fetch = []
        mapping = {}
        if var_temp:
            vars_to_fetch.append("T2M")
            mapping["T2M"] = ("Temperature (°C)", "°C")
        if var_precip:
            vars_to_fetch.append("PRECTOTCORR")
            mapping["PRECTOTCORR"] = ("Precipitation (mm/day)", "mm/day")
        if var_wind:
            vars_to_fetch.append("WS2M")
            mapping["WS2M"] = ("Wind speed (m/s)", "m/s")

        df = fetch_nasa_power_daily(lat, lon, start_year=years_range[0], end_year=years_range[1], params_list=vars_to_fetch)
        if df.empty:
            st.error("No data returned for this location/time range.")
        else:
            st.success(f"Loaded {len(df)} daily records ({df.index.year.min()}–{df.index.year.max()}).")

            results = {}
            if "T2M" in df.columns:
                r = day_of_year_stats(df, month, day, "T2M", temp_threshold, threshold_direction="gt")
                if r: results["Temperature"] = (r, "T2M")
            if "PRECTOTCORR" in df.columns:
                r = day_of_year_stats(df, month, day, "PRECTOTCORR", precip_threshold, threshold_direction="gt")
                if r: results["Precipitation"] = (r, "PRECTOTCORR")
            if "WS2M" in df.columns:
                r = day_of_year_stats(df, month, day, "WS2M", wind_threshold, threshold_direction="gt")
                if r: results["Wind"] = (r, "WS2M")

            # Display summary cards
            cols = st.columns(len(results) if results else 1)
            for i, (label, (res, varcode)) in enumerate(results.items()):
                with cols[i]:
                    prob_pct = res["probability"] * 100
                    st.metric(label=f"Chance of '{label}'", value=f"{prob_pct:.1f}%", delta=f"based on {res['n_years']} years")
                    st.write(f"Trend (slope per year): {res['trend_slope_per_year']:.4f} {mapping[varcode][1]} / year")

            # Detailed plots per variable
            for label, (res, varcode) in results.items():
                st.subheader(f"{label} — historical values for {month}/{day}")
                vals = res["values"].sort_index()

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=vals.index.year, y=vals.values, mode='markers+lines', name=label))
                # add threshold line
                thr = temp_threshold if varcode=="T2M" else (precip_threshold if varcode=="PRECTOTCORR" else wind_threshold)
                fig.add_hline(y=thr, line_dash='dash', annotation_text=f"Threshold: {thr}")

                # linear fit
                yrs = vals.index.year.values.reshape(-1,1)
                try:
                    lr = LinearRegression().fit(yrs, vals.values)
                    pred = lr.predict(yrs)
                    fig.add_trace(go.Line(x=vals.index.year, y=pred, name='Linear trend'))
                except Exception:
                    pass

                fig.update_layout(xaxis_title='Year', yaxis_title=f'{mapping[varcode][0]}')
                st.plotly_chart(fig, use_container_width=True)

                # Show raw table and download
                table = vals.reset_index()
                table.columns = ['date', mapping[varcode][0]]
                csv = table.to_csv(index=False).encode('utf-8')
                st.download_button(label=f"Download {label} data (CSV)", data=csv, file_name=f"{label.lower()}_{month}_{day}.csv", mime='text/csv')

    except Exception as e:
        st.error(f"Failed to compute: {e}")


# Footer
st.markdown("---")
st.caption("Prototype uses NASA POWER daily point API. Improve by adding caching, better geocoding, choice of thresholds, multi-point aggregation, and UX polish.")
