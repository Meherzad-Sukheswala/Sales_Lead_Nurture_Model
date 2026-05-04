import streamlit as st
import pandas as pd
import joblib
import uuid
import os
import base64
import requests
from PIL import Image
import plotly.express as px

st.set_page_config(page_title="Sales Lead Nurture Model Dashboard", layout="wide")

# ── Model ──────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "xgboost_ptb_pipeline.pkl")
    return joblib.load(model_path)

model = load_model()

# ── Logo ───────────────────────────────────────────────────────────────────────
logo_path = os.path.join(os.path.dirname(__file__), "analytics_ai_logo.png")
if os.path.exists(logo_path):
    logo = Image.open(logo_path)
    st.image(logo, width=250)
else:
    st.warning("⚠️ Company logo not found.")

# ── Load input Excel from repo ─────────────────────────────────────────────────
@st.cache_data
def fetch_data():
    """
    Reads the input Excel file committed in the GitHub repo.
    Place your file at the repo root (or update INPUT_FILE_PATH in secrets/below).
    """
    file_path = st.secrets.get("INPUT_FILE_PATH", "Final_Dataset_10000_Rows.xlsx")
    full_path = os.path.join(os.path.dirname(__file__), file_path)

    if not os.path.exists(full_path):
        st.error(f"❌ Input file not found at: {full_path}\n"
                 f"Commit your Excel file to the repo or set INPUT_FILE_PATH in secrets.")
        st.stop()

    return pd.read_excel(full_path)


# ── GitHub push helper ─────────────────────────────────────────────────────────
def push_csv_to_github(csv_bytes: bytes, filename: str = "scored_leads.csv"):
    """
    Pushes/updates a CSV file to a GitHub repo using the Contents API.
    Requires these Streamlit secrets:
        GITHUB_TOKEN   – personal access token with repo write scope
        GITHUB_REPO    – e.g. "username/sales_lead_nurture_model"
        GITHUB_BRANCH  – e.g. "main"
        GITHUB_OUTPUT_PATH – e.g. "outputs/scored_leads.csv"
    """
    token     = st.secrets["GITHUB_TOKEN"]
    repo      = st.secrets["GITHUB_REPO"]
    branch    = st.secrets.get("GITHUB_BRANCH", "main")
    file_path = st.secrets.get("GITHUB_OUTPUT_PATH", f"outputs/{filename}")

    api_url = f"https://api.github.com/repos/{repo}/contents/{file_path}"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
    }

    # Check if file already exists (need its SHA to update)
    sha = None
    get_resp = requests.get(api_url, headers=headers, params={"ref": branch})
    if get_resp.status_code == 200:
        sha = get_resp.json().get("sha")

    payload = {
        "message": f"chore: update {file_path} via Streamlit app",
        "content": base64.b64encode(csv_bytes).decode(),
        "branch": branch,
    }
    if sha:
        payload["sha"] = sha

    put_resp = requests.put(api_url, headers=headers, json=payload)

    if put_resp.status_code in (200, 201):
        html_url = put_resp.json()["content"]["html_url"]
        st.success(f"✅ Pushed to GitHub: [{file_path}]({html_url})")
    else:
        st.error(f"❌ GitHub push failed ({put_resp.status_code}): {put_resp.json().get('message')}")


# ── Scored results stored in session state so all tabs can access it ───────────
if "scored_df" not in st.session_state:
    st.session_state.scored_df = None

# ── TABS ───────────────────────────────────────────────────────────────────────
tabs = st.tabs(["🤖 Score & Upload", "📊 KPIs", "📈 Charts", "📤 Export"])

# ══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.title("Sales Lead Nurture Model Dashboard")

    with st.spinner("Loading input data from Excel file..."):
        df = fetch_data()

    if df.empty:
        st.warning("⚠️ No input data found.")
    else:
        st.subheader("📄 Input Data")
        st.dataframe(df.head())

        required = [
            'Age', 'Gender', 'Annual Income', 'Income Bracket', 'Marital Status',
            'Employment Status', 'Region', 'Urban/Rural Flag', 'State', 'ZIP Code',
            'Plan Preference Type', 'Web Form Completion Rate', 'Quote Requested',
            'Application Started', 'Behavior Score', 'Application Submitted',
            'Application Applied'
        ]

        missing = [col for col in required if col not in df.columns]
        if missing:
            st.error(f"❌ Missing columns: {missing}")
        else:
            input_df = df[required]
            proba = model.predict_proba(input_df)[:, 1]
            df["PTB_Score"] = proba * 100

            def tier(score):
                if score >= 90:   return "Platinum"
                elif score >= 75: return "Gold"
                elif score >= 50: return "Silver"
                else:             return "Bronze"

            df["Lead_Tier"] = df["PTB_Score"].apply(tier)
            st.session_state.scored_df = df  # save for other tabs

            st.subheader("✅ Scored Results")
            display_df = df.copy()
            display_df["PTB_Score"] = display_df["PTB_Score"].round(2).astype(str) + "%"
            st.dataframe(display_df)

            csv_bytes = df.to_csv(index=False).encode("utf-8")

            col1, col2 = st.columns(2)

            with col1:
                st.download_button(
                    label="📥 Download Scored CSV",
                    data=csv_bytes,
                    file_name="scored_leads.csv",
                    mime="text/csv",
                )

            with col2:
                if st.button("🚀 Push Scored CSV to GitHub"):
                    push_csv_to_github(csv_bytes, filename="scored_leads.csv")


# ══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    dash_df = st.session_state.scored_df
    if dash_df is not None and not dash_df.empty:
        st.title("📊 KPI Dashboard")
        st.markdown("**Key Funnel Metrics**")

        total     = len(dash_df)
        purchased = dash_df["Policy Purchased"].sum() if "Policy Purchased" in dash_df.columns else 0
        rate      = (purchased / total) * 100 if total else 0

        quote_col = ("Quote Requested (website)"
                     if "Quote Requested (website)" in dash_df.columns
                     else "Quote Requested")
        quote_requested   = dash_df[quote_col].isin(["1", 1, "Yes", True]).sum()
        quote_rate        = (quote_requested / total) * 100 if total else 0

        app_started       = dash_df["Application Started"].isin(["1", 1, "Yes", True]).sum()
        app_started_rate  = (app_started / total) * 100 if total else 0

        app_submitted     = dash_df["Application Submitted"].isin(["1", 1, "Yes", True]).sum()
        app_submitted_rate = (app_submitted / total) * 100 if total else 0

        kpi_values = [
            ("Total Leads",          f"{total}"),
            ("Policies Purchased",   f"{int(purchased)}"),
            ("Conversion Rate",      f"{rate:.2f}%"),
            ("Quote Requested Rate", f"{quote_rate:.2f}%"),
            ("App Started Rate",     f"{app_started_rate:.2f}%"),
            ("App Submitted Rate",   f"{app_submitted_rate:.2f}%"),
        ]

        def build_kpi_row(row_data):
            return "".join([
                f"<div style='flex:1;min-width:180px;max-width:250px;border:1px solid #ddd;"
                f"border-radius:12px;padding:18px;margin:8px;background:#fff;'>"
                f"<div style='font-size:13px;font-weight:500;color:#333;'>{title}</div>"
                f"<div style='font-size:28px;font-weight:700;margin-top:6px;color:#111;'>{value}</div>"
                f"</div>"
                for title, value in row_data
            ])

        st.markdown(
            f"<div style='display:flex;justify-content:space-between;flex-wrap:wrap;'>"
            f"{build_kpi_row(kpi_values[:4])}</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div style='display:flex;justify-content:flex-start;flex-wrap:wrap;'>"
            f"{build_kpi_row(kpi_values[4:])}</div>",
            unsafe_allow_html=True,
        )

        st.markdown("---")
        st.markdown("**Lead Tier Distribution**")

        tier_counts = dash_df["Lead_Tier"].value_counts().to_dict()

        def render_bar(label, count, color):
            return (
                f"<div style='margin-bottom:12px;'>"
                f"<strong>{label}</strong>"
                f"<div style='background:#eee;border-radius:5px;overflow:hidden;'>"
                f"<div style='background:{color};width:{count}px;height:16px;'></div>"
                f"</div>"
                f"<div style='text-align:right;font-weight:bold;'>{count}</div>"
                f"</div>"
            )

        bar_html  = render_bar("🥉 Bronze",   tier_counts.get("Bronze",   0), "#d97c40")
        bar_html += render_bar("🥈 Silver",   tier_counts.get("Silver",   0), "#608cb6")
        bar_html += render_bar("🥇 Gold",     tier_counts.get("Gold",     0), "#f2c84b")
        bar_html += render_bar("🏆 Platinum", tier_counts.get("Platinum", 0), "#bb83f2")
        st.markdown(bar_html, unsafe_allow_html=True)

    else:
        st.info("ℹ️ Score your leads in the '🤖 Score & Upload' tab first.")


# ══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.title("📈 Charts Dashboard")
    dash_df = st.session_state.scored_df
    if dash_df is not None and not dash_df.empty:

        st.subheader("1️⃣ Lead Tier by State")
        states = st.multiselect("Filter by State:", dash_df["State"].dropna().unique())
        filtered1 = dash_df[dash_df["State"].isin(states)] if states else dash_df
        fig1 = px.histogram(filtered1, x="State", color="Lead_Tier", barmode="group")
        st.plotly_chart(fig1, use_container_width=True)

        st.subheader("2️⃣ Lead Tier by Income Bracket")
        fig2 = px.histogram(dash_df, x="Income Bracket", color="Lead_Tier", barmode="stack")
        st.plotly_chart(fig2, use_container_width=True)

        st.subheader("3️⃣ Lead Tier by Age Group")
        if "Age Group" in dash_df.columns:
            ages = st.multiselect("Filter by Age Group:", dash_df["Age Group"].dropna().unique())
            filtered3 = dash_df[dash_df["Age Group"].isin(ages)] if ages else dash_df
            fig3 = px.histogram(filtered3, x="Age Group", color="Lead_Tier", barmode="group")
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.warning("⚠️ 'Age Group' column not found in data.")

        st.subheader("4️⃣ Lead Tier by Gender (Filtered by Employment)")
        jobs = ["All"] + dash_df["Employment Status"].dropna().unique().tolist()
        emp_filter = st.selectbox("Employment Status:", jobs)
        filtered4 = dash_df if emp_filter == "All" else dash_df[dash_df["Employment Status"] == emp_filter]
        fig4 = px.histogram(filtered4, x="Gender", color="Lead_Tier", barmode="group")
        st.plotly_chart(fig4, use_container_width=True)

        st.subheader("5️⃣ Quote Requested vs Purchase Channel")
        quote_col = ("Quote Requested (website)"
                     if "Quote Requested (website)" in dash_df.columns
                     else "Quote Requested")
        gender_options  = dash_df["Gender"].dropna().unique().tolist()
        income_options  = dash_df["Income Bracket"].dropna().unique().tolist()
        quote_options   = dash_df[quote_col].dropna().unique().tolist()

        selected_genders = st.multiselect("Filter by Gender:",         gender_options, default=gender_options)
        selected_incomes = st.multiselect("Filter by Income Bracket:", income_options, default=income_options)
        selected_quotes  = st.multiselect("Filter by Quote Requested:", quote_options, default=quote_options)

        filtered5 = dash_df[
            dash_df["Gender"].isin(selected_genders) &
            dash_df["Income Bracket"].isin(selected_incomes) &
            dash_df[quote_col].isin(selected_quotes)
        ]

        if "Purchase Channel" in dash_df.columns:
            fig5 = px.histogram(
                filtered5, x="Purchase Channel", color=quote_col,
                barmode="group", title="Quote Requested vs Purchase Channel"
            )
            st.plotly_chart(fig5, use_container_width=True)
        else:
            st.warning("⚠️ 'Purchase Channel' column not found in data.")

    else:
        st.info("ℹ️ Score your leads in the '🤖 Score & Upload' tab first.")


# ══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.title("📤 Export Scored Data")
    dash_df = st.session_state.scored_df
    if dash_df is not None and not dash_df.empty:
        csv_bytes = dash_df.to_csv(index=False).encode("utf-8")

        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="📥 Download Scored Leads CSV",
                data=csv_bytes,
                file_name="scored_leads.csv",
                mime="text/csv",
            )
        with col2:
            if st.button("🚀 Push to GitHub", key="export_tab_push"):
                push_csv_to_github(csv_bytes, filename="scored_leads.csv")
    else:
        st.info("ℹ️ Score your leads in the '🤖 Score & Upload' tab first.")
