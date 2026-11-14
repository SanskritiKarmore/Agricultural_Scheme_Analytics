import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.colors as mcolors

# ==========================================================
# PAGE CONFIGURATION
# ==========================================================
st.set_page_config(
    page_title="Agricultural Scheme Analytics",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================================
# CUSTOM CSS
# ==========================================================
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f5f9f6;
    }

    header[data-testid="stHeader"] {
        display: none;
    }

    .block-container {
        max-width: 1200px;
        padding-top: 1.5rem !important;
        padding-bottom: 2rem !important;
    }

    h1 {
        color: #2f6b3f !important;
        font-size: 32px !important;
        font-weight: 700 !important;
        margin-bottom: 8px !important;
    }

    h2 {
        color: #2f6b3f !important;
        font-size: 24px !important;
        font-weight: 650 !important;
        margin-top: 25px !important;
        margin-bottom: 12px !important;
    }

    h3 {
        color: #2f6b3f !important;
        font-size: 19px !important;
        font-weight: 600 !important;
        margin-top: 10px !important;
        margin-bottom: 6px !important;
    }

    p {
        color: #37443a;
        font-size: 15px;
        line-height: 1.6;
    }

    section[data-testid="stSidebar"] {
        background-color: #eaf3eb;
        border-right: 1px solid #d3edd9;
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #2f6b3f !important;
    }

    section[data-testid="stSidebar"] p {
        color: #536356 !important;
    }

    section[data-testid="stSidebar"] label {
        color: #344638 !important;
        font-size: 14px !important;
    }

    [data-testid="stSelectbox"] label {
        color: #344638 !important;
        font-weight: 600 !important;
        font-size: 14px !important;
    }

    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #d3edd9;
        border-radius: 9px;
        padding: 15px;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.04);
    }

    div[data-testid="stMetricLabel"] {
        color: #657365 !important;
        font-size: 13px !important;
    }

    div[data-testid="stMetricValue"] {
        color: #2f6b3f !important;
        font-size: 25px !important;
        font-weight: 650 !important;
    }

    [data-testid="stDataFrame"] {
        border-radius: 7px;
        border: 1px solid #a5d6a7;
        background-color: #ffffff !important;
    }

    [data-testid="stDataFrame"] th,
    [data-testid="stDataFrame"] th * {
        background-color: #c8e6c9 !important;
        color: #000000 !important;
        font-weight: 600 !important;
    }

    [data-testid="stDataFrame"] td:first-child,
    [data-testid="stDataFrame"] th:first-child {
        background-color: #c8e6c9 !important;
        color: #000000 !important;
        font-weight: 600 !important;
    }

    [data-testid="stDataFrame"] td {
        color: #000000 !important;
    }

    .stat-card-light {
        background-color: #ffffff;
        border: 1px solid #d3edd9;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 14px;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.03);
    }

    .stat-card-light h4 {
        color: #2f6b3f;
        font-size: 17px;
        font-weight: 600;
        margin-top: 0;
        margin-bottom: 6px;
    }

    .stat-card-light p {
        color: #37443a;
        font-size: 14px;
        margin-bottom: 0;
        line-height: 1.5;
    }

    [data-testid="stAlert"] {
        border-radius: 8px;
    }

    hr {
        border-color: #d3edd9;
    }

    .footer {
        text-align: center;
        color: #6a766b;
        font-size: 13px;
        padding-top: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ==========================================================
# FILE PATHS
# ==========================================================
PM_KISAN_FILE = "data/RS_Session_265_AU_2109_A_to_D.csv"
HOLDINGS_FILE = "data/RS_Session_267_AU_3204_A_and_B.csv"
PMFBY_FILE = "data/RS_Session_267_AU_513_A_to_C_ii.csv"


# ==========================================================
# HELPER FUNCTIONS FOR FORMULA-BASED STATISTICS
# ==========================================================
def calculate_quartile(sorted_values, p):
    """Calculate a quartile using linear interpolation."""

    n = len(sorted_values)

    if n == 0:
        return np.nan

    if n == 1:
        return sorted_values[0]

    position = (n - 1) * p
    lower_index = int(position)
    upper_index = lower_index + 1

    if upper_index >= n:
        return sorted_values[lower_index]

    fraction = position - lower_index

    return (
        sorted_values[lower_index]
        + fraction * (
            sorted_values[upper_index] - sorted_values[lower_index]
        )
    )


def calculate_descriptive_statistics(series):
    """Calculate descriptive statistics using formulas and loops."""

    values = series.dropna().tolist()
    n = len(values)

    if n == 0:
        return None

    # Mean = sum of values / number of values
    total = 0

    for i in range(n):
        total += values[i]

    mean_value = total / n

    # Median = middle value after sorting
    sorted_values = sorted(values)

    if n % 2 == 1:
        median_value = sorted_values[n // 2]
    else:
        median_value = (
            sorted_values[n // 2 - 1] + sorted_values[n // 2]
        ) / 2

    # Mode = value with the highest frequency
    frequency = {}

    for i in range(n):
        value = values[i]

        if value in frequency:
            frequency[value] += 1
        else:
            frequency[value] = 1

    highest_frequency = 0

    for value in frequency:
        if frequency[value] > highest_frequency:
            highest_frequency = frequency[value]

    modes = []

    for value in frequency:
        if frequency[value] == highest_frequency:
            modes.append(value)

    if highest_frequency == 1:
        mode_text = "No repeated mode"
        mode_num = None
    else:
        modes = sorted(modes)
        mode_text = ", ".join(str(value) for value in modes)
        mode_num = modes[0]

    # Sample variance = sum of squared deviations / (n - 1)
    squared_difference_sum = 0

    for i in range(n):
        difference = values[i] - mean_value
        squared_difference_sum += difference * difference

    variance_value = (
        squared_difference_sum / (n - 1) if n > 1 else np.nan
    )

    # Standard deviation = square root of variance
    std_value = np.sqrt(variance_value) if n > 1 else np.nan

    # Quartiles, IQR and range
    q1 = calculate_quartile(sorted_values, 0.25)
    q3 = calculate_quartile(sorted_values, 0.75)

    iqr_value = q3 - q1
    range_value = sorted_values[-1] - sorted_values[0]

    return {
        "values": values,
        "n": n,
        "sorted_values": sorted_values,
        "mean": mean_value,
        "median": median_value,
        "mode_text": mode_text,
        "mode_num": mode_num,
        "variance": variance_value,
        "std": std_value,
        "q1": q1,
        "q3": q3,
        "iqr": iqr_value,
        "range": range_value
    }


def calculate_pair_statistics(x_values, y_values):
    """Calculate sample covariance and Pearson correlation using formulas."""

    pairs = []

    for i in range(min(len(x_values), len(y_values))):
        if pd.notna(x_values[i]) and pd.notna(y_values[i]):
            pairs.append((float(x_values[i]), float(y_values[i])))

    n = len(pairs)

    if n < 2:
        return np.nan, np.nan

    # Calculate the means
    sum_x = 0
    sum_y = 0

    for i in range(n):
        sum_x += pairs[i][0]
        sum_y += pairs[i][1]

    mean_x = sum_x / n
    mean_y = sum_y / n

    # Calculate covariance numerator and squared deviations
    covariance_sum = 0
    sum_sq_x = 0
    sum_sq_y = 0

    for i in range(n):
        dx = pairs[i][0] - mean_x
        dy = pairs[i][1] - mean_y

        covariance_sum += dx * dy
        sum_sq_x += dx * dx
        sum_sq_y += dy * dy

    # Sample covariance = sum of products / (n - 1)
    covariance_value = covariance_sum / (n - 1)

    # Pearson correlation formula
    denominator = np.sqrt(sum_sq_x * sum_sq_y)

    if denominator == 0:
        pearson_value = np.nan
    else:
        pearson_value = covariance_sum / denominator

    return covariance_value, pearson_value


def calculate_correlation_matrix(df, numeric_columns):
    """Build a Pearson correlation matrix using loops and formulas."""

    corr_matrix = pd.DataFrame(
        index=numeric_columns,
        columns=numeric_columns,
        dtype=float
    )

    for i in range(len(numeric_columns)):
        for j in range(len(numeric_columns)):
            x_values = df[numeric_columns[i]].tolist()
            y_values = df[numeric_columns[j]].tolist()

            _, correlation = calculate_pair_statistics(
                x_values,
                y_values
            )

            corr_matrix.iloc[i, j] = correlation

    return corr_matrix


# ==========================================================
# LOAD DATA
# ==========================================================
@st.cache_data
def load_data():
    pm_kisan = pd.read_csv(PM_KISAN_FILE)
    holdings = pd.read_csv(HOLDINGS_FILE)
    pmfby = pd.read_csv(PMFBY_FILE)

    def is_serial_column(column_name):
        name = str(column_name).strip().lower()

        normalized = (
            name.replace(".", "")
            .replace("_", " ")
            .replace("-", " ")
        )

        normalized = " ".join(normalized.split())

        serial_names = {
            "s no",
            "sno",
            "sl no",
            "slno",
            "serial no",
            "serial number",
            "serial"
        }

        return normalized in serial_names

    # Remove serial-number columns
    for df in [pm_kisan, holdings, pmfby]:
        columns_to_drop = [
            col for col in df.columns if is_serial_column(col)
        ]

        df.drop(columns=columns_to_drop, inplace=True)

    # Convert numerical columns to numerical values
    for df in [pm_kisan, holdings, pmfby]:
        for col in df.columns:
            if col != "State/UT":
                df[col] = pd.to_numeric(df[col], errors="coerce")

    # Combine datasets using State/UT
    combined = pd.merge(
        pm_kisan,
        holdings,
        on="State/UT",
        how="inner"
    )

    combined = pd.merge(
        combined,
        pmfby,
        on="State/UT",
        how="inner"
    )

    return pm_kisan, holdings, pmfby, combined


try:
    pm_kisan, holdings, pmfby, combined = load_data()

except FileNotFoundError as error:
    st.error(
        f"Could not find a dataset file: {error.filename}. "
        "Check that the three CSV files are inside the 'data' folder."
    )
    st.stop()

except Exception as error:
    st.error(f"Could not load the datasets: {error}")
    st.stop()


# ==========================================================
# SIDEBAR NAVIGATION
# ==========================================================
with st.sidebar:
    st.markdown("##")
    st.markdown("### Agricultural Analytics")
    st.caption("Statistical Analysis Dashboard")
    st.divider()

    page = st.radio(
        "Navigation",
        ["Home", "Dataset Overview", "Statistical Analysis"]
    )

    st.divider()
    st.caption("Agricultural Scheme Analytics")
    st.caption("Statistical analysis of agricultural datasets.")


# ==========================================================
# HOME PAGE
# ==========================================================
if page == "Home":

    st.title("Agricultural Scheme Analytics")

    st.write(
        "A statistical dashboard for analysing agricultural scheme data."
    )

    st.header("Project Details")

    st.info(
        """
        **About the Project:** This analytical platform evaluates Indian
        agricultural welfare schemes, including PM-KISAN, PMFBY, and
        agricultural landholding patterns. It uses descriptive statistics
        and relationship analysis to explore regional differences in scheme
        data.

        **Objective:** To evaluate state and union-territory data, compare
        central tendency and variability, and examine relationships between
        landholding patterns and scheme-related measures.

        **Data Sources:** The datasets are based on Rajya Sabha question
        records and agricultural census data. They are integrated using an
        inner join on the `State/UT` column.

        **Statistical Analysis:** Mean, median, mode, sample variance,
        sample standard deviation, interquartile range, range, covariance,
        and Pearson correlation are calculated using explicit formulas
        and loops.

        **Data Visualisation:** Interactive Plotly charts show comparisons,
        distributions, correlation matrices, and pairwise scatter plots.
        """
    )

    st.header("Datasets Used")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
            <div class="stat-card-light">
                <h4>PM-KISAN</h4>
                <p>Agricultural scheme data related to PM-KISAN beneficiaries.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            """
            <div class="stat-card-light">
                <h4>Operational Holdings</h4>
                <p>Data related to marginal operational agricultural holdings.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            """
            <div class="stat-card-light">
                <h4>PMFBY</h4>
                <p>Agricultural crop insurance scheme data.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.header("Statistical Concepts")

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.markdown(
            """
            <div class="stat-card-light">
                <h4>Mean</h4>
                <p>The sum of observations divided by the number of observations.</p>
            </div>

            <div class="stat-card-light">
                <h4>Median</h4>
                <p>The middle value after arranging observations in order.</p>
            </div>

            <div class="stat-card-light">
                <h4>Mode</h4>
                <p>The value or values that occur most frequently.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col_b:
        st.markdown(
            """
            <div class="stat-card-light">
                <h4>Variance</h4>
                <p>Measures squared deviations from the mean.</p>
            </div>

            <div class="stat-card-light">
                <h4>Standard Deviation</h4>
                <p>The square root of variance, showing spread in the original units.</p>
            </div>

            <div class="stat-card-light">
                <h4>IQR &amp; Range</h4>
                <p>Range = maximum − minimum. IQR = third quartile − first quartile.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col_c:
        st.markdown(
            """
            <div class="stat-card-light">
                <h4>Covariance</h4>
                <p>Indicates whether two numerical variables tend to vary together.</p>
            </div>

            <div class="stat-card-light">
                <h4>Pearson Correlation</h4>
                <p>Measures the direction and strength of a linear relationship from −1 to +1.</p>
            </div>
            """,
            unsafe_allow_html=True
        )


# ==========================================================
# DATASET OVERVIEW
# ==========================================================
elif page == "Dataset Overview":

    st.title("Dataset Overview")

    st.write(
        "Explore the structure and contents of the selected dataset."
    )

    dataset_name = st.selectbox(
        "Select Dataset",
        [
            "PM-KISAN",
            "Marginal Operational Holdings",
            "PMFBY",
            "Combined Dataset"
        ],
        key="overview_dataset"
    )

    dataset_map = {
        "PM-KISAN": pm_kisan,
        "Marginal Operational Holdings": holdings,
        "PMFBY": pmfby,
        "Combined Dataset": combined
    }

    df = dataset_map[dataset_name]

    st.header("Dataset Information")

    card1, card2, card3 = st.columns(3)

    with card1:
        st.metric("Dataset", dataset_name)

    with card2:
        st.metric("Number of Rows", df.shape[0])

    with card3:
        st.metric("Number of Columns", df.shape[1])

    st.header("Dataset Preview")

    st.dataframe(
        df.head(10),
        use_container_width=True,
        hide_index=True
    )

    st.header("Column Information")

    column_info = pd.DataFrame({
        "Column Name": df.columns,
        "Data Type": [
            str(df[col].dtype) for col in df.columns
        ],
        "Missing Values": [
            int(df[col].isnull().sum()) for col in df.columns
        ]
    })

    st.dataframe(
        column_info,
        use_container_width=True,
        hide_index=True
    )


# ==========================================================
# STATISTICAL ANALYSIS
# ==========================================================
elif page == "Statistical Analysis":

    st.title("Statistical Analysis")

    st.write(
        "Calculate descriptive statistics and analyse relationships "
        "between numerical variables using formulas."
    )

    dataset_name = st.selectbox(
        "Select Dataset",
        [
            "PM-KISAN",
            "Marginal Operational Holdings",
            "PMFBY",
            "Combined Dataset"
        ],
        key="analysis_dataset"
    )

    dataset_map = {
        "PM-KISAN": pm_kisan,
        "Marginal Operational Holdings": holdings,
        "PMFBY": pmfby,
        "Combined Dataset": combined
    }

    df = dataset_map[dataset_name]

    # Identify numerical columns and exclude serial-number fields
    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    def is_serial_column(column_name):
        name = str(column_name).strip().lower()

        normalized = (
            name.replace(".", "")
            .replace("_", " ")
            .replace("-", " ")
        )

        normalized = " ".join(normalized.split())

        return normalized in {
            "s no",
            "sno",
            "sl no",
            "slno",
            "serial no",
            "serial number",
            "serial"
        }

    numeric_columns = [
        col for col in numeric_columns
        if not is_serial_column(col)
    ]

    if len(numeric_columns) == 0:

        st.warning(
            "No numerical columns are available in this dataset."
        )

    else:

        # ==================================================
        # DESCRIPTIVE STATISTICS
        # ==================================================
        st.header("Descriptive Statistics")

        selected_column = st.selectbox(
            "Select Variable",
            numeric_columns,
            index=0
        )

        stats = calculate_descriptive_statistics(
            df[selected_column]
        )

        if stats is None:

            st.warning(
                "No valid numerical data is available for this variable."
            )

            st.stop()

        mean_val = stats["mean"]
        median_val = stats["median"]
        mode_text = stats["mode_text"]
        mode_num = stats["mode_num"]
        variance_val = stats["variance"]
        std_val = stats["std"]
        iqr_val = stats["iqr"]
        range_val = stats["range"]

        def display_number(value):
            if pd.isna(value):
                return "Not available"

            return f"{value:,.4f}"

        stats_df = pd.DataFrame({
            "Measure": [
                "Mean",
                "Median",
                "Mode",
                "Variance",
                "Standard Deviation",
                "IQR",
                "Range"
            ],
            "Value": [
                display_number(mean_val),
                display_number(median_val),
                mode_text,
                display_number(variance_val),
                display_number(std_val),
                display_number(iqr_val),
                display_number(range_val)
            ]
        })

        st.dataframe(
            stats_df,
            use_container_width=True,
            hide_index=True
        )

        black_text_config = dict(
            color="#000000",
            size=12
        )

        # ==================================================
        # CENTRAL TENDENCY
        # ==================================================
        st.header("Central Tendency Metrics")

        metrics_list = ["Mean", "Median", "Mode"]

        values_list = [
            mean_val,
            median_val,
            mode_num if mode_num is not None else 0
        ]

        labels_list = [
            f"{mean_val:,.2f}",
            f"{median_val:,.2f}",
            f"{mode_num:,.2f}" if mode_num is not None else "No Mode"
        ]

        fig_bar = go.Figure()

        fig_bar.add_trace(
            go.Bar(
                x=metrics_list,
                y=values_list,
                text=labels_list,
                textposition="outside",
                textfont=dict(
                    color="#000000",
                    size=12
                ),
                marker_color=[
                    "#66bb6a",
                    "#81c784",
                    "#a5d6a7"
                ],
                marker=dict(
                    line=dict(
                        color="#4cae51",
                        width=1.5
                    )
                ),
                hovertemplate=(
                    "<b>%{x}</b><br>"
                    "Value: %{y:,.2f}<extra></extra>"
                )
            )
        )

        positive_values = [
            value for value in values_list if value > 0
        ]

        use_log_central = False

        if len(positive_values) >= 2:
            if max(positive_values) / min(positive_values) >= 100:
                use_log_central = True

        if use_log_central:
            fig_bar.update_yaxes(type="log")

        fig_bar.update_layout(
            title=dict(
                text=(
                    "Central Tendency Comparison for "
                    f"<b>{selected_column}</b>"
                ),
                font=dict(
                    color="#000000",
                    size=16
                )
            ),
            yaxis=dict(
                title="Value",
                title_font=black_text_config,
                tickfont=black_text_config,
                gridcolor="#e5e5e5"
            ),
            xaxis=dict(
                title="Statistical Metric",
                title_font=black_text_config,
                tickfont=black_text_config
            ),
            template="plotly_white",
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
            font=dict(color="#000000"),
            height=420,
            margin=dict(
                l=60,
                r=40,
                t=70,
                b=60
            ),
            showlegend=False
        )

        st.plotly_chart(
            fig_bar,
            use_container_width=True
        )

        if use_log_central:
            st.caption(
                "A logarithmic scale is used because the positive statistics "
                "have very different magnitudes."
            )

        if mode_num is not None:
            st.caption(f"Mode = {mode_num:,.2f}")
        else:
            st.caption(
                "No repeated mode exists in the selected data."
            )

        # ==================================================
        # DISTRIBUTION ANALYSIS
        # ==================================================
        st.header("Distribution Analysis")

        data = pd.Series(stats["values"])

        data_min = min(stats["values"])
        data_max = max(stats["values"])
        data_median = median_val

        use_log_scale = (
            len(stats["values"]) > 0
            and data_min > 0
            and data_max / max(data_median, 1) >= 50
        )

        if use_log_scale and data_min < data_max:

            bin_count = 10

            log_edges = np.logspace(
                np.log10(data_min),
                np.log10(data_max),
                bin_count + 1
            )

            counts, edges = np.histogram(
                stats["values"],
                bins=log_edges
            )

            centers = np.sqrt(edges[:-1] * edges[1:])
            widths = edges[1:] - edges[:-1]

            fig_hist = go.Figure()

            fig_hist.add_trace(
                go.Bar(
                    x=centers,
                    y=counts,
                    width=widths * 0.92,
                    marker_color="#81c784",
                    marker_line=dict(
                        color="#4caf50",
                        width=1
                    ),
                    opacity=0.9,
                    hovertemplate=(
                        "Range: %{customdata[0]:,.0f} – "
                        "%{customdata[1]:,.0f}"
                        "<br>Frequency: %{y}<extra></extra>"
                    ),
                    customdata=np.column_stack((
                        edges[:-1],
                        edges[1:]
                    ))
                )
            )

            fig_hist.add_vline(
                x=mean_val,
                line_dash="dash",
                line_color="#e57373",
                annotation_text=f"Mean: {mean_val:,.0f}",
                annotation_position="top right",
                annotation_font=dict(
                    color="#000000",
                    size=11
                )
            )

            fig_hist.add_vline(
                x=median_val,
                line_dash="dot",
                line_color="#2196f3",
                annotation_text=f"Median: {median_val:,.0f}",
                annotation_position="top left",
                annotation_font=dict(
                    color="#000000",
                    size=11
                )
            )

            fig_hist.update_xaxes(
                type="log",
                exponentformat="SI",
                showexponent="all",
                title=selected_column,
                title_font=black_text_config,
                tickfont=black_text_config,
                gridcolor="#e5e5e5"
            )

            st.caption(
                "A logarithmic histogram is used because the selected "
                "variable is highly skewed. The bars use logarithmically "
                "spaced intervals."
            )

        else:

            fig_hist = px.histogram(
                data,
                x=data,
                nbins=12,
                color_discrete_sequence=["#81c784"],
                opacity=0.85
            )

            fig_hist.add_vline(
                x=mean_val,
                line_dash="dash",
                line_color="#e57373",
                annotation_text=f"Mean: {mean_val:,.2f}",
                annotation_position="top right",
                annotation_font=dict(
                    color="#000000",
                    size=11
                )
            )

            fig_hist.add_vline(
                x=median_val,
                line_dash="dot",
                line_color="#2196f3",
                annotation_text=f"Median: {median_val:,.2f}",
                annotation_position="top left",
                annotation_font=dict(
                    color="#000000",
                    size=11
                )
            )

            fig_hist.update_xaxes(
                title=selected_column,
                title_font=black_text_config,
                tickfont=black_text_config,
                gridcolor="#e5e5e5"
            )

        fig_hist.update_layout(
            title=dict(
                text=(
                    "Data Distribution & Frequency for "
                    f"<b>{selected_column}</b>"
                ),
                font=dict(
                    color="#000000",
                    size=16
                )
            ),
            yaxis=dict(
                title="Count / Frequency",
                title_font=black_text_config,
                tickfont=black_text_config,
                gridcolor="#e5e5e5",
                rangemode="tozero"
            ),
            template="plotly_white",
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
            font=dict(color="#000000"),
            height=450,
            bargap=0.08,
            margin=dict(
                l=60,
                r=30,
                t=80,
                b=70
            ),
            showlegend=False
        )

        st.plotly_chart(
            fig_hist,
            use_container_width=True
        )

        # ==================================================
        # PEARSON CORRELATION MATRIX
        # ==================================================
        st.header("Pearson Correlation Matrix Table")

        corr_matrix = calculate_correlation_matrix(
            df,
            numeric_columns
        ).round(4)

        lightest_green_cmap = mcolors.LinearSegmentedColormap.from_list(
            "lightest_green",
            [
                "#f4fbf6",
                "#e8f5e9",
                "#c8e6c9"
            ]
        )

        styled_corr = (
            corr_matrix.style
            .background_gradient(
                cmap=lightest_green_cmap,
                axis=None
            )
            .set_properties(
                **{"color": "#000000"}
            )
        )

        st.dataframe(
            styled_corr,
            use_container_width=True
        )

        st.caption(
            "Pearson correlation values are calculated with the formula "
            "using loops, not the built-in DataFrame corr() method."
        )

        # ==================================================
        # PAIRWISE COVARIANCE AND CORRELATION
        # ==================================================
        st.header("Pairwise Covariance and Correlation")

        if len(numeric_columns) >= 2:

            col1, col2 = st.columns(2)

            with col1:
                x_column = st.selectbox(
                    "Select First Variable",
                    numeric_columns,
                    key="x_variable"
                )

            with col2:
                y_column = st.selectbox(
                    "Select Second Variable",
                    numeric_columns,
                    index=1,
                    key="y_variable"
                )

            pair_data = df[[x_column, y_column]].dropna()

            if len(pair_data) < 2:

                st.warning(
                    "Not enough data for correlation analysis."
                )

            else:

                covariance_value, pearson_value = calculate_pair_statistics(
                    pair_data[x_column].tolist(),
                    pair_data[y_column].tolist()
                )

                correlation_df = pd.DataFrame({
                    "Measure": [
                        "Covariance",
                        "Pearson Correlation"
                    ],
                    "Value": [
                        display_number(covariance_value),
                        display_number(pearson_value)
                    ]
                })

                st.dataframe(
                    correlation_df,
                    use_container_width=True,
                    hide_index=True
                )

                st.info(
                    "Covariance is calculated as the sum of products of "
                    "deviations divided by n − 1. Pearson correlation divides "
                    "that numerator by the square root of the product of "
                    "the two sums of squared deviations."
                )

                # Scatter plot
                st.header("Scatter Plot")

                st.write(
                    f"Relationship between **{x_column}** and **{y_column}**"
                )

                fig_scatter = px.scatter(
                    pair_data,
                    x=x_column,
                    y=y_column,
                    title=f"Scatter Plot: {x_column} vs {y_column}",
                    color_discrete_sequence=["#2e7d32"],
                    opacity=0.75
                )

                x_positive = pair_data[x_column][
                    pair_data[x_column] > 0
                ]

                y_positive = pair_data[y_column][
                    pair_data[y_column] > 0
                ]

                use_log_x = False
                use_log_y = False

                if len(x_positive) > 1:
                    if x_positive.max() / x_positive.min() >= 100:
                        use_log_x = True

                if len(y_positive) > 1:
                    if y_positive.max() / y_positive.min() >= 100:
                        use_log_y = True

                if use_log_x:
                    fig_scatter.update_xaxes(type="log")

                if use_log_y:
                    fig_scatter.update_yaxes(type="log")

                fig_scatter.update_layout(
                    title=dict(
                        text=f"Scatter Plot: {x_column} vs {y_column}",
                        font=dict(
                            color="#000000",
                            size=16
                        )
                    ),
                    yaxis=dict(
                        title=y_column,
                        title_font=black_text_config,
                        tickfont=black_text_config,
                        gridcolor="#e5e5e5"
                    ),
                    xaxis=dict(
                        title=x_column,
                        title_font=black_text_config,
                        tickfont=black_text_config,
                        gridcolor="#e5e5e5"
                    ),
                    template="plotly_white",
                    paper_bgcolor="#ffffff",
                    plot_bgcolor="#ffffff",
                    font=dict(color="#000000"),
                    height=450,
                    margin=dict(
                        l=60,
                        r=30,
                        t=70,
                        b=70
                    ),
                    showlegend=False
                )

                st.plotly_chart(
                    fig_scatter,
                    use_container_width=True
                )

                if use_log_x and use_log_y:

                    st.caption(
                        "Logarithmic scales are used on both axes because "
                        "the variables have highly different magnitudes."
                    )

                elif use_log_x:

                    st.caption(
                        "A logarithmic scale is used on the X-axis because "
                        "the selected variable has a wide positive range."
                    )

                elif use_log_y:

                    st.caption(
                        "A logarithmic scale is used on the Y-axis because "
                        "the selected variable has a wide positive range."
                    )

                else:

                    st.caption(
                        "Each point represents one observation and helps "
                        "visualize the relationship between the two variables."
                    )

        else:

            st.info(
                "At least two numerical columns are required for covariance, "
                "Pearson correlation and the scatter plot."
            )


# ==========================================================
# FOOTER
# ==========================================================
st.divider()

st.markdown(
    """
    <div class="footer">
        Agricultural Scheme Analytics | Statistical Analysis Dashboard
    </div>
    """,
    unsafe_allow_html=True
)