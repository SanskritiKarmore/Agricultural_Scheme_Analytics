import math
from pathlib import Path
from statistics import NormalDist
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
    .stApp { background-color: #f5f9f6; }

    /* Keep the header (it holds the sidebar open/close button) but hide its bar */
    header[data-testid="stHeader"] { background: transparent; }

    .block-container {
        max-width: 1200px;
        padding-top: 1.5rem !important;
        padding-bottom: 2rem !important;
    }

    h1 { color: #2f6b3f !important; font-size: 32px !important;
         font-weight: 700 !important; margin-bottom: 8px !important; }
    h2 { color: #2f6b3f !important; font-size: 24px !important;
         font-weight: 650 !important; margin-top: 25px !important;
         margin-bottom: 12px !important; }
    h3 { color: #2f6b3f !important; font-size: 19px !important;
         font-weight: 600 !important; margin-top: 10px !important;
         margin-bottom: 6px !important; }

    p { color: #37443a; font-size: 15px; line-height: 1.6;     li, ol, ul, label, summary,
    [data-testid="stMarkdownContainer"],
    [data-testid="stMarkdownContainer"] li,
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stCaptionContainer"],
    [data-testid="stWidgetLabel"] p,
    button[data-baseweb="tab"] {
        color: #37443a !important;
    }
    [data-testid="stMarkdownContainer"] strong { color: #1f4a2b !important; }}


    section[data-testid="stSidebar"] {
        background-color: #eaf3eb;
        border-right: 1px solid #d3edd9;
    }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 { color: #2f6b3f !important; }
    section[data-testid="stSidebar"] p { color: #536356 !important; }
    section[data-testid="stSidebar"] label {
        color: #344638 !important; font-size: 14px !important;
    }

    [data-testid="stSelectbox"] label {
        color: #344638 !important; font-weight: 600 !important;
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
        color: #657365 !important; font-size: 13px !important;
    }
    div[data-testid="stMetricValue"] {
        color: #2f6b3f !important; font-size: 25px !important;
        font-weight: 650 !important;
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
        color: #2f6b3f; font-size: 17px; font-weight: 600;
        margin-top: 0; margin-bottom: 6px;
    }
    .stat-card-light p {
        color: #37443a; font-size: 14px; margin-bottom: 0; line-height: 1.5;
    }

    [data-testid="stAlert"] { border-radius: 8px; }
    hr { border-color: #d3edd9; }

    .footer {
        text-align: center; color: #6a766b; font-size: 13px; padding-top: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ==========================================================
# FILE NAMES AND SMALL SETTINGS
# ==========================================================
PM_KISAN_FILE = "RS_Session_265_AU_2109_A_to_D.csv"
HOLDINGS_FILE = "RS_Session_267_AU_3204_A_and_B.csv"
PMFBY_FILE = "RS_Session_267_AU_513_A_to_C_ii.csv"
SCHEMES_FILE = "maharashtra_agriculture_schemes.csv"

# These rows are totals, not states, so they must not be used in the statistics
TOTAL_ROWS = ["Grand Total", "Total", "All India"]

# Text style used inside the charts
black_text_config = dict(color="#000000", size=12)

# Standard normal distribution (mean 0, standard deviation 1)
normal = NormalDist()


def find_file(file_name):
    """Look for a data file in ./data, next to this script, or the current folder."""

    here = Path(__file__).resolve().parent
    candidates = [
        here / "data" / file_name,
        here / file_name,
        Path("data") / file_name,
        Path(file_name)
    ]

    for path in candidates:
        if path.exists():
            return str(path)

    raise FileNotFoundError(file_name)


# ==========================================================
# STREAMLIT VERSION-SAFE DISPLAY HELPERS
# (newer Streamlit uses width="stretch"; older uses use_container_width)
# ==========================================================
def show_chart(fig, key=None):
    try:
        st.plotly_chart(fig, width="stretch", key=key)
    except TypeError:
        st.plotly_chart(fig, use_container_width=True, key=key)


def show_table(data, **kwargs):
    try:
        st.dataframe(data, width="stretch", **kwargs)
    except TypeError:
        st.dataframe(data, use_container_width=True, **kwargs)


# ==========================================================
# HELPER FUNCTIONS FOR FORMULA-BASED STATISTICS (UNIT 1)
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
        + fraction * (sorted_values[upper_index] - sorted_values[lower_index])
    )


def calculate_descriptive_statistics(series):
    """Calculate descriptive statistics using formulas and loops."""

    values = [float(v) for v in series.dropna().tolist()]
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
        mode_list = []
    else:
        modes = sorted(modes)
        mode_text = ", ".join(f"{value:,.2f}" for value in modes)
        mode_num = modes[0]
        mode_list = modes

    # Sample variance = sum of squared deviations / (n - 1)
    squared_difference_sum = 0

    for i in range(n):
        difference = values[i] - mean_value
        squared_difference_sum += difference * difference

    variance_value = (
        squared_difference_sum / (n - 1) if n > 1 else np.nan
    )

    # Standard deviation = square root of variance
    std_value = math.sqrt(variance_value) if n > 1 else np.nan

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
        "mode_list": mode_list,
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

    sum_x = 0
    sum_y = 0

    for i in range(n):
        sum_x += pairs[i][0]
        sum_y += pairs[i][1]

    mean_x = sum_x / n
    mean_y = sum_y / n

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
    denominator = math.sqrt(sum_sq_x * sum_sq_y)

    if denominator == 0:
        pearson_value = np.nan
    else:
        pearson_value = covariance_sum / denominator

    return covariance_value, pearson_value


@st.cache_data(show_spinner=False)
def calculate_correlation_matrix(df, numeric_columns):
    """Build a Pearson correlation matrix using loops and formulas."""

    numeric_columns = list(numeric_columns)

    corr_matrix = pd.DataFrame(
        index=numeric_columns,
        columns=numeric_columns,
        dtype=float
    )

    columns_as_lists = {col: df[col].tolist() for col in numeric_columns}

    for i in range(len(numeric_columns)):
        for j in range(len(numeric_columns)):
            _, correlation = calculate_pair_statistics(
                columns_as_lists[numeric_columns[i]],
                columns_as_lists[numeric_columns[j]]
            )

            corr_matrix.iloc[i, j] = correlation

    return corr_matrix


# ==========================================================
# SMALL DISPLAY HELPERS
# ==========================================================
def display_number(value):
    """Format a number for the tables. Huge values (variance) use scientific form."""

    if value is None or pd.isna(value):
        return "Not available"

    if abs(value) >= 1e12:
        return f"{value:.4e}"

    return f"{value:,.4f}"


def short_number(value):
    """Big numbers get no decimals, small numbers get two."""

    if value is None or pd.isna(value):
        return "N/A"

    if abs(value) >= 1e12:
        return f"{value:.3e}"

    if abs(value) >= 1000:
        return f"{value:,.0f}"

    return f"{value:,.2f}"


def compact_label(x):
    """1000 -> 1K, 2500000 -> 2.5M. Used for log-axis tick labels."""

    for limit, suffix in [(1e9, "B"), (1e6, "M"), (1e3, "K")]:
        if abs(x) >= limit:
            return f"{x / limit:g}{suffix}"

    return f"{x:g}"


def show_p_value(p_value):
    """Very small p-values are shown as '< 0.0001'."""

    if p_value < 0.0001:
        return "< 0.0001"

    return f"{p_value:.4f}"


def describe_correlation(r):
    """Turn a correlation value into a plain sentence."""

    if pd.isna(r):
        return "The correlation could not be calculated for this pair."

    if abs(r) >= 0.7:
        strength = "strong"
    elif abs(r) >= 0.4:
        strength = "moderate"
    else:
        strength = "weak"

    direction = "positive" if r > 0 else "negative"

    return (
        f"The correlation is {r:.3f}, which is a {strength} {direction} "
        "linear relationship. Correlation only shows that two variables "
        "move together, not that one causes the other."
    )


def is_serial_column(column_name):
    """Check whether a column is just a serial number column."""

    name = str(column_name).strip().lower()

    normalized = (
        name.replace(".", "")
        .replace("_", " ")
        .replace("-", " ")
    )

    normalized = " ".join(normalized.split())

    serial_names = {
        "s no", "sno", "sl no", "slno",
        "serial no", "serial number", "serial"
    }

    return normalized in serial_names


def get_numeric_columns(df):
    """List the numerical columns, leaving out serial numbers."""

    numeric_columns = df.select_dtypes(include=np.number).columns.tolist()

    return [col for col in numeric_columns if not is_serial_column(col)]


def style_figure(fig, title, x_title, y_title, height=420):
    """Give every chart the same look."""

    fig.update_layout(
        title=dict(text=title, font=dict(color="#000000", size=16)),
        xaxis=dict(
            title=x_title,
            title_font=black_text_config,
            tickfont=black_text_config,
            gridcolor="#e5e5e5"
        ),
        yaxis=dict(
            title=y_title,
            title_font=black_text_config,
            tickfont=black_text_config,
            gridcolor="#e5e5e5"
        ),
        template="plotly_white",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(color="#000000"),
        height=height,
        margin=dict(l=60, r=30, t=80, b=60),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    return fig


def is_heavily_skewed(values):
    """True when a log scale will show the data much better than a linear one."""

    arr = np.asarray(values, dtype=float)

    if len(arr) < 3 or arr.min() <= 0:
        return False

    median = np.median(arr)

    return (arr.max() / median >= 50) or (arr.max() / arr.min() >= 1000)


def build_histogram(values, bins, column, mean_val, median_val,
                    show_mean, show_median):
    """
    Histogram of the values.

    Heavily skewed data is binned on log10(values) and drawn on a normal
    linear axis whose tick labels read 1K, 10K, 100K ... This keeps every
    bar inside the chart (a bar chart on a true log axis can get a bar edge
    at zero or below, which cannot be drawn).
    """

    values = np.asarray(values, dtype=float)
    use_log = is_heavily_skewed(values)

    if use_log:
        log_values = np.log10(values)
        counts, edges = np.histogram(log_values, bins=bins)

        centers = (edges[:-1] + edges[1:]) / 2
        widths = edges[1:] - edges[:-1]

        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=centers,
                y=counts,
                width=widths * 0.95,
                marker_color="#81c784",
                marker_line=dict(color="#4caf50", width=1),
                opacity=0.9,
                customdata=np.column_stack((10 ** edges[:-1], 10 ** edges[1:])),
                hovertemplate=(
                    "Range: %{customdata[0]:,.0f} – %{customdata[1]:,.0f}"
                    "<br>States/UTs: %{y}<extra></extra>"
                )
            )
        )

        first_power = int(math.floor(edges[0]))
        last_power = int(math.ceil(edges[-1]))
        powers = list(range(first_power, last_power + 1))

        fig.update_xaxes(
            tickmode="array",
            tickvals=powers,
            ticktext=[compact_label(10.0 ** p) for p in powers],
            title=f"{column} (log scale)",
            title_font=black_text_config,
            tickfont=black_text_config,
            gridcolor="#e5e5e5"
        )

        mean_x = math.log10(mean_val) if mean_val > 0 else None
        median_x = math.log10(median_val) if median_val > 0 else None
    else:
        fig = px.histogram(
            x=values,
            nbins=bins,
            color_discrete_sequence=["#81c784"],
            opacity=0.85
        )

        fig.update_traces(
            marker_line=dict(color="#4caf50", width=1),
            hovertemplate="Range: %{x}<br>States/UTs: %{y}<extra></extra>"
        )

        fig.update_xaxes(
            title=column,
            title_font=black_text_config,
            tickfont=black_text_config,
            gridcolor="#e5e5e5"
        )

        mean_x = mean_val
        median_x = median_val

    if show_mean and mean_x is not None:
        fig.add_vline(
            x=mean_x,
            line_dash="dash",
            line_color="#e57373",
            annotation_text=f"Mean: {short_number(mean_val)}",
            annotation_position="top right",
            annotation_font=dict(color="#000000", size=11)
        )

    if show_median and median_x is not None:
        fig.add_vline(
            x=median_x,
            line_dash="dot",
            line_color="#2196f3",
            annotation_text=f"Median: {short_number(median_val)}",
            annotation_position="top left",
            annotation_font=dict(color="#000000", size=11)
        )

    fig.update_layout(
        title=dict(
            text=f"Data Distribution & Frequency for <b>{column}</b>",
            font=dict(color="#000000", size=16)
        ),
        yaxis=dict(
            title="Number of States/UTs",
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
        margin=dict(l=60, r=30, t=80, b=70),
        showlegend=False
    )

    return fig, use_log


# ==========================================================
# HELPER FUNCTIONS FOR HYPOTHESIS TESTING (UNIT 3)
# ==========================================================
def t_pdf(x, df):
    """Height of the Student t curve at x (df = degrees of freedom)."""

    constant = math.exp(
        math.lgamma((df + 1) / 2) - math.lgamma(df / 2)
    ) / math.sqrt(df * math.pi)

    return constant * (1 + x ** 2 / df) ** (-(df + 1) / 2)


def z_pdf(x):
    """Height of the standard normal curve at x."""

    return np.exp(-0.5 * np.asarray(x, dtype=float) ** 2) / math.sqrt(2 * math.pi)


def _beta_continued_fraction(a, b, x):
    """Continued fraction used by the incomplete beta function."""

    tiny = 1e-300
    qab = a + b
    qap = a + 1
    qam = a - 1

    c = 1.0
    d = 1 - qab * x / qap

    if abs(d) < tiny:
        d = tiny

    d = 1 / d
    h = d

    for m in range(1, 300):
        m2 = 2 * m

        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1 + aa * d
        d = tiny if abs(d) < tiny else d
        c = 1 + aa / c
        c = tiny if abs(c) < tiny else c
        d = 1 / d
        h *= d * c

        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1 + aa * d
        d = tiny if abs(d) < tiny else d
        c = 1 + aa / c
        c = tiny if abs(c) < tiny else c
        d = 1 / d
        delta = d * c
        h *= delta

        if abs(delta - 1) < 3e-14:
            break

    return h


def _regularized_incomplete_beta(a, b, x):
    """I_x(a, b), the regularized incomplete beta function."""

    if x <= 0:
        return 0.0

    if x >= 1:
        return 1.0

    front = math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + a * math.log(x) + b * math.log1p(-x)
    )

    if x < (a + 1) / (a + b + 2):
        return front * _beta_continued_fraction(a, b, x) / a

    return 1 - front * _beta_continued_fraction(b, a, 1 - x) / b


def t_upper_tail(t_value, df):
    """
    P(T > t_value) for the t distribution with df degrees of freedom.

    Uses the exact relationship between the t distribution and the
    incomplete beta function, so even very small p-values are accurate.
    """

    x = df / (df + t_value * t_value)
    tail = 0.5 * _regularized_incomplete_beta(df / 2, 0.5, x)

    if t_value >= 0:
        return tail

    return 1 - tail


def t_critical_value(tail_probability, df):
    """Find the t value that leaves tail_probability in the upper tail."""

    low = 0.0
    high = 1e6

    for _ in range(80):
        middle = (low + high) / 2

        if t_upper_tail(middle, df) > tail_probability:
            low = middle
        else:
            high = middle

    return (low + high) / 2


def upper_tail_probability(x, test_name, df):
    """P(statistic > x) for either the Z-test or the t-test."""

    if test_name == "Z-test":
        return normal.cdf(-x)

    return t_upper_tail(x, df)


def calculate_p_value(statistic, tail, test_name, df):
    """P-value for a two-tailed, right-tailed or left-tailed test."""

    if tail == "two-tailed":
        p_value = 2 * upper_tail_probability(abs(statistic), test_name, df)
        return min(p_value, 1.0)

    if tail == "right-tailed":
        return upper_tail_probability(statistic, test_name, df)

    # Left-tailed: the curve is symmetric, so flip the sign
    return upper_tail_probability(-statistic, test_name, df)


def get_critical_values(tail, alpha, test_name, df):
    """Return (lower, upper) critical values. None means no cut-off there."""

    tail_probability = alpha / 2 if tail == "two-tailed" else alpha

    if test_name == "Z-test":
        value = normal.inv_cdf(1 - tail_probability)
    else:
        value = t_critical_value(tail_probability, df)

    if tail == "two-tailed":
        return -value, value

    if tail == "right-tailed":
        return None, value

    return -value, None


def shade_between(fig, pdf_function, start, end, color, name, show_legend):
    """Shade the area under a curve between start and end."""

    if end <= start:
        return

    x_part = np.linspace(start, end, 150)
    y_part = np.asarray(pdf_function(x_part), dtype=float)

    fig.add_trace(
        go.Scatter(
            x=x_part,
            y=y_part,
            mode="lines",
            line=dict(width=0),
            fill="tozeroy",
            fillcolor=color,
            name=name,
            showlegend=show_legend,
            hoverinfo="skip"
        )
    )


def show_mean_test(sample_mean, n, mu_0, test_name, tail, alpha, spread, label):
    """
    Run a one-sample Z-test or t-test and show every step on the page.

    spread = population standard deviation (Z-test)
             or sample standard deviation (t-test)
    """

    df = n - 1
    standard_error = spread / math.sqrt(n)
    statistic = (sample_mean - mu_0) / standard_error

    p_value = calculate_p_value(statistic, tail, test_name, df)
    lower_cut, upper_cut = get_critical_values(tail, alpha, test_name, df)

    reject = p_value < alpha

    if tail == "two-tailed":
        symbol = "≠"
        phrase = "different from"
    elif tail == "right-tailed":
        symbol = ">"
        phrase = "greater than"
    else:
        symbol = "<"
        phrase = "less than"

    # ---------------- Step 1 ----------------
    st.subheader("Step 1: State the hypotheses")

    st.markdown(
        f"""
        - **H₀:** μ = {mu_0:,.2f}
        - **Hₐ:** μ {symbol} {mu_0:,.2f}

        Here μ is the true mean of **{label}**.
        """
    )

    # ---------------- Step 2 ----------------
    st.subheader("Step 2: Calculate the test statistic")

    if test_name == "Z-test":
        st.latex(r"z = \frac{\bar{x} - \mu_0}{\sigma / \sqrt{n}}")
    else:
        st.latex(r"t = \frac{\bar{x} - \mu_0}{s / \sqrt{n}}")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Sample mean (x̄)", short_number(sample_mean))

    with col2:
        st.metric("Standard error", short_number(standard_error))

    with col3:
        if test_name == "Z-test":
            st.metric("z statistic", f"{statistic:.4f}")
        else:
            st.metric(f"t statistic (df = {df})", f"{statistic:.4f}")

    with col4:
        st.metric("p-value", show_p_value(p_value))

    # ---------------- Step 3 ----------------
    st.subheader("Step 3: Critical value and rejection region")

    if tail == "two-tailed":
        st.write(
            f"At α = {alpha}, reject H₀ if the statistic is below "
            f"**{lower_cut:.4f}** or above **{upper_cut:.4f}**."
        )
    elif tail == "right-tailed":
        st.write(
            f"At α = {alpha}, reject H₀ if the statistic is above "
            f"**{upper_cut:.4f}**."
        )
    else:
        st.write(
            f"At α = {alpha}, reject H₀ if the statistic is below "
            f"**{lower_cut:.4f}**."
        )

    # Work out how wide the chart should be
    limit = 4.5

    for value in [statistic, lower_cut, upper_cut]:
        if value is not None:
            limit = max(limit, abs(value) + 0.5)

    limit = min(limit, 10)

    x = np.linspace(-limit, limit, 600)

    if test_name == "Z-test":
        pdf_function = z_pdf
        curve_name = "Standard normal curve"
    else:
        def pdf_function(values):
            return t_pdf(np.asarray(values, dtype=float), df)

        curve_name = f"t curve (df = {df})"

    y = pdf_function(x)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines",
            line=dict(color="#2e7d32", width=2.5),
            name=curve_name
        )
    )

    shade_color = "rgba(229, 115, 115, 0.55)"
    legend_shown = False

    if upper_cut is not None and upper_cut < limit:
        shade_between(
            fig, pdf_function, upper_cut, limit,
            shade_color, "Rejection region", True
        )
        legend_shown = True

    if lower_cut is not None and lower_cut > -limit:
        shade_between(
            fig, pdf_function, -limit, lower_cut,
            shade_color, "Rejection region", not legend_shown
        )

    if abs(statistic) <= limit:
        fig.add_vline(
            x=statistic,
            line_dash="dash",
            line_color="#1565c0",
            annotation_text=f"Test statistic = {statistic:.2f}",
            annotation_position="top",
            annotation_font=dict(color="#000000", size=11)
        )

    fig = style_figure(
        fig,
        "Where the test statistic falls",
        "z value" if test_name == "Z-test" else "t value",
        "Density"
    )

    show_chart(fig)

    st.caption(
        "The curve shows how the test statistic behaves when H₀ is true. "
        "The red area is the rejection region, whose total area equals α. "
        "If the blue dashed line lands inside the red area, the sample "
        "result would be very unusual under H₀, so we reject H₀."
    )

    if abs(statistic) > limit:
        st.caption(
            "The test statistic is too far out to appear on this chart."
        )

    # ---------------- Step 4 ----------------
    st.subheader("Step 4: Decision and conclusion")

    if reject:
        decision = (
            f"Because the p-value ({show_p_value(p_value)}) is less than "
            f"α = {alpha}, we **reject H₀**."
        )
        meaning = (
            f"There is convincing evidence that the true mean of {label} "
            f"is {phrase} {mu_0:,.2f}."
        )
        st.success(decision + " " + meaning)
    else:
        decision = (
            f"Because the p-value ({show_p_value(p_value)}) is not less "
            f"than α = {alpha}, we **fail to reject H₀**."
        )
        meaning = (
            f"There is not convincing evidence that the true mean of "
            f"{label} is {phrase} {mu_0:,.2f}."
        )
        st.info(decision + " " + meaning)

    st.caption(
        "The conclusion follows the two-sentence format from class: first "
        "the decision about H₀ using the p-value and α, then what it means "
        "in context. We never say 'accept H₀'."
    )


# ==========================================================
# HELPER FUNCTIONS FOR TYPE I AND TYPE II ERRORS
# ==========================================================
def error_probabilities(mu_0, mu_1, sigma, n, alpha, tail):
    """
    Z-test with known sigma.

    Returns beta (the Type II error probability) and the cut-off values
    for the sample mean. beta = P(not rejecting H0 when the true mean is mu_1).
    """

    standard_error = sigma / math.sqrt(n)
    true_curve = NormalDist(mu_1, standard_error)

    lower_cut = None
    upper_cut = None

    if tail == "two-tailed":
        z_value = normal.inv_cdf(1 - alpha / 2)
        lower_cut = mu_0 - z_value * standard_error
        upper_cut = mu_0 + z_value * standard_error
        beta = true_curve.cdf(upper_cut) - true_curve.cdf(lower_cut)

    elif tail == "right-tailed":
        z_value = normal.inv_cdf(1 - alpha)
        upper_cut = mu_0 + z_value * standard_error
        beta = true_curve.cdf(upper_cut)

    else:
        z_value = normal.inv_cdf(1 - alpha)
        lower_cut = mu_0 - z_value * standard_error
        beta = 1 - true_curve.cdf(lower_cut)

    return beta, lower_cut, upper_cut


def simulate_rejection_rate(true_mean, mu_0, sigma, n, alpha, tail, repeats, rng):
    """Repeat the experiment many times and see how often H0 is rejected."""

    samples = rng.normal(true_mean, sigma, size=(repeats, n))
    sample_means = samples.mean(axis=1)

    z_values = (sample_means - mu_0) / (sigma / math.sqrt(n))

    if tail == "two-tailed":
        rejected = np.abs(z_values) > normal.inv_cdf(1 - alpha / 2)
    elif tail == "right-tailed":
        rejected = z_values > normal.inv_cdf(1 - alpha)
    else:
        rejected = z_values < -normal.inv_cdf(1 - alpha)

    return rejected.mean()


# ==========================================================
# NEW: HELPERS FOR UNIT II (PROBABILITY DISTRIBUTIONS)
# ==========================================================
def binomial_pmf(k, n, p):
    """P(X = k) for a binomial distribution."""

    if k < 0 or k > n:
        return 0.0

    return math.comb(n, k) * (p ** k) * ((1 - p) ** (n - k))


def poisson_pmf(k, lam):
    """P(X = k) for a Poisson distribution."""

    if k < 0:
        return 0.0

    return math.exp(-lam + k * math.log(lam) - math.lgamma(k + 1))


def bar_pmf_chart(ks, probs, highlight, title, x_title, key):
    """Bar chart of a discrete distribution; highlighted bars are darker."""

    colors = ["#2e7d32" if k in highlight else "#a5d6a7" for k in ks]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=list(ks),
            y=list(probs),
            marker_color=colors,
            marker_line=dict(color="#4caf50", width=1),
            hovertemplate="X = %{x}<br>P = %{y:.4f}<extra></extra>"
        )
    )

    fig = style_figure(fig, title, x_title, "Probability", height=400)
    fig.update_layout(showlegend=False)

    show_chart(fig, key=key)


# ==========================================================
# NEW: HELPERS FOR UNIT III (CONFIDENCE INTERVALS, F-TEST)
# ==========================================================
def f_upper_tail(f_value, df1, df2):
    """P(F > f_value) for the F distribution (used by one-way ANOVA)."""

    if f_value <= 0:
        return 1.0

    x = df2 / (df2 + df1 * f_value)

    return _regularized_incomplete_beta(df2 / 2, df1 / 2, x)


def confidence_interval_for_mean(mean_value, spread, n, confidence, method):
    """Return (lower, upper, critical value, margin of error)."""

    standard_error = spread / math.sqrt(n)

    if method == "Z (σ known)":
        critical = normal.inv_cdf(1 - (1 - confidence) / 2)
    else:
        critical = t_critical_value((1 - confidence) / 2, n - 1)

    margin = critical * standard_error

    return mean_value - margin, mean_value + margin, critical, margin


# ==========================================================
# NEW: HELPERS FOR UNIT IV (REGRESSION AND METRICS)
# ==========================================================
def regression_metrics(actual, predicted):
    """MAE, MSE, RMSE and R² from the formulas."""

    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    errors = actual - predicted
    n = len(actual)

    mae = float(np.sum(np.abs(errors)) / n)
    mse = float(np.sum(errors ** 2) / n)
    rmse = math.sqrt(mse)

    ss_res = float(np.sum(errors ** 2))
    ss_tot = float(np.sum((actual - actual.mean()) ** 2))

    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R²": r2}


def fit_ols(X, y):
    """
    Ordinary least squares with an intercept (normal equation).

    beta = (XᵀX)⁻¹ Xᵀy. Returns coefficients with standard errors,
    t statistics and p-values.
    """

    X = np.asarray(X, dtype=float).reshape(len(y), -1)
    y = np.asarray(y, dtype=float)

    n = len(y)
    design = np.column_stack([np.ones(n), X])
    k = design.shape[1] - 1

    xtx_inverse = np.linalg.pinv(design.T @ design)
    beta = xtx_inverse @ design.T @ y

    fitted = design @ beta
    residuals = y - fitted

    df_resid = n - k - 1
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))

    mse_resid = ss_res / df_resid if df_resid > 0 else float("nan")

    standard_errors = np.sqrt(np.maximum(np.diag(xtx_inverse) * mse_resid, 0))

    with np.errstate(divide="ignore", invalid="ignore"):
        t_values = beta / standard_errors

    p_values = [
        min(1.0, 2 * t_upper_tail(abs(t), df_resid))
        if (df_resid > 0 and np.isfinite(t)) else float("nan")
        for t in t_values
    ]

    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    adj_r2 = (
        1 - (1 - r2) * (n - 1) / df_resid
        if df_resid > 0 and not np.isnan(r2) else float("nan")
    )

    ss_reg = ss_tot - ss_res
    f_value = (
        (ss_reg / k) / mse_resid
        if k > 0 and df_resid > 0 and mse_resid > 0 else float("nan")
    )
    f_p = (
        f_upper_tail(f_value, k, df_resid) if not np.isnan(f_value) else float("nan")
    )

    return {
        "beta": beta,
        "se": standard_errors,
        "t": t_values,
        "p": p_values,
        "fitted": fitted,
        "residuals": residuals,
        "r2": r2,
        "adj_r2": adj_r2,
        "df_resid": df_resid,
        "f": f_value,
        "f_p": f_p,
        "k": k
    }


def predict_ols(beta, X):
    """Predictions from fitted coefficients (beta[0] is the intercept)."""

    X = np.asarray(X, dtype=float).reshape(-1, len(beta) - 1)

    return beta[0] + X @ beta[1:]


def durbin_watson(residuals):
    """Durbin–Watson statistic (about 2 means no first-order autocorrelation)."""

    residuals = np.asarray(residuals, dtype=float)
    denominator = float(np.sum(residuals ** 2))

    if denominator == 0 or len(residuals) < 2:
        return float("nan")

    return float(np.sum(np.diff(residuals) ** 2) / denominator)


def run_regression(df, x_columns, y_column, use_log, test_share, key):
    """Fit a regression, show the results, metrics, plots and interpretation."""

    columns = list(dict.fromkeys(list(x_columns) + [y_column]))
    data = df[["State/UT"] + columns].dropna().copy()

    if use_log:
        data = data[(data[columns] > 0).all(axis=1)].copy()

        for col in columns:
            data[col] = np.log10(data[col])

    k = len(x_columns)
    n = len(data)

    names = {col: (f"log10({col})" if use_log else col) for col in columns}

    if n < k + 4:
        st.warning(
            f"Only {n} States/UTs have complete data for these variables. "
            "Choose fewer predictors or different variables."
        )
        return None

    # Optional train / test split
    if test_share > 0:
        order = np.random.default_rng(42).permutation(n)
        n_test = max(2, int(round(n * test_share)))
        test_index = order[:n_test]
        train_index = order[n_test:]
    else:
        train_index = np.arange(n)
        test_index = np.array([], dtype=int)

    if len(train_index) < k + 3:
        st.warning("The training set is too small. Reduce the test share.")
        return None

    X_all = data[list(x_columns)].to_numpy(dtype=float)
    y_all = data[y_column].to_numpy(dtype=float)

    model = fit_ols(X_all[train_index], y_all[train_index])
    beta = model["beta"]

    train_pred = predict_ols(beta, X_all[train_index])
    train_metrics = regression_metrics(y_all[train_index], train_pred)

    st.write(
        f"Based on **{n}** States/UTs with complete data"
        + (" (log10 applied; rows with zero or negative values dropped)."
           if use_log else ".")
    )

    # ---------------- Equation ----------------
    equation = f"ŷ = {beta[0]:,.4g}"

    for i, col in enumerate(x_columns):
        sign = "+" if beta[i + 1] >= 0 else "−"
        equation += f" {sign} {abs(beta[i + 1]):,.4g} × [{names[col]}]"

    st.subheader("Fitted equation")
    st.code(equation, language=None)

    # ---------------- Coefficient table ----------------
    st.subheader("Coefficients")

    labels = ["Intercept"] + [names[c] for c in x_columns]

    coefficient_table = pd.DataFrame({
        "Term": labels,
        "Coefficient": [f"{b:,.4g}" for b in beta],
        "Std. error": [f"{s:,.4g}" for s in model["se"]],
        "t": [f"{t:,.3f}" for t in model["t"]],
        "p-value": [show_p_value(p) if not np.isnan(p) else "N/A"
                    for p in model["p"]]
    })

    show_table(coefficient_table, hide_index=True)

    # ---------------- Metrics ----------------
    st.subheader("Model evaluation metrics")

    metric_rows = {
        "Metric": ["MAE", "MSE", "RMSE", "R² score"],
        "Training data": [
            f"{train_metrics['MAE']:,.4g}", f"{train_metrics['MSE']:,.4g}",
            f"{train_metrics['RMSE']:,.4g}", f"{train_metrics['R²']:.4f}"
        ]
    }

    test_metrics = None

    if len(test_index) > 0:
        test_pred = predict_ols(beta, X_all[test_index])
        test_metrics = regression_metrics(y_all[test_index], test_pred)

        metric_rows["Test data"] = [
            f"{test_metrics['MAE']:,.4g}", f"{test_metrics['MSE']:,.4g}",
            f"{test_metrics['RMSE']:,.4g}", f"{test_metrics['R²']:.4f}"
        ]

    show_table(pd.DataFrame(metric_rows), hide_index=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("R²", f"{model['r2']:.4f}")

    with col2:
        st.metric("Adjusted R²", f"{model['adj_r2']:.4f}")

    with col3:
        st.metric(
            f"F statistic (df = {model['k']}, {model['df_resid']})",
            f"{model['f']:.3f}" if not np.isnan(model["f"]) else "N/A"
        )

    st.caption(
        "MAE = average absolute error. MSE = average squared error. "
        "RMSE = √MSE, in the same units as y. R² = share of the variation "
        "in y explained by the model. Adjusted R² penalises extra "
        "predictors. The overall F-test p-value is "
        f"{show_p_value(model['f_p']) if not np.isnan(model['f_p']) else 'N/A'}."
    )

    if test_metrics is not None:
        st.caption(
            "Test-data metrics show how well the model predicts States/UTs "
            "it was not trained on. If they are much worse than the "
            "training metrics, the model is over-fitting."
        )

    # ---------------- Plots ----------------
    plot_col1, plot_col2 = st.columns(2)

    with plot_col1:
        fig_fit = go.Figure()

        fig_fit.add_trace(
            go.Scatter(
                x=y_all[train_index], y=train_pred, mode="markers",
                text=data["State/UT"].to_numpy()[train_index],
                marker=dict(color="#2e7d32", size=9, opacity=0.75),
                name="Training",
                hovertemplate="<b>%{text}</b><br>Actual: %{x:,.3f}"
                              "<br>Predicted: %{y:,.3f}<extra></extra>"
            )
        )

        if len(test_index) > 0:
            fig_fit.add_trace(
                go.Scatter(
                    x=y_all[test_index], y=test_pred, mode="markers",
                    text=data["State/UT"].to_numpy()[test_index],
                    marker=dict(color="#1565c0", size=9, opacity=0.85,
                                symbol="diamond"),
                    name="Test",
                    hovertemplate="<b>%{text}</b><br>Actual: %{x:,.3f}"
                                  "<br>Predicted: %{y:,.3f}<extra></extra>"
                )
            )

        low = float(min(y_all.min(), train_pred.min()))
        high = float(max(y_all.max(), train_pred.max()))

        fig_fit.add_trace(
            go.Scatter(
                x=[low, high], y=[low, high], mode="lines",
                line=dict(color="#e57373", dash="dash"),
                name="Perfect prediction"
            )
        )

        fig_fit = style_figure(
            fig_fit, "Actual vs predicted",
            f"Actual {names[y_column]}", "Predicted", height=400
        )

        show_chart(fig_fit, key=f"{key}_fit")

    with plot_col2:
        fig_res = go.Figure()

        fig_res.add_trace(
            go.Scatter(
                x=model["fitted"], y=model["residuals"], mode="markers",
                text=data["State/UT"].to_numpy()[train_index],
                marker=dict(color="#2e7d32", size=9, opacity=0.75),
                name="Residuals",
                hovertemplate="<b>%{text}</b><br>Fitted: %{x:,.3f}"
                              "<br>Residual: %{y:,.3f}<extra></extra>"
            )
        )

        fig_res.add_hline(y=0, line_dash="dash", line_color="#e57373")

        fig_res = style_figure(
            fig_res, "Residuals vs fitted values",
            "Fitted value", "Residual", height=400
        )
        fig_res.update_layout(showlegend=False)

        show_chart(fig_res, key=f"{key}_res")

    if k == 1:
        x_name = x_columns[0]

        fig_line = go.Figure()

        fig_line.add_trace(
            go.Scatter(
                x=data[x_name], y=data[y_column], mode="markers",
                text=data["State/UT"],
                marker=dict(color="#2e7d32", size=9, opacity=0.75),
                name="States/UTs",
                hovertemplate="<b>%{text}</b><br>x: %{x:,.3f}"
                              "<br>y: %{y:,.3f}<extra></extra>"
            )
        )

        x_line = np.array([data[x_name].min(), data[x_name].max()])

        fig_line.add_trace(
            go.Scatter(
                x=x_line, y=beta[0] + beta[1] * x_line, mode="lines",
                line=dict(color="#e57373", width=2.5),
                name="Regression line"
            )
        )

        fig_line = style_figure(
            fig_line, "Regression line",
            names[x_name], names[y_column], height=420
        )

        show_chart(fig_line, key=f"{key}_line")

    # ---------------- Assumption checks ----------------
    st.subheader("Checking the regression assumptions")

    residuals = model["residuals"]
    fitted = model["fitted"]

    res_sd = residuals.std(ddof=1)

    skew = (
        float(np.mean(((residuals - residuals.mean()) / res_sd) ** 3))
        if res_sd > 0 else float("nan")
    )

    hetero = (
        float(np.corrcoef(np.abs(residuals), fitted)[0, 1])
        if res_sd > 0 and fitted.std() > 0 else float("nan")
    )

    dw = durbin_watson(residuals)

    check_table = pd.DataFrame({
        "Assumption": [
            "Linearity", "Normal residuals",
            "Constant variance (homoscedasticity)", "Independent errors"
        ],
        "How we check it": [
            "Residuals vs fitted plot should show no curve",
            f"Residual skewness = {skew:.2f} (near 0 is good)",
            f"Corr(|residual|, fitted) = {hetero:.2f} (near 0 is good)",
            f"Durbin–Watson = {dw:.2f} (about 2 is good)"
        ],
        "Looks OK?": [
            "Check the plot",
            "Yes" if abs(skew) < 1 else "Doubtful",
            "Yes" if abs(hetero) < 0.4 else "Doubtful",
            "Yes" if 1.5 <= dw <= 2.5 else "Doubtful"
        ]
    })

    show_table(check_table, hide_index=True)

    st.caption(
        "States/UTs are not a random sample and the raw data are strongly "
        "right-skewed, so assumptions often fail on the raw scale. Tick "
        "'Use log10 of the variables' to see if the fit improves."
    )

    if k > 1:
        st.caption(
            "With several predictors also check multicollinearity: if the "
            "predictors are strongly correlated with each other, the "
            "individual coefficients become unstable."
        )

    # ---------------- Interpretation ----------------
    st.subheader("Interpretation")

    r2_percent = model["r2"] * 100
    lines = [
        f"The model explains about **{r2_percent:.1f}%** of the variation in "
        f"{names[y_column]} (R² = {model['r2']:.3f}, adjusted R² = "
        f"{model['adj_r2']:.3f})."
    ]

    for i, col in enumerate(x_columns):
        slope = beta[i + 1]
        p_value = model["p"][i + 1]
        significant = (not np.isnan(p_value)) and p_value < 0.05
        direction = "increase" if slope > 0 else "decrease"
        holding = " holding the other predictors fixed" if k > 1 else ""

        if use_log:
            effect = (
                f"a 10-fold increase in {col} is linked with y being multiplied "
                f"by about {10 ** slope:,.3g}{holding}"
            )
        else:
            effect = (
                f"each 1-unit increase in {col} is linked with a "
                f"{abs(slope):,.4g}-unit {direction} in {y_column}{holding}"
            )

        status = (
            "This effect is statistically significant at the 5% level"
            if significant else
            "This effect is not statistically significant at the 5% level"
        )

        lines.append(f"- {effect}. {status} (p = {show_p_value(p_value)}).")

    lines.append(
        "Regression shows association, not causation, and with only a few "
        "dozen States/UTs the results are best used to practise the method."
    )

    st.markdown("\n\n".join(lines))

    return model


# ==========================================================
# NEW: SHARED DECISION HELPER
# ==========================================================
def show_decision(p_value, alpha, claim):
    """Two-sentence conclusion: decision about H0, then meaning in context."""

    if p_value < alpha:
        st.success(
            f"Because the p-value ({show_p_value(p_value)}) is less than "
            f"α = {alpha}, we **reject H₀**. There is convincing evidence "
            f"that {claim}."
        )
    else:
        st.info(
            f"Because the p-value ({show_p_value(p_value)}) is not less than "
            f"α = {alpha}, we **fail to reject H₀**. There is not convincing "
            f"evidence that {claim}."
        )


# ==========================================================
# LOAD DATA
# ==========================================================
def clean_column_name(name):
    """Tidy column names, e.g. '2024-25 ( as on 22-07-2024) - ...'."""

    name = " ".join(str(name).split())
    name = name.replace("( as on", "(as on")

    return name


@st.cache_data
def load_data():
    raw_frames = [
        pd.read_csv(find_file(PM_KISAN_FILE), na_values=["NA", "N/A", "-", ""]),
        pd.read_csv(find_file(HOLDINGS_FILE), na_values=["NA", "N/A", "-", ""]),
        pd.read_csv(find_file(PMFBY_FILE), na_values=["NA", "N/A", "-", ""])
    ]

    cleaned = []

    for df in raw_frames:
        df = df.copy()
        df.columns = [clean_column_name(col) for col in df.columns]

        # Remove serial-number columns
        columns_to_drop = [col for col in df.columns if is_serial_column(col)]
        df = df.drop(columns=columns_to_drop)

        # Remove the Total / Grand Total / All India rows
        df["State/UT"] = df["State/UT"].astype(str).str.strip()
        df = df[~df["State/UT"].isin(TOTAL_ROWS)].copy()

        # Convert numerical columns to numbers (stray text becomes missing)
        for col in df.columns:
            if col != "State/UT":
                df[col] = pd.to_numeric(df[col], errors="coerce")

        cleaned.append(df.reset_index(drop=True))

    pm_kisan, holdings, pmfby = cleaned

    # Combine datasets using State/UT (inner join: only states present in all three)
    combined = pd.merge(pm_kisan, holdings, on="State/UT", how="inner")
    combined = pd.merge(combined, pmfby, on="State/UT", how="inner")

    return pm_kisan, holdings, pmfby, combined


@st.cache_data
def load_schemes():
    schemes = pd.read_csv(find_file(SCHEMES_FILE))
    schemes.columns = [col.strip() for col in schemes.columns]

    return schemes


try:
    pm_kisan, holdings, pmfby, combined = load_data()

except FileNotFoundError as error:
    st.error(
        f"Could not find the dataset file: {error}. Put the CSV files in a "
        "'data' folder next to this script."
    )
    st.stop()

except Exception as error:
    st.error(f"Could not load the datasets: {error}")
    st.stop()

try:
    schemes = load_schemes()
except Exception:
    schemes = None

datasets = {
    "PM-KISAN": pm_kisan,
    "Marginal Operational Holdings": holdings,
    "PMFBY": pmfby,
    "Combined Dataset": combined
}

dataset_names = list(datasets.keys())


# ==========================================================
# SIDEBAR NAVIGATION
# ==========================================================
with st.sidebar:
    st.markdown("### Agricultural Analytics")
    st.caption("Statistical Analysis Dashboard")
    st.divider()

    page = st.radio(
        "Navigation",
        [
            "Home",
            "Dataset Overview",
            "Statistical Analysis",
            "Probability and Distributions",
            "Sampling and CLT",
            "Confidence Intervals and Group Tests",
            "Hypothesis Testing",
            "Type I and Type II Errors",
            "Regression Analysis",
            "Analytics Workflow",
            "Maharashtra Schemes"
        ]
    )

    st.divider()
    st.caption("Agricultural Scheme Analytics")
    st.caption("Statistical analysis of agricultural datasets.")


# ==========================================================
# HOME PAGE
# ==========================================================
if page == "Home":

    st.title("Agricultural Scheme Analytics")

    st.write("A statistical dashboard for analysing agricultural scheme data.")

    st.header("Project Details")

    st.info(
        f"""
        **About the Project:** This analytical platform evaluates Indian
        agricultural welfare schemes, including PM-KISAN, PMFBY, and
        agricultural landholding patterns. It uses descriptive statistics,
        probability, relationship analysis, statistical inference and
        regression to explore regional differences in scheme data.

        **Objective:** To evaluate state and union-territory data, compare
        central tendency and variability, examine relationships between
        landholding patterns and scheme-related measures, test claims
        about the data using sampling theory and hypothesis tests, and
        build simple predictive models.

        **Data Sources:** Three Rajya Sabha question tables: PM-KISAN
        beneficiaries and amounts, 2018-19 to 2024-25 (Session 265);
        marginal operational holdings (Session 267); and PMFBY farmer
        applications insured (Session 267). They are joined on `State/UT`
        with an inner join, which leaves **{len(combined)} States/UTs**
        that appear in all three tables. Rows that only hold totals
        (Grand Total, Total, All India) are removed before any calculation.

        **Statistical Analysis:** Mean, median, mode, sample variance,
        sample standard deviation, interquartile range, range, covariance,
        and Pearson correlation are calculated using explicit formulas
        and loops. Probability distributions (binomial, Poisson, normal),
        sampling distributions, the Central Limit Theorem, confidence
        intervals, Z and t hypothesis tests, independent and paired t-tests,
        one-way ANOVA, Type I and Type II errors, and simple and multiple
        linear regression with MAE, MSE, RMSE and R² are also covered.

        **Data Visualisation:** Interactive Plotly charts show comparisons,
        distributions, correlation matrices, sampling distributions,
        rejection regions and regression diagnostics.
        """
    )

    st.warning(
        "**Data notes:** PM-KISAN has no figures ('NA') for Delhi, Lakshadweep "
        "and Sikkim in 2018-19, and for West Bengal in 2018-19 to 2020-21. "
        "These are treated as missing and left out of calculations on that "
        "column. PMFBY lists only 25 States/UTs, so the combined dataset "
        "has 25 rows. The holdings table lists Dadra and Nagar Haveli and "
        "Daman and Diu separately and has no Ladakh row, so these names do "
        "not match the other tables."
    )

    st.header("Datasets Used")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"""
            <div class="stat-card-light">
                <h4>PM-KISAN</h4>
                <p>Beneficiaries and amount transferred (₹ crore) for
                {len(pm_kisan)} States/UTs, 2018-19 to 2024-25.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            f"""
            <div class="stat-card-light">
                <h4>Operational Holdings</h4>
                <p>Number of marginal holdings (below 1 ha) for
                {len(holdings)} States/UTs.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            f"""
            <div class="stat-card-light">
                <h4>PMFBY</h4>
                <p>Farmer applications insured under the crop insurance
                scheme for {len(pmfby)} States/UTs.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col4:
        scheme_count = len(schemes) if schemes is not None else 0

        st.markdown(
            f"""
            <div class="stat-card-light">
                <h4>Maharashtra Schemes</h4>
                <p>Catalogue of {scheme_count} agriculture schemes
                with category and government level.</p>
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
                <p>The value or values that occur most frequently. In this
                data almost every value is different, so a mode usually
                does not exist.</p>
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

    st.header("Probability and Inference Concepts")

    col_d, col_e, col_f = st.columns(3)

    with col_d:
        st.markdown(
            """
            <div class="stat-card-light">
                <h4>Probability &amp; Distributions</h4>
                <p>Conditional probability, Bayes' theorem, and the binomial, Poisson and normal distributions.</p>
            </div>

            <div class="stat-card-light">
                <h4>Sampling Distribution &amp; CLT</h4>
                <p>Sample means form a bell shape around the population mean, whatever the shape of the population, once the sample is large enough.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col_e:
        st.markdown(
            """
            <div class="stat-card-light">
                <h4>Confidence Intervals &amp; Group Tests</h4>
                <p>A range of plausible values for a mean, plus independent t, paired t and one-way ANOVA tests to compare groups.</p>
            </div>

            <div class="stat-card-light">
                <h4>Hypothesis Testing</h4>
                <p>A Z-test or t-test checks whether a claim about a population mean is supported by the sample, using a p-value and a significance level α.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col_f:
        st.markdown(
            """
            <div class="stat-card-light">
                <h4>Type I &amp; Type II Errors</h4>
                <p>Type I: rejecting a true H₀ (probability α). Type II: failing to reject a false H₀ (probability β).</p>
            </div>

            <div class="stat-card-light">
                <h4>Regression &amp; Predictive Analytics</h4>
                <p>Linear models that predict one measure from others, judged with MAE, MSE, RMSE and R².</p>
            </div>
            """,
            unsafe_allow_html=True
        )


# ==========================================================
# DATASET OVERVIEW
# ==========================================================
elif page == "Dataset Overview":

    st.title("Dataset Overview")

    st.write("Explore the structure and contents of the selected dataset.")

    dataset_name = st.selectbox(
        "Select Dataset",
        dataset_names,
        key="overview_dataset"
    )

    df = datasets[dataset_name]

    st.header("Dataset Information")

    card1, card2, card3 = st.columns(3)

    with card1:
        st.metric("Dataset", dataset_name)

    with card2:
        st.metric("Number of Rows", df.shape[0])

    with card3:
        st.metric("Number of Columns", df.shape[1])

    st.header("Dataset Preview")

    filter_col1, filter_col2 = st.columns(2)

    with filter_col1:
        search_text = st.text_input(
            "Search by State/UT name",
            value="",
            key="overview_search"
        )

    with filter_col2:
        complete_only = st.checkbox(
            "Show only rows without missing values",
            key="overview_complete"
        )

    filtered_df = df.copy()

    if search_text.strip() != "":
        filtered_df = filtered_df[
            filtered_df["State/UT"].str.contains(
                search_text.strip(),
                case=False,
                na=False,
                regex=False
            )
        ]

    if complete_only:
        filtered_df = filtered_df.dropna()

    if len(filtered_df) == 0:
        st.warning("No rows match the current filters.")

    else:
        if len(filtered_df) > 5:
            rows_to_show = st.slider(
                "Number of rows to display",
                min_value=5,
                max_value=len(filtered_df),
                value=min(10, len(filtered_df)),
                key=f"overview_rows_{dataset_name}_{len(filtered_df)}"
            )
        else:
            rows_to_show = len(filtered_df)

        st.write(
            f"Showing {min(rows_to_show, len(filtered_df))} of "
            f"{len(filtered_df)} matching rows."
        )

        show_table(
            filtered_df.head(rows_to_show),
            hide_index=True
        )

        st.download_button(
            "Download the matching rows as CSV",
            data=filtered_df.to_csv(index=False).encode("utf-8"),
            file_name="filtered_data.csv",
            mime="text/csv"
        )

    st.header("Column Information")

    column_info = pd.DataFrame({
        "Column Name": list(df.columns),
        "Missing Values": [int(df[col].isnull().sum()) for col in df.columns]
    })

    show_table(column_info, hide_index=True)

    # ------------------------------------------------------
    # State comparison chart
    # ------------------------------------------------------
    st.header("Compare States / UTs")

    overview_numeric = get_numeric_columns(df)

    compare_col1, compare_col2, compare_col3 = st.columns(3)

    with compare_col1:
        compare_column = st.selectbox(
            "Variable to compare",
            overview_numeric,
            key="compare_variable"
        )

    with compare_col2:
        compare_order = st.radio(
            "Show",
            ["Highest", "Lowest"],
            horizontal=True,
            key="compare_order"
        )

    chart_data = df[["State/UT", compare_column]].dropna()

    if len(chart_data) < 4:
        st.warning("Not enough data to draw this chart.")

    else:
        with compare_col3:
            max_states = min(15, len(chart_data))

            top_n = st.slider(
                "Number of states / UTs",
                min_value=3,
                max_value=max_states,
                value=min(8, max_states),
                key=f"compare_top_n_{max_states}"
            )

        chart_data = chart_data.sort_values(
            compare_column,
            ascending=(compare_order == "Lowest")
        ).head(top_n)

        fig_compare = go.Figure()

        fig_compare.add_trace(
            go.Bar(
                x=chart_data[compare_column],
                y=chart_data["State/UT"],
                orientation="h",
                marker_color="#81c784",
                marker_line=dict(color="#4cae51", width=1),
                hovertemplate="<b>%{y}</b><br>Value: %{x:,.2f}<extra></extra>"
            )
        )

        fig_compare = style_figure(
            fig_compare,
            f"{compare_order} {top_n} for <b>{compare_column}</b>",
            compare_column,
            "State / UT",
            height=450
        )

        fig_compare.update_yaxes(autorange="reversed")

        show_chart(fig_compare)

        st.caption(
            f"Each bar is one State/UT. The longest bar belongs to "
            f"{chart_data.sort_values(compare_column, ascending=False)['State/UT'].iloc[0]} "
            "in this view. Switch between Highest and Lowest, or change the "
            "number of states, to see how the ranking changes."
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
        dataset_names,
        key="analysis_dataset"
    )

    df = datasets[dataset_name]

    numeric_columns = get_numeric_columns(df)

    if len(numeric_columns) == 0:
        st.warning("No numerical columns are available in this dataset.")

    else:

        # ==================================================
        # DESCRIPTIVE STATISTICS
        # ==================================================
        st.header("Descriptive Statistics")

        selected_column = st.selectbox(
            "Select Variable",
            numeric_columns,
            index=0,
            key=f"analysis_variable_{dataset_name}"
        )

        stats = calculate_descriptive_statistics(df[selected_column])

        if stats is None:
            st.warning("No valid numerical data is available for this variable.")

        else:
            mean_val = stats["mean"]
            median_val = stats["median"]
            mode_text = stats["mode_text"]
            mode_num = stats["mode_num"]
            variance_val = stats["variance"]
            std_val = stats["std"]
            iqr_val = stats["iqr"]
            range_val = stats["range"]

            missing_count = int(df[selected_column].isna().sum())

            st.write(
                f"Based on **{stats['n']}** States/UTs with data"
                + (
                    f" ({missing_count} missing value(s) left out)."
                    if missing_count > 0 else "."
                )
            )

            stats_df = pd.DataFrame({
                "Measure": [
                    "Mean", "Median", "Mode", "Variance",
                    "Standard Deviation", "IQR", "Range"
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

            show_table(stats_df, hide_index=True)

            # ==============================================
            # CENTRAL TENDENCY
            # ==============================================
            st.header("Central Tendency Metrics")

            metrics_list = ["Mean", "Median"]
            values_list = [mean_val, median_val]
            colors_list = ["#66bb6a", "#81c784"]

            # Only draw a Mode bar when a mode really exists. Drawing a
            # zero-height bar for "no mode" is wrong, and a zero cannot be
            # shown on a log axis at all.
            if mode_num is not None:
                metrics_list.append("Mode")
                values_list.append(mode_num)
                colors_list.append("#a5d6a7")

            labels_list = [short_number(v) for v in values_list]

            use_log_central = (
                all(v > 0 for v in values_list)
                and max(values_list) / min(values_list) >= 100
            )

            fig_bar = go.Figure()

            fig_bar.add_trace(
                go.Bar(
                    x=metrics_list,
                    y=values_list,
                    text=labels_list,
                    textposition="outside",
                    textfont=dict(color="#000000", size=12),
                    marker_color=colors_list,
                    marker_line=dict(color="#4cae51", width=1.5),
                    cliponaxis=False,
                    hovertemplate="<b>%{x}</b><br>Value: %{y:,.2f}<extra></extra>"
                )
            )

            if use_log_central:
                fig_bar.update_yaxes(type="log")

            fig_bar.update_layout(
                title=dict(
                    text=f"Central Tendency Comparison for <b>{selected_column}</b>",
                    font=dict(color="#000000", size=16)
                ),
                yaxis=dict(
                    title="Value" + (" (log scale)" if use_log_central else ""),
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
                margin=dict(l=60, r=40, t=70, b=60),
                showlegend=False
            )

            show_chart(fig_bar)

            if use_log_central:
                st.caption(
                    "A logarithmic scale is used because the mean and median "
                    "are very different in size."
                )

            if mode_num is None:
                st.caption(
                    "No value repeats in the selected data, so there is no "
                    "mode and only the mean and median are drawn. (For "
                    "continuous measures like counts and amounts, this is "
                    "normal.)"
                )
            elif len(stats["mode_list"]) > 1:
                st.caption(
                    f"There are several modes ({mode_text}). The chart shows "
                    "the smallest one."
                )
            else:
                st.caption(f"Mode = {mode_num:,.2f}")

            if mean_val > median_val:
                st.caption(
                    "The mean is larger than the median. This usually means a few "
                    "very large states pull the average up (right-skewed data)."
                )
            elif mean_val < median_val:
                st.caption(
                    "The mean is smaller than the median. This usually means a "
                    "few very small values pull the average down (left-skewed data)."
                )
            else:
                st.caption(
                    "The mean and median are equal, so the data are balanced "
                    "around the centre."
                )

            # ==============================================
            # DISTRIBUTION ANALYSIS
            # ==============================================
            st.header("Distribution Analysis")

            option_col1, option_col2, option_col3 = st.columns(3)

            with option_col1:
                histogram_bins = st.slider(
                    "Number of histogram bins",
                    min_value=3,
                    max_value=30,
                    value=min(12, max(3, stats["n"] // 2)),
                    key="analysis_bins"
                )

            with option_col2:
                show_mean_line = st.checkbox(
                    "Show mean line", value=True, key="analysis_show_mean"
                )

            with option_col3:
                show_median_line = st.checkbox(
                    "Show median line", value=True, key="analysis_show_median"
                )

            fig_hist, hist_is_log = build_histogram(
                stats["values"],
                histogram_bins,
                selected_column,
                mean_val,
                median_val,
                show_mean_line,
                show_median_line
            )

            show_chart(fig_hist)

            if hist_is_log:
                st.caption(
                    "The values are very uneven, so the bars use equal steps "
                    "of powers of ten (1K, 10K, 100K ...). Each bar still "
                    "counts how many States/UTs fall in that range."
                )
            else:
                st.caption(
                    "Each bar counts how many States/UTs fall inside a value "
                    "range. Use the slider to make the bars wider or narrower."
                )

            # Box plot with the IQR rule for outliers
            box_data = df[["State/UT", selected_column]].dropna()

            fig_box = go.Figure()

            fig_box.add_trace(
                go.Box(
                    x=box_data[selected_column],
                    text=box_data["State/UT"],
                    name="",
                    boxpoints="all",
                    jitter=0.4,
                    pointpos=0,
                    quartilemethod="linear",
                    marker=dict(color="#2e7d32", size=6, opacity=0.75),
                    line=dict(color="#2e7d32"),
                    fillcolor="rgba(129, 199, 132, 0.45)",
                    hovertemplate="<b>%{text}</b><br>Value: %{x:,.2f}<extra></extra>"
                )
            )

            fig_box = style_figure(
                fig_box,
                f"Box Plot for <b>{selected_column}</b>",
                selected_column,
                "",
                height=300
            )

            fig_box.update_layout(showlegend=False)
            fig_box.update_yaxes(showticklabels=False)

            if is_heavily_skewed(stats["values"]):
                fig_box.update_xaxes(type="log")

            show_chart(fig_box)

            lower_fence = stats["q1"] - 1.5 * stats["iqr"]
            upper_fence = stats["q3"] + 1.5 * stats["iqr"]

            outlier_rows = box_data[
                (box_data[selected_column] < lower_fence)
                | (box_data[selected_column] > upper_fence)
            ].sort_values(selected_column, ascending=False)

            if len(outlier_rows) > 0:
                st.caption(
                    f"By the 1.5 × IQR rule (fences {short_number(lower_fence)} "
                    f"and {short_number(upper_fence)}), these States/UTs are "
                    "outliers: "
                    + ", ".join(
                        f"{row['State/UT']} ({short_number(row[selected_column])})"
                        for _, row in outlier_rows.iterrows()
                    )
                    + ". They pull the mean away from the median."
                )
            else:
                st.caption(
                    "No State/UT falls outside the 1.5 × IQR fences "
                    f"({short_number(lower_fence)} to {short_number(upper_fence)})."
                )

        # ==================================================
        # PEARSON CORRELATION MATRIX
        # ==================================================
        if len(numeric_columns) >= 2:

            st.header("Pearson Correlation Matrix Table")

            corr_matrix = calculate_correlation_matrix(
                df[numeric_columns],
                tuple(numeric_columns)
            ).round(4)

            lightest_green_cmap = mcolors.LinearSegmentedColormap.from_list(
                "lightest_green",
                ["#f4fbf6", "#e8f5e9", "#c8e6c9"]
            )

            try:
                styled_corr = (
                    corr_matrix.style
                    .background_gradient(cmap=lightest_green_cmap, axis=None)
                    .set_properties(**{"color": "#000000"})
                )
                show_table(styled_corr)
            except Exception:
                show_table(corr_matrix)

            st.caption(
                "Pearson correlation values are calculated with the formula "
                "using loops, not the built-in DataFrame corr() method. Values "
                "near +1 or -1 show a strong linear relationship, and values "
                "near 0 show a weak one. Each pair uses only the States/UTs "
                "that have both values."
            )

            # ==============================================
            # PAIRWISE COVARIANCE AND CORRELATION
            # ==============================================
            st.header("Pairwise Covariance and Correlation")

            col1, col2 = st.columns(2)

            with col1:
                x_column = st.selectbox(
                    "Select First Variable",
                    numeric_columns,
                    key=f"x_variable_{dataset_name}"
                )

            with col2:
                y_column = st.selectbox(
                    "Select Second Variable",
                    numeric_columns,
                    index=1,
                    key=f"y_variable_{dataset_name}"
                )

            pair_data = df[["State/UT", x_column, y_column]].dropna()

            if len(pair_data) < 2:
                st.warning("Not enough data for correlation analysis.")

            else:
                covariance_value, pearson_value = calculate_pair_statistics(
                    pair_data[x_column].tolist(),
                    pair_data[y_column].tolist()
                )

                correlation_df = pd.DataFrame({
                    "Measure": [
                        "States/UTs used", "Covariance", "Pearson Correlation"
                    ],
                    "Value": [
                        str(len(pair_data)),
                        display_number(covariance_value),
                        display_number(pearson_value)
                    ]
                })

                show_table(correlation_df, hide_index=True)

                st.info(
                    "Covariance is calculated as the sum of products of "
                    "deviations divided by n − 1. Pearson correlation divides "
                    "that numerator by the square root of the product of "
                    "the two sums of squared deviations."
                )

                st.header("Scatter Plot")

                st.write(f"Relationship between **{x_column}** and **{y_column}**")

                use_log_x = is_heavily_skewed(pair_data[x_column])
                use_log_y = is_heavily_skewed(pair_data[y_column])

                fig_scatter = go.Figure()

                fig_scatter.add_trace(
                    go.Scatter(
                        x=pair_data[x_column],
                        y=pair_data[y_column],
                        mode="markers",
                        text=pair_data["State/UT"],
                        marker=dict(color="#2e7d32", size=9, opacity=0.75),
                        name="States/UTs",
                        hovertemplate=(
                            "<b>%{text}</b><br>"
                            + f"{x_column}: " + "%{x:,.2f}<br>"
                            + f"{y_column}: " + "%{y:,.2f}<extra></extra>"
                        )
                    )
                )

                # Least-squares line from the same formulas: slope = cov / var(x)
                x_stats = calculate_descriptive_statistics(pair_data[x_column])
                y_stats = calculate_descriptive_statistics(pair_data[y_column])

                if (
                    not use_log_x and not use_log_y
                    and x_stats["variance"] > 0
                ):
                    slope = covariance_value / x_stats["variance"]
                    intercept = y_stats["mean"] - slope * x_stats["mean"]

                    x_line = np.array([
                        pair_data[x_column].min(), pair_data[x_column].max()
                    ])

                    fig_scatter.add_trace(
                        go.Scatter(
                            x=x_line,
                            y=intercept + slope * x_line,
                            mode="lines",
                            line=dict(color="#e57373", width=2, dash="dash"),
                            name="Least-squares line"
                        )
                    )

                fig_scatter = style_figure(
                    fig_scatter,
                    f"Scatter Plot: {x_column} vs {y_column}",
                    x_column,
                    y_column,
                    height=480
                )

                if use_log_x:
                    fig_scatter.update_xaxes(type="log")

                if use_log_y:
                    fig_scatter.update_yaxes(type="log")

                show_chart(fig_scatter)

                if use_log_x and use_log_y:
                    st.caption(
                        "Logarithmic scales are used on both axes because "
                        "the values range from very small to very large. "
                        "The straight line is left out because it would not "
                        "look straight on a log scale."
                    )
                elif use_log_x:
                    st.caption(
                        "A logarithmic scale is used on the X-axis because "
                        "the values range from very small to very large."
                    )
                elif use_log_y:
                    st.caption(
                        "A logarithmic scale is used on the Y-axis because "
                        "the values range from very small to very large."
                    )
                else:
                    st.caption(
                        "Each point is one State/UT (hover to see which). The "
                        "dashed line is the least-squares line."
                    )

                st.caption(describe_correlation(pearson_value))

                st.caption(
                    "Pearson r is measured on the original numbers, so a few "
                    "very large States/UTs (such as Uttar Pradesh or "
                    "Rajasthan) can dominate it even when the log-scale chart "
                    "looks different."
                )

        else:
            st.info(
                "At least two numerical columns are required for covariance, "
                "Pearson correlation and the scatter plot. Choose the "
                "PM-KISAN or Combined dataset to see them."
            )


# ==========================================================
# SAMPLING DISTRIBUTION AND CENTRAL LIMIT THEOREM
# ==========================================================
elif page == "Sampling and CLT":

    st.title("Sampling Distribution and Central Limit Theorem")

    st.write(
        "Treat all the States/UTs in a dataset as the population. Draw many "
        "random samples from it, calculate the mean of each sample, and see "
        "how those sample means behave."
    )

    pick_col1, pick_col2 = st.columns(2)

    with pick_col1:
        dataset_name = st.selectbox(
            "Select Dataset",
            dataset_names,
            key="clt_dataset"
        )

    df = datasets[dataset_name]
    numeric_columns = get_numeric_columns(df)

    with pick_col2:
        selected_column = st.selectbox(
            "Select Variable",
            numeric_columns,
            key=f"clt_variable_{dataset_name}"
        )

    population = df[selected_column].dropna().to_numpy(dtype=float)
    population_size = len(population)

    population_mean = population.mean() if population_size > 0 else np.nan
    population_sd = (
        np.sqrt(np.mean((population - population_mean) ** 2))
        if population_size > 0 else 0
    )

    if population_size < 2 or population_sd == 0:

        st.warning(
            "This variable needs at least two different values to run "
            "the sampling experiment."
        )

    else:

        st.header("Sampling Experiment")

        slider_col1, slider_col2 = st.columns(2)

        with slider_col1:
            sample_size = st.slider(
                "Sample size (n)",
                min_value=1,
                max_value=100,
                value=5,
                key="clt_sample_size"
            )

        with slider_col2:
            number_of_samples = st.slider(
                "Number of samples drawn",
                min_value=100,
                max_value=5000,
                value=1000,
                step=100,
                key="clt_number_of_samples"
            )

        seed_col1, seed_col2 = st.columns(2)

        with seed_col1:
            use_fixed_seed = st.checkbox(
                "Use a fixed random seed (same samples every time)",
                key="clt_seed"
            )

        with seed_col2:
            st.button("Draw new samples", key="clt_redraw")

        if use_fixed_seed:
            rng = np.random.default_rng(42)
        else:
            rng = np.random.default_rng()

        # Each row is one sample. Samples are drawn with replacement so that
        # the draws stay independent, as the theorem assumes.
        samples = rng.choice(
            population,
            size=(number_of_samples, sample_size),
            replace=True
        )

        sample_means = samples.mean(axis=1)

        theory_se = population_sd / math.sqrt(sample_size)

        means_stats = calculate_descriptive_statistics(pd.Series(sample_means))

        observed_se = means_stats["std"]

        metric1, metric2, metric3, metric4 = st.columns(4)

        with metric1:
            st.metric("Population mean (μ)", short_number(population_mean))

        with metric2:
            st.metric("Mean of sample means", short_number(means_stats["mean"]))

        with metric3:
            st.metric("σ / √n (theory)", short_number(theory_se))

        with metric4:
            st.metric("SD of sample means", short_number(observed_se))

        st.caption(
            f"The population has {population_size} values with σ = "
            f"{short_number(population_sd)}. The theory says the sample "
            "means should centre on μ with a spread of σ/√n. Compare the "
            "last two numbers above."
        )

        hist_col1, hist_col2 = st.columns(2)

        with hist_col1:
            fig_population = px.histogram(
                x=population,
                nbins=min(12, max(4, population_size // 2)),
                color_discrete_sequence=["#a5d6a7"],
                opacity=0.9
            )

            fig_population.update_traces(
                marker_line=dict(color="#4caf50", width=1)
            )

            fig_population = style_figure(
                fig_population,
                "Population (all values)",
                selected_column,
                "Count",
                height=400
            )

            fig_population.update_layout(showlegend=False)

            show_chart(fig_population, key="clt_population_chart")

            st.caption(
                "This is the original data. Notice its shape: it is often "
                "lopsided, with a few very large states."
            )

        with hist_col2:
            curve_start = min(population_mean - 4 * theory_se, sample_means.min())
            curve_end = max(population_mean + 4 * theory_se, sample_means.max())

            x_curve = np.linspace(curve_start, curve_end, 300)

            curve_model = NormalDist(population_mean, theory_se)
            y_curve = [curve_model.pdf(value) for value in x_curve]

            fig_means = go.Figure()

            fig_means.add_trace(
                go.Histogram(
                    x=sample_means,
                    nbinsx=30,
                    histnorm="probability density",
                    marker_color="#66bb6a",
                    marker_line=dict(color="#4caf50", width=1),
                    opacity=0.85,
                    name="Sample means"
                )
            )

            fig_means.add_trace(
                go.Scatter(
                    x=x_curve,
                    y=y_curve,
                    mode="lines",
                    line=dict(color="#e57373", width=2.5),
                    name="Normal curve from CLT"
                )
            )

            fig_means = style_figure(
                fig_means,
                f"Means of {number_of_samples} samples (n = {sample_size})",
                "Sample mean",
                "Density",
                height=400
            )

            show_chart(fig_means, key="clt_means_chart")

            st.caption(
                "Each bar comes from one sample mean. The red line is the "
                "normal curve the Central Limit Theorem predicts. Increase "
                "n and the bars should hug the curve more closely, while "
                "the whole shape gets narrower."
            )

        if sample_size < 30:
            st.info(
                "The usual guideline is n ≥ 30 for the normal approximation "
                "to work well. With a skewed population and a small n, the "
                "histogram can still look lopsided."
            )
        else:
            st.success(
                "With n ≥ 30 the sample means should look close to a "
                "normal curve, even though the population itself is skewed."
            )

        # --------------------------------------------------
        # Table: spread shrinks as n grows
        # --------------------------------------------------
        st.header("How the Spread Shrinks as n Grows")

        table_rows = []

        for n_value in [1, 5, 10, 30, 50, 100]:
            table_samples = rng.choice(
                population,
                size=(number_of_samples, n_value),
                replace=True
            )

            table_means = table_samples.mean(axis=1)

            table_rows.append({
                "Sample size (n)": n_value,
                "Mean of sample means": short_number(table_means.mean()),
                "Theoretical σ/√n": short_number(population_sd / math.sqrt(n_value)),
                "Observed SD of sample means": short_number(table_means.std(ddof=1))
            })

        show_table(pd.DataFrame(table_rows), hide_index=True)

        st.caption(
            "The mean of the sample means stays at μ for every n, but the "
            "spread falls as n rises. The observed values should stay close "
            "to the theoretical ones, with small differences because the "
            "samples are random."
        )

    # ------------------------------------------------------
    # Probability calculator for a sample mean
    # ------------------------------------------------------
    st.header("Probability Calculator for a Sample Mean")

    st.write(
        "If X̄ is the mean of n observations from a population with mean μ "
        "and standard deviation σ, then X̄ is approximately normal with mean "
        "μ and standard deviation σ/√n. The starting values are the light "
        "bulb example from the lecture notes."
    )

    calc_col1, calc_col2, calc_col3, calc_col4 = st.columns(4)

    with calc_col1:
        calc_mu = st.number_input(
            "Population mean (μ)", value=800.0, key="calc_mu"
        )

    with calc_col2:
        calc_sigma = st.number_input(
            "Population SD (σ)", value=40.0, min_value=0.01, key="calc_sigma"
        )

    with calc_col3:
        calc_n = st.number_input(
            "Sample size (n)", value=16, min_value=1, step=1, key="calc_n"
        )

    with calc_col4:
        calc_x = st.number_input(
            "Sample mean value (x̄)", value=775.0, key="calc_x"
        )

    calc_direction = st.radio(
        "Probability to find",
        ["P(X̄ < x̄)", "P(X̄ > x̄)"],
        horizontal=True,
        key="calc_direction"
    )

    calc_se = calc_sigma / math.sqrt(calc_n)
    calc_z = (calc_x - calc_mu) / calc_se

    if calc_direction == "P(X̄ < x̄)":
        calc_probability = normal.cdf(calc_z)
    else:
        calc_probability = normal.cdf(-calc_z)

    result1, result2, result3 = st.columns(3)

    with result1:
        st.metric("Standard error (σ/√n)", f"{calc_se:.4f}")

    with result2:
        st.metric("z value", f"{calc_z:.4f}")

    with result3:
        st.metric(calc_direction, f"{calc_probability:.4f}")

    st.caption(
        "The z value says how many standard errors x̄ is away from μ. The "
        "probability is the area under the standard normal curve on the "
        "chosen side of that z value."
    )


# ==========================================================
# HYPOTHESIS TESTING
# ==========================================================
elif page == "Hypothesis Testing":

    st.title("Hypothesis Testing")

    st.write(
        "Test a claim about a population mean using a Z-test or a t-test. "
        "The page follows the same steps used in class: state the "
        "hypotheses, calculate the test statistic, find the critical value "
        "and p-value, then make a decision."
    )

    tab_data, tab_practice = st.tabs([
        "Test a dataset variable",
        "Practice with your own numbers"
    ])

    # ------------------------------------------------------
    # Tab 1: test using the dataset
    # ------------------------------------------------------
    with tab_data:

        pick_col1, pick_col2 = st.columns(2)

        with pick_col1:
            dataset_name = st.selectbox(
                "Select Dataset",
                dataset_names,
                key="test_dataset"
            )

        df = datasets[dataset_name]
        numeric_columns = get_numeric_columns(df)

        with pick_col2:
            selected_column = st.selectbox(
                "Select Variable",
                numeric_columns,
                key=f"test_variable_{dataset_name}"
            )

        stats = calculate_descriptive_statistics(df[selected_column])

        if stats is None or stats["n"] < 2 or stats["std"] == 0:

            st.warning(
                "This variable needs at least two different values to run "
                "a test."
            )

        else:

            st.write(
                f"Sample size n = **{stats['n']}**, sample mean = "
                f"**{short_number(stats['mean'])}**, sample standard "
                f"deviation = **{short_number(stats['std'])}**."
            )

            key_suffix = f"{dataset_name}_{selected_column}"

            input_col1, input_col2 = st.columns(2)

            with input_col1:
                mu_0 = st.number_input(
                    "Claimed population mean (μ₀)",
                    value=float(round(stats["median"], 2)),
                    step=float(max(stats["std"] / 10, 0.01)),
                    key=f"mu0_{key_suffix}"
                )

            with input_col2:
                test_name = st.radio(
                    "Type of test",
                    ["t-test", "Z-test"],
                    horizontal=True,
                    key="test_type_data",
                    help="Use the t-test when σ is unknown. Use the Z-test "
                         "only when σ is known."
                )

            if test_name == "Z-test":
                spread = st.number_input(
                    "Known population standard deviation (σ)",
                    value=float(max(round(stats["std"], 2), 0.01)),
                    min_value=0.01,
                    key=f"sigma_{key_suffix}"
                )
            else:
                spread = stats["std"]

            input_col3, input_col4 = st.columns(2)

            with input_col3:
                tail = st.radio(
                    "Alternative hypothesis",
                    ["two-tailed", "right-tailed", "left-tailed"],
                    horizontal=True,
                    key="tail_data"
                )

            with input_col4:
                alpha = st.radio(
                    "Significance level (α)",
                    [0.01, 0.05, 0.10],
                    index=1,
                    horizontal=True,
                    key="alpha_data"
                )

            st.caption(
                "The States/UTs are treated as the sample here, so use the "
                "result for practice with the method rather than as a "
                "final claim about agriculture. The data are strongly "
                "right-skewed, so the t-test's normality assumption is only "
                "roughly met."
            )

            show_mean_test(
                stats["mean"],
                stats["n"],
                mu_0,
                test_name,
                tail,
                alpha,
                spread,
                selected_column
            )

    # ------------------------------------------------------
    # Tab 2: practice with summary numbers
    # ------------------------------------------------------
    with tab_practice:

        st.write(
            "Type in the summary numbers of any problem. The starting "
            "values are the vacuum cleaner example from the lecture notes "
            "(answer: t ≈ −1.16, p ≈ 0.135, fail to reject H₀)."
        )

        practice_col1, practice_col2, practice_col3, practice_col4 = st.columns(4)

        with practice_col1:
            practice_mean = st.number_input(
                "Sample mean (x̄)", value=42.0, key="practice_mean"
            )

        with practice_col2:
            practice_spread = st.number_input(
                "Standard deviation (s or σ)",
                value=11.9, min_value=0.01, key="practice_spread"
            )

        with practice_col3:
            practice_n = st.number_input(
                "Sample size (n)", value=12, min_value=2, step=1,
                key="practice_n"
            )

        with practice_col4:
            practice_mu0 = st.number_input(
                "Claimed mean (μ₀)", value=46.0, key="practice_mu0"
            )

        practice_col5, practice_col6, practice_col7 = st.columns(3)

        with practice_col5:
            practice_test = st.radio(
                "Type of test",
                ["t-test", "Z-test"],
                horizontal=True,
                key="test_type_practice"
            )

        with practice_col6:
            practice_tail = st.radio(
                "Alternative hypothesis",
                ["two-tailed", "right-tailed", "left-tailed"],
                index=2,
                horizontal=True,
                key="tail_practice"
            )

        with practice_col7:
            practice_alpha = st.radio(
                "Significance level (α)",
                [0.01, 0.05, 0.10],
                index=1,
                horizontal=True,
                key="alpha_practice"
            )

        show_mean_test(
            practice_mean,
            int(practice_n),
            practice_mu0,
            practice_test,
            practice_tail,
            practice_alpha,
            practice_spread,
            "the population"
        )


# ==========================================================
# TYPE I AND TYPE II ERRORS
# ==========================================================
elif page == "Type I and Type II Errors":

    st.title("Type I and Type II Errors")

    st.write(
        "Every test can go wrong in two different ways. This page lets you "
        "change the test settings and watch both error probabilities move."
    )

    error_col1, error_col2 = st.columns(2)

    with error_col1:
        st.markdown(
            """
            <div class="stat-card-light">
                <h4>Type I error (probability α)</h4>
                <p>The test rejects H₀ when H₀ is actually true. Example from
                class: a truckload of good potatoes is sent back because the
                sample looked bad.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with error_col2:
        st.markdown(
            """
            <div class="stat-card-light">
                <h4>Type II error (probability β)</h4>
                <p>The test fails to reject H₀ when H₀ is actually false.
                Example from class: a truckload with too many blemished
                potatoes is accepted.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.header("Error Simulator (Z-test for a mean)")

    st.write(
        "H₀ says the population mean is μ₀. The population standard "
        "deviation σ is known. The starting values come from the life span "
        "example in the notes (μ₀ = 70, σ = 8.9)."
    )

    setting_col1, setting_col2, setting_col3 = st.columns(3)

    with setting_col1:
        mu_0 = st.number_input("Mean under H₀ (μ₀)", value=70.0, key="err_mu0")

    with setting_col2:
        sigma = st.number_input(
            "Population SD (σ)", value=8.9, min_value=0.1, key="err_sigma"
        )

    with setting_col3:
        tail = st.radio(
            "Alternative hypothesis",
            ["right-tailed", "left-tailed", "two-tailed"],
            key="err_tail"
        )

    slider_col1, slider_col2 = st.columns(2)

    with slider_col1:
        n = st.slider(
            "Sample size (n)", min_value=2, max_value=200, value=30, key="err_n"
        )

    with slider_col2:
        alpha = st.slider(
            "Significance level (α)",
            min_value=0.01, max_value=0.20, value=0.05, step=0.01,
            key="err_alpha"
        )

    # The default true mean sits on the side the alternative hypothesis points to
    default_shift = -0.5 * sigma if tail == "left-tailed" else 0.5 * sigma

    mu_1 = st.slider(
        "True population mean (μ₁), the value H₀ is really up against",
        min_value=float(mu_0 - 3 * sigma),
        max_value=float(mu_0 + 3 * sigma),
        value=float(mu_0 + default_shift),
        step=float(sigma / 20),
        # The key includes the settings that set the slider's range, so the
        # slider resets cleanly instead of keeping an out-of-range value.
        key=f"err_mu1_{mu_0}_{sigma}_{tail}"
    )

    beta, lower_cut, upper_cut = error_probabilities(
        mu_0, mu_1, sigma, n, alpha, tail
    )

    power = 1 - beta

    result1, result2, result3 = st.columns(3)

    with result1:
        st.metric("Type I error (α)", f"{alpha:.4f}")

    with result2:
        st.metric("Type II error (β)", f"{beta:.4f}")

    with result3:
        st.metric("Power (1 − β)", f"{power:.4f}")

    st.caption(
        "Power is the chance of correctly rejecting H₀ when it is false. "
        "Because the true mean can be anything, β changes with μ₁, while α "
        "is set by you."
    )

    tab_curves, tab_true_mean, tab_sample, tab_alpha = st.tabs([
        "Two curves",
        "β vs true mean",
        "β vs sample size",
        "α vs β trade-off"
    ])

    standard_error = sigma / math.sqrt(n)

    # ------------------------------------------------------
    # Tab: the two overlapping curves
    # ------------------------------------------------------
    with tab_curves:

        x_start = min(mu_0, mu_1) - 4 * standard_error
        x_end = max(mu_0, mu_1) + 4 * standard_error

        x = np.linspace(x_start, x_end, 600)

        null_curve = NormalDist(mu_0, standard_error)
        true_curve = NormalDist(mu_1, standard_error)

        def null_pdf(values):
            return np.array([null_curve.pdf(v) for v in np.atleast_1d(values)])

        def true_pdf(values):
            return np.array([true_curve.pdf(v) for v in np.atleast_1d(values)])

        fig_errors = go.Figure()

        fig_errors.add_trace(
            go.Scatter(
                x=x, y=null_pdf(x), mode="lines",
                line=dict(color="#2e7d32", width=2.5),
                name="If H₀ is true (mean μ₀)"
            )
        )

        fig_errors.add_trace(
            go.Scatter(
                x=x, y=true_pdf(x), mode="lines",
                line=dict(color="#1565c0", width=2.5),
                name="If the true mean is μ₁"
            )
        )

        # Rejection region under the H0 curve is alpha
        alpha_shown = False

        if upper_cut is not None:
            shade_between(
                fig_errors, null_pdf, upper_cut, x_end,
                "rgba(229, 115, 115, 0.55)", "α (Type I error)", True
            )
            alpha_shown = True

        if lower_cut is not None:
            shade_between(
                fig_errors, null_pdf, x_start, lower_cut,
                "rgba(229, 115, 115, 0.55)", "α (Type I error)",
                not alpha_shown
            )

        # Non-rejection region under the true curve is beta
        accept_start = lower_cut if lower_cut is not None else x_start
        accept_end = upper_cut if upper_cut is not None else x_end

        shade_between(
            fig_errors, true_pdf,
            max(accept_start, x_start), min(accept_end, x_end),
            "rgba(100, 181, 246, 0.45)", "β (Type II error)", True
        )

        for cut in [upper_cut, lower_cut]:
            if cut is not None:
                fig_errors.add_vline(
                    x=cut,
                    line_dash="dash",
                    line_color="#555555",
                    annotation_text=f"Cut-off {cut:,.2f}",
                    annotation_position="top",
                    annotation_font=dict(color="#000000", size=11)
                )

        fig_errors = style_figure(
            fig_errors,
            "Sampling distribution of the sample mean",
            "Sample mean (x̄)",
            "Density",
            height=450
        )

        show_chart(fig_errors, key="err_curves_chart")

        st.caption(
            "The green curve shows where sample means land when H₀ is true. "
            "The blue curve shows where they land when the true mean is μ₁. "
            "The dashed line is the cut-off: sample means beyond it lead to "
            "rejecting H₀. The red area under the green curve past the "
            "cut-off is α. The light blue area under the blue curve on the "
            "other side of the cut-off is β, the chance of missing a real "
            "difference."
        )

    # ------------------------------------------------------
    # Tab: beta against the true mean
    # ------------------------------------------------------
    with tab_true_mean:

        mean_grid = []
        beta_grid = []

        for step in range(121):
            candidate_mean = (mu_0 - 3 * sigma) + step * (6 * sigma / 120)
            candidate_beta, _, _ = error_probabilities(
                mu_0, candidate_mean, sigma, n, alpha, tail
            )
            mean_grid.append(candidate_mean)
            beta_grid.append(candidate_beta)

        fig_beta_mean = go.Figure()

        fig_beta_mean.add_trace(
            go.Scatter(
                x=mean_grid, y=beta_grid, mode="lines",
                line=dict(color="#e57373", width=2.5), name="β"
            )
        )

        fig_beta_mean.add_trace(
            go.Scatter(
                x=[mu_1], y=[beta], mode="markers",
                marker=dict(color="#1565c0", size=11),
                name="Your current μ₁"
            )
        )

        fig_beta_mean = style_figure(
            fig_beta_mean,
            "Type II error (β) for different true means",
            "True population mean (μ₁)",
            "β",
            height=400
        )

        show_chart(fig_beta_mean, key="err_beta_mean_chart")

        st.caption(
            "β is largest when the true mean sits right next to μ₀, because "
            "the two curves almost overlap and the test cannot tell them "
            "apart. The further the true mean is from μ₀ (on the side the "
            "alternative points to), the smaller β becomes. The blue dot is "
            "the value you picked with the slider."
        )

    # ------------------------------------------------------
    # Tab: beta against the sample size
    # ------------------------------------------------------
    with tab_sample:

        n_grid = []
        beta_by_n = []

        for n_value in range(2, 201):
            candidate_beta, _, _ = error_probabilities(
                mu_0, mu_1, sigma, n_value, alpha, tail
            )
            n_grid.append(n_value)
            beta_by_n.append(candidate_beta)

        fig_beta_n = go.Figure()

        fig_beta_n.add_trace(
            go.Scatter(
                x=n_grid, y=beta_by_n, mode="lines",
                line=dict(color="#e57373", width=2.5), name="β"
            )
        )

        fig_beta_n.add_trace(
            go.Scatter(
                x=[n], y=[beta], mode="markers",
                marker=dict(color="#1565c0", size=11),
                name="Your current n"
            )
        )

        fig_beta_n = style_figure(
            fig_beta_n,
            "Type II error (β) for different sample sizes",
            "Sample size (n)",
            "β",
            height=400
        )

        show_chart(fig_beta_n, key="err_beta_n_chart")

        st.caption(
            "With α fixed, collecting more data makes the sampling "
            "distributions narrower, so they overlap less and β falls. A "
            "bigger sample is the one way to lower β without raising α."
        )

    # ------------------------------------------------------
    # Tab: alpha against beta
    # ------------------------------------------------------
    with tab_alpha:

        trade_rows = []

        for alpha_value in [0.01, 0.05, 0.10, 0.20]:
            trade_beta, _, _ = error_probabilities(
                mu_0, mu_1, sigma, n, alpha_value, tail
            )

            trade_rows.append({
                "α (Type I error)": f"{alpha_value:.2f}",
                "β (Type II error)": f"{trade_beta:.4f}",
                "Power (1 − β)": f"{1 - trade_beta:.4f}"
            })

        show_table(pd.DataFrame(trade_rows), hide_index=True)

        st.caption(
            "Going down the table, α increases and β decreases. Making it "
            "harder to reject H₀ (small α) makes it easier to miss a real "
            "effect (large β), so the two errors have to be balanced. The "
            "table uses your current μ₁, n and type of test."
        )

    # ------------------------------------------------------
    # Simulation
    # ------------------------------------------------------
    st.header("Simulation Check")

    st.write(
        "The numbers above come from formulas. Here the experiment is "
        "actually repeated on random samples to see if the same numbers "
        "appear."
    )

    sim_col1, sim_col2 = st.columns(2)

    with sim_col1:
        repeats = st.slider(
            "Number of repeated experiments",
            min_value=500, max_value=5000, value=2000, step=500,
            key="err_repeats"
        )

    with sim_col2:
        st.button("Run the simulation again", key="err_rerun")

    sim_rng = np.random.default_rng()

    # Case 1: H0 is true, so every rejection is a Type I error
    type_one_rate = simulate_rejection_rate(
        mu_0, mu_0, sigma, n, alpha, tail, repeats, sim_rng
    )

    # Case 2: the true mean is mu_1, so every non-rejection is a Type II error
    rejection_rate_true = simulate_rejection_rate(
        mu_1, mu_0, sigma, n, alpha, tail, repeats, sim_rng
    )

    type_two_rate = 1 - rejection_rate_true

    sim_result1, sim_result2 = st.columns(2)

    with sim_result1:
        st.metric(
            "Rejected H₀ although H₀ was true (Type I)",
            f"{type_one_rate:.4f}"
        )
        st.caption(f"Formula value: α = {alpha:.4f}")

    with sim_result2:
        st.metric(
            "Failed to reject H₀ although μ₁ was true (Type II)",
            f"{type_two_rate:.4f}"
        )
        st.caption(f"Formula value: β = {beta:.4f}")

    st.caption(
        "In the first case every sample comes from a population whose mean "
        "really is μ₀, so each rejection is a Type I error. In the second "
        "case the samples come from a population with mean μ₁, so each "
        "failure to reject is a Type II error. The simulated rates will "
        "wobble a little from run to run but stay close to α and β."
    )


# ==========================================================
# MAHARASHTRA SCHEMES (categorical catalogue)
# ==========================================================
elif page == "Maharashtra Schemes":

    st.title("Maharashtra Agriculture Schemes")

    st.write(
        "A catalogue of agriculture schemes available to Maharashtra "
        "farmers, plus where Maharashtra stands among States/UTs in the "
        "three Rajya Sabha datasets."
    )

    # ------------------------------------------------------
    # Maharashtra's position in the numeric datasets
    # ------------------------------------------------------
    st.header("Maharashtra in the State Data")

    snapshot_items = [
        ("PM-KISAN beneficiaries, 2023-24", pm_kisan,
         "2023-24 - No. of Beneficiaries"),
        ("PM-KISAN amount, 2023-24 (₹ crore)", pm_kisan,
         "2023-24 - Amount Transferred (in Cr)"),
        ("Marginal holdings (below 1 ha)", holdings,
         "Marginal (Below 1.00 ha.) (In Numbers)"),
        ("PMFBY farmer applications insured", pmfby,
         "Farmer Applications Insured")
    ]

    snapshot_columns = st.columns(len(snapshot_items))

    for column_box, (title, frame, column_name) in zip(snapshot_columns, snapshot_items):
        with column_box:
            if column_name not in frame.columns:
                st.metric(title, "N/A")
                continue

            row = frame[frame["State/UT"] == "Maharashtra"]

            if len(row) == 0 or pd.isna(row[column_name].iloc[0]):
                st.metric(title, "N/A")
                continue

            value = float(row[column_name].iloc[0])
            valid = frame[column_name].dropna()
            rank = int((valid > value).sum()) + 1

            st.metric(title, short_number(value))
            st.caption(f"Rank {rank} of {len(valid)} States/UTs")

    # ------------------------------------------------------
    # Scheme catalogue
    # ------------------------------------------------------
    if schemes is None:
        st.info(
            f"Put '{SCHEMES_FILE}' in the data folder to see the scheme "
            "catalogue."
        )

    else:
        st.header("Scheme Catalogue")

        metric1, metric2, metric3, metric4 = st.columns(4)

        with metric1:
            st.metric("Schemes listed", len(schemes))

        with metric2:
            st.metric("Categories", schemes["category"].nunique())

        with metric3:
            st.metric("State-level schemes", int((schemes["government"] == "State").sum()))

        with metric4:
            st.metric(
                "Central or joint schemes",
                int((schemes["government"] != "State").sum())
            )

        chart_col1, chart_col2 = st.columns([3, 2])

        with chart_col1:
            category_counts = (
                schemes["category"].value_counts().rename_axis("category")
                .reset_index(name="count")
            )

            fig_category = go.Figure()

            fig_category.add_trace(
                go.Bar(
                    x=category_counts["count"],
                    y=category_counts["category"],
                    orientation="h",
                    marker_color="#81c784",
                    marker_line=dict(color="#4cae51", width=1),
                    text=category_counts["count"],
                    textposition="outside",
                    textfont=dict(color="#000000", size=12),
                    cliponaxis=False,
                    hovertemplate="<b>%{y}</b><br>Schemes: %{x}<extra></extra>"
                )
            )

            fig_category = style_figure(
                fig_category,
                "Schemes per category",
                "Number of schemes",
                "",
                height=420
            )

            fig_category.update_yaxes(autorange="reversed")
            fig_category.update_layout(margin=dict(l=20, r=40, t=80, b=60))

            show_chart(fig_category, key="scheme_category_chart")

        with chart_col2:
            government_counts = (
                schemes["government"].value_counts().rename_axis("government")
                .reset_index(name="count")
            )

            fig_government = go.Figure()

            fig_government.add_trace(
                go.Pie(
                    labels=government_counts["government"],
                    values=government_counts["count"],
                    hole=0.45,
                    marker=dict(colors=["#66bb6a", "#a5d6a7", "#c8e6c9"]),
                    textfont=dict(color="#000000", size=13),
                    hovertemplate="<b>%{label}</b><br>Schemes: %{value}<extra></extra>"
                )
            )

            fig_government.update_layout(
                title=dict(
                    text="Who runs the scheme",
                    font=dict(color="#000000", size=16)
                ),
                paper_bgcolor="#ffffff",
                font=dict(color="#000000"),
                height=420,
                margin=dict(l=20, r=20, t=80, b=40),
                legend=dict(orientation="h", y=-0.05)
            )

            show_chart(fig_government, key="scheme_government_chart")

        st.caption(
            "Most schemes in the catalogue are run by the State government. "
            "Every scheme in this file is marked 'Active', 'Online', and has "
            "the same generic eligibility and benefit text, so category and "
            "government level are the only columns that separate the schemes."
        )

        st.subheader("Browse the schemes")

        filter_col1, filter_col2, filter_col3 = st.columns(3)

        with filter_col1:
            category_filter = st.selectbox(
                "Category",
                ["All"] + sorted(schemes["category"].unique().tolist()),
                key="scheme_category_filter"
            )

        with filter_col2:
            government_filter = st.selectbox(
                "Government",
                ["All"] + sorted(schemes["government"].unique().tolist()),
                key="scheme_government_filter"
            )

        with filter_col3:
            scheme_search = st.text_input(
                "Search scheme name",
                key="scheme_search"
            )

        shown = schemes.copy()

        if category_filter != "All":
            shown = shown[shown["category"] == category_filter]

        if government_filter != "All":
            shown = shown[shown["government"] == government_filter]

        if scheme_search.strip() != "":
            shown = shown[
                shown["scheme_name"].str.contains(
                    scheme_search.strip(), case=False, na=False, regex=False
                )
            ]

        st.write(f"Showing {len(shown)} of {len(schemes)} schemes.")

        show_table(
            shown[[
                "scheme_id", "scheme_name", "category",
                "government", "department", "official_link"
            ]],
            hide_index=True
        )

        st.download_button(
            "Download the shown schemes as CSV",
            data=shown.to_csv(index=False).encode("utf-8"),
            file_name="maharashtra_schemes_filtered.csv",
            mime="text/csv"
        )


# ==========================================================
# UNIT II: PROBABILITY AND PROBABILITY DISTRIBUTIONS
# ==========================================================
elif page == "Probability and Distributions":

    st.title("Probability and Probability Distributions")

    st.write(
        "Probability concepts, conditional probability, Bayes' theorem, "
        "random variables and the binomial, Poisson and normal "
        "distributions, with applications to the agricultural data."
    )

    (tab_cond, tab_bin, tab_poi, tab_norm,
     tab_approx, tab_app) = st.tabs([
        "Probability & Bayes",
        "Binomial",
        "Poisson",
        "Normal & Z",
        "Normal approx. to Binomial",
        "Applications"
    ])

    # ------------------------------------------------------
    # Probability concepts, conditional probability, Bayes
    # ------------------------------------------------------
    with tab_cond:

        st.header("Probability Concepts")

        st.markdown(
            """
            - **Probability** of an event A = favourable outcomes ÷ total outcomes, always between 0 and 1.
            - **Random variable:** a numerical outcome of a random process. *Discrete* variables take countable values (number of insured farmers in a sample); *continuous* variables take any value in a range (average yield).
            - **Conditional probability:** P(A | B) = P(A ∩ B) / P(B), the chance of A once we know B happened.
            - **Bayes' theorem:** P(A | B) = P(B | A) × P(A) / P(B).
            """
        )

        st.subheader("Try it on the data")

        pick1, pick2, pick3 = st.columns(3)

        with pick1:
            prob_dataset = st.selectbox(
                "Select Dataset", dataset_names, key="prob_dataset"
            )

        prob_df = datasets[prob_dataset]
        prob_numeric = get_numeric_columns(prob_df)

        if len(prob_numeric) < 2:
            st.warning("This dataset needs at least two numerical columns.")

        else:
            with pick2:
                prob_a = st.selectbox(
                    "Event A: this variable is above its median",
                    prob_numeric, key=f"prob_a_{prob_dataset}"
                )

            with pick3:
                prob_b = st.selectbox(
                    "Event B: this variable is above its median",
                    prob_numeric, index=1, key=f"prob_b_{prob_dataset}"
                )

            pair = prob_df[["State/UT", prob_a, prob_b]].dropna()

            if prob_a == prob_b or len(pair) < 4:
                st.warning("Choose two different variables with enough data.")

            else:
                event_a = (pair[prob_a] > pair[prob_a].median()).to_numpy()
                event_b = (pair[prob_b] > pair[prob_b].median()).to_numpy()

                total = len(pair)
                p_a = event_a.mean()
                p_b = event_b.mean()
                p_ab = (event_a & event_b).mean()

                table = pd.DataFrame(
                    {
                        "B: above median": [
                            int((event_a & event_b).sum()),
                            int((~event_a & event_b).sum())
                        ],
                        "B: not above median": [
                            int((event_a & ~event_b).sum()),
                            int((~event_a & ~event_b).sum())
                        ]
                    },
                    index=["A: above median", "A: not above median"]
                )

                st.write(f"Counts of States/UTs (total = {total}):")
                show_table(table)

                m1, m2, m3 = st.columns(3)

                with m1:
                    st.metric("P(A)", f"{p_a:.4f}")

                with m2:
                    st.metric("P(B)", f"{p_b:.4f}")

                with m3:
                    st.metric("P(A ∩ B)", f"{p_ab:.4f}")

                if p_a > 0 and p_b > 0:
                    p_a_given_b = p_ab / p_b
                    p_b_given_a = p_ab / p_a
                    bayes_value = p_b_given_a * p_a / p_b

                    c1, c2, c3 = st.columns(3)

                    with c1:
                        st.metric("P(A | B)", f"{p_a_given_b:.4f}")

                    with c2:
                        st.metric("P(B | A)", f"{p_b_given_a:.4f}")

                    with c3:
                        st.metric("P(A | B) by Bayes", f"{bayes_value:.4f}")

                    st.caption(
                        "The last number uses P(B | A) × P(A) / P(B) and "
                        "matches P(A | B), which is exactly Bayes' theorem."
                    )

                    if abs(p_ab - p_a * p_b) < 0.05:
                        st.info(
                            f"P(A ∩ B) = {p_ab:.3f} is close to P(A) × P(B) = "
                            f"{p_a * p_b:.3f}, so A and B look roughly independent."
                        )
                    else:
                        st.info(
                            f"P(A ∩ B) = {p_ab:.3f} differs from P(A) × P(B) = "
                            f"{p_a * p_b:.3f}, so A and B are not independent: "
                            "knowing one changes the chance of the other."
                        )

        st.subheader("Bayes' theorem calculator")

        st.write(
            "Example: 2% of fields have a crop disease. A test detects the "
            "disease 90% of the time, but also gives a false alarm on 8% of "
            "healthy fields. If a test is positive, how likely is the field "
            "really diseased?"
        )

        b1, b2, b3 = st.columns(3)

        with b1:
            prior = st.slider("P(disease)", 0.001, 0.5, 0.02, 0.001,
                              format="%.3f", key="bayes_prior")

        with b2:
            sensitivity = st.slider("P(positive | disease)", 0.5, 1.0, 0.90,
                                    0.01, key="bayes_sens")

        with b3:
            false_alarm = st.slider("P(positive | healthy)", 0.0, 0.5, 0.08,
                                    0.01, key="bayes_fp")

        p_positive = sensitivity * prior + false_alarm * (1 - prior)
        posterior = sensitivity * prior / p_positive if p_positive > 0 else float("nan")

        r1, r2 = st.columns(2)

        with r1:
            st.metric("P(positive) (total probability)", f"{p_positive:.4f}")

        with r2:
            st.metric("P(disease | positive)", f"{posterior:.4f}")

        st.caption(
            "P(positive) = P(+|D)P(D) + P(+|healthy)P(healthy). Even with a "
            "good test, a rare disease means many positives are false alarms. "
            "Raise P(disease) and watch the posterior jump."
        )

    # ------------------------------------------------------
    # Binomial
    # ------------------------------------------------------
    with tab_bin:

        st.header("Binomial Distribution")

        st.write(
            "A fixed number n of independent trials, each with success "
            "probability p. Example: n farmers are checked, and each is "
            "insured with probability p."
        )

        st.latex(r"P(X=k)=\binom{n}{k}p^{k}(1-p)^{n-k}")

        bn1, bn2, bn3 = st.columns(3)

        with bn1:
            bin_n = st.slider("Number of trials (n)", 1, 200, 20, key="bin_n")

        with bn2:
            bin_p = st.slider("Success probability (p)", 0.01, 0.99, 0.30,
                              0.01, key="bin_p")

        with bn3:
            bin_k = st.slider("Value of k", 0, bin_n, min(6, bin_n), key=f"bin_k_{bin_n}")

        bin_mode = st.radio(
            "Probability to find", ["P(X = k)", "P(X ≤ k)", "P(X ≥ k)"],
            horizontal=True, key="bin_mode"
        )

        bin_all = [binomial_pmf(i, bin_n, bin_p) for i in range(bin_n + 1)]

        if bin_mode == "P(X = k)":
            bin_value = bin_all[bin_k]
            bin_highlight = {bin_k}
        elif bin_mode == "P(X ≤ k)":
            bin_value = sum(bin_all[: bin_k + 1])
            bin_highlight = set(range(0, bin_k + 1))
        else:
            bin_value = sum(bin_all[bin_k:])
            bin_highlight = set(range(bin_k, bin_n + 1))

        bin_mean = bin_n * bin_p
        bin_var = bin_n * bin_p * (1 - bin_p)

        s1, s2, s3 = st.columns(3)

        with s1:
            st.metric(bin_mode, f"{bin_value:.4f}")

        with s2:
            st.metric("Mean = np", f"{bin_mean:.3f}")

        with s3:
            st.metric("Variance = np(1−p)", f"{bin_var:.3f}")

        low = max(0, int(bin_mean - 5 * math.sqrt(bin_var)) - 1)
        high = min(bin_n, int(bin_mean + 5 * math.sqrt(bin_var)) + 1)
        window = list(range(low, high + 1))

        bar_pmf_chart(
            window, [bin_all[i] for i in window], bin_highlight,
            f"Binomial(n = {bin_n}, p = {bin_p})", "Number of successes (X)",
            "bin_chart"
        )

        st.caption(
            "Dark bars add up to the probability you selected. The "
            "distribution is symmetric when p = 0.5 and skewed otherwise."
        )

    # ------------------------------------------------------
    # Poisson
    # ------------------------------------------------------
    with tab_poi:

        st.header("Poisson Distribution")

        st.write(
            "Counts of events in a fixed interval when events happen "
            "independently at a constant average rate λ. Example: the number "
            "of crop-insurance claims arriving at a district office per day."
        )

        st.latex(r"P(X=k)=\frac{e^{-\lambda}\lambda^{k}}{k!}")

        po1, po2 = st.columns(2)

        with po1:
            poi_lambda = st.slider("Average rate (λ)", 0.5, 30.0, 4.0, 0.5,
                                   key="poi_lambda")

        with po2:
            poi_k = st.slider("Value of k", 0, 60, 5, key="poi_k")

        poi_mode = st.radio(
            "Probability to find", ["P(X = k)", "P(X ≤ k)", "P(X ≥ k)"],
            horizontal=True, key="poi_mode"
        )

        poi_limit = int(poi_lambda + 6 * math.sqrt(poi_lambda) + 10)
        poi_all = [poisson_pmf(i, poi_lambda) for i in range(poi_limit + 1)]

        if poi_mode == "P(X = k)":
            poi_value = poisson_pmf(poi_k, poi_lambda)
            poi_highlight = {poi_k}
        elif poi_mode == "P(X ≤ k)":
            poi_value = sum(poisson_pmf(i, poi_lambda) for i in range(poi_k + 1))
            poi_highlight = set(range(0, poi_k + 1))
        else:
            poi_value = 1 - sum(poisson_pmf(i, poi_lambda) for i in range(poi_k))
            poi_highlight = set(range(poi_k, poi_limit + 1))

        q1, q2, q3 = st.columns(3)

        with q1:
            st.metric(poi_mode, f"{poi_value:.4f}")

        with q2:
            st.metric("Mean = λ", f"{poi_lambda:.2f}")

        with q3:
            st.metric("Variance = λ", f"{poi_lambda:.2f}")

        bar_pmf_chart(
            list(range(poi_limit + 1)), poi_all, poi_highlight,
            f"Poisson(λ = {poi_lambda})", "Number of events (X)", "poi_chart"
        )

        st.caption(
            "A Poisson distribution has its mean equal to its variance. For "
            "large λ it starts to look like a bell curve."
        )

    # ------------------------------------------------------
    # Normal and standard normal
    # ------------------------------------------------------
    with tab_norm:

        st.header("Normal and Standard Normal (Z) Distribution")

        st.latex(r"Z=\frac{X-\mu}{\sigma}")

        n1, n2 = st.columns(2)

        with n1:
            norm_mu = st.number_input("Mean (μ)", value=800.0, key="norm_mu")

        with n2:
            norm_sigma = st.number_input(
                "Standard deviation (σ)", value=40.0, min_value=0.01,
                key="norm_sigma"
            )

        norm_mode = st.radio(
            "Probability to find",
            ["P(X < a)", "P(X > a)", "P(a < X < b)"],
            horizontal=True, key="norm_mode"
        )

        na1, na2 = st.columns(2)

        with na1:
            norm_a = st.number_input("a", value=775.0, key="norm_a")

        with na2:
            norm_b = st.number_input("b (used for the interval)",
                                     value=850.0, key="norm_b")

        z_a = (norm_a - norm_mu) / norm_sigma
        z_b = (norm_b - norm_mu) / norm_sigma

        if norm_mode == "P(X < a)":
            norm_value = normal.cdf(z_a)
        elif norm_mode == "P(X > a)":
            norm_value = 1 - normal.cdf(z_a)
        else:
            norm_value = normal.cdf(z_b) - normal.cdf(z_a)

        if norm_mode == "P(a < X < b)" and norm_b <= norm_a:
            st.warning("b must be larger than a for an interval.")

        w1, w2, w3 = st.columns(3)

        with w1:
            st.metric("z for a", f"{z_a:.4f}")

        with w2:
            st.metric("z for b", f"{z_b:.4f}")

        with w3:
            st.metric(norm_mode, f"{max(norm_value, 0):.4f}")

        def norm_pdf(values):
            return z_pdf((np.asarray(values, dtype=float) - norm_mu) / norm_sigma) / norm_sigma

        x_low = norm_mu - 4 * norm_sigma
        x_high = norm_mu + 4 * norm_sigma
        x_grid = np.linspace(x_low, x_high, 500)

        fig_norm = go.Figure()

        fig_norm.add_trace(
            go.Scatter(
                x=x_grid, y=norm_pdf(x_grid), mode="lines",
                line=dict(color="#2e7d32", width=2.5), name="Normal curve"
            )
        )

        fill = "rgba(229, 115, 115, 0.55)"

        if norm_mode == "P(X < a)":
            shade_between(fig_norm, norm_pdf, x_low, min(norm_a, x_high),
                          fill, "Probability", True)
        elif norm_mode == "P(X > a)":
            shade_between(fig_norm, norm_pdf, max(norm_a, x_low), x_high,
                          fill, "Probability", True)
        else:
            shade_between(fig_norm, norm_pdf, max(norm_a, x_low),
                          min(norm_b, x_high), fill, "Probability", True)

        fig_norm = style_figure(
            fig_norm, "Normal distribution", "X", "Density", height=400
        )

        show_chart(fig_norm, key="norm_chart")

        rule_rows = pd.DataFrame({
            "Range": ["μ ± 1σ", "μ ± 2σ", "μ ± 3σ"],
            "Probability": [
                f"{normal.cdf(k) - normal.cdf(-k):.4f}" for k in (1, 2, 3)
            ]
        })

        st.write("The empirical (68–95–99.7) rule:")
        show_table(rule_rows, hide_index=True)

    # ------------------------------------------------------
    # Normal approximation to the binomial
    # ------------------------------------------------------
    with tab_approx:

        st.header("Normal Approximation to the Binomial")

        st.write(
            "When n is large, Binomial(n, p) is close to a normal curve with "
            "mean np and standard deviation √(np(1−p)). A continuity "
            "correction (±0.5) improves the match."
        )

        ap1, ap2 = st.columns(2)

        with ap1:
            ap_n = st.slider("Number of trials (n)", 10, 500, 100, key="ap_n")

        with ap2:
            ap_p = st.slider("Success probability (p)", 0.01, 0.99, 0.30,
                             0.01, key="ap_p")

        ap_k = st.slider(
            "Find P(X ≤ k) for k =", 0, ap_n, int(ap_n * ap_p),
            key=f"ap_k_{ap_n}"
        )

        ap_mean = ap_n * ap_p
        ap_sd = math.sqrt(ap_n * ap_p * (1 - ap_p))

        exact = sum(binomial_pmf(i, ap_n, ap_p) for i in range(ap_k + 1))
        with_correction = normal.cdf((ap_k + 0.5 - ap_mean) / ap_sd)
        without_correction = normal.cdf((ap_k - ap_mean) / ap_sd)

        e1, e2, e3 = st.columns(3)

        with e1:
            st.metric("Exact binomial", f"{exact:.4f}")

        with e2:
            st.metric("Normal, with correction", f"{with_correction:.4f}")

        with e3:
            st.metric("Normal, no correction", f"{without_correction:.4f}")

        if ap_n * ap_p >= 5 and ap_n * (1 - ap_p) >= 5:
            st.success(
                f"np = {ap_n * ap_p:.1f} and n(1−p) = {ap_n * (1 - ap_p):.1f} "
                "are both at least 5, so the approximation is reliable."
            )
        else:
            st.warning(
                f"np = {ap_n * ap_p:.1f} and n(1−p) = {ap_n * (1 - ap_p):.1f}: "
                "at least one is below 5, so the normal approximation may be poor."
            )

        ap_low = max(0, int(ap_mean - 4 * ap_sd))
        ap_high = min(ap_n, int(ap_mean + 4 * ap_sd) + 1)
        ap_ks = list(range(ap_low, ap_high + 1))
        ap_curve_x = np.linspace(ap_low - 0.5, ap_high + 0.5, 300)

        fig_ap = go.Figure()

        fig_ap.add_trace(
            go.Bar(
                x=ap_ks, y=[binomial_pmf(i, ap_n, ap_p) for i in ap_ks],
                marker_color="#a5d6a7",
                marker_line=dict(color="#4caf50", width=1),
                name="Binomial"
            )
        )

        fig_ap.add_trace(
            go.Scatter(
                x=ap_curve_x,
                y=z_pdf((ap_curve_x - ap_mean) / ap_sd) / ap_sd,
                mode="lines", line=dict(color="#e57373", width=2.5),
                name="Normal curve"
            )
        )

        fig_ap = style_figure(
            fig_ap, "Binomial bars and the normal curve",
            "X", "Probability", height=400
        )

        show_chart(fig_ap, key="ap_chart")

    # ------------------------------------------------------
    # Applications
    # ------------------------------------------------------
    with tab_app:

        st.header("Applications of Probability in Data Analytics")

        st.write(
            "Probability turns data into risk statements. Below, a variable "
            "is used to answer 'how likely is a State/UT to exceed a "
            "threshold?' in two ways: counting, and using a fitted normal "
            "model."
        )

        ap_c1, ap_c2 = st.columns(2)

        with ap_c1:
            app_dataset = st.selectbox(
                "Select Dataset", dataset_names, key="app_dataset"
            )

        app_df = datasets[app_dataset]
        app_numeric = get_numeric_columns(app_df)

        with ap_c2:
            app_column = st.selectbox(
                "Select Variable", app_numeric, key=f"app_var_{app_dataset}"
            )

        app_stats = calculate_descriptive_statistics(app_df[app_column])

        if app_stats is None or app_stats["n"] < 5 or app_stats["std"] == 0:
            st.warning("Not enough varied data for this variable.")

        else:
            threshold = st.number_input(
                "Threshold (c)", value=float(round(app_stats["median"], 2)),
                step=float(max(app_stats["std"] / 10, 0.01)),
                key=f"app_c_{app_dataset}_{app_column}"
            )

            values_array = np.array(app_stats["values"])
            empirical = float(np.mean(values_array > threshold))
            model_value = 1 - NormalDist(
                app_stats["mean"], app_stats["std"]
            ).cdf(threshold)

            a1, a2, a3 = st.columns(3)

            with a1:
                st.metric("Empirical P(X > c)", f"{empirical:.4f}")

            with a2:
                st.metric("Normal model P(X > c)", f"{model_value:.4f}")

            with a3:
                st.metric("z of threshold",
                          f"{(threshold - app_stats['mean']) / app_stats['std']:.3f}")

            st.caption(
                "If the two probabilities differ a lot, the data are not "
                "normal. Counts and amounts across States/UTs are usually "
                "right-skewed, so the normal model can mislead."
            )

            st.subheader("Binomial application")

            m_col1, m_col2 = st.columns(2)

            with m_col1:
                m_states = st.slider("States/UTs picked at random (m)", 2, 30,
                                     10, key="app_m")

            with m_col2:
                j_states = st.slider("At least j exceed the threshold", 0,
                                     m_states, min(3, m_states),
                                     key=f"app_j_{m_states}")

            p_at_least = sum(
                binomial_pmf(i, m_states, empirical)
                for i in range(j_states, m_states + 1)
            )

            st.metric(
                f"P(at least {j_states} of {m_states} exceed c)",
                f"{p_at_least:.4f}"
            )

            st.caption(
                f"Using p = {empirical:.3f} from the data and a binomial "
                "model (picks treated as independent)."
            )


# ==========================================================
# UNIT III: CONFIDENCE INTERVALS AND GROUP TESTS
# ==========================================================
elif page == "Confidence Intervals and Group Tests":

    st.title("Confidence Intervals and Group Tests")

    st.write(
        "Estimate a population mean with a confidence interval, and compare "
        "groups with the independent t-test, paired t-test and one-way ANOVA."
    )

    gt_dataset = st.selectbox(
        "Select Dataset", dataset_names, key="gt_dataset"
    )

    gt_df = datasets[gt_dataset]
    gt_numeric = get_numeric_columns(gt_df)

    tab_ci, tab_ind, tab_pair, tab_anova = st.tabs([
        "Confidence interval",
        "Independent t-test",
        "Paired t-test",
        "One-way ANOVA"
    ])

    # ------------------------------------------------------
    # Confidence interval
    # ------------------------------------------------------
    with tab_ci:

        ci_column = st.selectbox(
            "Select Variable", gt_numeric, key=f"ci_var_{gt_dataset}"
        )

        ci_stats = calculate_descriptive_statistics(gt_df[ci_column])

        if ci_stats is None or ci_stats["n"] < 3 or ci_stats["std"] == 0:
            st.warning("This variable needs at least three varied values.")

        else:
            ci1, ci2 = st.columns(2)

            with ci1:
                ci_level = st.radio(
                    "Confidence level", [0.90, 0.95, 0.99], index=1,
                    horizontal=True, key="ci_level"
                )

            with ci2:
                ci_method = st.radio(
                    "Method", ["t (σ unknown)", "Z (σ known)"],
                    horizontal=True, key="ci_method"
                )

            if ci_method == "Z (σ known)":
                ci_spread = st.number_input(
                    "Known population standard deviation (σ)",
                    value=float(max(round(ci_stats["std"], 2), 0.01)),
                    min_value=0.01, key=f"ci_sigma_{gt_dataset}_{ci_column}"
                )
            else:
                ci_spread = ci_stats["std"]

            lower, upper, critical, margin = confidence_interval_for_mean(
                ci_stats["mean"], ci_spread, ci_stats["n"], ci_level, ci_method
            )

            st.latex(
                r"\bar{x}\ \pm\ (z^{*}\ \text{or}\ t^{*})\times\frac{s}{\sqrt{n}}"
            )

            k1, k2, k3, k4 = st.columns(4)

            with k1:
                st.metric("Sample mean", short_number(ci_stats["mean"]))

            with k2:
                st.metric("Critical value", f"{critical:.4f}")

            with k3:
                st.metric("Margin of error", short_number(margin))

            with k4:
                st.metric(
                    f"{int(ci_level * 100)}% interval",
                    f"{short_number(lower)} to {short_number(upper)}"
                )

            fig_ci = go.Figure()

            fig_ci.add_trace(
                go.Scatter(
                    x=[ci_stats["mean"]], y=["Mean"], mode="markers",
                    marker=dict(color="#2e7d32", size=12),
                    error_x=dict(type="constant", value=margin,
                                 color="#2e7d32", thickness=3),
                    name="Interval"
                )
            )

            fig_ci = style_figure(
                fig_ci, f"{int(ci_level * 100)}% confidence interval",
                ci_column, "", height=220
            )
            fig_ci.update_layout(showlegend=False)

            show_chart(fig_ci, key="ci_chart")

            st.caption(
                f"We are {int(ci_level * 100)}% confident that the true mean "
                "lies in this interval. The confidence is in the method: if "
                "we repeated the sampling many times, about "
                f"{int(ci_level * 100)}% of such intervals would contain the "
                "true mean."
            )

            width_rows = []

            for level in [0.90, 0.95, 0.99]:
                lo, hi, crit, mar = confidence_interval_for_mean(
                    ci_stats["mean"], ci_spread, ci_stats["n"], level, ci_method
                )
                width_rows.append({
                    "Confidence level": f"{int(level * 100)}%",
                    "Critical value": f"{crit:.4f}",
                    "Lower": short_number(lo),
                    "Upper": short_number(hi),
                    "Width": short_number(hi - lo)
                })

            show_table(pd.DataFrame(width_rows), hide_index=True)

            st.caption(
                "A higher confidence level gives a wider interval: more "
                "certainty costs precision. A larger sample makes it narrower."
            )

    # ------------------------------------------------------
    # Independent t-test
    # ------------------------------------------------------
    with tab_ind:

        if len(gt_numeric) < 2:
            st.warning("This dataset needs at least two numerical columns.")

        else:
            st.write(
                "Compare the means of two independent groups. Here each "
                "selected column is one group of State/UT values."
            )

            i1, i2 = st.columns(2)

            with i1:
                ind_a = st.selectbox("Group 1", gt_numeric,
                                     key=f"ind_a_{gt_dataset}")

            with i2:
                ind_b = st.selectbox("Group 2", gt_numeric, index=1,
                                     key=f"ind_b_{gt_dataset}")

            i3, i4, i5 = st.columns(3)

            with i3:
                ind_equal = st.radio(
                    "Variances", ["Unequal (Welch)", "Equal (pooled)"],
                    horizontal=True, key="ind_equal"
                )

            with i4:
                ind_tail = st.radio(
                    "Alternative hypothesis",
                    ["two-tailed", "right-tailed", "left-tailed"],
                    horizontal=True, key="ind_tail"
                )

            with i5:
                ind_alpha = st.radio(
                    "Significance level (α)", [0.01, 0.05, 0.10], index=1,
                    horizontal=True, key="ind_alpha"
                )

            use_log_ind = st.checkbox(
                "Use log10 of the values (helps with skewed data)",
                value=False, key="ind_log"
            )

            group_a = gt_df[ind_a].dropna()
            group_b = gt_df[ind_b].dropna()

            if use_log_ind:
                group_a = np.log10(group_a[group_a > 0])
                group_b = np.log10(group_b[group_b > 0])

            stats_a = calculate_descriptive_statistics(pd.Series(group_a))
            stats_b = calculate_descriptive_statistics(pd.Series(group_b))

            if (
                ind_a == ind_b or stats_a is None or stats_b is None
                or stats_a["n"] < 2 or stats_b["n"] < 2
            ):
                st.warning("Choose two different columns with at least two values each.")

            else:
                n_a, n_b = stats_a["n"], stats_b["n"]
                var_a, var_b = stats_a["variance"], stats_b["variance"]

                if ind_equal == "Equal (pooled)":
                    pooled = ((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2)
                    ind_se = math.sqrt(pooled * (1 / n_a + 1 / n_b))
                    ind_df = n_a + n_b - 2
                else:
                    ind_se = math.sqrt(var_a / n_a + var_b / n_b)
                    ind_df = (
                        (var_a / n_a + var_b / n_b) ** 2
                        / ((var_a / n_a) ** 2 / (n_a - 1)
                           + (var_b / n_b) ** 2 / (n_b - 1))
                    )

                difference = stats_a["mean"] - stats_b["mean"]
                t_stat = difference / ind_se if ind_se > 0 else float("nan")

                if math.isnan(t_stat):
                    st.warning("The standard error is zero, so the test cannot run.")

                else:
                    p_ind = calculate_p_value(t_stat, ind_tail, "t-test", ind_df)

                    symbol = {"two-tailed": "≠", "right-tailed": ">",
                              "left-tailed": "<"}[ind_tail]
                    phrase = {"two-tailed": "different from",
                              "right-tailed": "greater than",
                              "left-tailed": "less than"}[ind_tail]

                    st.subheader("Hypotheses")
                    st.markdown(f"- **H₀:** μ₁ = μ₂\n- **Hₐ:** μ₁ {symbol} μ₂")

                    st.latex(
                        r"t=\frac{\bar{x}_1-\bar{x}_2}{SE},\quad "
                        r"SE=\sqrt{\frac{s_1^2}{n_1}+\frac{s_2^2}{n_2}}\ \text{(Welch)}"
                    )

                    r1, r2, r3, r4 = st.columns(4)

                    with r1:
                        st.metric("Mean 1 − Mean 2", short_number(difference))

                    with r2:
                        st.metric(f"t (df = {ind_df:.1f})", f"{t_stat:.4f}")

                    with r3:
                        st.metric("p-value", show_p_value(p_ind))

                    with r4:
                        crit_two = t_critical_value(0.025, ind_df)
                        st.metric(
                            "95% CI for difference",
                            f"{short_number(difference - crit_two * ind_se)} to "
                            f"{short_number(difference + crit_two * ind_se)}"
                        )

                    show_table(
                        pd.DataFrame({
                            "Group": [ind_a, ind_b],
                            "n": [n_a, n_b],
                            "Mean": [short_number(stats_a["mean"]),
                                     short_number(stats_b["mean"])],
                            "Std. deviation": [short_number(stats_a["std"]),
                                               short_number(stats_b["std"])]
                        }),
                        hide_index=True
                    )

                    fig_ind = go.Figure()

                    for label, values in [("Group 1", group_a), ("Group 2", group_b)]:
                        fig_ind.add_trace(
                            go.Box(y=values, name=label, boxpoints="all",
                                   jitter=0.4, pointpos=0)
                        )

                    fig_ind = style_figure(
                        fig_ind, "Comparing the two groups", "",
                        "log10(value)" if use_log_ind else "Value", height=380
                    )

                    show_chart(fig_ind, key="ind_chart")

                    show_decision(
                        p_ind, ind_alpha,
                        f"the mean of group 1 ({ind_a}) is {phrase} the mean "
                        f"of group 2 ({ind_b})"
                    )

                    st.caption(
                        "The independent t-test assumes the two groups are "
                        "unrelated. Columns for the same States/UTs are "
                        "really paired, so use the Paired t-test tab for "
                        "those. The data are skewed, so also try the log10 "
                        "option."
                    )

    # ------------------------------------------------------
    # Paired t-test
    # ------------------------------------------------------
    with tab_pair:

        if len(gt_numeric) < 2:
            st.warning("This dataset needs at least two numerical columns.")

        else:
            st.write(
                "Compare two measurements on the same States/UTs (for "
                "example, beneficiaries in two years). The test works on the "
                "difference for each State/UT."
            )

            p1, p2 = st.columns(2)

            with p1:
                pair_a = st.selectbox("First measurement", gt_numeric,
                                      key=f"pair_a_{gt_dataset}")

            with p2:
                pair_b = st.selectbox("Second measurement", gt_numeric, index=1,
                                      key=f"pair_b_{gt_dataset}")

            p3, p4, p5 = st.columns(3)

            with p3:
                pair_tail = st.radio(
                    "Alternative hypothesis",
                    ["two-tailed", "right-tailed", "left-tailed"],
                    horizontal=True, key="pair_tail"
                )

            with p4:
                pair_alpha = st.radio(
                    "Significance level (α)", [0.01, 0.05, 0.10], index=1,
                    horizontal=True, key="pair_alpha"
                )

            with p5:
                pair_log = st.checkbox(
                    "Use log10 of the values", value=False, key="pair_log"
                )

            paired = gt_df[["State/UT", pair_a, pair_b]].dropna()

            if pair_log:
                paired = paired[(paired[pair_a] > 0) & (paired[pair_b] > 0)].copy()
                paired[pair_a] = np.log10(paired[pair_a])
                paired[pair_b] = np.log10(paired[pair_b])

            differences = paired[pair_a] - paired[pair_b]
            diff_stats = (
                calculate_descriptive_statistics(differences)
                if len(paired) >= 2 else None
            )

            if pair_a == pair_b or diff_stats is None or diff_stats["std"] == 0:
                st.warning(
                    "Choose two different columns with at least two complete "
                    "pairs whose differences are not all identical."
                )

            else:
                st.write(
                    f"Using **{diff_stats['n']}** complete pairs. Differences "
                    "are (first − second)."
                )

                fig_pair = go.Figure()

                fig_pair.add_trace(
                    go.Scatter(
                        x=paired[pair_b], y=paired[pair_a], mode="markers",
                        text=paired["State/UT"],
                        marker=dict(color="#2e7d32", size=9, opacity=0.75),
                        name="States/UTs",
                        hovertemplate="<b>%{text}</b><br>Second: %{x:,.2f}"
                                      "<br>First: %{y:,.2f}<extra></extra>"
                    )
                )

                axis_min = float(min(paired[pair_a].min(), paired[pair_b].min()))
                axis_max = float(max(paired[pair_a].max(), paired[pair_b].max()))

                fig_pair.add_trace(
                    go.Scatter(
                        x=[axis_min, axis_max], y=[axis_min, axis_max],
                        mode="lines", line=dict(color="#e57373", dash="dash"),
                        name="No difference (y = x)"
                    )
                )

                fig_pair = style_figure(
                    fig_pair, "Paired values (points on the line = no change)",
                    "Second measurement", "First measurement", height=400
                )

                show_chart(fig_pair, key="pair_chart")

                show_mean_test(
                    diff_stats["mean"], diff_stats["n"], 0.0, "t-test",
                    pair_tail, pair_alpha, diff_stats["std"],
                    f"the difference ({pair_a} − {pair_b})"
                )

    # ------------------------------------------------------
    # One-way ANOVA
    # ------------------------------------------------------
    with tab_anova:

        st.write(
            "One-way ANOVA tests whether three or more group means are all "
            "equal. Each selected column is treated as one group."
        )

        if len(gt_numeric) < 3:
            st.warning("This dataset needs at least three numerical columns.")

        else:
            anova_columns = st.multiselect(
                "Choose the groups (columns)", gt_numeric,
                default=gt_numeric[:3], key=f"anova_cols_{gt_dataset}"
            )

            a1, a2 = st.columns(2)

            with a1:
                anova_alpha = st.radio(
                    "Significance level (α)", [0.01, 0.05, 0.10], index=1,
                    horizontal=True, key="anova_alpha"
                )

            with a2:
                anova_log = st.checkbox(
                    "Use log10 of the values (recommended for skewed data)",
                    value=True, key="anova_log"
                )

            groups = []

            for column in anova_columns:
                series = gt_df[column].dropna()

                if anova_log:
                    series = np.log10(series[series > 0])

                if len(series) >= 2:
                    groups.append((column, np.asarray(series, dtype=float)))

            if len(groups) < 3:
                st.info("Select at least three columns that each have two or more values.")

            else:
                group_count = len(groups)
                total_count = sum(len(values) for _, values in groups)
                grand_mean = sum(values.sum() for _, values in groups) / total_count

                ss_between = sum(
                    len(values) * (values.mean() - grand_mean) ** 2
                    for _, values in groups
                )
                ss_within = sum(
                    float(np.sum((values - values.mean()) ** 2))
                    for _, values in groups
                )

                df_between = group_count - 1
                df_within = total_count - group_count

                ms_between = ss_between / df_between
                ms_within = ss_within / df_within if df_within > 0 else float("nan")

                if not ms_within or math.isnan(ms_within) or ms_within <= 0:
                    st.warning("The within-group variation is zero, so ANOVA cannot run.")

                else:
                    f_stat = ms_between / ms_within
                    p_anova = f_upper_tail(f_stat, df_between, df_within)

                    st.subheader("Hypotheses")
                    st.markdown(
                        "- **H₀:** all group means are equal\n"
                        "- **Hₐ:** at least one group mean is different"
                    )

                    st.latex(r"F=\frac{MS_{between}}{MS_{within}}")

                    anova_table = pd.DataFrame({
                        "Source": ["Between groups", "Within groups", "Total"],
                        "Sum of squares": [f"{ss_between:,.4f}", f"{ss_within:,.4f}",
                                           f"{ss_between + ss_within:,.4f}"],
                        "df": [df_between, df_within, df_between + df_within],
                        "Mean square": [f"{ms_between:,.4f}", f"{ms_within:,.4f}", ""],
                        "F": [f"{f_stat:.4f}", "", ""],
                        "p-value": [show_p_value(p_anova), "", ""]
                    })

                    show_table(anova_table, hide_index=True)

                    fig_anova = go.Figure()

                    for column, values in groups:
                        fig_anova.add_trace(
                            go.Box(y=values, name=column[:28], boxpoints="all",
                                   jitter=0.4, pointpos=0)
                        )

                    fig_anova = style_figure(
                        fig_anova, "Distribution of each group", "",
                        "log10(value)" if anova_log else "Value", height=420
                    )
                    fig_anova.update_layout(showlegend=False)

                    show_chart(fig_anova, key="anova_chart")

                    show_decision(
                        p_anova, anova_alpha,
                        "at least one group mean is different from the others"
                    )

                    st.caption(
                        "ANOVA assumes independent groups, roughly normal "
                        "data and similar spread. Columns holding different "
                        "years for the same States/UTs are not independent, "
                        "so treat this as practice with the method. A "
                        "significant F only says that some means differ, not "
                        "which ones."
                    )


# ==========================================================
# UNIT IV: REGRESSION ANALYSIS
# ==========================================================
elif page == "Regression Analysis":

    st.title("Regression Analysis and Predictive Analytics")

    st.write(
        "Build simple and multiple linear regression models, judge them "
        "with MAE, MSE, RMSE and R², and check the regression assumptions."
    )

    st.markdown(
        """
        - **Review of correlation:** r measures the strength of a straight-line relationship.
        - **Simple linear regression:** ŷ = b₀ + b₁x, with slope b₁ = Cov(x, y) / Var(x).
        - **Multiple linear regression:** ŷ = b₀ + b₁x₁ + b₂x₂ + … fitted by least squares.
        - **Assumptions:** linearity, independent errors, normal residuals, constant variance (and low multicollinearity for several predictors).
        """
    )

    reg_dataset = st.selectbox(
        "Select Dataset", dataset_names, key="reg_dataset"
    )

    reg_df = datasets[reg_dataset]
    reg_numeric = get_numeric_columns(reg_df)

    if len(reg_numeric) < 2:
        st.warning("This dataset needs at least two numerical columns.")

    else:
        opt1, opt2 = st.columns(2)

        with opt1:
            reg_log = st.checkbox(
                "Use log10 of the variables (recommended: the data are strongly right-skewed)",
                value=True, key="reg_log"
            )

        with opt2:
            reg_test_share = st.slider(
                "Share of States/UTs kept aside as a test set", 0.0, 0.4, 0.0,
                0.05, key="reg_test_share"
            )

        tab_simple, tab_multiple = st.tabs([
            "Simple linear regression", "Multiple linear regression"
        ])

        with tab_simple:

            s1, s2 = st.columns(2)

            with s1:
                simple_x = st.selectbox(
                    "Predictor (x)", reg_numeric, key=f"simple_x_{reg_dataset}"
                )

            with s2:
                simple_y = st.selectbox(
                    "Response (y)", reg_numeric, index=1,
                    key=f"simple_y_{reg_dataset}"
                )

            if simple_x == simple_y:
                st.warning("Choose two different variables.")
            else:
                run_regression(
                    reg_df, [simple_x], simple_y, reg_log, reg_test_share,
                    key="simple"
                )

        with tab_multiple:

            multi_y = st.selectbox(
                "Response (y)", reg_numeric, key=f"multi_y_{reg_dataset}"
            )

            multi_options = [c for c in reg_numeric if c != multi_y]

            multi_x = st.multiselect(
                "Predictors (x)", multi_options,
                default=multi_options[:2],
                key=f"multi_x_{reg_dataset}_{multi_y}"
            )

            if len(multi_x) < 2:
                st.info("Select at least two predictors for multiple regression.")
            else:
                run_regression(
                    reg_df, multi_x, multi_y, reg_log, reg_test_share,
                    key="multi"
                )


# ==========================================================
# END-TO-END DATA ANALYTICS WORKFLOW
# ==========================================================
elif page == "Analytics Workflow":

    st.title("End-to-End Data Analytics Workflow")

    st.write(
        "Follow the full process on one dataset: select the data, explore "
        "it, analyse it statistically, build a regression model, and "
        "interpret the result."
    )

    wf_c1, wf_c2 = st.columns(2)

    with wf_c1:
        wf_dataset = st.selectbox(
            "1. Dataset selection", dataset_names, key="wf_dataset"
        )

    wf_df = datasets[wf_dataset]
    wf_numeric = get_numeric_columns(wf_df)

    if len(wf_numeric) < 2:
        st.warning("This dataset needs at least two numerical columns.")

    else:
        with wf_c2:
            wf_target = st.selectbox(
                "Target variable (y)", wf_numeric, key=f"wf_target_{wf_dataset}"
            )

        wf_options = [c for c in wf_numeric if c != wf_target]

        wf_predictors = st.multiselect(
            "Predictor variables (x)", wf_options,
            default=wf_options[:1], key=f"wf_pred_{wf_dataset}_{wf_target}"
        )

        wf_log = st.checkbox(
            "Use log10 of the variables in the model (skewed data)",
            value=True, key="wf_log"
        )

        (step1, step2, step3, step4, step5) = st.tabs([
            "1. Dataset", "2. EDA", "3. Statistical analysis",
            "4. Regression modelling", "5. Result interpretation"
        ])

        wf_model = None

        with step1:
            st.write(
                f"**{wf_dataset}**: {wf_df.shape[0]} rows and "
                f"{wf_df.shape[1]} columns. Target: **{wf_target}**."
            )
            show_table(wf_df.head(10), hide_index=True)

        with step2:
            st.subheader("Missing values")

            missing = pd.DataFrame({
                "Column": list(wf_df.columns),
                "Missing": [int(wf_df[c].isna().sum()) for c in wf_df.columns]
            })

            show_table(missing[missing["Missing"] > 0]
                       if (missing["Missing"] > 0).any() else missing,
                       hide_index=True)

            st.subheader("Summary statistics")

            summary_rows = []

            for column in wf_numeric:
                col_stats = calculate_descriptive_statistics(wf_df[column])

                if col_stats is not None:
                    summary_rows.append({
                        "Variable": column,
                        "n": col_stats["n"],
                        "Mean": short_number(col_stats["mean"]),
                        "Median": short_number(col_stats["median"]),
                        "Std. deviation": short_number(col_stats["std"]),
                        "Min": short_number(col_stats["sorted_values"][0]),
                        "Max": short_number(col_stats["sorted_values"][-1])
                    })

            show_table(pd.DataFrame(summary_rows), hide_index=True)

            target_stats = calculate_descriptive_statistics(wf_df[wf_target])

            if target_stats is not None and target_stats["n"] >= 3:
                fig_wf_hist, _ = build_histogram(
                    target_stats["values"], min(12, max(3, target_stats["n"] // 2)),
                    wf_target, target_stats["mean"], target_stats["median"],
                    True, True
                )
                show_chart(fig_wf_hist, key="wf_hist")

            st.subheader(f"Correlation of every variable with {wf_target}")

            corr_rows = []

            for column in wf_options:
                _, r_value = calculate_pair_statistics(
                    wf_df[column].tolist(), wf_df[wf_target].tolist()
                )
                if pd.notna(r_value):
                    corr_rows.append((column, r_value))

            corr_rows.sort(key=lambda item: item[1])

            if corr_rows:
                fig_wf_corr = go.Figure()

                fig_wf_corr.add_trace(
                    go.Bar(
                        x=[r for _, r in corr_rows], y=[c for c, _ in corr_rows],
                        orientation="h", marker_color="#81c784",
                        marker_line=dict(color="#4cae51", width=1),
                        hovertemplate="<b>%{y}</b><br>r = %{x:.3f}<extra></extra>"
                    )
                )

                fig_wf_corr = style_figure(
                    fig_wf_corr, "Pearson r with the target", "r", "",
                    height=max(300, 28 * len(corr_rows) + 120)
                )
                fig_wf_corr.update_layout(margin=dict(l=20, r=30, t=80, b=60))
                fig_wf_corr.update_xaxes(range=[-1, 1])

                show_chart(fig_wf_corr, key="wf_corr")

        with step3:
            st.subheader("Confidence interval for the mean of the target")

            if target_stats is None or target_stats["n"] < 3 or target_stats["std"] == 0:
                st.warning("The target needs at least three varied values.")

            else:
                lo, hi, crit, mar = confidence_interval_for_mean(
                    target_stats["mean"], target_stats["std"],
                    target_stats["n"], 0.95, "t (σ unknown)"
                )

                st.write(
                    f"95% t-interval for the mean of **{wf_target}**: "
                    f"**{short_number(lo)} to {short_number(hi)}** "
                    f"(mean {short_number(target_stats['mean'])}, n = "
                    f"{target_stats['n']})."
                )

                st.subheader("Is each relationship statistically significant?")

                sig_rows = []

                for column in wf_options:
                    pairs = wf_df[[column, wf_target]].dropna()

                    if len(pairs) < 4:
                        continue

                    _, r_value = calculate_pair_statistics(
                        pairs[column].tolist(), pairs[wf_target].tolist()
                    )

                    if pd.isna(r_value) or abs(r_value) >= 1:
                        continue

                    t_corr = r_value * math.sqrt((len(pairs) - 2) / (1 - r_value ** 2))
                    p_corr = min(1.0, 2 * t_upper_tail(abs(t_corr), len(pairs) - 2))

                    sig_rows.append({
                        "Variable": column,
                        "n": len(pairs),
                        "r": f"{r_value:.3f}",
                        "t": f"{t_corr:.3f}",
                        "p-value": show_p_value(p_corr),
                        "Significant at 5%?": "Yes" if p_corr < 0.05 else "No"
                    })

                if sig_rows:
                    show_table(pd.DataFrame(sig_rows), hide_index=True)

                st.caption(
                    "Test of H₀: ρ = 0 using t = r√((n−2)/(1−r²)) with n−2 "
                    "degrees of freedom. Outliers can dominate r on the raw "
                    "scale, so interpret with care."
                )

        with step4:
            if len(wf_predictors) == 0:
                st.info("Select at least one predictor above.")
            else:
                wf_model = run_regression(
                    wf_df, wf_predictors, wf_target, wf_log, 0.0, key="wf"
                )

        with step5:
            if wf_model is None:
                st.info("Fit a model in step 4 to see the interpretation.")

            else:
                st.markdown(
                    f"""
                    **Summary of the analysis**

                    - **Data:** {wf_dataset}, target **{wf_target}**, predictors: {', '.join(wf_predictors)}.
                    - **Exploration:** the data are mostly right-skewed with a few very large States/UTs, which is why the log10 transformation is offered.
                    - **Model fit:** R² = {wf_model['r2']:.3f} and adjusted R² = {wf_model['adj_r2']:.3f}. The overall F-test p-value is {show_p_value(wf_model['f_p']) if not np.isnan(wf_model['f_p']) else 'N/A'}.
                    """
                )

                if not np.isnan(wf_model["f_p"]) and wf_model["f_p"] < 0.05:
                    st.success(
                        "The model as a whole is statistically significant at "
                        "the 5% level: at least one predictor is related to "
                        f"{wf_target}."
                    )
                else:
                    st.info(
                        "The model as a whole is not statistically "
                        "significant at the 5% level, so these predictors do "
                        "not give convincing evidence of a linear relationship."
                    )

                st.markdown(
                    """
                    **Cautions**

                    - Regression shows association, not causation.
                    - Only a few dozen States/UTs are available, so estimates are uncertain and large States can dominate.
                    - Check the residual plots and the assumption table in step 4 before trusting the p-values.
                    - A test set (Regression Analysis page) shows how well the model predicts unseen States/UTs.
                    """
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
