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

    p { color: #37443a; font-size: 15px; line-height: 1.6; }

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
            "Sampling and CLT",
            "Hypothesis Testing",
            "Type I and Type II Errors",
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
        relationship analysis and statistical inference to explore regional
        differences in scheme data.

        **Objective:** To evaluate state and union-territory data, compare
        central tendency and variability, examine relationships between
        landholding patterns and scheme-related measures, and test claims
        about the data using sampling theory and hypothesis tests.

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
        and loops. Sampling distributions, the Central Limit Theorem, Z and
        t hypothesis tests, and Type I and Type II errors are also covered.

        **Data Visualisation:** Interactive Plotly charts show comparisons,
        distributions, correlation matrices, sampling distributions and
        rejection regions.
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

    st.header("Inference Concepts")

    col_d, col_e, col_f = st.columns(3)

    with col_d:
        st.markdown(
            """
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