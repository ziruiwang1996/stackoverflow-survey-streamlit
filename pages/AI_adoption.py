import streamlit as st
from pyspark.sql import SparkSession
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from dotenv import load_dotenv
from pyspark.sql.functions import when, col, sum, lit

load_dotenv()
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT", "5432")  # Default PostgreSQL port
SSL_MODE = os.getenv("SSL_MODE", "prefer")  # Default SSL mode
JDBC_URL = f"jdbc:postgresql://{DB_HOST}:{DB_PORT}/{DB_NAME}"

# JDBC connection options
JDBC_OPTS = {
    "user": DB_USER,
    "password": DB_PASSWORD,
    "driver": "org.postgresql.Driver",
    "ssl": "true" if SSL_MODE in ["require", "prefer"] else "false",
    "sslmode": SSL_MODE,
    "fetchsize": "1000",  # Optimize fetch size for better performance
    "batchsize": "1000"   # Optimize batch size for better performance
}
# JDBC JAR path - you'll need to download the PostgreSQL JDBC driver
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
JDBC_JAR_PATH = os.path.join(BASE_DIR, "postgresql-42.7.6.jar")

spark = SparkSession.builder.appName("ai_adoption_analysis")\
            .config("spark.jars", JDBC_JAR_PATH)\
            .getOrCreate()

def fetch_data(spark, query):
    """
    Helper function to fetch data from the database using a SQL query.
    """
    try:
        df = (spark.read
              .format("jdbc")
              .option("url", JDBC_URL)
              .option("query", query)
              .option("user", JDBC_OPTS["user"])
              .option("password", JDBC_OPTS["password"])
              .option("driver", JDBC_OPTS["driver"])
              .option("ssl", JDBC_OPTS["ssl"])
              .option("sslmode", JDBC_OPTS["sslmode"])
              .option("fetchsize", JDBC_OPTS["fetchsize"])
              .option("batchsize", JDBC_OPTS["batchsize"])
              .load())
        return df
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        raise

@st.cache_data
def ai_use_by_industry_data(_spark):
    query = """
    WITH industry_totals AS (
        SELECT 
            "Industry", 
            COUNT(*) AS total_respondents
        FROM "Respondent"
        WHERE "Industry" IS NOT NULL AND "Industry" != 'NA' AND "Industry" != 'Other:'
        GROUP BY "Industry"
    )
    SELECT 
        r."Industry", 
        r."AISelect", 
        COUNT(*) * 100.0 / it.total_respondents AS percentage
    FROM "Respondent" r
    INNER JOIN industry_totals it
    ON r."Industry" = it."Industry"
    WHERE r."Industry" IS NOT NULL AND r."AISelect" IS NOT NULL
    GROUP BY r."Industry", r."AISelect", it.total_respondents
    """
    df = fetch_data(_spark, query)
    # Pivot the data so that AISelect becomes columns
    df = df.groupBy("Industry").pivot("AISelect").agg({"percentage": "max"}).fillna(0)
    # Sort by 'Yes' column in descending order
    if "Yes" in df.columns:
        df = df.orderBy(df["Yes"].desc())
    return df.toPandas()

@st.cache_data
def ai_use_by_devtype_data(_spark):
    query = """
    WITH dev_type_responses AS (
        SELECT
            "DevType", COUNT(*) AS total_respondents
        FROM "Respondent"
        WHERE "DevType" IS NOT NULL AND "DevType" != 'NA' AND "DevType" != 'Other (please specify):'
        GROUP BY "DevType"
    )
    SELECT 
        R."DevType", R."AISelect", COUNT(*) * 100.0 / DTR.total_respondents AS percentage
    FROM "Respondent" R
    INNER JOIN dev_type_responses DTR
    ON R."DevType" = DTR."DevType"
    WHERE R."AISelect" IS NOT NULL AND R."AISelect" != 'NA'
    GROUP BY R."DevType", R."AISelect", DTR.total_respondents
    """
    df = fetch_data(_spark, query)
    # Pivot the data so that AISelect becomes columns
    df = df.groupBy("DevType").pivot("AISelect").agg({"percentage": "max"}).fillna(0)
    # Sort by 'Yes' column in descending order
    if "Yes" in df.columns:
        df = df.orderBy(df["Yes"].desc())
    return df.toPandas()

def stacked_bar_plot(df_pd, feature, title, x_title, y_title, legend_title): 
    # Create a stacked bar chart using Plotly
    fig = go.Figure()
    dev_types = df_pd[feature] 
    ai_selects = ["Yes", "No, but I plan to soon", "No, and I don't plan to"]  # Order of stacking
    for ai_select in ai_selects:
        if ai_select in df_pd.columns:
            fig.add_trace(go.Bar(
                x=dev_types,
                y=df_pd[ai_select],
                name=ai_select,
                hoverinfo="x+y+name"  # Show details on hover
            ))
    # Update layout for better readability
    fig.update_layout(
        title=title,
        xaxis_title=x_title,
        yaxis_title=y_title,
        barmode="stack",  # Stacked bar chart
        legend_title=legend_title,
        xaxis_tickangle=45
    )
    # Display the plot in Streamlit
    st.plotly_chart(fig)

@st.cache_data
def ai_trust_data(_spark):
    query = """
    SELECT "AIAcc", COUNT(*) AS count
    FROM "Respondent"
    WHERE "AIAcc" IS NOT NULL AND "WorkExp" IS NOT NULL AND "AIAcc" != 'NA'
    GROUP BY "AIAcc"
    """
    df = fetch_data(_spark, query)

    order_mapping = {
        "Highly trust": 1,
        "Somewhat trust": 2,
        "Neither trust nor distrust": 3,
        "Somewhat distrust": 4,
        "Highly distrust": 5
    }

    # Add a new column for sorting based on the desired order
    df = df.withColumn(
        "order",
        when(col("AIAcc") == "Highly trust", order_mapping["Highly trust"])
        .when(col("AIAcc") == "Somewhat trust", order_mapping["Somewhat trust"])
        .when(col("AIAcc") == "Neither trust nor distrust", order_mapping["Neither trust nor distrust"])
        .when(col("AIAcc") == "Somewhat distrust", order_mapping["Somewhat distrust"])
        .when(col("AIAcc") == "Highly distrust", order_mapping["Highly distrust"])
        .otherwise(6)  # Default order for unexpected values
    )

    # Sort the data based on the custom order
    df = df.orderBy("order")
    return df.select("AIAcc", "count").toPandas()

@st.cache_data
def ai_threat_data(_spark):
    query = """
    SELECT "AIThreat", COUNT(*) AS count
    FROM "Respondent"
    WHERE "AIThreat" IS NOT NULL AND "AIThreat" != 'NA'
    GROUP BY "AIThreat"
    """
    df = fetch_data(_spark, query)
    return df.toPandas()
    
def bar_plot(df_pd, x_col, y_col, title, x_title, y_title):
    # Create an interactive bar chart using Plotly
    fig = go.Figure(data=[
        go.Bar(
            x=df_pd[x_col],
            y=df_pd[y_col],
            marker_color="skyblue",
            hoverinfo="x+y"  # Show details on hover
        )
    ])
    # Update layout for better readability
    fig.update_layout(
        title=title,
        xaxis_title=x_title,
        yaxis_title=y_title,
        xaxis_tickangle=45
    )
    # Display the plot in Streamlit
    st.plotly_chart(fig)


# Function to group <1% categories into "Other"
def group_small_categories(df, label_col, value_col, threshold=1):
    # Calculate the total count
    total_count = df.select(sum(col(value_col))).collect()[0][0]
    # Add a percentage column
    df = df.withColumn("percentage", (col(value_col) / lit(total_count)) * 100)
    # Separate large and small categories
    large_categories = df.filter(col("percentage") >= threshold)
    small_categories = df.filter(col("percentage") < threshold)
    # Aggregate small categories into "Other"
    if small_categories.count() > 0:
        other_row = small_categories.groupBy().agg(
            lit("Other").alias(label_col),
            sum(col(value_col)).alias(value_col),
            sum(col("percentage")).alias("percentage")
        )
        # Combine large categories and "Other"
        df = large_categories.union(other_row)
    # Select only the label and value columns
    return df.select(label_col, value_col)

def ai_trust_details_data(spark, ai_acc):
    ai_acc_escaped = ai_acc.replace("'", "''")
    query_devtype = f"""
    SELECT "DevType", COUNT(*) AS count
    FROM "Respondent"
    WHERE "AIAcc" = '{ai_acc_escaped}' AND "DevType" IS NOT NULL AND "DevType" != 'NA' AND "DevType" != 'Other (please specify):'
    GROUP BY "DevType"
    """
    query_industry = f"""
    SELECT "Industry", COUNT(*) AS count
    FROM "Respondent"
    WHERE "AIAcc" = '{ai_acc_escaped}' AND "Industry" IS NOT NULL AND "Industry" != 'NA' AND "Industry" != 'Other:'
    GROUP BY "Industry"
    """
    # Load data from the database
    df_devtype = fetch_data(spark, query_devtype)
    df_industry = fetch_data(spark, query_industry)
    return df_devtype, df_industry

def ai_threat_details_data(spark, ai_threat):
    ai_threat_escaped = ai_threat.replace("'", "''")
    query_devtype = f"""
    SELECT "DevType", COUNT(*) AS count
    FROM "Respondent"
    WHERE "AIThreat" = '{ai_threat_escaped}' AND "DevType" IS NOT NULL AND "DevType" != 'NA' AND "DevType" != 'Other (please specify):'
    GROUP BY "DevType"
    """
    query_industry = f"""
    SELECT "Industry", COUNT(*) AS count
    FROM "Respondent"
    WHERE "AIThreat" = '{ai_threat_escaped}' AND "Industry" IS NOT NULL AND "Industry" != 'NA' AND "Industry" != 'Other:'
    GROUP BY "Industry"
    """
    # Load data from the database
    df_devtype = fetch_data(spark, query_devtype)
    df_industry = fetch_data(spark, query_industry)
    return df_devtype, df_industry

@st.cache_data
def data_caching(_df_devtype, _df_industry, arg):
    devtype_data = group_small_categories(_df_devtype, "DevType", "count")
    industry_data = group_small_categories(_df_industry, "Industry", "count")
    return devtype_data.toPandas(), industry_data.toPandas()

def pie_plot(df_pd_devtype, df_pd_industry, trust_or_threat, level):
    # Create side-by-side pie charts
    fig = make_subplots(
        rows=1, cols=2, specs=[[{'type': 'domain'}, {'type': 'domain'}]],
        subplot_titles=["Developer Type Breakdown", "Industry Breakdown"]
    )
    # Add pie chart for developer type
    fig.add_trace(
        go.Pie(
            labels=df_pd_devtype["DevType"],
            values=df_pd_devtype["count"],
            name="Developer Type",
            showlegend=False  # Remove legend
        ),
        row=1, col=1
    )
    # Add pie chart for industry
    fig.add_trace(
        go.Pie(
            labels=df_pd_industry["Industry"],
            values=df_pd_industry["count"],
            name="Industry",
            showlegend=False  # Remove legend
        ),
        row=1, col=2
    )
    # Update layout for better readability
    fig.update_layout(
        title_text=f"Details for AI {trust_or_threat} Level: {level}",
        title_x=0.5,  # Center the title
        height=500  # Adjust height for side-by-side layout
    )
    # Display the plot in Streamlit
    st.plotly_chart(fig)



st.set_page_config(page_title="AI Adoption Analysis", page_icon="📈", layout="wide")
st.markdown("# 📊 AI Adoption Analysis")
st.write(
    """
    This analysis explores the adoption and perception of AI across different industries and developer roles. 
    It provides insights into how developers are using AI tools, their trust levels in AI, and their views on AI as a potential threat.

    ### Key Highlights:
    - **AI Use by Industry:** A breakdown of AI adoption across various industries, showing the percentage of respondents who currently use AI, plan to use it, or do not intend to use it.
    - **AI Use by Developer Type:** An analysis of AI adoption among different developer roles, highlighting how AI usage varies by profession.
    - **AI Trust Levels:** A distribution of trust levels in AI, ranging from "Highly trust" to "Highly distrust," providing insights into developers' confidence in AI technologies.
    - **AI Threat Levels:** An exploration of how developers perceive AI as a potential threat, categorized into different levels of concern.

    These visualizations aim to provide a comprehensive understanding of the current state of AI adoption and attitudes toward AI in the developer community.
    """
)

# Section: AI Use by Industry
st.markdown("## 🏭 AI Use by Industry")
st.write("Explore how AI adoption varies across different industries.")
df_pd_industry = ai_use_by_industry_data(spark)
stacked_bar_plot(df_pd_industry, "Industry", "AI Adoption by Industry", "Industry", "Percentage (%)", "AI Use")

# Section: AI Use by Developer Type
st.markdown("## 👩‍💻 AI Use by Developer Type")
st.write("Analyze how AI adoption differs among various developer roles.")
df_pd_devtype = ai_use_by_devtype_data(spark)
stacked_bar_plot(df_pd_devtype, "DevType", "AI Adoption by Developer Type", "Developer Type", "Percentage (%)", "AI Use")

# Section: AI Trust Levels
st.markdown("## 🤝 AI Trust Levels")
st.write("Understand developers' trust levels in AI technologies.")
ai_trust_df = ai_trust_data(spark)
bar_plot(ai_trust_df, "AIAcc", "count",
          title="AI Trust Levels Distribution",
          x_title="AI Trust Levels",
          y_title="Number of Respondents")  

# Detailed Analysis: AI Trust Levels
st.markdown("### 🔍 Detailed Analysis: AI Trust Levels")
st.write("Select an AI trust level to view a detailed breakdown by developer type and industry.")
aiacc = st.selectbox(
    "Select AI Trust Level:",
    ["Highly trust", "Somewhat trust", "Neither trust nor distrust", "Somewhat distrust", "Highly distrust"]
)
trust_by_devtype_df, trust_by_industry_df = ai_trust_details_data(spark, aiacc)
trust_by_devtype_df_pd, trust_by_industry_df_pd = data_caching(trust_by_devtype_df, trust_by_industry_df, aiacc)
pie_plot(trust_by_devtype_df_pd, trust_by_industry_df_pd, "Trust", aiacc)

# Section: AI Threat Levels
st.markdown("## ⚠️ AI Threat Levels")
st.write("Explore how developers perceive AI as a potential threat.")
ai_threat_df = ai_threat_data(spark)
bar_plot(ai_threat_df, "AIThreat", "count",
         title="AI Threat Levels Distribution",
         x_title="AI Threat Levels",
         y_title="Number of Respondents")    

# Detailed Analysis: AI Threat Levels
st.markdown("### 🔍 Detailed Analysis: AI Threat Levels")
st.write("Select an AI threat level to view a detailed breakdown by developer type and industry.")
aithreat = st.selectbox(
    "Select AI Threat Level:",
    ["Yes", "No", "I'm not sure"]
)
threat_by_devtype_df, threat_by_industry_df = ai_threat_details_data(spark, aithreat)
threat_by_devtype_df_pd, threat_by_industry_df_pd = data_caching(threat_by_devtype_df, threat_by_industry_df, aithreat)
pie_plot(threat_by_devtype_df_pd, threat_by_industry_df_pd, "Threat", aithreat)

spark.stop()  # Stop the Spark session when done