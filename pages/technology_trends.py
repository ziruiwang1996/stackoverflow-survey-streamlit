import streamlit as st
from pyspark.sql import SparkSession
import plotly.graph_objects as go
import os
from dotenv import load_dotenv

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

@st.cache_resource
def get_spark_session():
    """
    Cache the Spark session to avoid reinitializing it when switching pages.
    """
    return (SparkSession.builder
            .appName("Technology Trends Analysis")
            .config("spark.executor.memory", "2g")
            .config("spark.driver.memory", "2g")
            .config("spark.jars", JDBC_JAR_PATH)
            .getOrCreate())

@st.cache_data
def fetch_data(_spark, table_name, lookup_table):
    query = f"""
    WITH T AS (
        SELECT "id" AS t_id, COUNT(DISTINCT "ResponseId") AS cnt
        FROM "{table_name}"
        GROUP BY "id"
    ),
    W AS (
        SELECT "id" AS w_id, COUNT(DISTINCT "ResponseId") AS want_cnt
        FROM "{table_name}Want"
        GROUP BY "id"
    )
    SELECT LU.name, T.cnt, W.want_cnt
    FROM T INNER JOIN W ON T.t_id = W.w_id
    INNER JOIN "{lookup_table}" LU ON T.t_id = LU.id
    ORDER BY T.cnt + W.want_cnt DESC
    """
    try:
        df = (_spark.read
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
        return df.toPandas()
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        raise

def plot(df_pd, feature):
    # Create the stacked bar chart
    fig = go.Figure()

    # Add the `cnt` bar (bottom layer)
    fig.add_trace(go.Bar(
        x=df_pd["name"],
        y=df_pd["cnt"],
        name="Worked with in PAST year",
        marker_color="blue"
    ))

    # Add the `want_cnt` bar (stacked on top)
    fig.add_trace(go.Bar(
        x=df_pd["name"],
        y=df_pd["want_cnt"],
        name="Want to work with NEXT year",
        marker_color="orange"
    ))

    # Update layout for better visualization
    fig.update_layout(
        barmode="stack",  # Stacked bar chart
        title=f"{feature} Popularity and Demand",
        xaxis_title=feature,
        yaxis_title="Count",
        legend_title="Metric",
        xaxis=dict(
            tickangle=-45,  # Rotate x-axis labels
            automargin=True,  # Adjust margins automatically
            tickmode="linear"  # Ensure all labels are shown
        ),
        template="plotly_white",
        height=600,
        width=1200  # Increase width for better readability
    )

    # Show the plot
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    # Streamlit configuration
    st.set_page_config(page_title="Technology Trends", layout="wide")

    # Page Title and Description
    st.title("Technology Trends in Software Development")
    st.markdown("""
    This page explores the popularity and demand for various technologies among developers, 
    based on survey data. The analysis includes programming languages, databases, cloud platforms, 
    web frameworks, and more. Each chart shows the number of developers who have worked with a 
    technology in the past year and those who want to work with it in the next year.
    """)

    # Initialize Spark session
    spark = get_spark_session()

    # Fetch and Plot Data
    st.header("Programming Languages")
    st.markdown("The chart below shows the popularity and demand for programming languages.")
    pl_df = fetch_data(spark, "RespondentLanguage", "ProgrammingLanguage")
    plot(pl_df, "Programming Language")

    st.header("Databases")
    st.markdown("The chart below shows the popularity and demand for databases.")
    db_df = fetch_data(spark, "RespondentDatabase", "Database")
    plot(db_df, "Database")

    st.header("Cloud Platforms")
    st.markdown("The chart below shows the popularity and demand for cloud platforms.")
    cloud_df = fetch_data(spark, "RespondentCloud", "Cloud")
    plot(cloud_df, "Cloud Platform")

    st.header("Web Frameworks")
    st.markdown("The chart below shows the popularity and demand for web frameworks and technologies.")
    web_df = fetch_data(spark, "RespondentWebFramework", "WebFramework")
    plot(web_df, "Web Framework/Technology")

    st.header("Embedded Systems")
    st.markdown("The chart below shows the popularity and demand for embedded systems and technologies.")
    embed_df = fetch_data(spark, "RespondentEmbeddedSystem", "EmbeddedSystem")
    plot(embed_df, "Embedded Systems/Technologies")

    st.header("Miscellaneous Technologies")
    st.markdown("The chart below shows the popularity and demand for other frameworks and technologies.")
    misc_df = fetch_data(spark, "RespondentMiscTech", "MiscTech")
    plot(misc_df, "Other Frameworks/Technologies")

    st.header("Developer Tools")
    st.markdown("The chart below shows the popularity and demand for developer tools used for compiling, building, and testing.")
    dev_df = fetch_data(spark, "RespondentDevTool", "DevTool")
    plot(dev_df, "Developer Tools for Compiling, Building and Testing")

    st.header("Integrated Development Environments (IDEs)")
    st.markdown("The chart below shows the popularity and demand for IDEs.")
    ide_df = fetch_data(spark, "RespondentIDE", "IDE")
    plot(ide_df, "Integrated Development Environment")

    st.header("AI-Powered Tools")
    st.markdown("The chart below shows the popularity and demand for AI-powered search and developer tools.")
    ai_tool_df = fetch_data(spark, "RespondentAITool", "AITool")
    plot(ai_tool_df, "AI-powered search and developer tools")

    # Stop Spark session
    spark.stop()