import streamlit as st
from pyspark.sql import SparkSession
from pyspark.sql.functions import lit, sum, count
from pyspark.ml.feature import VectorAssembler, PCA
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import Normalizer
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import plotly.graph_objects as go
import plotly.express as px
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

spark = SparkSession.builder \
            .appName("education_pathway_analysis") \
            .config("spark.jars", JDBC_JAR_PATH) \
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

def fetch_respondent(spark):
    query = """
    SELECT "ResponseId", "EdLevel","DevType", "Industry"
    FROM "Respondent"
    WHERE "EdLevel" IS NOT NULL AND "DevType" IS NOT NULL AND "Industry" IS NOT NULL
        AND "EdLevel" != 'NA' AND "DevType" != 'NA' AND "Industry" != 'NA'
        AND "DevType" NOT LIKE 'Other (please specify):%' AND "Industry" NOT LIKE 'Other:%'
    """
    return fetch_data(spark, query)

def fetch_learn_code_data(spark):
    query = """
    SELECT "ResponseId", "LearnCode"."name"
    FROM "RespondentLearnCode"
    INNER JOIN "LearnCode" ON "RespondentLearnCode"."id" = "LearnCode"."id"
    """
    return fetch_data(spark, query) 

def create_sankey(flows, title):
    # Remove duplicate rows in flows
    flows = flows.drop_duplicates()
    # Strip whitespace and clean source and target columns
    flows["source"] = flows["source"].str.strip()
    flows["target"] = flows["target"].str.strip()
    # Create a list of unique nodes
    nodes = pd.Index(pd.concat([flows["source"], flows["target"]]).unique())
    nodes = nodes.str.strip()  # Ensure all labels are clean
    node_map = {node: i for i, node in enumerate(nodes)}  # Map nodes to indices
    # Map source and target to their indices
    flows["source_idx"] = flows["source"].map(node_map)
    flows["target_idx"] = flows["target"].map(node_map)
    # Create the Sankey Diagram
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=20,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=nodes.tolist()  # Convert Index to list
        ),
        link=dict(
            source=flows["source_idx"],
            target=flows["target_idx"],
            value=flows["count"]
        )
    )])
    # Update layout
    fig.update_layout(
        title_text=title,
        font_size=10,
        height=600,  # Adjust height for better readability
        width=1000,  # Adjust width for better readability
        template="plotly_white",  # Use a clean template
        font=dict(size=12, color="black")  # Set consistent font size and color
    )
    # Show the Sankey Diagram in Streamlit
    st.plotly_chart(fig, use_container_width=True)

@st.cache_data
def data_to_pandas(_df_learn_code, _df_respondent):
    # Merge the two DataFrames on ResponseId
    combined_df = _df_learn_code.join(_df_respondent, on="ResponseId", how="inner")
    # Prepare the data for the Sankey Diagrams
    # Group by each pair of columns to calculate the flow counts
    edlevel_to_learn = combined_df.groupBy("EdLevel", "name").agg(count("*").alias("count")).toPandas()
    learn_to_devtype = combined_df.groupBy("name", "DevType").agg(count("*").alias("count")).toPandas()
    learn_to_industry = combined_df.groupBy("name", "Industry").agg(count("*").alias("count")).toPandas()
    return edlevel_to_learn, learn_to_devtype, learn_to_industry

def create_sankey_plots(edlevel_to_learn, learn_to_devtype, learn_to_industry):
    # Plot 1: Education → Learn Code → Dev Type
    flows_1 = pd.concat([
        edlevel_to_learn.rename(columns={"EdLevel": "source", "name": "target"}),
        learn_to_devtype.rename(columns={"name": "source", "DevType": "target"})
    ])
    create_sankey(flows_1, "Education → Learn Code → Dev Type")
    # Plot 2: Education → Learn Code → Industry
    flows_2 = pd.concat([
        edlevel_to_learn.rename(columns={"EdLevel": "source", "name": "target"}),
        learn_to_industry.rename(columns={"name": "source", "Industry": "target"})
    ])
    create_sankey(flows_2, "Education → Learn Code → Industry")

def association_analysis_data(spark):
    df_respondent = fetch_respondent(spark)
    df_respondent = df_respondent.withColumn("value", lit(1))
    edlevel_pivot = df_respondent.groupBy("ResponseId").pivot("EdLevel").agg(sum("value")).fillna(0)
    devtype_pivot = df_respondent.groupBy("ResponseId").pivot("DevType").agg(sum("value")).fillna(0)
    industry_pivot = df_respondent.groupBy("ResponseId").pivot("Industry").agg(sum("value")).fillna(0)
     # Combine all pivoted DataFrames
    df_respondent_pivot_combined = edlevel_pivot.join(devtype_pivot, on="ResponseId", how="inner") \
                    .join(industry_pivot, on="ResponseId", how="inner")
    rename_map = {
        "Associate degree (A.A., A.S., etc.)": "Associate degree",
        "Bachelor’s degree (B.A., B.S., B.Eng., etc.)": "Bachelor degree",
        "Master’s degree (M.A., M.S., M.Eng., MBA, etc.)": "Master degree",
        "Primary/elementary school": "Primary or elementary school",
        "Professional degree (JD, MD, Ph.D, Ed.D, etc.)": "Professional degree",
        "Secondary school (e.g. American high school, German Realschule or Gymnasium, etc.)": "Secondary school",
        "Some college/university study without earning a degree": "Some college or university study without earning a degree",
        "Books / Physical media": "Books or physical media",
        "Other online resources (e.g., videos, blogs, forum, online community)": "Other online resources",
        "School (i.e., University, College, etc)": "School (University or College)",
        "Senior Executive (C-Suite, VP, etc.)": "Senior Executive"
    }
    for old_name, new_name in rename_map.items():
        df_respondent_pivot_combined = df_respondent_pivot_combined.withColumnRenamed(old_name, new_name)
    df_respondent_pivot_combined = df_respondent_pivot_combined.fillna(0)

    df_learn_code = fetch_learn_code_data(spark)
    df_learn_code = df_learn_code.withColumn("value", lit(1))
    # Pivot the DataFrame to reshape it
    learn_code_pivot = df_learn_code.groupBy("ResponseId").pivot("name").agg(sum("value"))
    rename_map = {
        "Books / Physical media": "Books or physical media",
        "Other online resources (e.g., videos, blogs, forum, online community)": "Other online resources",
        "School (i.e., University, College, etc)": "School (University or College)",
    }
    for old_name, new_name in rename_map.items():
        learn_code_pivot = learn_code_pivot.withColumnRenamed(old_name, new_name)
    learn_code_pivot = learn_code_pivot.fillna(0)

    df = df_respondent_pivot_combined.join(learn_code_pivot, on="ResponseId", how="inner")
    return df

def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2, p, dof, ex = chi2_contingency(confusion_matrix, correction=False)
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    return np.sqrt(phi2 / min(k-1, r-1))

@st.cache_data
def association_analysis(_df, features, targets):
    df_pd = _df.toPandas()
    results = []
    for f in features:
        for t in targets:
            v = cramers_v(df_pd[f], df_pd[t])
            results.append((f, t, v))
    # Create a DataFrame to store the results
    correlation_df = pd.DataFrame(results, columns=["Feature", "Target", "CramersV"])
    correlation_df = correlation_df.sort_values("CramersV", ascending=False).head(10)
    return correlation_df

def association_plot(df_pd, f, t):
    # Create a table using Plotly
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=[f, t, "Cramér's V"],
            fill_color="lightgrey",
            align="left",
            font=dict(size=12, color="black")
        ),
        cells=dict(
            values=[df_pd["Feature"], df_pd["Target"], df_pd["CramersV"]],
            fill_color="white",
            align="left",
            font=dict(size=11),
            format=["", "", ".4f"]  # Format Cramér's V to 4 decimal places
        )
    )])
    # Set the title and layout
    fig.update_layout(
        title=f"Top 10 {f}-{t} Associations by Cramér's V",
        title_x=0.5,  # Center the title
        margin=dict(l=20, r=20, t=40, b=20)
    )
    st.plotly_chart(fig)


def clustering_analysis(df, feature_cols):
    assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol="features"
    )
    df_vectorized = assembler.transform(df)
    # Normalize the features
    normalizer = Normalizer(inputCol="features", outputCol="normalized_features", p=2.0)
    df_normalized = normalizer.transform(df_vectorized)
    # Perform KMeans clustering 
    kmeans = KMeans(featuresCol="normalized_features", k=3) 
    model = kmeans.fit(df_normalized)
    clustered = model.transform(df_normalized)
    return clustered

@st.cache_data
def pca_analysis(_df):
    # Perform PCA
    pca = PCA(k=2, inputCol="normalized_features", outputCol="pca_features")
    pca_model = pca.fit(_df)
    df_pca = pca_model.transform(_df)
    # Convert PCA features and cluster predictions to Pandas
    df_pandas = df_pca.select("pca_features", "prediction").toPandas()
    df_pandas[["pca_x", "pca_y"]] = pd.DataFrame(df_pandas["pca_features"].tolist(), index=df_pandas.index)
    return df_pandas

def pca_plot(df_pd):
    # Plot the PCA visualization using Plotly
    fig = px.scatter(
        df_pd,
        x="pca_x",
        y="pca_y",
        color="prediction",
        title="PCA Visualization of Clusters",
        labels={"pca_x": "PCA Component 1", "pca_y": "PCA Component 2", "prediction": "Cluster"},
        color_continuous_scale="Viridis",
        template="plotly"
    )
    # Update layout for better visualization
    fig.update_layout(
        title_x=0.5,  # Center the title
        width=800,
        height=600,
        legend_title="Cluster"
    )
    st.plotly_chart(fig)



# Streamlit page configuration
st.set_page_config(page_title="Education Pathway Analysis", page_icon="📚", layout="wide")

# Page title and description
st.title("📚 Education Pathway Analysis")
st.markdown(
    """
    This page explores the relationships between education pathways, learning methods, and career outcomes 
    using data from the Stack Overflow Developer Survey. Key analyses include:
    - **Sankey Diagrams**: Visualize the flow from education levels to learning methods and career outcomes.
    - **Association Analysis**: Identify the strongest relationships between education, industries, and developer roles.
    - **Clustering and PCA**: Group respondents based on their education and learning methods, and visualize clusters using PCA.
    """
)

st.subheader("🎓 Education Pathways to Career Outcomes")
st.markdown(
    """
    The Sankey diagrams below illustrate the flow from education levels to learning methods and career outcomes.
    """
)
# Create and display the Sankey Diagrams
df_learn_code = fetch_learn_code_data(spark)
df_respondent = fetch_respondent(spark)
edlevel_to_learn, learn_to_devtype, learn_to_industry = data_to_pandas(df_learn_code, df_respondent)
create_sankey_plots(edlevel_to_learn, learn_to_devtype, learn_to_industry)

learn_code_features = [
    "Books or physical media", "Coding Bootcamp", "Colleague", "Friend or family member",
    "On the job training", "Online Courses or Certification", "Other online resources", "School (University or College)"
]
education_features = [
    "Associate degree", "Bachelor degree", "Master degree", "Primary or elementary school", "Professional degree",
    "Secondary school", "Some college or university study without earning a degree"
]
industry_targets = [
    "Banking/Financial Services", "Computer Systems Design and Services", "Energy", "Fintech", "Government", "Healthcare", 
    "Higher Education", "Insurance", "Internet, Telecomm or Information Services", "Manufacturing", 
    "Media & Advertising Services", "Retail and Consumer Services", "Software Development", "Transportation, or Supply Chain"
]
dev_type_targets = [
    "Academic researcher", "Blockchain", "Cloud infrastructure engineer", "Data engineer", "Data or business analyst", "Student", "System administrator",
    "Data scientist or machine learning specialist", "Database administrator", "Designer", "DevOps specialist", "Developer Advocate", 
    "Developer Experience", "Developer, AI", "Developer, QA or test", "Developer, back-end", "Developer, desktop or enterprise applications", 
    "Developer, embedded applications or devices", "Developer, front-end", "Developer, full-stack", "Developer, game or graphics", 
    "Developer, mobile", "Educator", "Engineer, site reliability", "Engineering manager", "Hardware Engineer", "Marketing or sales professional", 
    "Product manager", "Project manager", "Research & Development role", "Scientist", "Security professional", "Senior Executive"
]

st.subheader("🔗 Association Analysis")
st.markdown(
    """
    This analysis identifies the strongest relationships between education levels, industries, and developer roles 
    using Cramér's V statistic.
    """
)
features_pd = association_analysis_data(spark)
indus_association = association_analysis(features_pd, education_features, industry_targets)
association_plot(indus_association, "Education", "Industry")
dev_association = association_analysis(features_pd, education_features, dev_type_targets)
association_plot(dev_association, "Education", "DevType")


st.subheader("📊 PCA Visualization of Clusters")
st.markdown(
    """
    The PCA plot below visualizes clusters of respondents based on their education levels and learning methods.
    """
)
clustered = clustering_analysis(features_pd, learn_code_features + education_features)
clustered = pca_analysis(clustered)
pca_plot(clustered)

spark.stop()


# from pyspark.ml.evaluation import ClusteringEvaluator
# # Loop through k values from 3 to 51
# for k in range(3, 51):
#     print(f"Training KMeans with k={k}...")
    
#     # Train the KMeans model
#     kmeans = KMeans(featuresCol="normalized_features", k=k, seed=42)  # Set seed for reproducibility
#     model = kmeans.fit(df_normalized)
    
#     # Transform the data to assign clusters
#     clustered = model.transform(df_normalized)
    
#     # Evaluate the model using Silhouette Score with cosine distance
#     evaluator = ClusteringEvaluator(featuresCol="normalized_features", metricName="silhouette", distanceMeasure="cosine")
#     silhouette_score = evaluator.evaluate(clustered)
    
#     print(f"Silhouette Score for k={k}: {silhouette_score}")