import streamlit as st

st.set_page_config(
    page_title="2024 Developer Survey Analysis",
    page_icon="👋",
)

st.write("# Analysis of Stack Overflow Developer Survey 2024")

st.sidebar.success("Select a demo above.")

st.markdown(
    """
    This study provides a comprehensive exploration of developer trends, career outcomes, education pathways, 
    and technology adoption based on the [Stack Overflow Developer Survey 2024](https://survey.stackoverflow.co). 
    Below are the key areas of analysis:
    - **AI Adoption:** Examine how developers are adopting AI tools and their attitudes toward AI.
    - **Career Outcomes:** Understand how factors like education, years of professional coding experience, developer roles, and industries influence salaries.
    - **Education Pathways:** Explore how self-taught developers differ from university-trained developers in terms of roles and industries.
    - **Technology Adoption:** Identify technologies with the highest growth potential among developers.
    ### How to Explore the Study
    - **👈 Select a demo from the sidebar** to navigate through the different sections of the analysis.
    - Each section includes interactive visualizations and insights to help you understand the trends and patterns in the developer community.
    ### Want to learn more?
    - Check out [github](https://github.com/ziruiwang1996)
"""
)