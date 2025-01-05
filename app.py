import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from langchain_openai import ChatOpenAI
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.tools import Tool
import os
from dotenv import load_dotenv

# Page configuration
st.set_page_config(page_title="Titanic Dataset Analyzer", layout="wide")

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Initialize the model
def init_model():
    return ChatOpenAI(
        model_name="gpt-3.5-turbo-0125",
        openai_api_key=OPENAI_API_KEY,
        temperature=0.3
    )

# Custom functions for visualizations
def create_age_distribution(df_input):
    fig = px.histogram(
        df_input, 
        x='Age',
        nbins=20,
        title='Age Distribution in Titanic Dataset',
        labels={'Age': 'Age (years)', 'count': 'Number of Passengers'}
    )
    st.plotly_chart(fig, use_container_width=True)
    return "Age distribution plot has been created and displayed. The histogram shows the distribution of passenger ages across different age groups."

def create_survival_by_class(df_input):
    survival_by_class = df_input.groupby('Pclass')['Survived'].mean().reset_index()
    fig = px.bar(
        survival_by_class,
        x='Pclass',
        y='Survived',
        title='Survival Rate by Passenger Class',
        labels={'Pclass': 'Passenger Class', 'Survived': 'Survival Rate'}
    )
    st.plotly_chart(fig, use_container_width=True)
    return "Survival rate by class plot has been created and displayed. The bar chart shows the survival rates for each passenger class."

def create_fare_distribution(df_input):
    fig = px.box(
        df_input,
        x='Survived',
        y='Fare',
        title='Fare Distribution by Survival Status',
        labels={'Survived': 'Survival Status', 'Fare': 'Fare Price'}
    )
    st.plotly_chart(fig, use_container_width=True)
    return "Fare distribution plot has been created and displayed. The box plot shows the distribution of fares for survivors and non-survivors."

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        margin-top: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and description
st.title("🚢 Titanic Dataset Analyzer")
st.markdown("""
This app allows you to analyze the Titanic dataset through natural language queries.
Simply ask questions about passenger survival rates, demographics, or ticket information.
""")

# Load the Titanic dataset
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('titanic.csv')
        return df
    except:
        st.error("Please ensure the Titanic dataset is in the same directory as the app.")
        return None

df = load_data()

if df is not None:
    # Initialize the LangChain agent
    llm = init_model()
    
    # Create tools list with proper function wrappers
    tools = [
        Tool(
            name="Age Distribution Plot",
            func=lambda x: create_age_distribution(df),
            description="Creates a histogram showing the distribution of passenger ages"
        ),
        Tool(
            name="Survival by Class Plot",
            func=lambda x: create_survival_by_class(df),
            description="Creates a bar chart showing survival rates by passenger class"
        ),
        Tool(
            name="Fare Distribution Plot",
            func=lambda x: create_fare_distribution(df),
            description="Creates a box plot showing fare distribution by survival status"
        )
    ]
    
    # Create the agent with tools
    agent = create_pandas_dataframe_agent(
        llm,
        df,
        verbose=True,
        handle_parsing_errors=True,
        allow_dangerous_code=True,
        extra_tools=tools
    )

    # Sidebar with example questions
    st.sidebar.header("📝 Example Questions")
    example_questions = [
        "Show me the age distribution of passengers",
        "What percentage of survivors were females?",
        "Show me the survival rate by passenger class",
        "What is the fare distribution for survivors vs non-survivors?",
        "How many males survived?",
        "What was the average age of survivors vs non-survivors?",
    ]
    
    # Create columns for better button layout
    col1, col2 = st.sidebar.columns(2)
    for i, q in enumerate(example_questions):
        if i % 2 == 0:
            if col1.button(q, key=f"q{i}"):
                st.session_state['user_question'] = q
        else:
            if col2.button(q, key=f"q{i}"):
                st.session_state['user_question'] = q

    # Main query interface
    st.header("💭 Ask your question")
    
    # Initialize session state for user question if it doesn't exist
    if 'user_question' not in st.session_state:
        st.session_state['user_question'] = ""

    # Text input for user question
    user_question = st.text_input(
        "Enter your question about the Titanic dataset:",
        value=st.session_state.get('user_question', ''),
        key="question_input"
    )

    if st.button("Analyze"):
        if user_question:
            try:
                with st.spinner('Analyzing the data...'):
                    # Add context to the question
                    enhanced_question = f"""
                    Analyze this question about the Titanic dataset: {user_question}
                    
                    If the question is about:
                    - age distribution or age patterns: use the Age Distribution Plot tool
                    - survival rates by class or passenger class analysis: use the Survival by Class Plot tool
                    - fare analysis or ticket prices: use the Fare Distribution Plot tool
                    - If the explicit mention about the plot of anything use the Plot tool
                    
                    For other questions, provide numerical results and insights in natural language.
                    Make sure to explain the insights from any visualizations created.
                    """
                    
                    response = agent.run(enhanced_question)
                    
                    # Display the response
                    st.markdown("### 📊 Analysis Results")
                    st.write(response)

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.info("Try rephrasing your question or use one of the example questions from the sidebar.")
        else:
            st.warning("Please enter a question or select one from the examples.")

    # Display dataset preview
    with st.expander("👀 Preview Dataset"):
        st.dataframe(df.head())
        st.markdown(f"**Dataset Shape:** {df.shape[0]} rows and {df.shape[1]} columns")

else:
    st.error("Failed to load the dataset. Please check if the file exists and is accessible.")

# Footer
st.markdown("---")
st.markdown("Built with Streamlit, LangChain, and OpenAI")