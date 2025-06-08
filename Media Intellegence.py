import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import json
import io
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.enums import TA_CENTER, TA_LEFT

# --- Configuration ---
# Set page config for a wider layout and title
st.set_page_config(layout="wide", page_title="Interactive Media Intelligence Dashboard")

# Colors for charts (Tailwind-inspired equivalents)
CHART_COLORS = {
    'primary': '#3B82F6',  # Blue-500
    'secondary': '#6366F1', # Indigo-500
    'tertiary': '#0EA5E9', # Sky-500
    'sentiment': ['#10B981', '#FCD34D', '#EF4444'], # Green, Yellow, Red for Positive, Neutral, Negative
    'media_type': ['#8B5CF6', '#EC4899', '#F97316', '#14B8A6', '#60A5FA', '#DC2626', '#EAB308'], # Various vibrant colors
}

# --- Utility Functions ---

def parse_csv_and_clean_data(uploaded_file):
    """
    Parses an uploaded CSV file into a DataFrame and cleans the data.
    - Converts 'Date' to datetime.
    - Fills missing 'Engagements' with 0.
    - Normalizes column names (basic).
    - Fills missing categorical data with 'Unknown'.
    """
    try:
        # Read CSV with pandas
        df = pd.read_csv(uploaded_file)

        # Normalize column names by stripping whitespace and converting to Title Case for display,
        # but using a consistent lowercase_underscore for internal processing.
        # Create a mapping for easy access
        column_mapping = {col.strip(): col.strip().replace(' ', '_').lower() for col in df.columns}
        df.rename(columns=column_mapping, inplace=True)

        # Ensure required columns exist after normalization
        required_columns = ['date', 'platform', 'sentiment', 'location', 'engagements', 'media_type', 'influencer_brand', 'post_type']
        for col in required_columns:
            if col not in df.columns:
                st.error(f"Missing required column after cleaning: '{col}'. Please check your CSV file headers.")
                return pd.DataFrame() # Return empty DataFrame on critical error

        # Convert 'date' to datetime, coercing errors to NaT (Not a Time)
        df['date'] = pd.to_datetime(df['date'], errors='coerce')

        # Fill missing 'engagements' with 0 and convert to integer
        df['engagements'] = pd.to_numeric(df['engagements'], errors='coerce').fillna(0).astype(int)

        # Fill missing categorical data with 'Unknown'
        for col in ['platform', 'sentiment', 'location', 'media_type', 'influencer_brand', 'post_type']:
            if col in df.columns:
                df[col] = df[col].fillna('Unknown').astype(str)

        # Filter out rows where 'date' could not be parsed
        df.dropna(subset=['date'], inplace=True)

        return df
    except Exception as e:
        st.error(f"Error processing CSV: {e}")
        return pd.DataFrame() # Return empty DataFrame on error

def get_insights(chart_type, data):
    """Generates top 3 insights for a given chart type and data."""
    insights = []
    if data.empty:
        return ["No data available to generate insights for this chart."]

    if chart_type == 'Sentiment Breakdown':
        sentiment_counts = data['sentiment'].value_counts(normalize=True) * 100
        if not sentiment_counts.empty:
            insights.append(f"The most dominant sentiment is '{sentiment_counts.index[0]}' with {sentiment_counts.iloc[0]:.2f}% of posts.")
        if len(sentiment_counts) > 1:
            insights.append(f"The second most common sentiment is '{sentiment_counts.index[1]}' representing {sentiment_counts.iloc[1]:.2f}% of posts.")
        if len(sentiment_counts) > 2:
            insights.append(f"The least common sentiment among the top three is '{sentiment_counts.index[-1]}' with {sentiment_counts.iloc[-1]:.2f}%.")
        elif len(sentiment_counts) == 2:
            insights.append(f"The second sentiment, '{sentiment_counts.index[1]}', is notably less frequent than the dominant one.")

    elif chart_type == 'Engagement Trend over time':
        engagements_by_date = data.groupby(data['date'].dt.date)['engagements'].sum().sort_index()
        if not engagements_by_date.empty:
            peak_date = engagements_by_date.idxmax()
            peak_engagements = engagements_by_date.max()
            insights.append(f"Peak engagement occurred on {peak_date} with {peak_engagements} total engagements, indicating a significant event or campaign around that time.")

            lowest_date = engagements_by_date.idxmin()
            lowest_engagements = engagements_by_date.min()
            insights.append(f"Lowest engagement occurred on {lowest_date} with {lowest_engagements} total engagements, potentially due to low activity or off-peak periods.")

            if len(engagements_by_date) >= 2:
                first_engagement = engagements_by_date.iloc[0]
                last_engagement = engagements_by_date.iloc[-1]
                if last_engagement > first_engagement * 1.1:
                    insights.append("Overall, there appears to be an increasing trend in engagements over the analyzed period.")
                elif last_engagement < first_engagement * 0.9:
                    insights.append("Overall, there appears to be a decreasing trend in engagements over the analyzed period.")
                else:
                    insights.append("Engagements show a relatively stable trend over time.")
        else:
            insights.append("Not enough data points to determine a clear engagement trend.")

    elif chart_type == 'Platform Engagements':
        engagements_by_platform = data.groupby('platform')['engagements'].sum().nlargest(3)
        if not engagements_by_platform.empty:
            insights.append(f"The platform '{engagements_by_platform.index[0]}' generates the highest engagement with {engagements_by_platform.iloc[0]} total engagements, making it the most effective channel.")
        if len(engagements_by_platform) > 1:
            insights.append(f"'{engagements_by_platform.index[1]}' is the second highest platform, indicating its significant contribution to overall engagement.")
        if len(engagements_by_platform) > 2:
            insights.append(f"The top three platforms ('{engagements_by_platform.index[0]}', '{engagements_by_platform.index[1]}', '{engagements_by_platform.index[2]}') collectively capture a large majority of total engagements.")
        elif not engagements_by_platform.empty:
            insights.append("Engagement is heavily concentrated on a limited number of platforms.")

    elif chart_type == 'Media Type Mix':
        media_type_counts = data['media_type'].value_counts(normalize=True) * 100
        if not media_type_counts.empty:
            insights.append(f"'{media_type_counts.index[0]}' is the most frequently used media type, accounting for {media_type_counts.iloc[0]:.2f}% of content.")
        if len(media_type_counts) > 1:
            insights.append(f"The second most common media type is '{media_type_counts.index[1]}', suggesting its importance in content strategy.")
        if len(media_type_counts) > 2:
            insights.append(f"There's a diverse mix of media types, but the top three ('{media_type_counts.index[0]}', '{media_type_counts.index[1]}', '{media_type_counts.index[2]}') dominate content creation.")
        elif not media_type_counts.empty:
            insights.append("The content strategy appears focused on a few primary media types.")

    elif chart_type == 'Top 5 Locations':
        engagements_by_location = data.groupby('location')['engagements'].sum().nlargest(5)
        if not engagements_by_location.empty:
            insights.append(f"The top location by engagement is '{engagements_by_location.index[0]}' with {engagements_by_location.iloc[0]} total engagements, highlighting a key geographic market.")
        if len(engagements_by_location) > 1:
            insights.append(f"'{engagements_by_location.index[1]}' is the second most engaging location, indicating strong audience presence there.")
        if len(engagements_by_location) > 2:
            insights.append(f"The top locations show concentrated engagement, suggesting specific regional marketing efforts could be highly effective.")
        elif not engagements_by_location.empty:
            insights.append("Engagement is highly concentrated in a very small number of locations.")
    
    return insights if insights else ["No specific insights available for this chart type."]


def generate_our_model_analysis(data):
    """Generates summary and recommendations based on built-in logic."""
    if data.empty:
        return "No data available to generate a summary.", []

    summary_parts = []
    recommendations = []

    total_engagements = data['engagements'].sum()
    summary_parts.append(f"Analyzed a total of {len(data)} posts with {total_engagements} engagements.")

    # Sentiment
    sentiment_counts = data['sentiment'].value_counts(normalize=True) * 100
    if not sentiment_counts.empty:
        dominant_sentiment = sentiment_counts.index[0]
        summary_parts.append(f"The dominant sentiment is '{dominant_sentiment}' ({sentiment_counts.iloc[0]:.1f}%).")
        if dominant_sentiment == 'Positive' and sentiment_counts.iloc[0] > 60:
            recommendations.append("Leverage positive sentiment: Focus on replicating strategies from high-performing positive content. Consider user-generated content campaigns showcasing positive experiences.")
        elif dominant_sentiment == 'Negative' and sentiment_counts.iloc[0] > 30:
            recommendations.append("Address negative sentiment: Actively monitor and respond to negative feedback. Investigate root causes and consider a proactive PR strategy.")
        elif dominant_sentiment == 'Neutral' and sentiment_counts.iloc[0] > 50:
            recommendations.append("Boost engagement for neutral content: Experiment with more emotive language, compelling visuals, and clear calls to action to shift neutral sentiment towards positive.")

    # Top Platform
    engagements_by_platform = data.groupby('platform')['engagements'].sum().sort_values(ascending=False)
    if not engagements_by_platform.empty:
        top_platform = engagements_by_platform.index[0]
        summary_parts.append(f"'{top_platform}' is the highest engaging platform, contributing {engagements_by_platform.iloc[0]} engagements.")
        recommendations.append(f"Optimize for '{top_platform}': Allocate more resources to content creation and advertising on this platform, as it currently delivers the highest engagement.")
        if len(engagements_by_platform) > 1 and engagements_by_platform.iloc[0] > engagements_by_platform.iloc[1] * 2:
            recommendations.append(f"Explore underperforming platforms: Investigate why platforms like '{engagements_by_platform.index[1]}' have significantly lower engagement compared to the top performer. Could there be an audience mismatch or content style issue?")

    # Top Media Type
    media_type_counts = data['media_type'].value_counts().sort_values(ascending=False)
    if not media_type_counts.empty:
        top_media_type = media_type_counts.index[0]
        summary_parts.append(f"'{top_media_type}' is the most frequently used media type.")
        recommendations.append(f"Double down on '{top_media_type}': Since this is the most used media type, ensure its quality is top-notch and explore variations within this format.")
        if len(media_type_counts) > 1 and media_type_counts.iloc[0] > media_type_counts.iloc[1] * 1.5:
            recommendations.append("Diversify media types: If your content is heavily skewed towards one media type, consider experimenting with other formats to reach different audience segments or cater to varied consumption preferences.")

    # Engagement Trend
    engagements_by_date = data.groupby(data['date'].dt.date)['engagements'].sum().sort_index()
    if len(engagements_by_date) >= 2:
        first_engagement = engagements_by_date.iloc[0]
        last_engagement = engagements_by_date.iloc[-1]
        if last_engagement > first_engagement * 1.1:
            summary_parts.append("Engagements show an increasing trend over time.")
            recommendations.append("Capitalize on growth: Identify factors contributing to the increasing engagement trend (e.g., successful campaigns, trending topics) and scale those efforts.")
        elif last_engagement < first_engagement * 0.9:
            summary_parts.append("Engagements show a decreasing trend over time.")
            recommendations.append("Reverse declining trends: Analyze periods of low engagement to understand potential causes (e.g., content fatigue, competitive activity) and devise strategies to re-engage the audience.")
        else:
            summary_parts.append("Engagements show a relatively stable trend over time.")

    summary_text = " ".join(summary_parts)
    if not summary_text:
        summary_text = "No specific summary could be generated based on the current data."
    if not recommendations:
        recommendations.append("No specific campaign recommendations could be generated at this time. Consider uploading more data or adjusting filters.")

    return summary_text, recommendations

def generate_openrouter_analysis(data_sample, api_key, model_name):
    """
    Generates summary and recommendations using OpenRouter AI.
    """
    if not api_key:
        return "", [], "Please enter your OpenRouter API Key."
    if data_sample.empty:
        return "", [], "No data available to send to AI for analysis."

    # Convert DataFrame sample to a list of dictionaries for JSON serialization
    data_list = data_sample.to_dict(orient='records')

    # Limit data to 50 rows for brevity and API limits
    limited_data = data_list[:50]

    prompt = f"""
    You are an expert media intelligence analyst.
    I will provide you with cleaned social media data. Each entry represents a post with the following details:
    {json.dumps(list(data_sample.columns), indent=2)}

    Analyze the following data and provide a concise overall summary of the media performance and specific campaign recommendations to optimize future strategies.
    Your response MUST be in JSON format, with two keys: "summary" (a string) and "recommendations" (an array of strings).

    Here is a sample of the data (first {len(limited_data)} rows):
    {json.dumps(limited_data, indent=2)}
    """

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://media-intelligence-dashboard.streamlit.app", # Replace with your actual app URL if deployed
        "X-Title": "Interactive Media Intelligence Dashboard"
    }

    payload = {
        "model": model_name,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "response_format": {"type": "json_object"}
    }

    try:
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=60)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        result = response.json()

        ai_content = result.get('choices', [])[0].get('message', {}).get('content', '{}')
        parsed_content = json.loads(ai_content)
        
        summary = parsed_content.get('summary', 'AI did not provide a summary.')
        recommendations = parsed_content.get('recommendations', ['AI did not provide recommendations.'])

        return summary, recommendations, None # No error

    except requests.exceptions.Timeout:
        return "", [], "OpenRouter AI request timed out. Please try again."
    except requests.exceptions.RequestException as e:
        error_message = f"OpenRouter AI request failed: {e}. Check API key and network."
        if hasattr(e, 'response') and e.response is not None:
            try:
                error_details = e.response.json()
                error_message += f" Details: {error_details.get('message', error_details)}"
            except json.JSONDecodeError:
                error_message += f" Raw response: {e.response.text}"
        return "", [], error_message
    except json.JSONDecodeError as e:
        return "", [], f"Failed to parse AI response as JSON: {e}. Raw content: {ai_content}"
    except Exception as e:
        return "", [], f"An unexpected error occurred during AI analysis: {e}"


def create_pdf_report(summary_text, recommendations):
    """Generates a basic text-based PDF report of the summary and recommendations."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()

    # Custom style for title
    title_style = ParagraphStyle(
        name='TitleStyle',
        fontName='Helvetica-Bold',
        fontSize=24,
        leading=28,
        alignment=TA_CENTER,
        spaceAfter=20,
    )

    # Custom style for section headers
    heading_style = ParagraphStyle(
        name='HeadingStyle',
        fontName='Helvetica-Bold',
        fontSize=16,
        leading=18,
        alignment=TA_LEFT,
        spaceAfter=10,
        spaceBefore=20,
    )

    # Custom style for body text
    body_style = ParagraphStyle(
        name='BodyStyle',
        fontName='Helvetica',
        fontSize=12,
        leading=14,
        alignment=TA_LEFT,
        spaceAfter=5,
    )

    flowables = []

    flowables.append(Paragraph("Media Intelligence Report", title_style))
    flowables.append(Paragraph("---", styles['Normal'])) # Separator
    flowables.append(Spacer(1, 12))

    flowables.append(Paragraph("Overall Summary:", heading_style))
    flowables.append(Paragraph(summary_text, body_style))
    flowables.append(Spacer(1, 24))

    flowables.append(Paragraph("Campaign Recommendations:", heading_style))
    if recommendations:
        for i, rec in enumerate(recommendations):
            flowables.append(Paragraph(f"{i+1}. {rec}", body_style))
    else:
        flowables.append(Paragraph("No specific recommendations were provided.", body_style))
    flowables.append(Spacer(1, 24))

    flowables.append(Paragraph("Powered by Gemini AI", body_style))
    flowables.append(Paragraph("Copyright Media Intelligence Vokasi UI", body_style))

    try:
        doc.build(flowables)
        buffer.seek(0)
        return buffer
    except Exception as e:
        st.error(f"Error generating PDF: {e}")
        return None

# --- Streamlit App Layout ---

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;500;600;700&display=swap');
    html, body, [class*="st-"] {
        font-family: 'Montserrat', sans-serif;
    }
    .stApp {
        background-color: #f3f4f6; /* Gray-100 */
    }
    .main-header {
        text-align: center;
        padding: 1.5rem;
        background-color: #ffffff;
        border-radius: 0.75rem; /* rounded-xl */
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05); /* shadow-lg */
        margin-bottom: 2rem;
        color: #1f2937; /* Gray-800 */
    }
    .content-card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 0.75rem; /* rounded-xl */
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06); /* shadow-md */
        margin-bottom: 2rem;
        border: 1px solid #e5e7eb; /* border-gray-200 */
    }
    .section-header {
        font-size: 1.5rem; /* text-2xl */
        font-weight: 600; /* font-semibold */
        color: #1f2937; /* Gray-800 */
        margin-bottom: 1rem;
    }
    .text-gray-600 {
        color: #4b5563;
    }
    .font-mono {
        font-family: monospace;
    }
    .bg-gray-200 {
        background-color: #e5e7eb;
    }
    .px-1 { padding-left: 0.25rem; padding-right: 0.25rem; }
    .py-0\.5 { padding-top: 0.125rem; padding-bottom: 0.125rem; }
    .rounded { border-radius: 0.25rem; }
    .text-sm { font-size: 0.875rem; }
    .block { display: block; }
    .w-full { width: 100%; }
    .text-gray-900 { color: #111827; }
    .border-gray-300 { border-color: #d1d5db; }
    .rounded-lg { border-radius: 0.5rem; }
    .cursor-pointer { cursor: pointer; }
    .bg-gray-50 { background-color: #f9fafb; }
    .focus\:outline-none:focus { outline: 2px solid transparent; outline-offset: 2px; }
    .file\:mr-4 file\:py-2 file\:px-4 file\:rounded-md file\:border-0 file\:text-sm file\:font-semibold file\:bg-blue-50 file\:text-blue-700 hover\:file\:bg-blue-100 {
        /* Custom file input styling */
    }
    .text-blue-600 { color: #2563eb; }
    .text-green-600 { color: #059669; }
    .text-red-600 { color: #dc2626; }
    .text-yellow-600 { color: #f59e0b; }

    /* Custom styles for buttons */
    .stButton>button {
        padding: 0.75rem 1.5rem; /* px-6 py-3 */
        border-radius: 0.375rem; /* rounded-md */
        font-weight: 500; /* font-medium */
        transition: all 0.2s ease-in-out;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05); /* shadow-sm */
        border: none; /* remove default button border */
    }
    .stButton>button.blue-btn {
        background-color: #2563eb; /* bg-blue-600 */
        color: white;
    }
    .stButton>button.blue-btn:hover {
        background-color: #1d4ed8; /* hover:bg-blue-700 */
    }
    .stButton>button.purple-btn {
        background-color: #9333ea; /* bg-purple-600 */
        color: white;
    }
    .stButton>button.purple-btn:hover {
        background-color: #7e22ce; /* hover:bg-purple-700 */
    }
    .stButton>button.gray-btn {
        background-color: #e5e7eb; /* bg-gray-200 */
        color: #4b5563; /* text-gray-700 */
    }
    .stButton>button.gray-btn:hover {
        background-color: #d1d5db; /* hover:bg-gray-300 */
    }
    .stButton>button.red-btn {
        background-color: #dc2626; /* bg-red-600 */
        color: white;
    }
    .stButton>button.red-btn:hover {
        background-color: #b91c1c; /* hover:bg-red-700 */
    }
    .stButton>button.green-btn {
        background-color: #10B981; /* bg-green-600 */
        color: white;
    }
    .stButton>button.green-btn:hover {
        background-color: #059669; /* hover:bg-green-700 */
    }
    /* Animation for loading spinner */
    .animate-spin {
        animation: spin 1s linear infinite;
    }
    @keyframes spin {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-header'>Interactive Media Intelligence Dashboard</h1>", unsafe_allow_html=True)

# --- State Management (using st.session_state) ---
if 'original_data' not in st.session_state:
    st.session_state.original_data = pd.DataFrame()
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = pd.DataFrame()
if 'filtered_data' not in st.session_state:
    st.session_state.filtered_data = pd.DataFrame()
if 'our_model_summary' not in st.session_state:
    st.session_state.our_model_summary = ""
if 'our_model_recommendations' not in st.session_state:
    st.session_state.our_model_recommendations = []
if 'ai_generated_summary' not in st.session_state:
    st.session_state.ai_generated_summary = ""
if 'ai_generated_recommendations' not in st.session_state:
    st.session_state.ai_generated_recommendations = []
if 'current_analysis_source' not in st.session_state:
    st.session_state.current_analysis_source = 'our_model'
if 'openrouter_api_key' not in st.session_state:
    st.session_state.openrouter_api_key = ""
if 'openrouter_selected_model' not in st.session_state:
    st.session_state.openrouter_selected_model = 'openai/gpt-3.5-turbo'


# --- Step 1: Upload CSV File ---
st.markdown("<div class='content-card'>", unsafe_allow_html=True)
st.markdown("<h2 class='section-header'>1. Upload your CSV file</h2>", unsafe_allow_html=True)
st.markdown("""
    <p class='text-gray-600 mb-4'>Please upload a CSV file with the following columns:
    <code class='font-mono bg-gray-200 px-1 py-0.5 rounded text-sm'>Date</code>,
    <code class='font-mono bg-gray-200 px-1 py-0.5 rounded text-sm'>Platform</code>,
    <code class='font-mono bg-gray-200 px-1 py-0.5 rounded text-sm'>Sentiment</code>,
    <code class='font-mono bg-gray-200 px-1 py-0.5 rounded text-sm'>Location</code>,
    <code class='font-mono bg-gray-200 px-1 py-0.5 rounded text-sm'>Engagements</code>,
    <code class='font-mono bg-gray-200 px-1 py-0.5 rounded text-sm'>Media Type</code>,
    <code class='font-mono bg-gray-200 px-1 py-0.5 rounded text-sm'>Influencer Brand</code>,
    <code class='font-mono bg-gray-200 px-1 py-0.5 rounded text-sm'>Post Type</code>.
    </p>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Select CSV File:", type=["csv"], key="csv_uploader")

if uploaded_file is not None:
    # Check if a new file is uploaded or existing file is re-uploaded
    if st.session_state.get('last_uploaded_file_id') != uploaded_file.file_id:
        with st.spinner("Processing file..."):
            processed_df = parse_csv_and_clean_data(uploaded_file)
            st.session_state.original_data = pd.read_csv(uploaded_file) # Store original for count display
            st.session_state.processed_data = processed_df
            st.session_state.filtered_data = processed_df.copy() # Initialize filtered data
            st.session_state.last_uploaded_file_id = uploaded_file.file_id
            st.session_state.our_model_summary = "" # Clear previous summaries
            st.session_state.our_model_recommendations = []
            st.session_state.ai_generated_summary = ""
            st.session_state.ai_generated_recommendations = []
            st.session_state.current_analysis_source = 'our_model' # Reset analysis source to default

        if not st.session_state.processed_data.empty:
            st.success(f"File uploaded and parsed successfully. Records found: {len(st.session_state.original_data)}. Valid records after cleaning: {len(st.session_state.processed_data)}.")
        else:
            st.error("Error parsing CSV file. Please ensure it is correctly formatted or has valid data.")

st.markdown("</div>", unsafe_allow_html=True)

# --- Step 2: Data Cleaning Summary ---
st.markdown("<div class='content-card'>", unsafe_allow_html=True)
st.markdown("<h2 class='section-header'>2. Data Cleaning Summary</h2>", unsafe_allow_html=True)
st.markdown("""
    <p class='text-gray-600'>
    The uploaded data is automatically cleaned and prepared for visualization:
    </p>
    <ul class='list-disc list-inside text-gray-600 mt-2 space-y-1'>
        <li>The 'Date' column is converted to proper datetime objects for accurate trend analysis.</li>
        <li>Any missing 'Engagements' values are automatically set to zero to ensure all data points are included.</li>
        <li>Column names are internally normalized for consistent processing across all charts.</li>
        <li>Rows with invalid or unparseable dates are filtered out to maintain data integrity.</li>
    </ul>
""", unsafe_allow_html=True)

if not st.session_state.processed_data.empty:
    st.markdown(f"<p class='mt-4 text-green-600 font-medium'>Data cleaned. Valid records for charting: {len(st.session_state.processed_data)} (out of {len(st.session_state.original_data)} original records).</p>", unsafe_allow_html=True)
elif st.session_state.original_data.empty and uploaded_file is not None: # only show if file was attempted
    st.markdown("<p class='mt-4 text-yellow-600 font-medium'>No valid records found after cleaning. Please ensure your CSV has correctly formatted dates and engagements.</p>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)


# --- Data Filters Section ---
if not st.session_state.processed_data.empty:
    st.markdown("<div class='content-card'>", unsafe_allow_html=True)
    st.markdown("<h2 class='section-header'>Data Filters</h2>", unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    # Date Range Filter
    min_date = st.session_state.processed_data['date'].min()
    max_date = st.session_state.processed_data['date'].max()

    with col1:
        st.write("Start Date:")
        # Provide default values only if min_date and max_date are not NaT
        default_start_date = min_date if pd.notna(min_date) else pd.to_datetime('2020-01-01')
        default_end_date = max_date if pd.notna(max_date) else pd.to_datetime('2024-12-31')

        start_date_filter = st.date_input(
            "Start Date",
            value=default_start_date,
            min_value=min_date if pd.notna(min_date) else None,
            max_value=max_date if pd.notna(max_date) else None,
            key="start_date_filter"
        )
    with col2:
        st.write("End Date:")
        end_date_filter = st.date_input(
            "End Date",
            value=default_end_date,
            min_value=min_date if pd.notna(min_date) else None,
            max_value=max_date if pd.notna(max_date) else None,
            key="end_date_filter"
        )

    # Convert date_input to pandas datetime for filtering
    if start_date_filter:
        start_date_filter = pd.to_datetime(start_date_filter)
    if end_date_filter:
        end_date_filter = pd.to_datetime(end_date_filter)

    # Filter options for selectboxes
    platform_options = ['All'] + sorted(st.session_state.processed_data['platform'].unique().tolist())
    sentiment_options = ['All'] + sorted(st.session_state.processed_data['sentiment'].unique().tolist())
    location_options = ['All'] + sorted(st.session_state.processed_data['location'].unique().tolist())
    media_type_options = ['All'] + sorted(st.session_state.processed_data['media_type'].unique().tolist())

    col1_select, col2_select, col3_select, col4_select = st.columns(4)

    with col1_select:
        selected_platform = st.selectbox("Platform:", platform_options, key="platform_select")
    with col2_select:
        selected_sentiment = st.selectbox("Sentiment:", sentiment_options, key="sentiment_select")
    with col3_select:
        selected_location = st.selectbox("Location:", location_options, key="location_select")
    with col4_select:
        selected_media_type = st.selectbox("Media Type:", media_type_options, key="media_type_select")

    # Apply filters
    filtered_df = st.session_state.processed_data.copy()

    if start_date_filter and end_date_filter:
        filtered_df = filtered_df[
            (filtered_df['date'] >= start_date_filter) &
            (filtered_df['date'] <= end_date_filter)
        ]
    elif start_date_filter:
        filtered_df = filtered_df[filtered_df['date'] >= start_date_filter]
    elif end_date_filter:
        filtered_df = filtered_df[filtered_df['date'] <= end_date_filter]

    if selected_platform != 'All':
        filtered_df = filtered_df[filtered_df['platform'] == selected_platform]
    if selected_sentiment != 'All':
        filtered_df = filtered_df[filtered_df['sentiment'] == selected_sentiment]
    if selected_location != 'All':
        filtered_df = filtered_df[filtered_df['location'] == selected_location]
    if selected_media_type != 'All':
        filtered_df = filtered_df[filtered_df['media_type'] == selected_media_type]

    st.session_state.filtered_data = filtered_df

    if st.button("Reset Filters", key="reset_filters_btn", help="Clear all filter selections"):
        st.session_state.processed_data = parse_csv_and_clean_data(uploaded_file) # Re-process original to reset
        st.session_state.filtered_data = st.session_state.processed_data.copy()
        # Reset selectbox values, date inputs need to be explicitly set or refreshed
        st.session_state.platform_select = 'All'
        st.session_state.sentiment_select = 'All'
        st.session_state.location_select = 'All'
        st.session_state.media_type_select = 'All'
        st.experimental_rerun() # Rerun to update date inputs and selectboxes

    st.markdown(f"<p class='mt-4 text-blue-600 font-medium'>Showing {len(st.session_state.filtered_data)} records after applying filters.</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# --- Step 3 & 4: Interactive Charts and Insights ---
st.markdown("<div class='content-card'>", unsafe_allow_html=True)
st.markdown("<h2 class='section-header'>3. Interactive Charts & 4. Top 3 Insights</h2>", unsafe_allow_html=True)

if st.session_state.filtered_data.empty:
    st.info("Upload a CSV file and apply filters to see the interactive charts and their insights.")
else:
    # Chart 1: Sentiment Breakdown
    st.markdown("### Sentiment Breakdown")
    sentiment_data = st.session_state.filtered_data['sentiment'].value_counts().reset_index()
    sentiment_data.columns = ['Sentiment', 'Count']
    fig1 = px.pie(
        sentiment_data,
        values='Count',
        names='Sentiment',
        title='Sentiment Breakdown',
        hole=0.4,
        color='Sentiment',
        color_discrete_map={
            'Positive': CHART_COLORS['sentiment'][0],
            'Neutral': CHART_COLORS['sentiment'][1],
            'Negative': CHART_COLORS['sentiment'][2]
        }
    )
    fig1.update_traces(textinfo='percent+label', marker=dict(line=dict(color='#000', width=1)))
    fig1.update_layout(font_family="Montserrat", title_x=0.5, margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig1, use_container_width=True)
    st.markdown("<div class='mt-4 p-4 bg-gray-50 rounded-md border border-gray-200'>", unsafe_allow_html=True)
    st.markdown("<h4 class='text-lg font-medium text-gray-700 mb-2'>Top 3 Insights:</h4>", unsafe_allow_html=True)
    for insight in get_insights('Sentiment Breakdown', st.session_state.filtered_data):
        st.markdown(f"<p class='text-gray-600'>{insight}</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


    # Chart 2: Engagement Trend over time
    st.markdown("### Engagement Trend over time")
    engagements_by_date = st.session_state.filtered_data.groupby(st.session_state.filtered_data['date'].dt.date)['engagements'].sum().reset_index()
    engagements_by_date.columns = ['Date', 'Total Engagements']
    fig2 = px.line(
        engagements_by_date,
        x='Date',
        y='Total Engagements',
        title='Engagement Trend over time',
        markers=True,
        line_shape='spline',
        color_discrete_sequence=[CHART_COLORS['primary']]
    )
    fig2.update_layout(font_family="Montserrat", title_x=0.5, margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown("<div class='mt-4 p-4 bg-gray-50 rounded-md border border-gray-200'>", unsafe_allow_html=True)
    st.markdown("<h4 class='text-lg font-medium text-gray-700 mb-2'>Top 3 Insights:</h4>", unsafe_allow_html=True)
    for insight in get_insights('Engagement Trend over time', st.session_state.filtered_data):
        st.markdown(f"<p class='text-gray-600'>{insight}</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


    # Chart 3: Platform Engagements
    st.markdown("### Platform Engagements")
    platform_engagements = st.session_state.filtered_data.groupby('platform')['engagements'].sum().reset_index()
    platform_engagements.columns = ['Platform', 'Total Engagements']
    platform_engagements = platform_engagements.sort_values('Total Engagements', ascending=False)
    fig3 = px.bar(
        platform_engagements,
        x='Platform',
        y='Total Engagements',
        title='Platform Engagements',
        color_discrete_sequence=[CHART_COLORS['secondary']]
    )
    fig3.update_layout(font_family="Montserrat", title_x=0.5, margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig3, use_container_width=True)
    st.markdown("<div class='mt-4 p-4 bg-gray-50 rounded-md border border-gray-200'>", unsafe_allow_html=True)
    st.markdown("<h4 class='text-lg font-medium text-gray-700 mb-2'>Top 3 Insights:</h4>", unsafe_allow_html=True)
    for insight in get_insights('Platform Engagements', st.session_state.filtered_data):
        st.markdown(f"<p class='text-gray-600'>{insight}</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


    # Chart 4: Media Type Mix
    st.markdown("### Media Type Mix")
    media_type_data = st.session_state.filtered_data['media_type'].value_counts().reset_index()
    media_type_data.columns = ['Media Type', 'Count']
    fig4 = px.pie(
        media_type_data,
        values='Count',
        names='Media Type',
        title='Media Type Mix',
        hole=0.4,
        color='Media Type',
        color_discrete_sequence=CHART_COLORS['media_type']
    )
    fig4.update_traces(textinfo='percent+label', marker=dict(line=dict(color='#000', width=1)))
    fig4.update_layout(font_family="Montserrat", title_x=0.5, margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig4, use_container_width=True)
    st.markdown("<div class='mt-4 p-4 bg-gray-50 rounded-md border border-gray-200'>", unsafe_allow_html=True)
    st.markdown("<h4 class='text-lg font-medium text-gray-700 mb-2'>Top 3 Insights:</h4>", unsafe_allow_html=True)
    for insight in get_insights('Media Type Mix', st.session_state.filtered_data):
        st.markdown(f"<p class='text-gray-600'>{insight}</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


    # Chart 5: Top 5 Locations
    st.markdown("### Top 5 Locations by Engagement")
    location_engagements = st.session_state.filtered_data.groupby('location')['engagements'].sum().nlargest(5).reset_index()
    location_engagements.columns = ['Location', 'Total Engagements']
    fig5 = px.bar(
        location_engagements,
        x='Location',
        y='Total Engagements',
        title='Top 5 Locations by Engagement',
        color_discrete_sequence=[CHART_COLORS['tertiary']]
    )
    fig5.update_layout(font_family="Montserrat", title_x=0.5, margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig5, use_container_width=True)
    st.markdown("<div class='mt-4 p-4 bg-gray-50 rounded-md border border-gray-200'>", unsafe_allow_html=True)
    st.markdown("<h4 class='text-lg font-medium text-gray-700 mb-2'>Top 3 Insights:</h4>", unsafe_allow_html=True)
    for insight in get_insights('Top 5 Locations', st.session_state.filtered_data):
        st.markdown(f"<p class='text-gray-600'>{insight}</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)


# --- Data Summary & Campaign Recommendations ---
if not st.session_state.filtered_data.empty:
    st.markdown("<div class='content-card'>", unsafe_allow_html=True)
    st.markdown("<h2 class='section-header'>Data Summary & Campaign Recommendations</h2>", unsafe_allow_html=True)

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("Analysis from Us", key="our_model_analysis_btn", help="Generate summary and recommendations from our built-in model.",
                     use_container_width=True, type="secondary" if st.session_state.current_analysis_source != 'our_model' else "primary"):
            summary, recommendations = generate_our_model_analysis(st.session_state.filtered_data)
            st.session_state.our_model_summary = summary
            st.session_state.our_model_recommendations = recommendations
            st.session_state.current_analysis_source = 'our_model'
            st.toast("Analysis from Us generated!", icon="✅")

    with col_btn2:
        if st.button("Analysis from OpenRouter AI", key="openrouter_analysis_btn", help="Generate summary and recommendations using OpenRouter AI. Requires API Key.",
                     use_container_width=True, type="secondary" if st.session_state.current_analysis_source != 'openrouter_ai' else "primary"):
            st.session_state.openrouter_analysis_loading = True
            with st.spinner("Generating AI analysis..."):
                summary, recommendations, err = generate_openrouter_analysis(
                    st.session_state.filtered_data,
                    st.session_state.openrouter_api_key,
                    st.session_state.openrouter_selected_model
                )
                st.session_state.ai_generated_summary = summary
                st.session_state.ai_generated_recommendations = recommendations
                if err:
                    st.session_state.openrouter_analysis_error = err
                    st.error(err)
                else:
                    st.session_state.current_analysis_source = 'openrouter_ai'
                    st.session_state.openrouter_analysis_error = "" # Clear previous errors
                    st.toast("AI Analysis generated!", icon="🤖")
            st.session_state.openrouter_analysis_loading = False
            st.rerun() # Rerun to update button state correctly

    st.markdown("<div class='p-4 bg-gray-50 rounded-md border border-gray-200 mb-6'>", unsafe_allow_html=True)
    st.markdown("<h3 class='text-lg font-semibold text-gray-700 mb-3'>OpenRouter AI Configuration</h3>", unsafe_allow_html=True)
    st.session_state.openrouter_api_key = st.text_input(
        "OpenRouter API Key:",
        type="password",
        value=st.session_state.openrouter_api_key,
        placeholder="sk-or-...",
        help="Get your key from https://openrouter.ai/keys",
        key="openrouter_api_key_input"
    )
    st.session_state.openrouter_selected_model = st.selectbox(
        "Select AI Model:",
        options=openRouterModels,
        index=openRouterModels.index(st.session_state.openrouter_selected_model) if st.session_state.openrouter_selected_model in openRouterModels else 0,
        key="openrouter_model_select"
    )
    if st.session_state.get('openrouter_analysis_error'):
        st.markdown(f"<p class='mt-4 text-red-600 font-medium'>{st.session_state.openrouter_analysis_error}</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Display Analysis
    if st.session_state.current_analysis_source == 'our_model':
        st.markdown("<h3 class='text-xl font-semibold text-gray-700 mb-2'>Overall Summary (Our Model):</h3>", unsafe_allow_html=True)
        st.markdown(f"<p class='text-gray-600 leading-relaxed'>{st.session_state.our_model_summary}</p>", unsafe_allow_html=True)
        st.markdown("<h3 class='text-xl font-semibold text-gray-700 mb-2 mt-4'>Campaign Recommendations (Our Model):</h3>", unsafe_allow_html=True)
        st.markdown("<ul class='list-disc list-inside text-gray-600 space-y-2'>", unsafe_allow_html=True)
        if st.session_state.our_model_recommendations:
            for rec in st.session_state.our_model_recommendations:
                st.markdown(f"<li>{rec}</li>", unsafe_allow_html=True)
        else:
            st.markdown("<li>No specific recommendations could be generated with the current data. Try uploading more data or adjusting filters.</li>", unsafe_allow_html=True)
        st.markdown("</ul>", unsafe_allow_html=True)
    elif st.session_state.current_analysis_source == 'openrouter_ai':
        st.markdown("<h3 class='text-xl font-semibold text-gray-700 mb-2'>Overall Summary (OpenRouter AI):</h3>", unsafe_allow_html=True)
        if st.session_state.ai_generated_summary:
            st.markdown(f"<p class='text-gray-600 leading-relaxed'>{st.session_state.ai_generated_summary}</p>", unsafe_allow_html=True)
        else:
            st.markdown("<p class='text-gray-600 leading-relaxed'>Click 'Analysis from OpenRouter AI' to get insights.</p>", unsafe_allow_html=True)

        st.markdown("<h3 class='text-xl font-semibold text-gray-700 mb-2 mt-4'>Campaign Recommendations (OpenRouter AI):</h3>", unsafe_allow_html=True)
        st.markdown("<ul class='list-disc list-inside text-gray-600 space-y-2'>", unsafe_allow_html=True)
        if st.session_state.ai_generated_recommendations:
            for rec in st.session_state.ai_generated_recommendations:
                st.markdown(f"<li>{rec}</li>", unsafe_allow_html=True)
        else:
            st.markdown("<li>No specific recommendations were generated by the AI, or there was an error.</li>", unsafe_allow_html=True)
        st.markdown("</ul>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# --- Download Report Button ---
if not st.session_state.processed_data.empty:
    st.markdown("<div class='mb-8 text-center mt-8'>", unsafe_allow_html=True)
    # Check if a summary and recommendations are available before allowing PDF download
    can_download = (
        (st.session_state.current_analysis_source == 'our_model' and st.session_state.our_model_summary) or
        (st.session_state.current_analysis_source == 'openrouter_ai' and st.session_state.ai_generated_summary)
    )
    if can_download:
        pdf_buffer = None
        current_summary = ""
        current_recommendations = []

        if st.session_state.current_analysis_source == 'our_model':
            current_summary = st.session_state.our_model_summary
            current_recommendations = st.session_state.our_model_recommendations
        elif st.session_state.current_analysis_source == 'openrouter_ai':
            current_summary = st.session_state.ai_generated_summary
            current_recommendations = st.session_state.ai_generated_recommendations

        pdf_buffer = create_pdf_report(current_summary, current_recommendations)

        if pdf_buffer:
            st.download_button(
                label="Download Report (PDF)",
                data=pdf_buffer,
                file_name="media_intelligence_report.pdf",
                mime="application/pdf",
                help="Download the current analysis summary and recommendations as a PDF.",
                key="download_pdf_button",
                use_container_width=False,
                type="primary"
            )
        else:
            st.warning("Could not generate PDF. Please ensure content is available for analysis.")
    else:
        st.info("Generate an analysis (from 'Us' or 'OpenRouter AI') to enable PDF download.")
    st.markdown("</div>", unsafe_allow_html=True)


# --- Branding Footer ---
st.markdown("<div class='text-center text-gray-500 text-sm mt-12 p-4 border-t border-gray-200'>", unsafe_allow_html=True)
st.markdown("<p class='mb-1'>Powered by **Gemini AI**</p>", unsafe_allow_html=True)
st.markdown("<p>Copyright **Media Intelligence Vokasi UI**</p>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

