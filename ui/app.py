import streamlit as st
import requests
import pandas as pd
from bs4 import BeautifulSoup
import os
import time

# --- ✅ MUST BE FIRST STREAMLIT COMMAND ---
st.set_page_config(
    page_title="✨ SHL Assessment Recommender", 
    page_icon="📊", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Get API URL from environment ---
API_URL = os.environ.get("API_URL", "http://localhost:8000")

# --- Custom CSS for Styling ---
st.markdown("""
    <style>
        /* Global Styles */
        body {
            background-color: #f9fafc;
        }
        .main-title {
            text-align: center;
            color: #2b2d42;
            font-weight: 800;
            font-size: 2.4rem;
            margin-bottom: 0.5rem;
        }
        .subtitle {
            text-align: center;
            color: #6c757d;
            font-size: 1.1rem;
            margin-bottom: 2rem;
        }
        .stTextArea textarea, .stTextInput input {
            border-radius: 10px;
            border: 1px solid #ced4da;
            font-size: 1rem;
        }
        .stButton>button {
            background: linear-gradient(90deg, #0072ff, #00c6ff);
            color: white;
            border-radius: 8px;
            padding: 0.6rem 1.2rem;
            font-weight: 600;
            border: none;
            transition: 0.3s;
            width: 100%;
        }
        .stButton>button:hover {
            transform: translateY(-2px);
            background: linear-gradient(90deg, #0056b3, #00a1c9);
            box-shadow: 0 4px 12px rgba(0, 114, 255, 0.3);
        }
        
        /* ✅ FIXED TABLE STYLING - Make text visible */
        .recommend-table {
            margin-top: 2rem;
            border-collapse: collapse;
            width: 100%;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            border-radius: 8px;
            overflow: hidden;
        }
        .recommend-table th {
            background-color: #0072ff !important;
            color: white !important;
            padding: 14px 12px !important;
            text-align: left !important;
            font-weight: 600 !important;
            font-size: 0.95rem !important;
            border: none !important;
        }
        .recommend-table td {
            background-color: #ffffff !important;
            color: #2b2d42 !important;  /* ✅ Dark text color */
            padding: 12px !important;
            border-bottom: 1px solid #e0e0e0 !important;
            font-size: 0.9rem !important;
        }
        .recommend-table tr:nth-child(even) td {
            background-color: #f8f9fa !important;
        }
        .recommend-table tr:hover td {
            background-color: #e8f4ff !important;
        }
        .recommend-table tbody tr:last-child td {
            border-bottom: none !important;
        }
        
        /* ✅ Fix link styling */
        .recommend-table a {
            color: #0072ff !important;
            text-decoration: none !important;
            font-weight: 500 !important;
            display: inline-flex !important;
            align-items: center !important;
            gap: 4px !important;
        }
        .recommend-table a:hover {
            color: #0056b3 !important;
            text-decoration: underline !important;
        }
        
        /* General link styling */
        a {
            color: #0072ff;
            text-decoration: none;
        }
        a:hover {
            text-decoration: underline;
        }
        
        .info-box {
            background-color: #e7f3ff;
            border-left: 4px solid #0072ff;
            padding: 1rem;
            border-radius: 4px;
            margin: 1rem 0;
        }
        .error-box {
            background-color: #ffe7e7;
            border-left: 4px solid #ff4444;
            padding: 1rem;
            border-radius: 4px;
            margin: 1rem 0;
        }
        .success-box {
            background-color: #e7ffe7;
            border-left: 4px solid #44ff44;
            padding: 1rem;
            border-radius: 4px;
            margin: 1rem 0;
        }
        
        /* Hide Streamlit branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown("<h1 class='main-title'>✨ SHL Assessment Recommender</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Enter a job description or provide a job posting URL to get the most relevant SHL assessments tailored for your role.</p>", unsafe_allow_html=True)

# --- Info Section ---
with st.expander("ℹ️ How to use this tool"):
    st.markdown("""
    ### Getting Started
    1. **Option 1**: Paste or type a job description in the text area
    2. **Option 2**: Provide a URL to a job posting (we'll extract the description automatically)
    3. Click **"Get Recommendations"** to receive 5-10 relevant SHL assessments
    
    ### What you'll get
    - Assessment names and descriptions
    - Duration and test types
    - Remote testing and adaptive support information
    - Direct links to assessment details
    
    ### Tips for best results
    - Include specific skills, technologies, and role requirements
    - Mention experience level (entry, mid, senior)
    - Specify any time constraints or test preferences
    """)

# --- Tabs for input methods ---
job_description_tab, url_tab = st.tabs(["📝 Job Description", "🔗 Job URL"])

job_description = ""
url = ""

with job_description_tab:
    job_description = st.text_area(
        "Enter Job Description:",
        height=200,
        placeholder="Example: We are seeking a Senior Java Developer with 5+ years of experience in Spring Boot, microservices, and SQL. The candidate should have strong problem-solving skills and experience with cloud platforms like AWS...",
        help="Paste the complete job description here. The more detailed, the better the recommendations!"
    )

with url_tab:
    url = st.text_input(
        "Enter Job Description URL:",
        placeholder="https://example.com/job-posting",
        help="Provide a direct link to a job posting. We'll extract the description automatically."
    )
    
    if url:
        with st.spinner('🔍 Extracting job description from URL...'):
            try:
                # Add timeout to prevent hanging
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                }
                page = requests.get(url, headers=headers, timeout=10)
                page.raise_for_status()  # Raise exception for bad status codes
                
                soup = BeautifulSoup(page.content, 'html.parser')
                
                # Remove script and style elements
                for script in soup(["script", "style", "nav", "footer", "header"]):
                    script.decompose()
                
                # Get text
                job_description = soup.get_text(separator=' ', strip=True)
                
                # Clean up whitespace
                job_description = ' '.join(job_description.split())
                
                if len(job_description) > 50:
                    st.success(f"✅ Successfully extracted {len(job_description)} characters from the URL!")
                    with st.expander("📄 Preview extracted content"):
                        st.text_area("Extracted Job Description", job_description[:1000] + "..." if len(job_description) > 1000 else job_description, height=200, disabled=True)
                else:
                    st.error("❌ Failed to extract meaningful content from the URL. Please try the job description tab instead.")
                    job_description = ""
                    
            except requests.exceptions.Timeout:
                st.error("❌ Request timeout. The website took too long to respond. Please try again or use the job description tab.")
                job_description = ""
            except requests.exceptions.RequestException as e:
                st.error(f"❌ Failed to fetch the URL: {str(e)}")
                st.info("💡 Try copying and pasting the job description manually in the other tab.")
                job_description = ""
            except Exception as e:
                st.error(f"❌ Error parsing job description from URL: {str(e)}")
                st.info("💡 Try copying and pasting the job description manually in the other tab.")
                job_description = ""

# --- Add some spacing ---
st.markdown("<br>", unsafe_allow_html=True)

# --- Recommendation Button ---
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    recommend_button = st.button("🚀 Get Recommendations", use_container_width=True)

if recommend_button:
    if not job_description.strip() and not url.strip():
        st.error("⚠️ Please enter a job description or provide a valid URL.")
    elif len(job_description.strip()) < 20:
        st.error("⚠️ Job description is too short. Please provide more details for better recommendations.")
    else:
        # Show loading message
        with st.spinner('🔍 Analyzing job description and fetching SHL recommendations...'):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Update progress
                status_text.text("📊 Processing job description...")
                progress_bar.progress(20)
                
                # Make API request
                status_text.text("🤖 Querying recommendation engine...")
                progress_bar.progress(40)
                
                response = requests.post(
                    f"{API_URL}/recommend",
                    json={"job_description": job_description},
                    timeout=300  # 5 minutes timeout
                )
                
                progress_bar.progress(80)
                status_text.text("📦 Processing recommendations...")

                if response.status_code == 200:
                    try:
                        response_json = response.json()
                        recommendations = response_json.get("recommendations", [])
                        
                        progress_bar.progress(100)
                        status_text.empty()
                        progress_bar.empty()

                        if recommendations:
                            st.success(f"✅ Found {len(recommendations)} relevant SHL assessments!")
                            
                            # Add summary metrics
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("📊 Total Assessments", len(recommendations))
                            with col2:
                                avg_duration = "N/A"
                                durations = [r.get('duration', '') for r in recommendations if r.get('duration') and r.get('duration') != 'N/A']
                                if durations:
                                    # Extract numeric values and calculate average
                                    numeric_durations = []
                                    for d in durations:
                                        try:
                                            nums = [int(s) for s in d.split() if s.isdigit()]
                                            if nums:
                                                numeric_durations.append(nums[0])
                                        except:
                                            pass
                                    if numeric_durations:
                                        avg_duration = f"{sum(numeric_durations) // len(numeric_durations)} min"
                                st.metric("⏱️ Avg Duration", avg_duration)
                            with col3:
                                remote_count = sum(1 for r in recommendations if r.get('remote_testing_support', '').lower() == 'yes')
                                st.metric("🌐 Remote Testing", f"{remote_count}/{len(recommendations)}")

                            # Convert to DataFrame
                            df = pd.DataFrame(recommendations)
                            
                            # Reorder and rename columns
                            column_mapping = {
                                "name": "Assessment Name",
                                "url": "URL",
                                "duration": "Duration",
                                "test_types": "Test Types",
                                "remote_testing_support": "Remote Testing",
                                "adaptive_irt_support": "Adaptive/IRT",
                            }
                            
                            # Keep only relevant columns
                            df = df[[col for col in column_mapping.keys() if col in df.columns]]
                            df = df.rename(columns=column_mapping)

                            # Format test_types
                            if 'Test Types' in df.columns:
                                df['Test Types'] = df['Test Types'].apply(
                                    lambda x: ', '.join(x) if isinstance(x, list) else str(x)
                                )

                            # Make URLs clickable with icon
                            if 'URL' in df.columns:
                                df['URL'] = df['URL'].apply(
                                    lambda x: f'<a href="{x}" target="_blank">🔗 View Details</a>' if x else 'N/A'
                                )

                            # Display table with custom styling
                            st.markdown("### 📋 Recommended Assessments")
                            st.markdown(
                                df.to_html(escape=False, index=False, classes='recommend-table'),
                                unsafe_allow_html=True
                            )

                            # Download button
                            st.markdown("<br>", unsafe_allow_html=True)
                            
                            # Prepare CSV (with original URLs, not HTML)
                            df_csv = pd.DataFrame(recommendations)
                            df_csv = df_csv[[col for col in column_mapping.keys() if col in df_csv.columns]]
                            df_csv = df_csv.rename(columns=column_mapping)
                            if 'Test Types' in df_csv.columns:
                                df_csv['Test Types'] = df_csv['Test Types'].apply(
                                    lambda x: ', '.join(x) if isinstance(x, list) else str(x)
                                )
                            
                            csv = df_csv.to_csv(index=False).encode('utf-8')
                            col1, col2, col3 = st.columns([1, 2, 1])
                            with col2:
                                st.download_button(
                                    label="📥 Download Recommendations as CSV",
                                    data=csv,
                                    file_name="shl_recommendations.csv",
                                    mime="text/csv",
                                    use_container_width=True
                                )
                        else:
                            st.warning("⚠️ No matching assessments found. Try adding more details about the job role, required skills, and experience level.")
                            st.info("💡 **Tips for better results:**\n- Include specific technologies or skills\n- Mention the seniority level\n- Specify any time or format preferences")
                            
                    except ValueError as ve:
                        progress_bar.empty()
                        status_text.empty()
                        st.error("⚠️ Error processing recommendations. The response format was unexpected.")
                        with st.expander("🔍 Debug Info"):
                            st.code(response.text[:500])
                    except Exception as e:
                        progress_bar.empty()
                        status_text.empty()
                        st.error(f"⚠️ Error processing recommendations: {str(e)}")
                        
                elif response.status_code == 504:
                    progress_bar.empty()
                    status_text.empty()
                    st.error("⏰ Request timeout. The backend service took too long to respond.")
                    st.info("💡 This can happen with complex job descriptions. Try:\n- Shortening the description\n- Using more specific keywords\n- Trying again in a moment")
                    
                else:
                    progress_bar.empty()
                    status_text.empty()
                    st.error(f"⚠️ Backend error (Status {response.status_code}). Please try again later.")
                    if response.text:
                        with st.expander("🔍 Error Details"):
                            st.code(response.text[:500])
                            
            except requests.exceptions.Timeout:
                progress_bar.empty()
                status_text.empty()
                st.error("⏰ Request timeout. The recommendation service took too long to respond.")
                st.info("💡 The service might be starting up (cold start). Please try again in a moment.")
                
            except requests.exceptions.ConnectionError:
                progress_bar.empty()
                status_text.empty()
                st.error("🚫 Connection error: Unable to reach the recommendation service.")
                st.info(f"💡 Please check if the backend is running at: {API_URL}")
                
            except requests.exceptions.RequestException as e:
                progress_bar.empty()
                status_text.empty()
                st.error(f"🚫 Network error: {str(e)}")
                st.info("💡 Please check your internet connection and try again.")
                
            except Exception as e:
                progress_bar.empty()
                status_text.empty()
                st.error(f"⚠️ Unexpected error: {str(e)}")
                with st.expander("🔍 Technical Details"):
                    st.code(str(e))

# --- Footer ---
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("---")

# Footer with additional info
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("""
        <div style='text-align: center; color: #6c757d;'>
            <p>💡 Built for SHL Recommendation System Assignment</p>
            <p>Powered by FastAPI + Streamlit + Google Gemini + Qdrant</p>
        </div>
    """, unsafe_allow_html=True)

# --- Sidebar (Optional) ---
with st.sidebar:
    st.markdown("### ℹ️ About")
    st.markdown("""
    This tool uses advanced AI to match job descriptions with relevant SHL assessments.
    
    **Features:**
    - 🤖 AI-powered analysis
    - 🔍 Semantic search
    - 📊 1000+ assessments
    - ⚡ Real-time recommendations
    """)
    
    st.markdown("### 🔧 Configuration")
    st.code(f"API: {API_URL}", language="text")
    
    st.markdown("### 📞 Support")
    st.markdown("""
    Having issues?
    - Check API connection
    - Verify backend status
    - Review error messages
    """)