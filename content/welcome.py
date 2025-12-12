"""Welcome page for TOPPView-Lite.

A viewer for mass spectrometry data visualization.
"""

import streamlit as st
from pathlib import Path

from src.common.common import page_setup


# Page setup
page_setup()


def inject_custom_css():
    """Inject custom CSS for the welcome page styling."""
    st.markdown(
        """
        <style>
        /* Hero section styling */
        .hero-section {
            text-align: center;
            margin-bottom: 2rem;
        }

        .hero-title {
            font-size: 2.5rem;
            font-weight: 700;
            color: #29379b;
            margin-bottom: 0.5rem;
        }

        .hero-subtitle {
            font-size: 1.25rem;
            color: #a0a0a0;
            margin-bottom: 2rem;
        }

        /* Feature card styling */
        .feature-card {
            background: linear-gradient(135deg, #262730 0%, #1e1e2e 100%);
            border: 2px solid #444;
            border-radius: 12px;
            padding: 1.5rem;
            text-align: center;
            height: 100%;
            box-shadow: 0 2px 8px rgba(0,0,0,0.3);
        }

        .feature-icon {
            font-size: 2.5rem;
            margin-bottom: 0.75rem;
        }

        .feature-title {
            font-size: 1.1rem;
            font-weight: 600;
            color: #6b7fff;
            margin-bottom: 0.5rem;
        }

        .feature-desc {
            font-size: 0.9rem;
            color: #a0a0a0;
        }

        /* Navigation button styling */
        .nav-button-container {
            display: flex;
            justify-content: center;
            gap: 1rem;
            margin: 2rem 0;
        }

        /* Info box styling */
        .info-box {
            background: linear-gradient(135deg, #1e2a3a 0%, #1a2530 100%);
            border: 1px solid #2a4a6a;
            border-radius: 8px;
            padding: 1rem 1.5rem;
            margin: 1.5rem 0;
        }

        .info-box h4 {
            color: #64b5f6;
            margin-bottom: 0.5rem;
        }

        .info-box p {
            color: #90caf9;
            margin: 0;
        }

        /* Footer styling */
        .footer-text {
            text-align: center;
            color: #a0a0a0;
            font-size: 0.9rem;
        }

        .footer-text a {
            color: #6b7fff;
        }

        /* Download box styling */
        .download-box {
            background: linear-gradient(135deg, #1a2a3a 0%, #152030 100%);
            border: 2px solid #29379b;
            border-radius: 12px;
            padding: 2rem;
            text-align: center;
            margin: 2rem 0;
        }

        .download-box-title {
            font-size: 1.3rem;
            font-weight: 600;
            color: #6b7fff;
            margin-bottom: 0.5rem;
        }

        .download-box-desc {
            color: #a0a0a0;
            margin-bottom: 1.5rem;
        }

        .download-box-hint {
            color: #808080;
            font-size: 0.85rem;
            margin-top: 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def create_navigation_button(emoji, title, subtitle, page_path):
    """Create a styled navigation button."""
    button_key = f"{title.lower().replace(' ', '_')}_nav_btn"

    if st.button(
        label=f"{emoji} {title}",
        key=button_key,
        help=subtitle,
        use_container_width=True,
        type="primary"
    ):
        st.switch_page(page_path)

    # Apply custom styling
    st.markdown(
        f"""
        <style>
        .st-key-{button_key} button {{
            background: linear-gradient(135deg, #262730 0%, #1e1e2e 100%) !important;
            border: 2px solid #444 !important;
            border-radius: 12px !important;
            padding: 2rem 1.5rem !important;
            height: 200px !important;
            min-height: 200px !important;
            box-shadow: 0 2px 8px rgba(0,0,0,0.3) !important;
            transition: all 0.3s ease !important;
            color: #6b7fff !important;
            font-size: 1.25rem !important;
            font-weight: 700 !important;
        }}

        .st-key-{button_key} button p {{
            color: #6b7fff !important;
            font-size: 1.25rem !important;
            font-weight: 700 !important;
        }}

        .st-key-{button_key} button:hover {{
            background: linear-gradient(135deg, #3a4abf 0%, #29379b 100%) !important;
            border-color: #4a5acf !important;
            transform: translateY(-4px) !important;
            box-shadow: 0 8px 24px rgba(41, 55, 155, 0.4) !important;
        }}

        .st-key-{button_key} button:hover p {{
            color: white !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero_section():
    """Render the hero section with title and logo."""
    st.markdown('<div class="hero-section">', unsafe_allow_html=True)

    # Title and logo columns
    spacer1, title_col, logo_col, spacer2 = st.columns([0.5, 4, 1.5, 0.5])

    with title_col:
        st.markdown(
            """
            <h1 class="hero-title">TOPPView-Lite</h1>
            <p class="hero-subtitle">A viewer for mass spectrometry data</p>
            """,
            unsafe_allow_html=True,
        )

    with logo_col:
        logo_path = Path("assets/openms-logo.svg")
        if logo_path.exists():
            st.image(str(logo_path), width=150)

    st.markdown('</div>', unsafe_allow_html=True)


def render_navigation_buttons():
    """Render the main navigation buttons."""
    spacer1, col1, col2, spacer2 = st.columns([1, 2, 2, 1], gap="medium")

    with col1:
        create_navigation_button(
            "📂",
            "Upload Files",
            "Upload and preprocess mzML files",
            "content/upload.py"
        )

    with col2:
        create_navigation_button(
            "👀",
            "View Data",
            "Interactive visualization",
            "content/viewer.py"
        )


def render_windows_download():
    """Render the Windows download section if the installer is available."""
    zip_path = Path("TOPPView-Lite.zip")
    if not zip_path.exists():
        return False

    st.markdown("---")

    spacer1, center_col, spacer2 = st.columns([1, 3, 1])

    with center_col:
        st.markdown(
            """
            <div class="download-box">
                <div class="download-box-title">📥 TOPPView-Lite for Windows</div>
                <div class="download-box-desc">
                    Download the offline version to use without an internet connection.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Create columns for centered button
        btn_spacer1, btn_col, btn_spacer2 = st.columns([1, 2, 1])
        with btn_col:
            with open(zip_path, "rb") as file:
                st.download_button(
                    label="Download for Windows",
                    data=file,
                    file_name="TOPPView-Lite.zip",
                    mime="application/zip",
                    type="primary",
                    use_container_width=True,
                )

        st.markdown(
            """
            <div class="download-box-hint">
                Extract the zip file and run the installer (.msi) to install.
            </div>
            """,
            unsafe_allow_html=True,
        )

    return True


def render_features():
    """Render the features section."""
    st.markdown("---")
    st.markdown("### Features")

    col1, col2, col3, col4 = st.columns(4)

    features = [
        ("🗺️", "Peak Map", "Interactive 2D heatmap with zoom-based resolution"),
        ("📊", "Spectrum View", "Click to view individual mass spectra"),
        ("📋", "Data Tables", "Browse spectra and peaks with sorting/filtering"),
        ("⚡", "Fast Loading", "Preprocessed data for instant visualization"),
    ]

    for col, (icon, title, desc) in zip([col1, col2, col3, col4], features):
        with col:
            st.markdown(
                f"""
                <div class="feature-card">
                    <div class="feature-icon">{icon}</div>
                    <div class="feature-title">{title}</div>
                    <div class="feature-desc">{desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_quick_start():
    """Render the quick start guide."""
    st.markdown("---")
    st.markdown("### Quick Start")

    st.markdown(
        """
        <div class="info-box">
            <h4>Getting Started</h4>
            <p>
                1. <strong>Upload</strong> your mzML files on the Upload page<br>
                2. Preprocess files for fast visualization<br>
                3. Go to the <strong>Viewer</strong> to explore your data interactively
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Supported Data:**")
        st.markdown("""
        - mzML files (MS1 and MS2)
        - FAIMS ion mobility data
        - Large datasets (millions of peaks)
        """)

    with col2:
        st.markdown("**Interactive Features:**")
        st.markdown("""
        - Click heatmap to select spectrum
        - Zoom for higher resolution
        - Cross-linked tables and plots
        """)


def main():
    """Main function to render the welcome page."""
    inject_custom_css()
    render_hero_section()
    render_navigation_buttons()
    render_windows_download()
    render_features()
    render_quick_start()


main()
