"""Traffic Analytics Dashboard using Streamlit."""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime, timedelta
from sqlalchemy import func

from src.core.config import get_settings
from src.core.database import Detection, LicensePlate, Vehicle, VideoSource, get_db_manager


# Page configuration
st.set_page_config(
    page_title="ALPR Traffic Analytics",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize
settings = get_settings()
db_manager = get_db_manager()


def load_data():
    """Load data from database."""
    db = db_manager.get_session()

    try:
        # Load all data
        videos_df = pd.read_sql(db.query(VideoSource).statement, db.bind)
        detections_df = pd.read_sql(db.query(Detection).statement, db.bind)
        vehicles_df = pd.read_sql(db.query(Vehicle).statement, db.bind)
        plates_df = pd.read_sql(db.query(LicensePlate).statement, db.bind)

        return videos_df, detections_df, vehicles_df, plates_df
    finally:
        db.close()


def get_summary_stats(videos_df, detections_df, vehicles_df, plates_df):
    """Calculate summary statistics."""
    stats = {
        "total_videos": len(videos_df),
        "total_detections": len(detections_df),
        "total_vehicles": len(vehicles_df),
        "unique_plates": plates_df["plate_number"].nunique() if len(plates_df) > 0 else 0,
        "detection_rate": (
            (detections_df["plate_text"].notna().sum() / len(detections_df) * 100)
            if len(detections_df) > 0
            else 0
        ),
        "avg_confidence": (
            detections_df["plate_text_confidence"].mean() if len(detections_df) > 0 else 0
        ),
    }
    return stats


def plot_vehicle_count_over_time(detections_df):
    """Plot vehicle count over time."""
    if len(detections_df) == 0:
        return None

    # Group by frame and count unique vehicle IDs
    detections_df["timestamp"] = pd.to_datetime(
        detections_df["detection_time"], errors="coerce"
    )

    # Resample by hour
    hourly_counts = (
        detections_df.set_index("timestamp")
        .resample("H")["vehicle_id"]
        .nunique()
        .reset_index()
    )
    hourly_counts.columns = ["timestamp", "vehicle_count"]

    fig = px.line(
        hourly_counts,
        x="timestamp",
        y="vehicle_count",
        title="Vehicle Traffic Over Time",
        labels={"timestamp": "Time", "vehicle_count": "Number of Vehicles"},
    )

    fig.update_layout(hovermode="x unified")
    return fig


def plot_vehicle_type_distribution(vehicles_df):
    """Plot vehicle type distribution."""
    if len(vehicles_df) == 0:
        return None

    type_counts = vehicles_df["vehicle_type"].value_counts()

    fig = px.pie(
        values=type_counts.values,
        names=type_counts.index,
        title="Vehicle Type Distribution",
        hole=0.4,
    )

    return fig


def plot_detection_rate_by_hour(detections_df):
    """Plot detection rate by hour of day."""
    if len(detections_df) == 0:
        return None

    detections_df["timestamp"] = pd.to_datetime(
        detections_df["detection_time"], errors="coerce"
    )
    detections_df["hour"] = detections_df["timestamp"].dt.hour

    # Calculate detection rate by hour
    hourly_stats = (
        detections_df.groupby("hour")
        .agg(
            total=("id", "count"),
            detected=("plate_text", lambda x: x.notna().sum()),
        )
        .reset_index()
    )

    hourly_stats["detection_rate"] = (
        hourly_stats["detected"] / hourly_stats["total"] * 100
    )

    fig = px.bar(
        hourly_stats,
        x="hour",
        y="detection_rate",
        title="Plate Detection Rate by Hour",
        labels={"hour": "Hour of Day", "detection_rate": "Detection Rate (%)"},
    )

    fig.update_layout(xaxis=dict(tickmode="linear", tick0=0, dtick=1))
    return fig


def plot_top_plates(plates_df):
    """Plot most frequently detected plates."""
    if len(plates_df) == 0:
        return None

    top_plates = plates_df.nlargest(10, "detection_count")[
        ["plate_number", "detection_count"]
    ]

    fig = px.bar(
        top_plates,
        x="plate_number",
        y="detection_count",
        title="Top 10 Most Frequently Detected Plates",
        labels={"plate_number": "License Plate", "detection_count": "Detection Count"},
    )

    return fig


def plot_confidence_distribution(detections_df):
    """Plot confidence score distribution."""
    if len(detections_df) == 0:
        return None

    # Filter out null confidences
    conf_data = detections_df[detections_df["plate_text_confidence"].notna()]

    if len(conf_data) == 0:
        return None

    fig = px.histogram(
        conf_data,
        x="plate_text_confidence",
        nbins=50,
        title="OCR Confidence Score Distribution",
        labels={"plate_text_confidence": "Confidence Score", "count": "Frequency"},
    )

    return fig


def plot_daily_statistics(vehicles_df):
    """Plot daily traffic statistics."""
    if len(vehicles_df) == 0:
        return None

    vehicles_df["date"] = pd.to_datetime(vehicles_df["first_seen_time"]).dt.date

    daily_stats = vehicles_df.groupby("date").agg(
        vehicle_count=("id", "count"),
    ).reset_index()

    fig = px.bar(
        daily_stats,
        x="date",
        y="vehicle_count",
        title="Daily Vehicle Count",
        labels={"date": "Date", "vehicle_count": "Number of Vehicles"},
    )

    return fig


def main():
    """Main dashboard function."""
    # Header
    st.title("🚗 ALPR Traffic Analytics Dashboard")
    st.markdown("---")

    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Page",
        [
            "Overview",
            "Traffic Analysis",
            "License Plates",
            "Vehicle Details",
            "Performance Metrics",
        ],
    )

    # Load data
    with st.spinner("Loading data..."):
        videos_df, detections_df, vehicles_df, plates_df = load_data()

    # Calculate summary stats
    stats = get_summary_stats(videos_df, detections_df, vehicles_df, plates_df)

    # Overview Page
    if page == "Overview":
        st.header("📊 System Overview")

        # KPI Cards
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Videos Processed", stats["total_videos"])

        with col2:
            st.metric("Total Detections", f"{stats['total_detections']:,}")

        with col3:
            st.metric("Unique Vehicles", stats["total_vehicles"])

        with col4:
            st.metric("Unique License Plates", stats["unique_plates"])

        col5, col6 = st.columns(2)

        with col5:
            st.metric(
                "Plate Detection Rate",
                f"{stats['detection_rate']:.1f}%",
            )

        with col6:
            st.metric(
                "Average OCR Confidence",
                f"{stats['avg_confidence']:.2f}",
            )

        st.markdown("---")

        # Recent Activity
        st.subheader("📹 Recent Videos")

        if len(videos_df) > 0:
            recent_videos = videos_df.nlargest(5, "upload_date")[
                ["filename", "upload_date", "duration", "frame_count", "processed"]
            ]
            st.dataframe(recent_videos, use_container_width=True)
        else:
            st.info("No videos processed yet.")

    # Traffic Analysis Page
    elif page == "Traffic Analysis":
        st.header("📈 Traffic Analysis")

        # Vehicle count over time
        st.subheader("Vehicle Traffic Trends")
        fig = plot_vehicle_count_over_time(detections_df)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data available for traffic trends.")

        col1, col2 = st.columns(2)

        with col1:
            # Vehicle type distribution
            st.subheader("Vehicle Type Distribution")
            fig = plot_vehicle_type_distribution(vehicles_df)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No vehicle type data available.")

        with col2:
            # Daily statistics
            st.subheader("Daily Vehicle Count")
            fig = plot_daily_statistics(vehicles_df)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No daily statistics available.")

        # Detection rate by hour
        st.subheader("Detection Performance by Hour")
        fig = plot_detection_rate_by_hour(detections_df)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No hourly data available.")

    # License Plates Page
    elif page == "License Plates":
        st.header("🔢 License Plate Analysis")

        # Search functionality
        st.subheader("Search License Plates")
        search_query = st.text_input("Enter license plate number", "")

        if search_query:
            results = plates_df[
                plates_df["plate_number"].str.contains(search_query, case=False, na=False)
            ]
            st.write(f"Found {len(results)} results:")
            st.dataframe(results, use_container_width=True)

        # Top plates
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Most Frequently Detected Plates")
            fig = plot_top_plates(plates_df)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No plate data available.")

        with col2:
            st.subheader("Recent Detections")
            if len(plates_df) > 0:
                recent_plates = plates_df.nlargest(10, "detected_at")[
                    ["plate_number", "confidence", "detection_count", "is_valid_format"]
                ]
                st.dataframe(recent_plates, use_container_width=True)
            else:
                st.info("No plate data available.")

        # Plate statistics
        st.subheader("License Plate Statistics")
        if len(plates_df) > 0:
            col1, col2, col3 = st.columns(3)

            with col1:
                valid_count = plates_df["is_valid_format"].sum()
                st.metric("Valid Format Plates", valid_count)

            with col2:
                avg_detections = plates_df["detection_count"].mean()
                st.metric("Avg Detections per Plate", f"{avg_detections:.1f}")

            with col3:
                avg_conf = plates_df["confidence"].mean()
                st.metric("Avg Plate Confidence", f"{avg_conf:.2f}")

    # Vehicle Details Page
    elif page == "Vehicle Details":
        st.header("🚙 Vehicle Details")

        # Filter options
        col1, col2 = st.columns(2)

        with col1:
            video_filter = st.selectbox(
                "Filter by Video",
                ["All"] + videos_df["filename"].tolist() if len(videos_df) > 0 else ["All"],
            )

        with col2:
            type_filter = st.selectbox(
                "Filter by Vehicle Type",
                ["All"] + vehicles_df["vehicle_type"].unique().tolist()
                if len(vehicles_df) > 0
                else ["All"],
            )

        # Apply filters
        filtered_vehicles = vehicles_df.copy()

        if video_filter != "All" and len(videos_df) > 0:
            video_id = videos_df[videos_df["filename"] == video_filter]["id"].iloc[0]
            filtered_vehicles = filtered_vehicles[
                filtered_vehicles["video_source_id"] == video_id
            ]

        if type_filter != "All":
            filtered_vehicles = filtered_vehicles[
                filtered_vehicles["vehicle_type"] == type_filter
            ]

        st.write(f"Showing {len(filtered_vehicles)} vehicles")

        # Display vehicles
        if len(filtered_vehicles) > 0:
            display_df = filtered_vehicles[
                [
                    "track_id",
                    "vehicle_type",
                    "first_seen_frame",
                    "last_seen_frame",
                    "total_frames_detected",
                    "confidence_avg",
                ]
            ]
            st.dataframe(display_df, use_container_width=True)
        else:
            st.info("No vehicles found with the selected filters.")

    # Performance Metrics Page
    elif page == "Performance Metrics":
        st.header("⚡ Performance Metrics")

        # Confidence distribution
        st.subheader("OCR Confidence Distribution")
        fig = plot_confidence_distribution(detections_df)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No confidence data available.")

        # System performance
        st.subheader("System Performance")

        if len(videos_df) > 0 and len(detections_df) > 0:
            col1, col2, col3 = st.columns(3)

            with col1:
                avg_fps = videos_df["fps"].mean()
                st.metric("Average Video FPS", f"{avg_fps:.1f}")

            with col2:
                total_frames = videos_df["frame_count"].sum()
                st.metric("Total Frames Processed", f"{total_frames:,}")

            with col3:
                detection_rate = stats["detection_rate"]
                st.metric("Overall Detection Rate", f"{detection_rate:.1f}%")

        # Model performance table
        st.subheader("Model Performance Summary")

        performance_data = {
            "Metric": [
                "Total Detections",
                "Successful Plate Reads",
                "Detection Rate",
                "Average Confidence",
                "Unique Vehicles Tracked",
                "Unique Plates Recognized",
            ],
            "Value": [
                f"{stats['total_detections']:,}",
                f"{detections_df['plate_text'].notna().sum():,}",
                f"{stats['detection_rate']:.2f}%",
                f"{stats['avg_confidence']:.4f}",
                f"{stats['total_vehicles']:,}",
                f"{stats['unique_plates']:,}",
            ],
        }

        st.table(pd.DataFrame(performance_data))

    # Footer
    st.markdown("---")
    st.markdown(
        "<center>ALPR System v2.0 | Built with Streamlit, FastAPI, YOLOv11, and ❤️</center>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
