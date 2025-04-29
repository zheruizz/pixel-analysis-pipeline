import streamlit as st
import subprocess
import os

st.title("Data Pipeline Interface")

# --- User Inputs ---
uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx"])
measurement_frequency = st.number_input("Measurement Frequency (Hz)", min_value=1, value=50)
breath_cycle = st.number_input("Breath Cycle Duration (seconds)", min_value=1, value=2)
recording_duration = st.number_input("Recording Duration (minutes)", min_value=1, value=15)

# --- Run Button ---
if st.button("Run Pipeline"):
    if uploaded_file is not None:
        # Save the uploaded file to a temporary location
        input_path = os.path.join("temp", uploaded_file.name)
        os.makedirs("temp", exist_ok=True)
        with open(input_path, "wb") as f:
            f.write(uploaded_file.read())

        # Run pipeline.py with arguments
        cmd = [
            "python", "pipeline.py", input_path,
            "--freq", str(measurement_frequency),
            "--cycle", str(breath_cycle),
            "--duration", str(recording_duration)
        ]
        st.write("Running pipeline with the following command:")
        st.code(" ".join(cmd))

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            st.success("Pipeline completed successfully!")
            st.text(result.stdout)
        except subprocess.CalledProcessError as e:
            st.error("Pipeline failed!")
            st.text(e.stderr)
    else:
        st.warning("Please upload an Excel file before running the pipeline.")
