import streamlit as st
import psutil

st.title("🖥️ System Health")

cpu = psutil.cpu_percent()

memory = psutil.virtual_memory().percent

disk = psutil.disk_usage('/').percent

st.metric("CPU Usage %", cpu)

st.metric("Memory Usage %", memory)

st.metric("Disk Usage %", disk)