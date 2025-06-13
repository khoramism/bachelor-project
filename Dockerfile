FROM bachelor-project-hafez-search:latest

RUN pip install qdrant-client==1.14.2

# Run embedding generation during build
RUN python run_embedding.py

# Expose the Streamlit port
EXPOSE 7777

# Command to run the application with persistence
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=7777", "--server.address=0.0.0.0"]