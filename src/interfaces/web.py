"""Streamlit web interface for Local RAG Assistant."""

import streamlit as st
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.core.pipeline import RAGPipeline
from src.utils.config import load_config
from src.utils.logging import setup_logging


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Local RAG Assistant",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 Local RAG Assistant")
    st.subtitle("Offline Document Q&A System")
    
    # Initialize pipeline
    if "pipeline" not in st.session_state:
        try:
            config = load_config()
            st.session_state.pipeline = RAGPipeline(config)
            st.session_state.pipeline.initialize_pipeline()
            st.success("✅ RAG Pipeline initialized successfully!")
        except Exception as e:
            st.error(f"❌ Failed to initialize pipeline: {e}")
            st.stop()
    
    # Chat interface
    st.header("💬 Ask Questions")
    
    # Query input
    query = st.text_input("Enter your question:", placeholder="What would you like to know?")
    
    if st.button("🔍 Search") and query:
        with st.spinner("Searching documents..."):
            try:
                response = st.session_state.pipeline.query(query)
                
                st.subheader("📝 Answer")
                st.write(response.get("answer", "No answer generated"))
                
                if "sources" in response and response["sources"]:
                    st.subheader("📚 Sources")
                    for i, source in enumerate(response["sources"], 1):
                        with st.expander(f"Source {i}: {source.get('title', 'Unknown')}"):
                            st.write(f"**Path:** {source.get('path', 'N/A')}")
                            st.write(f"**Score:** {source.get('score', 0):.3f}")
                            if 'content' in source:
                                st.write("**Content:**")
                                st.write(source['content'])
                
            except Exception as e:
                st.error(f"❌ Error processing query: {e}")
    
    # Sidebar with stats
    with st.sidebar:
        st.header("📊 Statistics")
        try:
            retriever = st.session_state.pipeline.retriever
            doc_count = retriever.get_document_count()
            st.metric("Documents Indexed", doc_count)
        except:
            st.metric("Documents Indexed", "N/A")


if __name__ == "__main__":
    main()
