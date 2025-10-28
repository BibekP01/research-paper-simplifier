"""
PaperFetcher Module for Research Paper Q&A Assistant

This module provides functionality to fetch research papers from arXiv API,
download PDFs, and save metadata.
"""

import arxiv
import json
import logging
import time
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PaperFetcher:
    """
    A class to fetch research papers from arXiv API.
    
    This class provides methods to search for papers by keywords,
    fetch papers by arXiv ID, download PDFs, and save metadata.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize the PaperFetcher with configuration settings.
        
        Args:
            config_path: Path to the configuration YAML file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
        # Set up directories
        self.papers_dir = Path("data/papers")
        self.processed_dir = Path("data/processed")
        self._ensure_directories()
        
        # Configure arXiv client
        self.max_results = self.config.get('arxiv', {}).get('max_results', 10)
        self.categories = self.config.get('arxiv', {}).get('categories', ['cs.AI'])
        
        # Set up requests session with retry logic
        self.session = self._create_session()
        
        logger.info(f"PaperFetcher initialized with max_results={self.max_results}")
    
    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Returns:
            Dictionary containing configuration settings
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If config file is invalid
        """
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {self.config_path}")
            return config
        except FileNotFoundError:
            logger.error(f"Config file not found: {self.config_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Error parsing config file: {e}")
            raise
    
    def _ensure_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        self.papers_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        logger.debug("Ensured data directories exist")
    
    def _create_session(self) -> requests.Session:
        """
        Create a requests session with retry logic for robust downloads.
        
        Returns:
            Configured requests Session object
        """
        session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session
    
    def _sanitize_filename(self, filename: str) -> str:
        """
        Sanitize filename by removing invalid characters.
        
        Args:
            filename: Original filename
            
        Returns:
            Sanitized filename safe for all operating systems
        """
        # Remove or replace invalid characters
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            filename = filename.replace(char, '_')
        
        # Limit length and strip whitespace
        filename = filename.strip()[:200]
        return filename
    
    def search_papers(
        self, 
        query: str, 
        max_results: Optional[int] = None,
        sort_by: arxiv.SortCriterion = arxiv.SortCriterion.Relevance
    ) -> List[Dict[str, Any]]:
        """
        Search for papers on arXiv by keywords.
        
        Args:
            query: Search query string (e.g., "machine learning", "attention mechanism")
            max_results: Maximum number of results to return (uses config default if None)
            sort_by: Sort criterion (Relevance, LastUpdatedDate, SubmittedDate)
            
        Returns:
            List of dictionaries containing paper metadata
            
        Raises:
            Exception: If API request fails
        """
        if max_results is None:
            max_results = self.max_results
        
        logger.info(f"Searching arXiv for: '{query}' (max_results={max_results})")
        
        try:
            # Create search object
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=sort_by
            )
            
            papers = []
            for result in search.results():
                paper_data = self._extract_paper_data(result)
                papers.append(paper_data)
                logger.debug(f"Found paper: {paper_data['title']}")
            
            logger.info(f"Successfully retrieved {len(papers)} papers")
            return papers
            
        except Exception as e:
            logger.error(f"Error searching arXiv: {e}")
            raise
    
    def fetch_by_id(self, arxiv_id: str) -> Dict[str, Any]:
        """
        Fetch a specific paper by its arXiv ID.
        
        Args:
            arxiv_id: arXiv identifier (e.g., "2103.14030" or "arXiv:2103.14030")
            
        Returns:
            Dictionary containing paper metadata
            
        Raises:
            ValueError: If paper not found
            Exception: If API request fails
        """
        # Clean the arXiv ID (remove "arXiv:" prefix if present)
        clean_id = arxiv_id.replace("arXiv:", "").strip()
        
        logger.info(f"Fetching paper by ID: {clean_id}")
        
        try:
            # Search by ID
            search = arxiv.Search(id_list=[clean_id])
            
            # Get the first (and should be only) result
            result = next(search.results(), None)
            
            if result is None:
                raise ValueError(f"Paper with ID {clean_id} not found")
            
            paper_data = self._extract_paper_data(result)
            logger.info(f"Successfully fetched paper: {paper_data['title']}")
            return paper_data
            
        except StopIteration:
            logger.error(f"Paper with ID {clean_id} not found")
            raise ValueError(f"Paper with ID {clean_id} not found")
        except Exception as e:
            logger.error(f"Error fetching paper by ID: {e}")
            raise
    
    def _extract_paper_data(self, result: arxiv.Result) -> Dict[str, Any]:
        """
        Extract relevant data from an arxiv.Result object.
        
        Args:
            result: arxiv.Result object
            
        Returns:
            Dictionary containing paper metadata
        """
        return {
            'arxiv_id': result.entry_id.split('/')[-1],
            'title': result.title,
            'abstract': result.summary,
            'authors': [author.name for author in result.authors],
            'published_date': result.published.strftime('%Y-%m-%d'),
            'updated_date': result.updated.strftime('%Y-%m-%d'),
            'pdf_url': result.pdf_url,
            'categories': result.categories,
            'primary_category': result.primary_category,
            'comment': result.comment,
            'journal_ref': result.journal_ref,
            'doi': result.doi,
            'links': [link.href for link in result.links]
        }
    
    def download_pdf(
        self, 
        paper_url: str, 
        save_path: Optional[Path] = None,
        arxiv_id: Optional[str] = None
    ) -> Path:
        """
        Download a PDF from the given URL.
        
        Args:
            paper_url: URL of the PDF to download
            save_path: Custom path to save the PDF (optional)
            arxiv_id: arXiv ID for filename generation (optional)
            
        Returns:
            Path object pointing to the downloaded PDF
            
        Raises:
            requests.RequestException: If download fails
        """
        try:
            # Generate filename if save_path not provided
            if save_path is None:
                if arxiv_id:
                    filename = f"{arxiv_id.replace('/', '_')}.pdf"
                else:
                    # Extract filename from URL
                    filename = paper_url.split('/')[-1]
                    if not filename.endswith('.pdf'):
                        filename += '.pdf'
                
                filename = self._sanitize_filename(filename)
                save_path = self.papers_dir / filename
            
            logger.info(f"Downloading PDF from {paper_url}")
            
            # Download with timeout and retry logic
            response = self.session.get(
                paper_url, 
                timeout=30,
                stream=True
            )
            response.raise_for_status()
            
            # Save PDF
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            file_size = save_path.stat().st_size / (1024 * 1024)  # Size in MB
            logger.info(f"PDF downloaded successfully: {save_path} ({file_size:.2f} MB)")
            
            return save_path
            
        except requests.RequestException as e:
            logger.error(f"Error downloading PDF: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during PDF download: {e}")
            raise
    
    def save_metadata(
        self, 
        paper_data: Dict[str, Any], 
        filepath: Optional[Path] = None
    ) -> Path:
        """
        Save paper metadata as JSON file.
        
        Args:
            paper_data: Dictionary containing paper metadata
            filepath: Custom path to save the JSON (optional)
            
        Returns:
            Path object pointing to the saved JSON file
            
        Raises:
            IOError: If file writing fails
        """
        try:
            # Generate filepath if not provided
            if filepath is None:
                arxiv_id = paper_data.get('arxiv_id', 'unknown')
                filename = f"{arxiv_id.replace('/', '_')}_metadata.json"
                filename = self._sanitize_filename(filename)
                filepath = self.processed_dir / filename
            
            # Add timestamp
            paper_data['fetched_at'] = datetime.now().isoformat()
            
            # Save as JSON with pretty printing
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(paper_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Metadata saved to: {filepath}")
            return filepath
            
        except IOError as e:
            logger.error(f"Error saving metadata: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error saving metadata: {e}")
            raise
    
    def fetch_and_download(
        self, 
        arxiv_id: str, 
        download_pdf: bool = True
    ) -> Dict[str, Any]:
        """
        Convenience method to fetch paper metadata and optionally download PDF.
        
        Args:
            arxiv_id: arXiv identifier
            download_pdf: Whether to download the PDF (default: True)
            
        Returns:
            Dictionary containing paper metadata with added 'pdf_path' key if downloaded
        """
        # Fetch metadata
        paper_data = self.fetch_by_id(arxiv_id)
        
        # Save metadata
        metadata_path = self.save_metadata(paper_data)
        paper_data['metadata_path'] = str(metadata_path)
        
        # Download PDF if requested
        if download_pdf:
            pdf_path = self.download_pdf(
                paper_data['pdf_url'],
                arxiv_id=paper_data['arxiv_id']
            )
            paper_data['pdf_path'] = str(pdf_path)
        
        return paper_data
    
    def search_and_download(
        self, 
        query: str, 
        max_results: Optional[int] = None,
        download_pdfs: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Search for papers and optionally download them.
        
        Args:
            query: Search query string
            max_results: Maximum number of results
            download_pdfs: Whether to download PDFs (default: True)
            
        Returns:
            List of dictionaries containing paper metadata
        """
        papers = self.search_papers(query, max_results)
        
        for i, paper in enumerate(papers, 1):
            logger.info(f"Processing paper {i}/{len(papers)}: {paper['title'][:50]}...")
            
            # Save metadata
            metadata_path = self.save_metadata(paper)
            paper['metadata_path'] = str(metadata_path)
            
            # Download PDF if requested
            if download_pdfs:
                try:
                    pdf_path = self.download_pdf(
                        paper['pdf_url'],
                        arxiv_id=paper['arxiv_id']
                    )
                    paper['pdf_path'] = str(pdf_path)
                    
                    # Rate limiting to be respectful to arXiv
                    if i < len(papers):
                        time.sleep(3)
                        
                except Exception as e:
                    logger.warning(f"Failed to download PDF for {paper['arxiv_id']}: {e}")
                    paper['pdf_path'] = None
        
        return papers


def main():
    """Example usage of the PaperFetcher class."""
    
    # Initialize fetcher
    fetcher = PaperFetcher()
    
    print("\n" + "="*70)
    print("Research Paper Fetcher - Example Usage")
    print("="*70)
    
    # Example 1: Fetch a specific paper by ID
    print("\n[Example 1] Fetching paper by arXiv ID...")
    try:
        paper = fetcher.fetch_and_download(
            arxiv_id="2103.14030",  # CLIP paper
            download_pdf=True
        )
        print(f"✓ Title: {paper['title']}")
        print(f"✓ Authors: {', '.join(paper['authors'][:3])}...")
        print(f"✓ Published: {paper['published_date']}")
        print(f"✓ PDF saved to: {paper.get('pdf_path', 'N/A')}")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # Example 2: Search for papers by keyword
    print("\n[Example 2] Searching for papers on 'large language models'...")
    try:
        papers = fetcher.search_papers(
            query="large language models",
            max_results=3
        )
        print(f"✓ Found {len(papers)} papers:")
        for i, paper in enumerate(papers, 1):
            print(f"\n  {i}. {paper['title']}")
            print(f"     ID: {paper['arxiv_id']}")
            print(f"     Published: {paper['published_date']}")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # Example 3: Search and download papers
    print("\n[Example 3] Searching and downloading papers on 'attention mechanism'...")
    try:
        papers = fetcher.search_and_download(
            query="attention mechanism in transformers",
            max_results=2,
            download_pdfs=True
        )
        print(f"✓ Downloaded {len(papers)} papers")
        for paper in papers:
            print(f"  - {paper['title'][:60]}...")
            print(f"    PDF: {paper.get('pdf_path', 'Not downloaded')}")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    print("\n" + "="*70)
    print("Examples completed! Check data/papers/ and data/processed/")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
