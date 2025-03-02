import os
import google.generativeai as genai
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import requests
import pandas as pd
from tqdm import tqdm
import time
import random
import logging
import anthropic
from langchain_anthropic import ChatAnthropic
from langchain.schema import HumanMessage
from langchain.prompts import ChatPromptTemplate

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class HTMLTextExtractor:
    def __init__(self, max_retries=5, initial_backoff=2, max_backoff=60):
        # Get AI provider from environment variables
        self.ai_provider = os.getenv('AI_PROVIDER', 'gemini').lower()
        
        if self.ai_provider == 'claude':
            # Configure the Claude API with LangChain
            api_key = os.getenv('CLAUDE_API_KEY')
            if not api_key:
                raise ValueError("CLAUDE_API_KEY not found in environment variables")
            
            self.claude_model = os.getenv('CLAUDE_MODEL', 'claude-3-7-sonnet-latest')
            # Initialize LangChain Anthropic client
            self.langchain_claude = ChatAnthropic(
                anthropic_api_key=api_key,
                model=self.claude_model,
                max_tokens=2048
            )
            logger.info(f"Using LangChain with Claude API model {self.claude_model} for text extraction")
        elif self.ai_provider == 'gemini':
            # Configure the Gemini API
            api_key = os.getenv('GEMINI_API_KEY')
            if not api_key:
                raise ValueError("GEMINI_API_KEY not found in environment variables")
                
            genai.configure(api_key=api_key)
            self.gemini_model = os.getenv('GEMINI_MODEL', 'gemini-2.0-flash-lite')
            logger.info(f"Using Gemini API with model {self.gemini_model} for text extraction")
        else:
            raise ValueError(f"Unsupported AI_PROVIDER: {self.ai_provider}. Use 'claude' or 'gemini'")
        
        # Rate limiting and retry parameters
        self.max_retries = max_retries
        self.initial_backoff = initial_backoff
        self.max_backoff = max_backoff
        
        try:
            if self.ai_provider == 'claude':
                # No need to initialize a specific model for Claude
                # The model is specified during LangChain client initialization
                pass
            elif self.ai_provider == 'gemini':
                # List available models
                for m in genai.list_models():
                    if 'generateContent' in m.supported_generation_methods:
                        logger.info(f"Found model: {m.name}")
                
                # Initialize the model with safety settings
                self.model = genai.GenerativeModel(model_name=self.gemini_model,
                                                 safety_settings=[
                                                     {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                                                     {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                                                     {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                                                     {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
                                                 ])
        except Exception as e:
            raise RuntimeError(f"Failed to initialize model: {str(e)}")

    def extract_text_from_html(self, html_content):
        """
        Extract text content from HTML using BeautifulSoup for initial parsing
        and AI model for intelligent text extraction.
        """
        try:
            if pd.isna(html_content):
                return ""
                
            # First pass: Use BeautifulSoup to get raw text
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
                
            # Get text content
            text = soup.get_text()
            
            # Use AI model to extract and clean English text
            prompt = f"""Extract only the English text content from the following, ignoring any code, 
            metadata, or non-English content. Format it as clean, readable text:

            {text}"""
            
            # Implement retry logic with exponential backoff
            retry_count = 0
            backoff_time = self.initial_backoff
            
            while retry_count <= self.max_retries:
                try:
                    if self.ai_provider == 'claude':
                        # Use LangChain with Claude
                        messages = [HumanMessage(content=prompt)]
                        response = self.langchain_claude.invoke(messages)
                        return response.content
                    elif self.ai_provider == 'gemini':
                        response = self.model.generate_content(prompt)
                    
                    if self.ai_provider == 'gemini':
                        if response.prompt_feedback.block_reason:
                            return f"Content was blocked: {response.prompt_feedback.block_reason}"
                        
                        # Fix for the Gemini response structure
                        # Instead of using response.text, access the text content through parts
                        return response.candidates[0].content.parts[0].text
                except Exception as e:
                    error_message = str(e)
                    
                    # Check if it's a rate limit error
                    rate_limit_error = False
                    if self.ai_provider == 'gemini' and "429" in error_message and "Resource has been exhausted" in error_message:
                        rate_limit_error = True
                    elif self.ai_provider == 'claude' and ("rate_limit" in error_message.lower() or "429" in error_message):
                        rate_limit_error = True
                        
                    if rate_limit_error:
                        retry_count += 1
                        
                        if retry_count > self.max_retries:
                            logger.warning(f"Max retries exceeded for rate limit. Final error: {error_message}")
                            return f"Error extracting text after {self.max_retries} retries: {error_message}"
                        
                        # Add jitter to backoff time to prevent synchronized retries
                        jitter = random.uniform(0.1, 0.5) * backoff_time
                        sleep_time = backoff_time + jitter
                        
                        logger.info(f"Rate limit hit. Retrying in {sleep_time:.2f} seconds (attempt {retry_count}/{self.max_retries})")
                        time.sleep(sleep_time)
                        
                        # Exponential backoff with cap
                        backoff_time = min(backoff_time * 2, self.max_backoff)
                    else:
                        # For non-rate-limit errors, don't retry
                        return f"Error extracting text: {error_message}"
            
            # If we get here, we've exceeded retries
            return f"Error extracting text after {self.max_retries} retries: Rate limit exceeded"
        except Exception as e:
            return f"Error extracting text: {str(e)}"

    def process_csv(self, input_csv_path, output_csv_path, batch_size=10, batch_delay=2):
        """
        Process a CSV file containing HTML content in the 'text' column
        and output a new CSV with extracted English text.
        
        Args:
            input_csv_path: Path to input CSV file
            output_csv_path: Path to output CSV file
            batch_size: Number of items to process before pausing
            batch_delay: Delay in seconds between batches to avoid rate limits
        """
        try:
            # Read the CSV file
            logger.info(f"Reading CSV file: {input_csv_path}")
            
            df = pd.read_csv(input_csv_path)
            
            if 'text' not in df.columns:
                raise ValueError("CSV file must contain a 'text' column")
            
            # Process each row with a progress bar, but in batches
            logger.info("Processing HTML content...")
            
            # Create a new column for extracted text
            df['extracted_text'] = ""
            
            # Process in batches to avoid rate limits
            for i in tqdm(range(0, len(df), batch_size)):
                batch = df.iloc[i:i+batch_size]
                
                # Process each item in the batch
                for idx, row in batch.iterrows():
                    extracted_text = self.extract_text_from_html(row['text'])
                    df.at[idx, 'extracted_text'] = extracted_text
                
                # Save intermediate results after each batch
                df.to_csv(output_csv_path, index=False)
                
                # Add delay between batches to avoid rate limits
                if i + batch_size < len(df):
                    # Use different batch delays based on AI provider
                    if self.ai_provider == 'claude':
                        actual_delay = 0  # No delay for Claude
                    else:  # For Gemini and any other providers
                        actual_delay = batch_delay
                        
                    logger.info(f"Processed batch {i//batch_size + 1}/{(len(df)-1)//batch_size + 1}. Pausing for {actual_delay} seconds...")
                    time.sleep(actual_delay)
            
            # Replace the original text column with extracted text
            df['text'] = df['extracted_text']
            df = df.drop(columns=['extracted_text'])
            
            logger.info(f"Saving final results to: {output_csv_path}")
            df.to_csv(output_csv_path, index=False)
            logger.info("Processing complete!")
            
            return True
        except Exception as e:
            logger.error(f"Error processing CSV: {str(e)}")
            return False

def main():
    extractor = HTMLTextExtractor(max_retries=5, initial_backoff=2, max_backoff=60)
    
    # Example usage with CSV files
    input_csv = "input.csv"  # Replace with your input CSV file path
    
    # Add timestamp to output filename
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_csv = f"{timestamp}_output.csv"  # Output CSV with timestamp
    
    # Set batch delay based on AI provider
    if extractor.ai_provider == 'claude':
        batch_delay = 0  # No delay for Claude
    else:  # For Gemini and any other providers
        batch_delay = 3  # Use larger delay for Gemini
    
    logger.info("Starting CSV processing...")
    success = extractor.process_csv(input_csv, output_csv, batch_size=5, batch_delay=batch_delay)
    
    if success:
        logger.info(f"Successfully processed {input_csv} and saved results to {output_csv}")
    else:
        logger.error("Failed to process CSV file")

if __name__ == "__main__":
    main()