import os
import google.generativeai as genai
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import requests
import pandas as pd
from tqdm import tqdm

# Load environment variables
load_dotenv()

class HTMLTextExtractor:
    def __init__(self):
        # Configure the Gemini API
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
            
        genai.configure(api_key=api_key)
        
        try:
            # List available models
            for m in genai.list_models():
                if 'generateContent' in m.supported_generation_methods:
                    print(f"Found model: {m.name}")
            
            # Initialize the model with safety settings
            self.model = genai.GenerativeModel(model_name='gemini-2.0-flash',
                                             safety_settings=[
                                                 {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                                                 {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                                                 {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                                                 {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
                                             ])
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Gemini model: {str(e)}")

    def extract_text_from_html(self, html_content):
        """
        Extract text content from HTML using BeautifulSoup for initial parsing
        and Gemini for intelligent text extraction.
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
            
            # Use Gemini to extract and clean English text
            prompt = f"""Extract only the English text content from the following, ignoring any code, 
            metadata, or non-English content. Format it as clean, readable text:

            {text}"""
            
            response = self.model.generate_content(prompt)
            
            if response.prompt_feedback.block_reason:
                return f"Content was blocked: {response.prompt_feedback.block_reason}"
                
            return response.text
        except Exception as e:
            return f"Error extracting text: {str(e)}"

    def process_csv(self, input_csv_path, output_csv_path):
        """
        Process a CSV file containing HTML content in the 'text' column
        and output a new CSV with extracted English text.
        """
        try:
            # Read the CSV file
            print(f"Reading CSV file: {input_csv_path}")
            df = pd.read_csv(input_csv_path)
            
            if 'text' not in df.columns:
                raise ValueError("CSV file must contain a 'text' column")
            
            # Process each row with a progress bar
            print("Processing HTML content...")
            tqdm.pandas()
            df['text'] = df['text'].progress_apply(self.extract_text_from_html)
            
            print(f"Saving results to: {output_csv_path}")
            df.to_csv(output_csv_path, index=False)
            print("Processing complete!")
            
            return True
        except Exception as e:
            print(f"Error processing CSV: {str(e)}")
            return False

def main():
    try:
        extractor = HTMLTextExtractor()
        
        # Example usage with CSV files
        input_csv = "input.csv"  # Replace with your input CSV file path
        output_csv = "output.csv"  # Replace with your desired output CSV file path
        
        print("Starting CSV processing...")
        success = extractor.process_csv(input_csv, output_csv)
        
        if success:
            print(f"Successfully processed {input_csv} and saved results to {output_csv}")
        else:
            print("Failed to process CSV file")
            
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 