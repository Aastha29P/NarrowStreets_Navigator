import os
from pinecone import Pinecone, ServerlessSpec
from langchain_openai.embeddings import OpenAIEmbeddings
import openai
import pandas as pd
import json
import random
import time
import logging
import backoff
from openai import RateLimitError, OpenAIError

# Set up API keys
openai_api_key = "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"  # Replace with your OpenAI key
pinecone_api_key = "pcsk_3DWssQ_Snki6xhka6gTHhyNMxRNVQvu1zfnihLRFDcNqXvZTys74o54CZsoWXgqGVyvus2"  # Replace with your Pinecone API key
pinecone_environment = "us-east-1"  # Replace with your Pinecone environment region

# Set the OpenAI API key in the environment variable
os.environ["OPENAI_API_KEY"] = openai_api_key

# Create a Pinecone instance
pc = Pinecone(api_key=pinecone_api_key)

# Check if the index exists; if not, create one
index_name = "indian-tourist-destinations"
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,  # Example dimension for OpenAI embeddings
        metric="cosine",  # Metric to measure similarity
        spec=ServerlessSpec(cloud="aws", region=pinecone_environment)
    )

# Connect to the index
index = pc.Index(index_name)

# Debug: Ensure `index` is not None
if index is None:
    raise ValueError("Failed to initialize or connect to Pinecone index.")

print(f"Connected to Pinecone index: {index_name}")

# Load your CSV data
file_path = "Top_Indian_Places_to_Visit.csv"  # Replace with the actual CSV file path
df = pd.read_csv(file_path)


# Combine relevant columns into a single text representation
df["combined_text"] = df.apply(
    lambda row: f"{row['Zone']}, {row['State']}, {row['City']}, {row['Name']}, {row['Type']}, {row['Significance']}, Best time to visit: {row['Best Time to visit']}",
    axis=1
)

# Create embeddings for the data and upload to Pinecone
embeddings = OpenAIEmbeddings()  # Automatically picks up the API key from the environment
for i, row in df.iterrows():
    text = row["combined_text"]
    embedding = embeddings.embed_query(text)
    index.upsert([
        {"id": str(i), "values": embedding, "metadata": {"name": row["Name"]}}  # Format for Pinecone upsert
    ])

# Retrieval function
def retrieve_relevant_data(user_query):
    query_embedding = embeddings.embed_query(user_query)  # Create the query embedding
    results = index.query(
        vector=query_embedding,  # Use the query embedding
        top_k=5,                # Number of top matches to retrieve
        include_metadata=True   # Include metadata in the results
    )
    return results

# Example usage
user_query = "Best places to visit in Delhi during the morning"
results = retrieve_relevant_data(user_query)

# Display the results
print("Top results:")
for result in results["matches"]:
    destination_name = result["metadata"]["name"]  # Retrieve the destination name
    print(result['id'], destination_name, result['score'])



test_query = [0.1] * 1536  # Mock query vector of appropriate dimension
test_results = index.query(vector=test_query, top_k=5, include_metadata=True)
print(test_results)



csv_file = "Top_Indian_Places_to_Visit.csv"  # Replace with your actual file path
data = pd.read_csv(csv_file)

# Define prompt and completion templates
prompt_templates = [
    "Tell me about {Name} in {City}.",
    "What is significant about {Name} in {City}?",
    "Where can I experience {Significance} significance in {City}?",
    "Which {Type} in {City} is worth visiting?",
    "What makes {Name} in {City} special?",
    "Suggest a {Significance} destination in {State}.",
]

completion_templates = [
    "{Name} is a {Type} in {City}, built in {Establishment Year}. It has a Google review rating of {Google review rating} and is best visited in the {Best Time to visit}.",
    "{Name} in {City} is known for its {Significance}. It was established in {Establishment Year} and has a rating of {Google review rating}. Ideal visit time: {Best Time to visit}.",
    "For {Significance} experiences in {City}, {Name} is a top choice. Built in {Establishment Year}, it has a Google review rating of {Google review rating}.",
    "{Name} in {City} is a {Type} rated {Google review rating} on Google. It's best visited in the {Best Time to visit}.",
    "Located in {City}, {Name} is a popular {Significance} destination built in {Establishment Year}. It is rated {Google review rating} on Google reviews.",
]


# Function to create varied prompts and completions
def create_varied_prompt_completion(row):
    # Safely access and preprocess row data
    row_data = {col: (str(row[col]).strip() if pd.notna(row[col]) else "") for col in data.columns}
    row_data = {k: v.lower() if isinstance(v, str) else v for k, v in row_data.items()}  # Convert strings to lowercase safely
    try:
        prompt = random.choice(prompt_templates).format(**row_data)
        completion = random.choice(completion_templates).format(**row_data)
    except KeyError as e:
        # Handle missing columns gracefully
        print(f"KeyError: Missing data for {e}")
        return None
    return {"prompt": prompt, "completion": completion}

# Generate prompts and completions
json_data = [create_varied_prompt_completion(row) for _, row in data.iterrows()]
json_data = [entry for entry in json_data if entry]  # Remove None entries

# Save the data to a JSON file
output_file = "output.json"
with open(output_file, "w") as file:
    for entry in json_data:
        file.write(json.dumps(entry) + "\n")

print(f"Varied JSON data has been written to {output_file}")


mv output.json output.jsonl



if not openai_api_key:
    logging.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
    print("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
    exit(1)
openai.api_key = openai_api_key

# Step 1: Upload the JSONL file
def upload_file(file_path):
    try:
        with open(file_path, "rb") as file:
            response = openai.File.create(
                file=file,
                purpose="fine-tune"
            )
        logging.info("File uploaded successfully.")
        logging.info(f"File ID: {response['id']}")
        return response["id"]
    except OpenAIError as e:
        logging.error(f"OpenAI API error during file upload: {e}")
    except Exception as e:
        logging.error(f"Unexpected error during file upload: {e}")
    return None

# Step 2: Start fine-tuning
def start_fine_tuning(training_file_id, model="gpt-3.5-turbo", n_epochs=4, learning_rate_multiplier=0.1):
    try:
        fine_tune_response = openai.FineTunes.create(
            training_file=training_file_id,
            model=model,
            n_epochs=n_epochs,
            learning_rate_multiplier=learning_rate_multiplier
        )
        logging.info("Fine-tuning started.")
        logging.info(f"Fine-tune ID: {fine_tune_response['id']}")
        return fine_tune_response["id"]
    except OpenAIError as e:
        logging.error(f"OpenAI API error during fine-tuning start: {e}")
    except Exception as e:
        logging.error(f"Unexpected error during fine-tuning start: {e}")
    return None

# Step 3: Check the fine-tuning status with backoff for rate limits
@backoff.on_exception(backoff.expo,
                      (RateLimitError, OpenAIError),
                      max_time=300)
def check_fine_tune_status(fine_tune_id):
    try:
        status_response = openai.FineTunes.retrieve(id=fine_tune_id)
        logging.info(f"Fine-tune status: {status_response['status']}")
        print("Fine-tune status:")
        print(json.dumps(status_response, indent=2))
        return status_response
    except OpenAIError as e:
        logging.error(f"OpenAI API error during status check: {e}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error during status check: {e}")
        raise

# Step 4: Use the fine-tuned model
def use_fine_tuned_model(model_name, prompt):
    try:
        response = openai.ChatCompletion.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=100,
            temperature=0.7
        )
        generated_text = response.choices[0].message["content"].strip()
        logging.info(f"Generated response: {generated_text}")
        print("Generated response:")
        print(generated_text)
    except OpenAIError as e:
        logging.error(f"OpenAI API error during model usage: {e}")
    except Exception as e:
        logging.error(f"Unexpected error during model usage: {e}")

# Main workflow
def main():
    print("Starting fine-tuning workflow...")
    logging.info("Starting fine-tuning workflow...")
    
    # Replace with your JSONL file path
    file_path = "output.jsonl"
    
    if not os.path.exists(file_path):
        logging.error(f"File {file_path} does not exist. Please provide a valid file path.")
        print(f"File {file_path} does not exist. Please provide a valid file path.")
        return
    
    training_file_id = upload_file(file_path)

    if not training_file_id:
        logging.error("Training file upload failed. Exiting.")
        print("Training file upload failed. Check the logs for more details.")
        return

    # Start fine-tuning with a supported model
    fine_tune_id = start_fine_tuning(training_file_id, model="gpt-3.5-turbo")

    if not fine_tune_id:
        logging.error("Fine-tuning initiation failed. Exiting.")
        print("Fine-tuning initiation failed. Check the logs for more details.")
        return

    # Monitor the fine-tuning process
    logging.info("Monitoring fine-tuning process. This may take some time...")
    print("Monitoring fine-tuning process. This may take some time...")

    while True:
        status = check_fine_tune_status(fine_tune_id)
        if status:
            if status["status"] == "succeeded":
                logging.info("Fine-tuning completed successfully.")
                print("Fine-tuning completed successfully.")
                fine_tuned_model = status["fine_tuned_model"]
                break
            elif status["status"] == "failed":
                logging.error("Fine-tuning failed. Check the logs for more details.")
                print("Fine-tuning failed. Check the logs for more details.")
                return
            else:
                logging.info(f"Current status: {status['status']}")
        time.sleep(60)  # Wait for 60 seconds before checking again

    # Use the fine-tuned model
    prompt = "Tell me about India Gate in Delhi."
    use_fine_tuned_model(fine_tuned_model, prompt)

if __name__ == "__main__":
    main()


